import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv, GATv2Conv, GraphUNet
except Exception as e:
    raise ImportError("topogeonet_lite requires torch-geometric. Install with: pip install torch-geometric") from e


class FourierFeatureEncoder(nn.Module):
    """
    Simple Fourier feature mapping for 3D coordinates.
    """
    def __init__(self, input_dim: int = 3, mapping_size: int = 64, scale: float = 10.0, passthrough: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.passthrough = passthrough
        # Random frequencies
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [N, 3]
        proj = 2.0 * math.pi * coords @ self.B  # [N, mapping_size]
        enc = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [N, 2*mapping]
        return torch.cat([coords, enc], dim=-1) if self.passthrough else enc


class TopoGeoNetLite(nn.Module):
    """
    Lightweight TopoGeoNet-inspired model for per-node regression on 3D point clouds.
    - Optional Fourier encoding of data.coords
    - MLP encoder to hidden dim
    - Stack of GNN layers (SAGE or GATv2) with LayerNorm, residuals
    - Optional GraphUNet head
    - Final MLP to output_dim (default 3 for XYZ)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 4,
        layer_type: str = 'sage',  # 'sage' or 'gatv2'
        dropout: float = 0.1,
        use_unet: bool = False,
        use_fourier_features: bool = True,
        fourier_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer_type = layer_type.lower()
        self.dropout = dropout
        self.use_unet = use_unet
        self.use_fourier_features = use_fourier_features

        # Fourier encoder on coords
        fourier_out = 0
        if self.use_fourier_features:
            mapping = fourier_dim if fourier_dim is not None else hidden_dim // 2
            self.fourier = FourierFeatureEncoder(input_dim=3, mapping_size=mapping, passthrough=True)
            # passthrough adds +3, and sin/cos adds 2*mapping
            fourier_out = 3 + 2 * mapping
        else:
            self.fourier = None

        encoder_in = input_dim + (fourier_out if self.use_fourier_features else 0)
        self.encoder = nn.Sequential(
            nn.Linear(encoder_in, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GNN stack
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if self.layer_type == 'gatv2':
                self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=1, edge_dim=None, dropout=dropout, concat=False))
            else:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, normalize=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

        if use_unet:
            self.unet = GraphUNet(hidden_dim, hidden_dim, hidden_dim, depth=3, pool_ratios=0.5)
        else:
            self.unet = None

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x  # [N, F]
        if self.use_fourier_features:
            coords = data.coords if hasattr(data, 'coords') else x[:, :3]
            x = torch.cat([x, self.fourier(coords)], dim=-1)

        h = self.encoder(x)
        for conv, ln in zip(self.convs, self.norms):
            h_in = h
            if self.layer_type == 'gatv2' and hasattr(data, 'edge_attr') and data.edge_attr is not None:
                h = conv(h, data.edge_index, edge_attr=None)  # edge_attr optional; distance can be injected later
            else:
                h = conv(h, data.edge_index)
            h = ln(h)
            h = F.silu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_in

        if self.unet is not None:
            h = self.unet(h, data.edge_index)

        out = self.head(h)
        return out


