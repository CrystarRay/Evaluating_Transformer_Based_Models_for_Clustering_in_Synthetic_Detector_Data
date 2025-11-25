"""
Adapted full-style TopoGeoNet for PyG Data graphs (homogeneous).

- Optional Fourier encoding using data.coords (preferred) or last 3 dims of x
- MLP node encoder to hidden_dim
- Configurable GNN stack (gcn/gin/gatv2/sage/transformer)
- Optional UNet omitted by default (requires assignment/grid); can be enabled later
- Final projection to output_dim (default 3 for XYZ regression)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import (
        GCNConv, GINConv, GATv2Conv, SAGEConv, TransformerConv
    )
except Exception as e:
    raise ImportError("topogeonet_full requires torch-geometric. Install with: pip install torch-geometric") from e


class FourierFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int = 3, mapping_size: int = 64, scale: float = 10.0, passthrough: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.passthrough = passthrough
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * coords @ self.B
        enc = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return torch.cat([coords, enc], dim=-1) if self.passthrough else enc


class UnifiedGNNLayerSimple(nn.Module):
    def __init__(self, layer_type: str, in_channels: int, out_channels: int, heads: int = 1, dropout: float = 0.0, edge_dim: Optional[int] = None):
        super().__init__()
        lt = layer_type.lower()
        self.layer_type = lt
        self.edge_dim = edge_dim
        if lt == 'gcn':
            self.layer = GCNConv(in_channels, out_channels)
        elif lt == 'gin':
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels), nn.SiLU(), nn.Linear(out_channels, out_channels)
            )
            self.layer = GINConv(mlp)
        elif lt == 'gatv2':
            self.layer = GATv2Conv(in_channels, out_channels, heads=heads, dropout=dropout, concat=False, edge_dim=edge_dim)
        elif lt == 'sage':
            self.layer = SAGEConv(in_channels, out_channels, normalize=True)
        elif lt == 'transformer':
            self.layer = TransformerConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=False, edge_dim=edge_dim)
        else:
            raise ValueError(f"Unsupported layer_type: {layer_type}")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.layer_type in ['gatv2', 'transformer'] and self.edge_dim is not None and edge_attr is not None:
            out = self.layer(x, edge_index, edge_attr=edge_attr)
        else:
            out = self.layer(x, edge_index)
        return self.dropout(out)


class VoxelConverter3D:
    """
    Lightweight 3D converter between node features and voxel grid [D, H, W].
    Expects data.assignment of shape [N, 3] with integer indices (z, y, x).
    """
    def __init__(self, assignment: torch.Tensor, depth: int, height: int, width: int, device: torch.device):
        self.assignment = assignment.to(device)
        self.depth = depth
        self.height = height
        self.width = width
        self.device = device

        self.flat_index = (
            self.assignment[:, 0].long() * (self.height * self.width)
            + self.assignment[:, 1].long() * self.width
            + self.assignment[:, 2].long()
        ).to(device)

    def to_util_map(self, node_features: torch.Tensor, reduce: str = 'sum') -> torch.Tensor:
        """Aggregate node features [N, C] into voxel grid [D, H, W, C]."""
        N, C = node_features.shape
        DHW = self.depth * self.height * self.width
        out = torch.zeros(DHW, C, device=node_features.device, dtype=node_features.dtype)

        reduce = (reduce or 'sum').lower()
        if reduce in ['sum', 'add']:
            out.index_add_(0, self.flat_index, node_features)
        elif reduce in ['mean', 'avg']:
            out.index_add_(0, self.flat_index, node_features)
            counts = torch.zeros(DHW, 1, device=node_features.device, dtype=node_features.dtype)
            ones = torch.ones(N, 1, device=node_features.device, dtype=node_features.dtype)
            counts.index_add_(0, self.flat_index, ones)
            counts = torch.clamp(counts, min=1.0)
            out = out / counts
        elif reduce in ['amax', 'max']:
            # Try scatter_reduce if available (PyTorch >= 1.12)
            if hasattr(torch.Tensor, 'scatter_reduce_'):
                out.fill_(float('-inf'))
                out.scatter_reduce_(0, self.flat_index.unsqueeze(-1).expand(-1, C), node_features, reduce='amax', include_self=True)
                out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
            else:
                # Fallback approximate with sum
                out.index_add_(0, self.flat_index, node_features)
        else:
            out.index_add_(0, self.flat_index, node_features)

        out = out.view(self.depth, self.height, self.width, C)
        return out

    def to_node_features(self, util_map: torch.Tensor) -> torch.Tensor:
        """Gather voxel features [D, H, W, C] back to nodes [N, C]."""
        D, H, W, C = util_map.shape
        assert D == self.depth and H == self.height and W == self.width
        flat = util_map.view(-1, C)
        return flat.index_select(0, self.flat_index)


class UNet3D(nn.Module):
    """
    Minimal 3D UNet with two downsampling and two upsampling stages.
    Input/Output: [B, C, D, H, W]. Channels preserved.
    """
    def __init__(self, in_channels: int, base_channels: int = 64, dropout: float = 0.1):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2

        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm3d(c1),
            nn.SiLU(),
            nn.Conv3d(c1, c1, kernel_size=3, padding=1),
            nn.BatchNorm3d(c1),
            nn.SiLU(),
            nn.Dropout3d(p=dropout),
        )
        self.down1 = nn.MaxPool3d(2)

        self.enc2 = nn.Sequential(
            nn.Conv3d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm3d(c2),
            nn.SiLU(),
            nn.Conv3d(c2, c2, kernel_size=3, padding=1),
            nn.BatchNorm3d(c2),
            nn.SiLU(),
            nn.Dropout3d(p=dropout),
        )

        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(c1 + c1, c1, kernel_size=3, padding=1),
            nn.BatchNorm3d(c1),
            nn.SiLU(),
            nn.Conv3d(c1, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        x = self.down1(e1)
        e2 = self.enc2(x)
        x = self.up1(e2)
        # Pad/crop if needed for odd sizes
        if x.shape[-3:] != e1.shape[-3:]:
            dz = e1.shape[-3] - x.shape[-3]
            dy = e1.shape[-2] - x.shape[-2]
            dx = e1.shape[-1] - x.shape[-1]
            pad = [max(dx // 2, 0), max(dx - dx // 2, 0), max(dy // 2, 0), max(dy - dy // 2, 0), max(dz // 2, 0), max(dz - dz // 2, 0)]
            x = F.pad(x, pad)
            # If larger, center-crop
            x = x[..., :e1.shape[-3], :e1.shape[-2], :e1.shape[-1]]
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)
        return x


class TopoGeoNetFull(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 4,
        layer_type: str = 'sage',
        dropout: float = 0.1,
        heads: int = 1,
        edge_dim: Optional[int] = None,
        use_unet: bool = False,
        use_fourier_features: bool = True,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.dropout = dropout
        self.heads = heads
        self.edge_dim = edge_dim
        self.use_unet = use_unet
        self.use_fourier_features = use_fourier_features
        self.config = config or {}
        # Runtime enable flags for executing specific UNet stages (not constructor args)
        self.enable_unet_first: bool = False
        self.enable_unet_second: bool = False

        # Fourier encoder
        fourier_out = 0
        if self.use_fourier_features:
            mapping = hidden_dim // 2
            self.fourier = FourierFeatureEncoder(input_dim=3, mapping_size=mapping, passthrough=True)
            fourier_out = 3 + 2 * mapping
        else:
            self.fourier = None

        encoder_in = input_dim + (fourier_out if self.use_fourier_features else 0)
        self.node_encoder = nn.Sequential(
            nn.Linear(encoder_in, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(
                UnifiedGNNLayerSimple(
                    layer_type=layer_type, in_channels=hidden_dim, out_channels=hidden_dim, heads=heads, dropout=dropout, edge_dim=edge_dim
                )
            )
            if i < num_layers - 1:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim)
        )

        # 3D UNet components (optional)
        if self.use_unet:
            self.unet3d_first = UNet3D(in_channels=self.hidden_dim, base_channels=max(16, self.hidden_dim // 2), dropout=self.dropout)
            self.unet3d_second = UNet3D(in_channels=self.hidden_dim, base_channels=max(16, self.hidden_dim // 2), dropout=self.dropout)
            self.alpha_unet1 = nn.Parameter(torch.tensor(1e-4))
            self.alpha_unet2 = nn.Parameter(torch.tensor(1e-4))
        else:
            self.unet3d_first = None
            self.unet3d_second = None
            self.alpha_unet1 = None
            self.alpha_unet2 = None

        # Converter placeholders
        self.converter3d = None
        self.spatial_dims_3d = None
        self.converter_config = self.config.get('converter', {})

    @staticmethod
    def _safe_residual_scale(alpha_param: nn.Parameter, base_tensor: torch.Tensor, residual_tensor: torch.Tensor) -> torch.Tensor:
        alpha_clamped = torch.clamp(alpha_param, min=1e-6, max=1e-2)
        return base_tensor + alpha_clamped * residual_tensor

    def _init_converter3d_if_needed(self, data: Data):
        if not self.use_unet or self.converter3d is not None:
            return
        if not hasattr(data, 'assignment') or data.assignment is None:
            return
        # Expect assignment as [N, 3] with (z,y,x)
        z_max = int(data.assignment[:, 0].max().item())
        y_max = int(data.assignment[:, 1].max().item())
        x_max = int(data.assignment[:, 2].max().item())
        D, H, W = z_max + 1, y_max + 1, x_max + 1
        self.spatial_dims_3d = (D, H, W)
        device = data.x.device
        self.converter3d = VoxelConverter3D(assignment=data.assignment, depth=D, height=H, width=W, device=device)

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x  # [N, F]
        # Coordinates for Fourier
        if self.use_fourier_features:
            if hasattr(data, 'coords') and data.coords is not None:
                coords = data.coords
            else:
                # Fallback to last 3 features as XYZ
                coords = x[:, -3:] if x.size(-1) >= 3 else F.pad(x, (3 - x.size(-1), 0))
            fourier = self.fourier(coords)
            x = torch.cat([x, fourier], dim=-1)

        # Encode to hidden
        h = self.node_encoder(x)

        # Initialize 3D converter lazily if needed
        self._init_converter3d_if_needed(data)

        # First UNet3D (pre-GNN)
        if self.use_unet and self.enable_unet_first and self.unet3d_first is not None and self.converter3d is not None:
            aggregation_method = self.converter_config.get('aggregation_method', 'amax')
            util = self.converter3d.to_util_map(h, reduce=aggregation_method)  # [D, H, W, C]
            util_bchw = util.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, D, H, W]
            util_refined = self.unet3d_first(util_bchw)
            util_refined = util_refined.squeeze(0).permute(1, 2, 3, 0)  # [D, H, W, C]
            h_unet_nodes = self.converter3d.to_node_features(util_refined)
            h = self._safe_residual_scale(self.alpha_unet1, h, h_unet_nodes)
            del util, util_bchw, util_refined, h_unet_nodes

        # Message passing
        for i, layer in enumerate(self.gnn_layers):
            h_in = h if i < self.num_layers - 1 else None
            # Second UNet3D (before last GNN layer) — parity with original
            if self.use_unet and self.enable_unet_second and self.unet3d_second is not None and self.converter3d is not None and i == self.num_layers - 1:
                aggregation_method = self.converter_config.get('aggregation_method', 'amax')
                util2 = self.converter3d.to_util_map(h, reduce=aggregation_method)
                util2_bchw = util2.permute(3, 0, 1, 2).unsqueeze(0)
                util2_refined = self.unet3d_second(util2_bchw)
                util2_refined = util2_refined.squeeze(0).permute(1, 2, 3, 0)
                h_unet2_nodes = self.converter3d.to_node_features(util2_refined)
                h = self._safe_residual_scale(self.alpha_unet2, h, h_unet2_nodes)
                del util2, util2_bchw, util2_refined, h_unet2_nodes
            edge_attr = getattr(data, 'edge_attr', None)
            h = layer(h, data.edge_index, edge_attr=edge_attr if self.edge_dim is not None else None)
            if i < len(self.layer_norms):
                h = self.layer_norms[i](h)
                h = F.silu(h)
                if h_in is not None:
                    h = h + h_in

        out = self.output_proj(h)
        return out


