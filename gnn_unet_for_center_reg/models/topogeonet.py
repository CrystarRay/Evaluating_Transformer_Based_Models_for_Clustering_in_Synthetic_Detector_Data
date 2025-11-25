"""
TopoGeoNet: Main model implementation.

This module contains the core TopoGeoNet neural network architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from .layers import UnifiedGNNLayer
from .components import MLPEncoder, FourierFeatureEncoder, UNet
from utils.converter import UtilMapConverter


class UNet3D(nn.Module):
    """
    Minimal 3D UNet with two downsampling and two upsampling stages.
    Input/Output: [B, C, D, H, W]. Channels preserved.
    """
    def __init__(self, in_channels: int, base_channels: int = 64, dropout: float = 0.1):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2

        # Use GroupNorm to avoid train/eval mismatch with batch size = 1
        def _gn(num_channels: int) -> nn.GroupNorm:
            groups = 8
            while groups > 1 and (num_channels % groups) != 0:
                groups -= 1
            return nn.GroupNorm(groups, num_channels)

        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, c1, kernel_size=3, padding=1),
            _gn(c1),
            nn.SiLU(),
            nn.Conv3d(c1, c1, kernel_size=3, padding=1),
            _gn(c1),
            nn.SiLU(),
            nn.Dropout3d(p=dropout),
        )
        self.down1 = nn.MaxPool3d(2)

        self.enc2 = nn.Sequential(
            nn.Conv3d(c1, c2, kernel_size=3, padding=1),
            _gn(c2),
            nn.SiLU(),
            nn.Conv3d(c2, c2, kernel_size=3, padding=1),
            _gn(c2),
            nn.SiLU(),
            nn.Dropout3d(p=dropout),
        )

        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(c1 + c1, c1, kernel_size=3, padding=1),
            _gn(c1),
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


class TopoGeoNet(nn.Module):
    """
    TopoGeoNet: A neural network architecture for topological and geometric data using PyG.
    
    This model uses unified GNN layers that can handle HeteroData format with
    different node and edge types. It includes:
    - MLP encoder for initial node features (data['node'].x)
    - Fourier feature encoder for 3D coordinates (optional)
    - First UNet for spatial feature enhancement (optional, before GNN layers)
    - Configurable GNN layers with residual connections
    - Second UNet for final spatial refinement (optional, after GNN layers)
    - Final MLP projection
    
    Args:
        input_dim (int): Input node feature dimension
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output feature dimension
        num_layers (int): Number of GNN layers
        layer_type (str): Type of GNN layer ('gcn', 'gin', 'gat', etc.)
        dropout (float): Dropout probability
        heads (int): Number of attention heads (for GAT-like layers)
        edge_dim (int, optional): Edge feature dimension
        use_unet (bool): Whether to use UNet layers for spatial feature processing
        use_fourier_features (bool): Whether to use Fourier feature encoder for coordinates
        config (Dict[str, Any], optional): Additional configuration parameters
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        layer_type: str = 'gcn',
        dropout: float = 0.1,
        heads: int = 1,
        edge_dim: Optional[int] = None,
        use_unet: bool = True,
        use_first_unet: bool = True,
        use_fourier_features: bool = True,
        use_3d: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        super(TopoGeoNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.dropout = dropout
        self.heads = heads
        self.edge_dim = edge_dim
        self.use_unet = use_unet
        self.use_first_unet = use_first_unet
        self.use_fourier_features = use_fourier_features
        self.use_3d = use_3d
        self.config = config or {}
        # Global warning control
        self.suppress_warnings = self.config.get('suppress_warnings', False)
        
        # Memory optimization settings
        self.use_gradient_checkpointing = self.config.get('use_gradient_checkpointing', False)
        self.use_amp = self.config.get('use_amp', False)
        self.memory_efficient_mode = self.config.get('memory_efficient_mode', True)
        
        # Initialize encoders
        self._build_encoders()
        
        # Initialize model components
        self._build_model()
        
        # Enable gradient checkpointing if requested
        if self.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
    def _build_encoders(self):
        """Build the encoders for node features and coordinates."""
        # Determine input dimension for node_encoder based on Fourier features usage
        if self.use_fourier_features:
            # Fourier features + raw coordinates + original node features (without coordinates)
            fourier_dim = self.hidden_dim // 2  # Use half of hidden_dim for Fourier features
            fourier_output_dim = fourier_dim + 3  # mapping_size + input_dim (3D coords)
            # x_features has (input_dim - 3) features since we remove the last 3 coordinates
            total_input_dim = fourier_output_dim + (self.input_dim - 3)  # Fourier + original features (without coords)
        else:
            # Just original node features
            total_input_dim = self.input_dim
        
        # Simple MLP encoder (same as TopoGeoNetFull)
        self.node_encoder = nn.Sequential(
            nn.Linear(total_input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        
        # Fourier feature encoder (only if enabled)
        if self.use_fourier_features:
            self.fourier_encoder = FourierFeatureEncoder(
                input_dim=3,  # 3D coordinates (x, y, z)
                mapping_size=fourier_dim,
                passthrough=True,  # Include raw coordinates
                normalize_coords=True,  # Normalize to [0,1] range
                multi_scale=True,  # Use multi-scale frequency bands
                seed=42  # Reproducible results
            )
        
        # Total encoder output dimension is now just hidden_dim
        self.encoder_output_dim = self.hidden_dim
        
    def _build_model(self):
        """Build the TopoGeoNet model architecture."""
        # Create GNN layers manually
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(self.num_layers):
            # Determine input and output dimensions for this layer
            if i == 0:
                in_dim = self.encoder_output_dim  # Use encoder output dimension for first layer
            else:
                in_dim = self.hidden_dim
            
            if i == self.num_layers - 1:
                out_dim = self.hidden_dim  # Keep hidden_dim for final projection
                layer_heads = 1 if self.layer_type in ['gat', 'gatv2', 'transformer'] else self.heads
            else:
                out_dim = self.hidden_dim
                layer_heads = self.heads
            
            # Create the GNN layer
            layer = UnifiedGNNLayer(
                layer_type=self.layer_type,
                in_channels=in_dim,
                out_channels=out_dim,
                heads=layer_heads,
                dropout=self.dropout,
                edge_dim=self.edge_dim
            )
            self.gnn_layers.append(layer)
            
            # Add layer normalization
            if i < self.num_layers - 1:  # No layer norm after last layer
                # Since we use concat=False for GAT layers, output dimension is always out_dim
                ln_dim = out_dim
                self.layer_norms.append(nn.LayerNorm(ln_dim))
        
        # UNet layers (optional)
        if self.use_unet:
            if self.use_3d:
                # First UNet3D (before first GNN layer) - optional
                if self.use_first_unet:
                    self.first_unet = UNet3D(
                        in_channels=self.hidden_dim,
                        base_channels=max(16, self.hidden_dim // 2),
                        dropout=self.dropout
                    )
                    # Learnable residual scale for first UNet - FIXED: Use safer initialization and constraints
                    self.alpha_unet1 = nn.Parameter(torch.tensor(1e-4))  # Reduced from 1e-3 to 1e-4
                else:
                    self.first_unet = None
                    self.alpha_unet1 = None
                
                # Second UNet3D (before final GNN layer)
                self.second_unet = UNet3D(
                    in_channels=self.hidden_dim,
                    base_channels=max(16, self.hidden_dim // 2),
                    dropout=self.dropout
                )
                # Learnable residual scale for second UNet - FIXED: Use safer initialization and constraints
                self.alpha_unet2 = nn.Parameter(torch.tensor(1e-4))  # Reduced from 1e-3 to 1e-4
            else:
                # First UNet (before first GNN layer) - optional
                if self.use_first_unet:
                    self.first_unet = UNet(
                        in_channels=self.hidden_dim,  # Now encoder_output_dim = hidden_dim
                        out_channels=self.hidden_dim,
                        base_channels=self.hidden_dim // 2,
                        depth=4,
                        dropout=self.dropout,
                        use_batch_norm=True,
                        upsampling_mode='bilinear',
                        activation='silu'
                    )
                    # Learnable residual scale for first UNet - FIXED: Use safer initialization and constraints
                    self.alpha_unet1 = nn.Parameter(torch.tensor(1e-4))  # Reduced from 1e-3 to 1e-4
                else:
                    self.first_unet = None
                    self.alpha_unet1 = None
                
                # Second UNet (before final GNN layer)
                self.second_unet = UNet(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    base_channels=self.hidden_dim // 2,
                    depth=4,
                    dropout=self.dropout,
                    use_batch_norm=True,
                    upsampling_mode='bilinear',
                    activation='silu'
                )
                # Learnable residual scale for second UNet - FIXED: Use safer initialization and constraints
                self.alpha_unet2 = nn.Parameter(torch.tensor(1e-4))  # Reduced from 1e-3 to 1e-4
        else:
            self.first_unet = None
            self.second_unet = None
        
        # Initialize converter as None - will be set when first data is processed
        self.converter = None
        self.converter_config = None
        self.spatial_dimensions = None
        
        # Final output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        # Node activity indicator head (binary logits per node)
        self.indicator_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
    def _safe_residual_scale(self, alpha_param: nn.Parameter, base_tensor: torch.Tensor, residual_tensor: torch.Tensor) -> torch.Tensor:
        """
        Safely apply residual scaling with constraints to prevent explosion.
        
        Args:
            alpha_param: Learnable scaling parameter
            base_tensor: Base tensor to add residual to
            residual_tensor: Residual tensor to scale and add
            
        Returns:
            Safely scaled and added tensor
        """
        # Apply constraints to alpha parameter
        # Clamp to prevent explosion: min=1e-6, max=1e-2
        alpha_clamped = torch.clamp(alpha_param, min=1e-6, max=1e-2)
        
        # Monitor for potential issues
        if (alpha_clamped > 1e-3) and (not self.suppress_warnings):
            # Log warning if alpha gets too large (can be suppressed via config)
            print(f"Warning: Residual scaling parameter {alpha_clamped.item():.2e} is getting large")
        
        # Apply residual with safe scaling
        return base_tensor + alpha_clamped * residual_tensor
    
    def forward(
        self, 
        data,
        node_type: str = 'node',
        edge_type: tuple = ('node', 'connect', 'node')
    ) -> torch.Tensor:
        """
        Forward pass of TopoGeoNet with memory optimizations.
        
        Args:
            data: Input data (HeteroData or dict)
            node_type: Node type to process
            edge_type: Edge type for connectivity
            
        Returns:
            torch.Tensor: Output predictions
        """
        x = data.x
        
        # Process features based on Fourier usage
        if self.use_fourier_features:
            # Prefer explicit coords if provided; otherwise use last 3 dims of x
            if hasattr(data, 'coords') and getattr(data, 'coords') is not None:
                coords = data.coords
            else:
                # For synthetic data: coordinates are the LAST 3 dimensions of x
                coords = x[:, -3:] if x.size(-1) >= 3 else F.pad(x, (0, 3 - x.size(-1)))
            x_features = x[:, :-3] if x.size(-1) >= 3 else x  # Remove last 3 dims (xyz)
            fourier_features = self.fourier_encoder(coords)  # [N, mapping_size + 3]
            
            # Concatenate Fourier features with remaining node features (without xyz)
            x_combined = torch.cat([fourier_features, x_features], dim=-1)  # [N, fourier_dim + 3 + (input_dim - 3)]
            
            # Clean up intermediate tensors
            del fourier_features, coords, x_features
        else:
            # Just use original node features
            x_combined = x
        
        # Process through single node encoder
        x = self.node_encoder(x_combined)  # [N, hidden_dim]
        
        # Initialize converter if needed (only once)
        self._initialize_converter(data)
        
        # First UNet (before first GNN layer) - optional
        if self.use_unet and self.first_unet is not None and self.converter is not None:
            # Downsample to spatial grid using stored aggregation method
            aggregation_method = self.converter_config.get('aggregation_method', 'amax')
            
            if self.use_3d:
                # 3D case
                x_coarsed = self.converter.to_util_map(x, reduce=aggregation_method)  # [D, H, W, C]
                x_coarsed = x_coarsed.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, D, H, W]
                
                # Apply first UNet3D
                x_unet1 = self.first_unet(x_coarsed)  # [1, hidden_dim, D, H, W]
                
                # Convert back to node features and add residual
                x_unet1_flat = x_unet1.squeeze(0).permute(1, 2, 3, 0)  # [D, H, W, hidden_dim]
                x_unet1_nodes = self.converter.to_node_features(x_unet1_flat)  # [num_nodes, hidden_dim]
            else:
                # 2D case
                x_coarsed = self.converter.to_util_map(x, reduce=aggregation_method)  # [H, W, encoder_output_dim]
                x_coarsed = x_coarsed.permute(2, 0, 1).unsqueeze(0)  # [1, encoder_output_dim, H, W]
                
                # Apply first UNet
                x_unet1 = self.first_unet(x_coarsed)  # [1, hidden_dim, H, W]
                
                # Convert back to node features and add residual
                x_unet1_flat = x_unet1.squeeze(0).permute(1, 2, 0)  # [H, W, hidden_dim]
                x_unet1_nodes = self.converter.to_node_features(x_unet1_flat)  # [num_nodes, hidden_dim]
            
            # Add learnable residual connection
            x = self._safe_residual_scale(self.alpha_unet1, x, x_unet1_nodes)
            
            # Clean up intermediate tensors
            del x_coarsed, x_unet1, x_unet1_flat, x_unet1_nodes
        
        # Pass through each GNN layer with memory optimization
        for i, layer in enumerate(self.gnn_layers):
            # Store input for residual connection (only if needed)
            x_input = x if i < self.num_layers - 1 else None
            
            # Apply second UNet before the last GNN layer
            if self.use_unet and self.second_unet is not None and i == self.num_layers - 1 and self.converter is not None:
                # Downsample to spatial grid using stored aggregation method
                aggregation_method = self.converter_config.get('aggregation_method', 'amax')
                
                if self.use_3d:
                    # 3D case
                    x_coarsed = self.converter.to_util_map(x, reduce=aggregation_method)  # [D, H, W, C]
                    x_coarsed = x_coarsed.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, D, H, W]
                    
                    # Apply second UNet3D
                    x_unet2 = self.second_unet(x_coarsed)  # [1, hidden_dim, D, H, W]
                    
                    # Convert back to node features and add residual
                    x_unet2_flat = x_unet2.squeeze(0).permute(1, 2, 3, 0)  # [D, H, W, hidden_dim]
                    x_unet2_nodes = self.converter.to_node_features(x_unet2_flat)  # [num_nodes, hidden_dim]
                else:
                    # 2D case
                    x_coarsed = self.converter.to_util_map(x, reduce=aggregation_method)  # [H, W, hidden_dim]
                    x_coarsed = x_coarsed.permute(2, 0, 1).unsqueeze(0)  # [1, hidden_dim, H, W]
                    
                    # Apply second UNet
                    x_unet2 = self.second_unet(x_coarsed)  # [1, hidden_dim, H, W]
                    
                    # Convert back to node features and add residual
                    x_unet2_flat = x_unet2.squeeze(0).permute(1, 2, 0)  # [H, W, hidden_dim]
                    x_unet2_nodes = self.converter.to_node_features(x_unet2_flat)  # [num_nodes, hidden_dim]
                
                # Add learnable residual connection
                x = self._safe_residual_scale(self.alpha_unet2, x, x_unet2_nodes)
                
                # Clean up intermediate tensors
                del x_coarsed, x_unet2, x_unet2_flat, x_unet2_nodes
            
            # Apply GNN layer with current node features
            x = layer(data, node_type=node_type, edge_type=edge_type, x=x)
            
            # Apply layer normalization (except for last layer)
            if i < len(self.layer_norms):
                x = self.layer_norms[i](x)
            
            # Apply activation (except for the last layer)
            if i < self.num_layers - 1:
                x = F.silu(x)
            
            # Add residual connection (only if we stored the input)
            if x_input is not None:
                x = x + x_input
                del x_input  # Clean up residual input
        
        # Apply final projections
        output_centers = self.output_proj(x)
        output_centers = output_centers + data.x[:, -3:]
        # Binary node indicator logits
        logits_indicator = self.indicator_head(x).squeeze(-1)
        # Clean up final intermediate tensor
        del x
        
        return output_centers, logits_indicator
    

    
    def get_encoder_dimensions(self) -> Dict[str, int]:
        """
        Get the dimensions of different encoder components for debugging.
        
        Returns:
            Dict containing dimension information
        """
        fourier_info = None
        if self.use_fourier_features:
            fourier_dim = self.hidden_dim // 2
            fourier_output_dim = fourier_dim + 3  # mapping_size + input_dim (3D coords)
            # x_features has (input_dim - 3) features since we remove the last 3 coordinates
            total_input_dim = fourier_output_dim + (self.input_dim - 3)
            fourier_info = {
                'fourier_dim': fourier_dim,
                'fourier_output_dim': fourier_output_dim,
                'total_input_dim': total_input_dim
            }
        
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'node_encoder_input_dim': self.node_encoder.input_dim,
            'node_encoder_output_dim': self.node_encoder.output_dim,
            'encoder_output_dim': self.encoder_output_dim,
            'final_output_dim': self.output_dim,
            'fourier_info': fourier_info
        }
    
    def get_residual_scaling_info(self) -> Dict[str, float]:
        """
        Get information about residual scaling parameters for monitoring.
        
        Returns:
            Dict containing residual scaling parameter values and status
        """
        info = {}
        
        if self.use_unet and self.first_unet is not None:
            alpha1_val = self.alpha_unet1.item()
            alpha1_clamped = torch.clamp(self.alpha_unet1, min=1e-6, max=1e-2).item()
            info['alpha_unet1'] = {
                'raw_value': alpha1_val,
                'clamped_value': alpha1_clamped,
                'is_clamped': alpha1_val != alpha1_clamped,
                'status': 'normal' if alpha1_val <= 1e-3 else 'warning' if alpha1_val <= 1e-2 else 'danger'
            }
        
        if self.use_unet and self.second_unet is not None:
            alpha2_val = self.alpha_unet2.item()
            alpha2_clamped = torch.clamp(self.alpha_unet2, min=1e-6, max=1e-2).item()
            info['alpha_unet2'] = {
                'raw_value': alpha2_val,
                'clamped_value': alpha2_clamped,
                'is_clamped': alpha2_val != alpha2_clamped,
                'status': 'normal' if alpha2_val <= 1e-3 else 'warning' if alpha2_val <= 1e-2 else 'danger'
            }
        
        return info
    
    def reset_residual_scaling_if_needed(self, threshold: float = 1e-2) -> bool:
        """
        Reset residual scaling parameters if they exceed the threshold.
        This is a safety mechanism to prevent loss explosion.
        
        Args:
            threshold: Threshold above which parameters will be reset
            
        Returns:
            True if parameters were reset, False otherwise
        """
        reset_occurred = False
        
        if self.use_unet and self.first_unet is not None:
            if self.alpha_unet1.item() > threshold:
                print(f"⚠️  Resetting alpha_unet1 from {self.alpha_unet1.item():.2e} to 1e-4")
                with torch.no_grad():
                    self.alpha_unet1.data.fill_(1e-4)
                reset_occurred = True
        
        if self.use_unet and self.second_unet is not None:
            if self.alpha_unet2.item() > threshold:
                print(f"⚠️  Resetting alpha_unet2 from {self.alpha_unet2.item():.2e} to 1e-4")
                with torch.no_grad():
                    self.alpha_unet2.data.fill_(1e-4)
                reset_occurred = True
        
        if reset_occurred:
            print("✅ Residual scaling parameters have been reset to safe values")
        
        return reset_occurred
    
    def get_current_residual_scaling_values(self) -> Dict[str, float]:
        """
        Get current values of residual scaling parameters for debugging.
        
        Returns:
            Dict containing current parameter values
        """
        values = {}
        if self.use_unet and self.first_unet is not None:
            values['alpha_unet1'] = self.alpha_unet1.item()
        if self.use_unet and self.second_unet is not None:
            values['alpha_unet2'] = self.alpha_unet2.item()
        return values
    
    def get_converter_status(self) -> Dict[str, Any]:
        """
        Get current status of the converter.
        
        Returns:
            Dict containing converter status information
        """
        if self.converter is None:
            return {'status': 'not_initialized', 'message': 'Converter has not been initialized yet'}
        
        return {
            'status': 'initialized',
            'spatial_dimensions': self.spatial_dimensions,
            'config': self.converter_config,
            'device': str(self.converter.assignment.device),
            'assignment_shape': self.converter.assignment.shape,
            'num_sites_x': self.converter.num_sites_x,
            'num_sites_y': self.converter.num_sites_y
        }
    
    def set_converter_manually(self, assignment: torch.Tensor, num_sites_x: int, num_sites_y: int, device: torch.device):
        """
        Manually set the converter (useful for testing or when you want to override the automatic initialization).
        
        Args:
            assignment: Assignment tensor
            num_sites_x: Number of sites in X dimension
            num_sites_y: Number of sites in Y dimension
            device: Device to place converter on
        """
        self.converter = UtilMapConverter(
            assignment=assignment,
            num_sites_x=num_sites_x,
            num_sites_y=num_sites_y,
            device=device
        )
        self.spatial_dimensions = (num_sites_x, num_sites_y)
        self.converter_config = self.config.get('converter', {})
        print(f"Converter manually set with spatial dimensions: {num_sites_x}x{num_sites_y}")
    
    def get_converter_info(self, data) -> Optional[Dict[str, Any]]:
        """
        Get information about the converter if assignment exists.
        
        Args:
            data: Input data object
            
        Returns:
            Dict containing converter information or None if no assignment
        """
        # Initialize converter if needed
        self._initialize_converter(data)
        
        if self.converter is None:
            return None
            
        try:
            return self.converter.get_spatial_info()
        except Exception as e:
            return {'error': str(e)}
    
    def test_converter_functionality(self, data) -> Dict[str, Any]:
        """
        Test the converter functionality to ensure proper downsampling/upsampling.
        
        Args:
            data: Input data object with assignment
            
        Returns:
            Dict containing test results
        """
        # Initialize converter if needed
        self._initialize_converter(data)
        
        if self.converter is None:
            return {'status': 'no_assignment', 'message': 'No assignment found in data'}
            
        try:
            # Test with a simple tensor
            device = self.converter.assignment.device
            
            # Get number of nodes from data - directly from dataset structure
            num_nodes = data.x.shape[0]
            
            test_features = torch.randn(num_nodes, 64, device=device)
            
            # Downsample
            util_map = self.converter.to_util_map(test_features, max_sum='sum')
            
            # Upsample back
            reconstructed_features = self.converter.to_node_features(util_map)
            
            # Check reconstruction error
            reconstruction_error = torch.mean((test_features - reconstructed_features) ** 2).item()
            
            return {
                'status': 'success',
                'spatial_dimensions': self.spatial_dimensions,
                'input_shape': test_features.shape,
                'util_map_shape': util_map.shape,
                'reconstruction_error': reconstruction_error,
                'converter_info': self.converter.get_spatial_info()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _get_spatial_dimensions(self, data, num_nodes: int) -> Tuple[int, int]:
        """
        Extract spatial dimensions H and W from data.assignment or fallback to data.vn.shape.
        
        Args:
            data: Input data object with direct attributes
            num_nodes: Number of nodes
            
        Returns:
            tuple: (H, W) spatial dimensions for 2D, (D, H, W) for 3D
        """
        # First try to get dimensions from assignment (preferred method)
        if hasattr(data, 'assignment') and data.assignment is not None:
            if self.use_3d:
                # 3D case: expect [N, 3] with (z, y, x)
                z_max = data.assignment[:, 0].max().item()
                y_max = data.assignment[:, 1].max().item()
                x_max = data.assignment[:, 2].max().item()
                return int(z_max + 1), int(y_max + 1), int(x_max + 1)
            else:
                # 2D case: expect [N, 2] with (y, x)
                y_max = data.assignment[:, 0].max().item()
                x_max = data.assignment[:, 1].max().item()
                return int(y_max + 1), int(x_max + 1)
        elif hasattr(data, 'vn'):
            # Fallback to data.vn.shape
            if self.use_3d:
                return data.vn.shape[0], data.vn.shape[1], data.vn.shape[2]
            else:
                return data.vn.shape[0], data.vn.shape[1]
        else:
            # Last fallback: try to infer from node features
            if self.use_3d:
                # Assume cubic grid
                side = int(num_nodes ** (1/3))
                return side, side, side
            else:
                # Assume square grid
                H = W = int(num_nodes ** 0.5)
                if H * W != num_nodes:
                    # If not perfect square, use closest approximation
                    H = int(num_nodes ** 0.5)
                    W = num_nodes // H
                return H, W
    
    def get_encoded_features(
        self, 
        data,
        node_type: str = 'node'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get encoded features separately for analysis.
        
        Args:
            data: Input data (HeteroData or dict)
            node_type: Node type to process
            
        Returns:
            tuple: (original_features, encoded_node_features, encoded_coord_features)
        """
        with torch.no_grad():
            # Get initial features - directly from dataset structure
            x = data.x
            
            # Process features based on Fourier usage
            if self.use_fourier_features:
                # Extract coordinates and compute Fourier features (consistent with forward method)
                if hasattr(data, 'coords') and getattr(data, 'coords') is not None:
                    coords = data.coords
                else:
                    coords = x[:, -3:] if x.size(-1) >= 3 else F.pad(x, (0, 3 - x.size(-1)))
                x_features = x[:, :-3] if x.size(-1) >= 3 else x  # Remove last 3 dims (xyz)
                
                fourier_features = self.fourier_encoder(coords)
                x_combined = torch.cat([fourier_features, x_features], dim=-1)
                encoded_features = self.node_encoder(x_combined)
            else:
                # Just use original node features
                encoded_features = self.node_encoder(x)
            
            return x, encoded_features, None  # No separate coord_features anymore
    
    def get_current_device(self) -> torch.device:
        """
        Get the current device that the model is on.
        
        Returns:
            PyTorch device
        """
        # Get device from model parameters
        if next(self.parameters()).device.type != 'cpu':
            return next(self.parameters()).device
        else:
            # Fallback to checking CUDA availability
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def ensure_device_consistency(self, data) -> torch.device:
        """
        Ensure that the model and data are on the same device.
        
        Args:
            data: Input data object
            
        Returns:
            Device that should be used
        """
        model_device = self.get_current_device()
        
        # Check if data needs to be moved to model's device - directly from dataset structure
        data_device = data.x.device
        if data_device != model_device:
            if not self.suppress_warnings:
                print(f"Warning: Data.x is on {data_device}, but model is on {model_device}")
                print(f"Consider moving data to {model_device} for optimal performance")
        
        if hasattr(data, 'assignment') and hasattr(data.assignment, 'device'):
            assignment_device = data.assignment.device
            if assignment_device != model_device:
                if not self.suppress_warnings:
                    print(f"Warning: Assignment tensor is on {assignment_device}, but model is on {model_device}")
                    print(f"Consider moving assignment tensor to {model_device} for optimal performance")
        
        return model_device
    
    def _initialize_converter(self, data):
        """
        Initialize the converter once when first data is processed.
        
        Args:
            data: Input data object with assignment tensor
        """
        if not hasattr(data, 'assignment') or data.assignment is None:
            return  # No assignment, no converter needed
        
        # If converter exists, verify spatial dims/assignment dtype; reinit if mismatch
        if self.converter is not None:
            try:
                if self.use_3d:
                    D, H, W = self._get_spatial_dimensions(data, data.assignment.shape[0])
                    if self.spatial_dimensions != (D, H, W):
                        self.converter = None
                else:
                    H, W = self._get_spatial_dimensions(data, data.assignment.shape[0])
                    if self.spatial_dimensions != (H, W):
                        self.converter = None
            except Exception:
                self.converter = None
            if self.converter is not None:
                return

        # Check if converter is enabled in config
        converter_enabled = self.config.get('converter', {}).get('enabled', True)
        if not converter_enabled:
            return
        num_nodes = data.assignment.shape[0]
        
        # Get converter configuration
        self.converter_config = self.config.get('converter', {})
        aggregation_method = self.converter_config.get('aggregation_method', 'amax')
        log_converter_init = self.converter_config.get('log_init', False)
        
        # Use the device consistency check to ensure optimal device selection
        device = self.ensure_device_consistency(data)
        
        if self.use_3d:
            # 3D case
            D, H, W = self._get_spatial_dimensions(data, num_nodes)
            self.spatial_dimensions = (D, H, W)
            
            # Create 3D converter
            self.converter = VoxelConverter3D(
                assignment=data.assignment,
                depth=D,
                height=H,
                width=W,
                device=device
            )
            if log_converter_init:
                print(f"3D Converter initialized with spatial dimensions: {D}x{H}x{W}, aggregation: {aggregation_method}")
        else:
            # 2D case
            H, W = self._get_spatial_dimensions(data, num_nodes)
            self.spatial_dimensions = (H, W)
            
            # Create 2D converter
            self.converter = UtilMapConverter(
                assignment=data.assignment,
                num_sites_x=H,
                num_sites_y=W,
                device=device
            )
            if log_converter_init:
                print(f"2D Converter initialized with spatial dimensions: {H}x{W}, aggregation: {aggregation_method}")
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(torch.utils.checkpoint, 'checkpoint'):
            # Enable gradient checkpointing for UNet layers if they exist
            if self.use_unet and self.first_unet is not None:
                self.first_unet.use_checkpoint = True
            if self.use_unet and self.second_unet is not None:
                self.second_unet.use_checkpoint = True
            
            print("Gradient checkpointing enabled for memory efficiency")
        else:
            print("Warning: Gradient checkpointing not available in this PyTorch version")
    
    def set_memory_efficient_mode(self, enabled: bool = True):
        """Enable or disable memory efficient mode."""
        self.memory_efficient_mode = enabled
        if enabled:
            print("Memory efficient mode enabled")
        else:
            print("Memory efficient mode disabled")
    
    def get_memory_usage_info(self) -> Dict[str, Any]:
        """Get information about current memory usage and optimization settings."""
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3  # GB
            memory_free = torch.cuda.get_device_properties(current_device).total_memory / 1024**3 - memory_reserved
        else:
            memory_allocated = memory_reserved = memory_free = 0.0
        
        return {
            'memory_efficient_mode': self.memory_efficient_mode,
            'use_gradient_checkpointing': self.use_gradient_checkpointing,
            'use_amp': self.use_amp,
            'gpu_memory_allocated_gb': round(memory_allocated, 2),
            'gpu_memory_reserved_gb': round(memory_reserved, 2),
            'gpu_memory_free_gb': round(memory_free, 2),
            'model_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free up memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")
    
    def optimize_for_inference(self):
        """Optimize the model for inference (reduces memory usage)."""
        self.eval()
        with torch.no_grad():
            # Convert to half precision if using AMP
            if self.use_amp:
                self.half()
                print("Model converted to half precision for inference")
            
            # Enable optimizations
            if hasattr(torch, 'jit') and self.config.get('use_jit', False):
                try:
                    self = torch.jit.optimize_for_inference(torch.jit.script(self))
                    print("Model optimized with TorchScript for inference")
                except Exception as e:
                    print(f"TorchScript optimization failed: {e}")
        
        return self
