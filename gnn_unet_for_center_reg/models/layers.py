"""
Unified GNN message passing layers for TopoGeoNet.

This module contains a unified layer that can use different PyG GNN layers
and selectively handle available attributes based on the data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any
import warnings

# Import PyTorch Geometric layers
try:
    from torch_geometric.nn import GCNConv, GINConv, GATConv, GATv2Conv, SAGEConv, TransformerConv
    from torch_geometric.data import HeteroData, Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    warnings.warn("PyTorch Geometric not available. GNN layers will not work.")


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer layer based on GraphGPS template.
    
    This layer implements:
    - Multi-head self-attention with positional encodings
    - Feed-forward network with residual connections
    - Layer normalization (Pre-LN style)
    - Dropout for regularization
    
    Args:
        hidden_dim (int): Hidden dimension size
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
        edge_dim (int, optional): Edge feature dimension
        use_edge_attr (bool): Whether to use edge attributes
        ff_multiplier (float): Multiplier for feed-forward hidden dimension
        use_pos_encoding (bool): Whether to use positional encodings
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        use_edge_attr: bool = True,
        ff_multiplier: float = 4.0,
        use_pos_encoding: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.use_edge_attr = use_edge_attr
        self.ff_multiplier = ff_multiplier
        self.use_pos_encoding = use_pos_encoding
        
        # Ensure hidden_dim is divisible by num_heads
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for attention
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network with memory optimization
        ff_hidden_dim = int(hidden_dim * ff_multiplier)
        self.ff_network = nn.Sequential(
            nn.Linear(hidden_dim, ff_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Edge feature processing (if edge attributes are used)
        if use_edge_attr and edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        else:
            self.edge_proj = None
        
        # Positional encoding (learnable) - reduced initialization scale for memory efficiency
        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01)  # Reduced from 0.02
        else:
            self.pos_encoding = None
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Learnable residual scaling - smaller initial values for memory efficiency
        self.alpha_attn = nn.Parameter(torch.tensor(1e-5))  # Reduced from 1e-4
        self.alpha_ff = nn.Parameter(torch.tensor(1e-5))    # Reduced from 1e-4
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Add positional encoding if enabled
        if self.pos_encoding is not None:
            x = x + self.pos_encoding.expand(x.size(0), -1, -1).squeeze(1)
        
        # Store input for residual connection
        x_input = x
        
        # Self-attention block with Pre-LN
        x_norm = self.norm1(x)
        
        # Process edge attributes if available
        if self.use_edge_attr and edge_attr is not None and self.edge_proj is not None:
            # Project edge features to hidden dimension
            if edge_attr.dim() == 1:
                edge_attr_reshaped = edge_attr.unsqueeze(1)  # [num_edges, 1]
            else:
                edge_attr_reshaped = edge_attr
            
            edge_features = self.edge_proj(edge_attr_reshaped)
            
            # Apply attention
            if x_norm.dim() == 2:
                x_norm_batch = x_norm.unsqueeze(0)
            else:
                x_norm_batch = x_norm
            
            attn_output, _ = self.attention(
                query=x_norm_batch,
                key=x_norm_batch,
                value=x_norm_batch
            )
            
            if x_norm.dim() == 2:
                attn_output = attn_output.squeeze(0)
            
            # Process edge integration
            if edge_index is not None and edge_index.numel() > 0:
                # Aggregate edge features to nodes using mean pooling
                edge_features_agg = torch.zeros(x.size(0), edge_features.size(1), device=x.device)
                edge_counts = torch.zeros(x.size(0), 1, device=x.device)
                
                # Count edges per node
                edge_counts.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), 1, device=x.device))
                
                # Sum edge features per node
                edge_features_agg.scatter_add_(0, edge_index[1], edge_features)
                
                # Average edge features per node (avoid division by zero)
                edge_features_agg = edge_features_agg / (edge_counts + 1e-8)
                
                # Ensure edge_features_agg has the right shape
                if edge_features_agg.shape[1] != attn_output.shape[1]:
                    edge_features_agg = edge_features_agg.expand(-1, attn_output.shape[1])
                
                attn_output = attn_output + 0.1 * edge_features_agg
                
                # Clean up intermediate tensors
                del edge_features, edge_features_agg, edge_counts
        else:
            # Standard self-attention
            if x_norm.dim() == 2:
                x_norm_batch = x_norm.unsqueeze(0)
            else:
                x_norm_batch = x_norm
            
            attn_output, _ = self.attention(
                query=x_norm_batch,
                key=x_norm_batch,
                value=x_norm_batch
            )
            
            if x_norm.dim() == 2:
                attn_output = attn_output.squeeze(0)
        
        # Apply dropout and residual connection with learnable scaling
        x = x_input + self.alpha_attn * self.dropout1(attn_output)
        
        # Feed-forward block with Pre-LN
        x_norm = self.norm2(x)
        ff_output = self.ff_network(x_norm)
        
        # Apply dropout and residual connection with learnable scaling
        x = x + self.alpha_ff * ff_output
        
        # Clean up intermediate tensors
        del x_input, x_norm, attn_output, ff_output
        
        return x
    
    def get_residual_scaling_values(self) -> Dict[str, float]:
        """Get current residual scaling parameter values for monitoring."""
        return {
            'alpha_attn': self.alpha_attn.item(),
            'alpha_ff': self.alpha_ff.item()
        }
    
    def reset_residual_scaling_if_needed(self, threshold: float = 1e-2) -> bool:
        """Reset residual scaling parameters if they exceed the threshold."""
        reset_occurred = False
        
        if self.alpha_attn.item() > threshold:
            print(f"⚠️  Resetting alpha_attn from {self.alpha_attn.item():.2e} to 1e-4")
            with torch.no_grad():
                self.alpha_attn.data.fill_(1e-4)
            reset_occurred = True
        
        if self.alpha_ff.item() > threshold:
            print(f"⚠️  Resetting alpha_ff from {self.alpha_ff.item():.2e} to 1e-4")
            with torch.no_grad():
                self.alpha_ff.data.fill_(1e-4)
            reset_occurred = True
        
        if reset_occurred:
            print("✅ GraphTransformer residual scaling parameters have been reset")
        
        return reset_occurred
    
    def optimize_memory_usage(self):
        """Optimize memory usage by reducing parameter precision and clearing cache."""
        # Reduce positional encoding precision if it's too large
        if self.pos_encoding is not None and self.pos_encoding.numel() > 1000:
            with torch.no_grad():
                self.pos_encoding.data = self.pos_encoding.data.half()
        
        # Clear any cached computations
        if hasattr(self.attention, 'reset_parameters'):
            self.attention.reset_parameters()
        
        print("GraphTransformer memory optimization completed")


class SMPNNBlock(nn.Module):
    """
    SMPNN block: Pre-LN -> GCN -> scaled residual -> Pre-LN -> FF(SiLU+Linear) -> scaled residual.
    Based on 'Scalable Message Passing Neural Networks' (SMPNN).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Pre-LN before graph conv
        self.ln1 = nn.LayerNorm(in_channels)
        self.gcn = GCNConv(in_channels, out_channels, cached=False, normalize=True)

        # Learnable residual scalars, initialized with smaller values for memory efficiency
        self.alpha1 = nn.Parameter(torch.tensor(1e-7))  # Reduced from 1e-6
        self.alpha2 = nn.Parameter(torch.tensor(1e-7))  # Reduced from 1e-6

        # Skip projection if dims change
        self.res_proj = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels, bias=False)

        # Pre-LN before pointwise FF (stays in out_channels)
        self.ln2 = nn.LayerNorm(out_channels)
        self.ff  = nn.Linear(out_channels, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Block 1: Pre-LN -> GCN -> SiLU -> scaled residual
        h1 = self.ln1(x)
        m = self.gcn(h1, edge_index)              # local message passing
        m = self.act(m)
        x1 = self.res_proj(x) + self.alpha1 * m    # residual with learnable scaling
        
        # Clean up intermediate tensors
        del h1, m

        # Block 2: Pre-LN -> FF -> SiLU -> scaled residual
        h2 = self.ln2(x1)
        f = self.act(self.ff(h2))
        x2 = x1 + self.alpha2 * f
        
        # Clean up intermediate tensors
        del h2, f
        
        return x2
    
    def optimize_memory_usage(self):
        """Optimize memory usage by clearing cached computations."""
        # Clear GCN cache
        if hasattr(self.gcn, 'reset_parameters'):
            self.gcn.reset_parameters()
        
        print("SMPNN memory optimization completed")


class UnifiedGNNLayer(nn.Module):
    """
    Unified GNN message passing layer that can use different PyG layers.
    
    This layer automatically handles:
    - Different GNN layer types (GCN, GIN, GAT, etc.)
    - Selective attribute usage (edge_attr when supported)
    - HeteroData input format
    
    Args:
        layer_type (str): Type of GNN layer ('gcn', 'gin', 'gat', 'gatv2', 'sage', 'transformer')
        in_channels (int): Input feature dimension
        out_channels (int): Output feature dimension
        heads (int, optional): Number of attention heads (for GAT/Transformer). Default: 1
        dropout (float, optional): Dropout rate. Default: 0.0
        edge_dim (int, optional): Edge feature dimension (when supported). Default: None
        **kwargs: Additional arguments passed to the specific GNN layer
    """
    
    def __init__(
        self,
        layer_type: str,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is required but not available. Install with: pip install torch-geometric")
        
        self.layer_type = layer_type.lower()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        
        # Define which layers support edge attributes
        self.edge_attr_supported = {
            'gcn': False,
            'gin': False,
            'gat': True,
            'gatv2': True,
            'sage': False,
            'transformer': True, 
            'smpnn': False,
            'graphtransformer': True
        }
        
        # Create the appropriate GNN layer
        self.gnn_layer = self._create_gnn_layer(**kwargs)
        
        # Add dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
    def _create_gnn_layer(self, **kwargs) -> nn.Module:
        """Create the appropriate GNN layer based on layer_type."""
        
        if self.layer_type == 'gcn':
            return GCNConv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                **kwargs
            )
        
        elif self.layer_type == 'gin':
            # GIN requires a neural network for the MLP
            mlp = nn.Sequential(
                nn.Linear(self.in_channels, self.out_channels),
                nn.SiLU(),
                nn.Linear(self.out_channels, self.out_channels)
            )
            return GINConv(nn=mlp, **kwargs)
        
        elif self.layer_type == 'gat':
            return GATConv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                heads=self.heads,
                dropout=self.dropout,
                edge_dim=self.edge_dim if self.edge_dim is not None else None,
                concat=False,  # Use sum instead of concat for multi-head attention
                **kwargs
            )
        
        elif self.layer_type == 'gatv2':
            return GATv2Conv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                heads=self.heads,
                dropout=self.dropout,
                edge_dim=self.edge_dim if self.edge_dim is not None else None,
                concat=False,  # Use sum instead of concat for multi-head attention
                **kwargs
            )
        
        elif self.layer_type == 'sage':
            return SAGEConv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                **kwargs
            )
        
        elif self.layer_type == 'transformer':
            return TransformerConv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                heads=self.heads,
                dropout=self.dropout,
                edge_dim=self.edge_dim if self.edge_dim is not None else None,
                concat=False,  # Use sum instead of concat for multi-head attention
                **kwargs
            )
        
        elif self.layer_type == 'smpnn':
            return SMPNNBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                **kwargs
            )
        elif self.layer_type == 'graphtransformer':
            return GraphTransformerLayer(
                hidden_dim=self.out_channels,
                num_heads=self.heads,
                dropout=self.dropout,
                edge_dim=self.edge_dim,
                use_edge_attr=self.edge_dim is not None,
                ff_multiplier=4.0,
                use_pos_encoding=True
            )
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_type}. "
                           f"Supported types: {list(self.edge_attr_supported.keys())}")
    
    def forward(
        self, 
        data: Union[HeteroData, Dict[str, torch.Tensor]], 
        node_type: str = 'node',
        edge_type: tuple = ('node', 'connect', 'node'),
        x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the unified GNN layer.
        
        Args:
            data: Input data (HeteroData or dict with required keys)
            node_type: Node type to extract features from. Default: 'node'
            edge_type: Edge type to extract connectivity from. Default: ('node', 'connect', 'node')
            x: Optional node features to use instead of extracting from data
            
        Returns:
            torch.Tensor: Processed node features
        """
        # Extract node features (use provided x if available)
        if x is None:
            if isinstance(data, HeteroData):
                x = data[node_type].x
            elif isinstance(data, (Data, Batch)):
                x = data.x
            elif isinstance(data, dict):
                x = data.get('x', data.get('node_features'))
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Extract edge information
        if isinstance(data, HeteroData):
            edge_index = data[edge_type].edge_index
            edge_attr = getattr(data[edge_type], 'edge_attr', None)
        elif isinstance(data, (Data, Batch)):
            edge_index = data.edge_index
            edge_attr = getattr(data, 'edge_attr', None)
        elif isinstance(data, dict):
            edge_index = data.get('edge_index')
            edge_attr = data.get('edge_attr', None)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        if x is None:
            raise ValueError(f"Node features not found for node type '{node_type}'")
        if edge_index is None:
            raise ValueError(f"Edge index not found for edge type '{edge_type}'")
        
        # Apply the GNN layer based on type
        if self.layer_type == 'graphtransformer':
            # GraphTransformer layer has its own forward method signature
            out = self.gnn_layer(x, edge_index, edge_attr=edge_attr)
        else:
            # Prepare arguments for standard GNN layers
            layer_args = {
                'x': x,
                'edge_index': edge_index
            }
            
            # Add edge attributes if supported and available
            if (self.edge_attr_supported.get(self.layer_type, False) and 
                edge_attr is not None and 
                self.edge_dim is not None):
                layer_args['edge_attr'] = edge_attr
            
            # Standard GNN layers
            out = self.gnn_layer(**layer_args)
        
        # Apply dropout
        out = self.dropout_layer(out)
        
        return out
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'layer_type={self.layer_type}, '
                f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'heads={self.heads}, '
                f'dropout={self.dropout}, '
                f'edge_dim={self.edge_dim})')
    
    def optimize_memory_usage(self):
        """Optimize memory usage by clearing cached computations."""
        # Clear GNN layer cache
        if hasattr(self.gnn_layer, 'reset_parameters'):
            self.gnn_layer.reset_parameters()
        
        # Clear dropout cache
        self.dropout_layer.reset_parameters()
        
        print(f"UnifiedGNNLayer ({self.layer_type}) memory optimization completed")