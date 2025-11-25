"""
Model components for TopoGeoNet.

This module contains reusable components like encoders, decoders, and graph operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class FourierFeatureEncoder(nn.Module):
    """
    Robust Fourier Feature Encoder for positional encoding with multi-scale bands.
    
    Maps input coordinates to higher-dimensional space using random Fourier features
    with multi-scale frequency bands optimized for terrain/spatial data.
    
    Args:
        input_dim (int): Input coordinate dimension (e.g., 3 for 3D coordinates)
        mapping_size (int): Size of the output feature mapping (must be even)
        passthrough (bool): Whether to append raw coordinates to output
        normalize_coords (bool): Whether to normalize coordinates to [0,1] range
        multi_scale (bool): Whether to use multi-scale frequency bands (recommended)
        sigmas (tuple): Multi-scale bandwidths (log-spaced for terrain features)
        seed (int, optional): Random seed for reproducibility
    """
    
    def __init__(
        self, 
        input_dim: int = 3, 
        mapping_size: int = 128, 
        passthrough: bool = True,  # Default to True to include raw coordinates
        normalize_coords: bool = True,  # Default to True for robust performance
        multi_scale: bool = True,  # Default to True for terrain features
        sigmas: tuple = (1/(4*math.pi), 1/(2*math.pi), 1/math.pi, 2/math.pi, 4/math.pi),  # Log-spaced bands
        seed: int = None
    ):
        super().__init__()
        assert mapping_size % 2 == 0, "mapping_size must be even"
        
        self.input_dim = input_dim
        self.passthrough = passthrough
        self.mapping_size = mapping_size
        self.normalize_coords = normalize_coords
        self.multi_scale = multi_scale
        
        if multi_scale:
            # Multi-scale implementation with log-spaced sigmas
            self.sigmas = tuple(sigmas)
            half = mapping_size // 2
            
            # Distribute mapping size evenly across bands
            per_band = max(1, half // len(self.sigmas))
            counts = [per_band] * len(self.sigmas)
            counts[-1] = half - per_band * (len(self.sigmas) - 1)  # Last band absorbs remainder
            
            rng = torch.Generator()
            if seed is not None:
                rng.manual_seed(seed)
            
            Bs = []
            for sigma_band, c in zip(self.sigmas, counts):
                # Generate random directions and scale by sigma
                B = torch.randn(input_dim, c, generator=rng) * sigma_band
                Bs.append(B)
            B = torch.cat(Bs, dim=1)
        else:
            # Single scale implementation (fallback)
            rng = torch.Generator()
            if seed is not None:
                rng.manual_seed(seed)
            # Use middle sigma as default for single scale
            default_sigma = 1/math.pi  # π^-1 as middle value
            B = torch.randn(input_dim, mapping_size // 2, generator=rng) * default_sigma
        
        self.register_buffer("B", B)  # device-safe, no grads
        
        # Log the configuration for debugging
        if multi_scale:
            print(f"FourierFeatureEncoder: Multi-scale with {len(self.sigmas)} bands")
            print(f"  Sigmas: {[f'{s:.4f}' for s in self.sigmas]}")
            print(f"  Covers wavelengths from ≈{min(self.sigmas)/math.pi:.3f} to ≈{max(self.sigmas)/math.pi:.3f} in normalized units")
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of robust Fourier feature encoder.
        
        Args:
            coords: Input coordinates [..., input_dim] (will be normalized if normalize_coords=True)
            
        Returns:
            torch.Tensor: Fourier features [..., mapping_size + input_dim if passthrough]
        """
        # Normalize coordinates to [0,1] range if enabled
        if self.normalize_coords:
            # Find min/max for each coordinate dimension
            coords_min = coords.min(dim=0, keepdim=True)[0]
            coords_max = coords.max(dim=0, keepdim=True)[0]
            # Avoid division by zero
            coords_range = coords_max - coords_min
            coords_range = torch.where(coords_range < 1e-8, 1.0, coords_range)
            coords_normalized = (coords - coords_min) / coords_range
        else:
            coords_normalized = coords
        
        # Promote dtype/device to match coords
        B = self.B.to(dtype=coords_normalized.dtype, device=coords_normalized.device)
        
        # Compute Fourier features
        x = 2 * math.pi * coords_normalized @ self.B   # [..., mapping_size/2]
        fourier_features = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # [..., mapping_size]
        
        # Include raw coordinates if passthrough is enabled
        if self.passthrough:
            return torch.cat([coords_normalized, fourier_features], dim=-1)  # [..., input_dim + mapping_size]
        else:
            return fourier_features  # [..., mapping_size]


def sinusoidal_2d(xy: torch.Tensor, K: int = 64) -> torch.Tensor:
    """
    2D sinusoidal positional encoding function.
    
    Creates sinusoidal positional encodings for 2D coordinates using different
    frequencies for x and y dimensions.
    
    Args:
        xy: Input coordinates [..., 2] with xy[...,0]=x, xy[...,1]=y
        K: Number of frequency components per dimension
        
    Returns:
        torch.Tensor: Sinusoidal encodings [..., 4*K]
    """
    # xy: [..., 2] with xy[...,0]=x, xy[...,1]=y
    d = K
    i = torch.arange(d, device=xy.device)
    omega = 1.0 / (10000 ** (2*i.float()/d))
    
    x, y = xy[..., 0:1], xy[..., 1:2]
    
    # Positional encoding for x dimension
    Px = torch.cat([torch.sin(x*omega), torch.cos(x*omega)], dim=-1)  # [..., 2K]
    
    # Positional encoding for y dimension  
    Py = torch.cat([torch.sin(y*omega), torch.cos(y*omega)], dim=-1)  # [..., 2K]
    
    return torch.cat([Px, Py], dim=-1)  # [..., 4K]


class SinusoidalEncoder(nn.Module):
    """
    Sinusoidal positional encoder for 2D coordinates.
    
    Wrapper class for the sinusoidal_2d function to make it compatible
    with PyTorch module structure.
    
    Args:
        K (int): Number of frequency components per dimension
    """
    
    def __init__(self, K: int = 64):
        super().__init__()
        self.K = K
        self.output_dim = 4 * K
    
    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of sinusoidal encoder.
        
        Args:
            xy: Input coordinates [..., 2]
            
        Returns:
            torch.Tensor: Sinusoidal encodings [..., 4*K]
        """
        return sinusoidal_2d(xy, self.K)


class MLPEncoder(nn.Module):
    """
    MLP encoder for initial node feature processing.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output feature dimension
        num_layers (int): Number of hidden layers
        dropout (float): Dropout rate
        use_batch_norm (bool): Whether to use batch normalization
        activation (str): Activation function ('silu', 'leaky_relu', 'gelu')
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = 'silu'
    ):
        super().__init__()
        
        # Choose activation function
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.SiLU()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(self.activation)
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MLP encoder.
        
        Args:
            x: Input features [..., input_dim]
            
        Returns:
            torch.Tensor: Encoded features [..., output_dim]
        """
        return self.mlp(x)


def _create_conv_block(in_channels: int, out_channels: int, dropout: float = 0.1, use_batch_norm: bool = True) -> nn.Sequential:
    """
    Create a convolution block with two convolutions.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization
        
    Returns:
        nn.Sequential: Convolution block
    """
    layers = []
    
    # First convolution
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.SiLU())
    layers.append(nn.Dropout2d(dropout))
    
    # Second convolution
    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.SiLU())
    layers.append(nn.Dropout2d(dropout))
    
    return nn.Sequential(*layers)


class UNet(nn.Module):
    """
    Configurable UNet model for segmentation and dense prediction tasks.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        base_channels (int): Base number of channels (doubled at each level)
        depth (int): Number of encoder/decoder levels
        dropout (float): Dropout rate
        use_batch_norm (bool): Whether to use batch normalization
        upsampling_mode (str): Upsampling mode ('transpose' or 'bilinear')
        activation (str): Final activation function ('sigmoid', 'softmax', 'none')
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        upsampling_mode: str = 'transpose',
        activation: str = 'none'
    ):
        super().__init__()
        
        self.depth = depth
        self.activation = activation
        
        # Input convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels) if use_batch_norm else nn.Identity(),
            nn.SiLU()
        )
        
        # Encoder blocks
        self.encoders = nn.ModuleList()
        in_ch = base_channels
        
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(_create_conv_block(in_ch, out_ch, dropout, use_batch_norm))
            in_ch = out_ch
        
        # Bottleneck
        bottleneck_channels = base_channels * (2 ** depth)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch, bottleneck_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_channels) if use_batch_norm else nn.Identity(),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_channels) if use_batch_norm else nn.Identity(),
            nn.SiLU(),
            nn.Dropout2d(dropout)
        )
        
        # Decoder blocks
        self.decoders = nn.ModuleList()
        
        for i in range(depth):
            level = depth - 1 - i  # Reverse order
            
            # Input channels from previous decoder (or bottleneck)
            if i == 0:
                dec_in_channels = bottleneck_channels
            else:
                dec_in_channels = base_channels * (2 ** (level + 1))
            
            # Skip connection channels from corresponding encoder
            skip_channels = base_channels * (2 ** level)
            
            # Output channels
            dec_out_channels = base_channels * (2 ** level)
            
            # Create upsampling layer
            if upsampling_mode == 'transpose':
                upsample = nn.Conv2d(
                    dec_in_channels, dec_in_channels // 2, 
                    kernel_size=2, stride=2
                )
            else:  # bilinear
                upsample = nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=True
                )
                upsample_conv = nn.Conv2d(dec_in_channels, dec_in_channels // 2, kernel_size=1)
            
            # Create convolution block after concatenation
            conv_in_channels = (dec_in_channels // 2) + skip_channels
            conv_block = _create_conv_block(conv_in_channels, dec_out_channels, dropout, use_batch_norm)
            
            # Create a simple decoder module
            decoder = nn.Module()
            decoder.upsample = upsample
            if upsampling_mode == 'bilinear':
                decoder.upsample_conv = upsample_conv
            decoder.conv_block = conv_block
            decoder.upsampling_mode = upsampling_mode
            
            self.decoders.append(decoder)
        
        # Final output convolution
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # Final activation
        if activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif activation == 'tanh':
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through UNet.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Output predictions [B, out_channels, H, W]
        """
        # Initial convolution
        x = self.input_conv(x)
        
        # Encoder path
        skip_connections = []
        current = x
        
        for encoder in self.encoders:
            # Apply convolution block
            current = encoder(current)
            # Store features before pooling for skip connection
            skip_connections.append(current)
            # Apply pooling
            current = F.max_pool2d(current, kernel_size=2, stride=2)
        
        # Bottleneck
        current = self.bottleneck(current)
        
        # Decoder path
        for i, decoder in enumerate(self.decoders):
            # Get corresponding skip connection (in reverse order)
            skip_idx = self.depth - 1 - i
            skip = skip_connections[skip_idx]
            
            # Upsample
            if decoder.upsampling_mode == 'transpose':
                current = decoder.upsample(current)
            else:  # bilinear
                current = decoder.upsample(current)
                current = decoder.upsample_conv(current)
            
            # Handle size mismatch between upsampled current and skip connection
            if current.size() != skip.size():
                # Pad or crop to match skip connection size
                diff_h = skip.size(2) - current.size(2)
                diff_w = skip.size(3) - current.size(3)
                
                if diff_h > 0 or diff_w > 0:
                    # Pad current to match skip
                    current = F.pad(current, [diff_w // 2, diff_w - diff_w // 2, 
                                         diff_h // 2, diff_h - diff_h // 2])
                elif diff_h < 0 or diff_w < 0:
                    # Crop current to match skip
                    current = current[:, :, :skip.size(2), :skip.size(3)]
            
            # Concatenate with skip connection
            current = torch.cat([current, skip], dim=1)
            
            # Apply convolution block
            current = decoder.conv_block(current)
        
        # Final output
        output = self.output_conv(current)
        output = self.final_activation(output)
        
        return output
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate feature maps for visualization/analysis.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing feature maps at different levels
        """
        features = {}
        
        # Initial convolution
        x = self.input_conv(x)
        features['input_conv'] = x
        
        # Encoder path
        skip_connections = []
        current = x
        
        for i, encoder in enumerate(self.encoders):
            # Apply convolution block
            current = encoder(current)
            # Store features before pooling for skip connection
            skip_connections.append(current)
            features[f'encoder_{i}'] = current
            # Apply pooling
            current = F.max_pool2d(current, kernel_size=2, stride=2)
            features[f'encoder_{i}_pooled'] = current
        
        # Bottleneck
        current = self.bottleneck(current)
        features['bottleneck'] = current
        
        # Decoder path
        for i, decoder in enumerate(self.decoders):
            skip_idx = self.depth - 1 - i
            skip = skip_connections[skip_idx]
            
            # Upsample
            if decoder.upsampling_mode == 'transpose':
                current = decoder.upsample(current)
            else:  # bilinear
                current = decoder.upsample(current)
                current = decoder.upsample_conv(current)
            
            # Handle size mismatch
            if current.size() != skip.size():
                diff_h = skip.size(2) - current.size(2)
                diff_w = skip.size(3) - current.size(3)
                
                if diff_h > 0 or diff_w > 0:
                    current = F.pad(current, [diff_w // 2, diff_w - diff_w // 2, 
                                         diff_h // 2, diff_h - diff_h // 2])
                elif diff_h < 0 or diff_w < 0:
                    current = current[:, :, :skip.size(2), :skip.size(3)]
            
            # Concatenate with skip connection
            current = torch.cat([current, skip], dim=1)
            
            # Apply convolution block
            current = decoder.conv_block(current)
            features[f'decoder_{i}'] = current
        
        # Final output
        output = self.output_conv(current)
        output = self.final_activation(output)
        features['output'] = output
        
        return features
