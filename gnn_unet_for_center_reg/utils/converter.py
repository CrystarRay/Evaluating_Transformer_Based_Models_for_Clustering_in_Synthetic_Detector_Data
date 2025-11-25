"""
Utility classes for converting between node features and utility maps.

This module contains the UtilMapConverter class that handles downsampling/upsampling
between node features and 2D utility maps for spatial processing.
"""

import torch
from typing import Optional, Union, Literal


class UtilMapConverter:
    """
    Utility class for converting between node features and 2D utility maps.
    
    This class handles the conversion between node features and 2D spatial representations
    (utility maps) for use with convolutional operations like UNet. It can perform
    both downsampling (node_features -> util_map) and upsampling (util_map -> node_features).
    
    Args:
        assignment (torch.Tensor): Assignment tensor mapping node indices to spatial positions
                                  Shape: [num_nodes, 2] where each row is [x_idx, y_idx]
        num_sites_x (int): Number of spatial sites in x dimension
        num_sites_y (int): Number of spatial sites in y dimension
        device (torch.device): Device to place tensors on
    """
    
    def __init__(self, assignment: torch.Tensor, num_sites_x: int, num_sites_y: int, device: torch.device):
        print(f"Debug: Converter constructor called with num_sites_x={num_sites_x} (type: {type(num_sites_x)}), num_sites_y={num_sites_y} (type: {type(num_sites_y)})")
        self.shape = (int(num_sites_x), int(num_sites_y))
        self.device = device
        self.flat_size = int(num_sites_x) * int(num_sites_y)
        print(f"Debug: Calculated flat_size={self.flat_size} (type: {type(self.flat_size)})")
        self.assignment = assignment.to(device)
        
        # Convert 2D indices to 1D flat indices for efficient scatter operations
        # assignment[:, 0] = x indices, assignment[:, 1] = y indices
        self.indices = (self.assignment[:, 0].long() * num_sites_y + self.assignment[:, 1].long()).to(device)
        self._expanded_indices = None
        
    def to_util_map(self, node_features: torch.Tensor, max_sum: Literal['sum', 'amax'] = 'sum') -> torch.Tensor:
        """
        Convert node features to a 2D utility map.
        
        Args:
            node_features (torch.Tensor): Node features tensor [num_nodes, num_channels]
            max_sum (str): Aggregation method - 'sum' for summation, 'amax' for max pooling
            
        Returns:
            torch.Tensor: 2D utility map [num_sites_x, num_sites_y, num_channels]
        """
        num_channels = node_features.shape[1]
        
        # Get or create expanded indices for broadcasting
        if self._expanded_indices is None or self._expanded_indices.shape[1] != num_channels:
            self._expanded_indices = self.indices.unsqueeze(1).expand(-1, num_channels)
            
        if max_sum == 'sum':
            # Sum aggregation: multiple nodes in same patch get summed
            util_map = torch.zeros(self.flat_size, num_channels, device=self.device)
            util_map.scatter_add_(0, self._expanded_indices, node_features)
        elif max_sum == 'amax':
            # Max aggregation: multiple nodes in same patch get max value
            min_value = torch.min(node_features).item()
            util_map = torch.full((self.flat_size, num_channels), min_value, device=self.device)
            util_map.scatter_reduce_(0, self._expanded_indices, node_features, reduce='amax')
        else:
            raise ValueError(f"Invalid max_sum value: {max_sum}. Must be 'sum' or 'amax'")
            
        # Reshape to 2D spatial format
        return util_map.view(*self.shape, num_channels)
    
    def to_node_features(self, util_map: torch.Tensor) -> torch.Tensor:
        """
        Convert 2D utility map back to node features using the stored assignment.
        
        Args:
            util_map (torch.Tensor): 2D utility map [num_sites_x, num_sites_y, num_channels]
                                    or flattened [num_sites_x * num_sites_y, num_channels]
            
        Returns:
            torch.Tensor: Node features [num_nodes, num_channels]
        """
        # Handle both 2D and flattened input
        if util_map.dim() == 3:
            # Input is [num_sites_x, num_sites_y, num_channels]
            util_map_flat = util_map.view(self.flat_size, -1)
        elif util_map.dim() == 2:
            # Input is already flattened [num_sites_x * num_sites_y, num_channels]
            util_map_flat = util_map
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {util_map.dim()}D")
            
        # Extract features at the assigned positions
        return util_map_flat[self.indices]
    
    def get_spatial_info(self) -> dict:
        """
        Get information about the spatial configuration.
        
        Returns:
            dict: Dictionary containing spatial information
        """
        return {
            'shape': self.shape,
            'flat_size': self.flat_size,
            'num_nodes': len(self.assignment),
            'device': self.device,
            'assignment_range': {
                'x_min': self.assignment[:, 0].min().item(),
                'x_max': self.assignment[:, 0].max().item(),
                'y_min': self.assignment[:, 1].min().item(),
                'y_max': self.assignment[:, 1].max().item()
            }
        }


def create_assignment_from_coordinates(
    coordinates: torch.Tensor, 
    num_sites_x: int = 250, 
    num_sites_y: int = 250,
    normalize: bool = True
) -> torch.Tensor:
    """
    Create assignment tensor from node coordinates.
    
    This function takes 2D coordinates and assigns each node to a spatial patch
    in a uniform grid of size num_sites_x × num_sites_y.
    
    Args:
        coordinates (torch.Tensor): Node coordinates [num_nodes, 2] (x, y)
        num_sites_x (int): Number of spatial sites in x dimension
        num_sites_y (int): Number of spatial sites in y dimension
        normalize (bool): Whether to normalize coordinates to [0, 1] before assignment
        
    Returns:
        torch.Tensor: Assignment tensor [num_nodes, 2] where each row is [x_idx, y_idx]
    """
    if coordinates.numel() == 0:
        raise ValueError("Cannot create assignment from empty coordinates tensor")
    
    if coordinates.shape[1] < 2:
        raise ValueError(f"Coordinates must have at least 2 dimensions, got {coordinates.shape[1]}")
    
    # Extract x, y coordinates (first two dimensions)
    x_coords = coordinates[:, 0]
    y_coords = coordinates[:, 1]
    
    if normalize:
        # Normalize coordinates to [0, 1] range
        x_min, x_max = x_coords.min(dim=0).values, x_coords.max(dim=0).values
        y_min, y_max = y_coords.min(dim=0).values, y_coords.max(dim=0).values
        
        if x_max > x_min:
            x_coords = (x_coords - x_min) / (x_max - x_min)
        if y_max > y_min:
            y_coords = (y_coords - y_min) / (y_max - y_min)
    
    # Convert to spatial indices
    x_indices = torch.floor(x_coords * (num_sites_x - 1)).clamp(0, num_sites_x - 1)
    y_indices = torch.floor(y_coords * (num_sites_y - 1)).clamp(0, num_sites_y - 1)
    
    # Stack into assignment tensor
    assignment = torch.stack([x_indices, y_indices], dim=1)
    
    return assignment
