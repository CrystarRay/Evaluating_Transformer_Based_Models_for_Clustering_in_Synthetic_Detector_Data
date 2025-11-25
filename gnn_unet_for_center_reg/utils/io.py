"""
Input/output utilities for TopoGeoNet.

This module contains functions for saving and loading models, results, and data.
"""

import os
import pickle
import torch
import numpy as np
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path


def save_model(
    model: torch.nn.Module,
    filepath: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save model checkpoint with optional optimizer and scheduler states.
    
    Args:
        model: PyTorch model to save
        filepath: Path to save the model
        optimizer: Optimizer to save (optional)
        scheduler: Learning rate scheduler to save (optional)
        epoch: Current epoch number (optional)
        metadata: Additional metadata to save (optional)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['optimizer_class'] = optimizer.__class__.__name__
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        checkpoint['scheduler_class'] = scheduler.__class__.__name__
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    # Save checkpoint
    torch.save(checkpoint, filepath)


def load_model(
    filepath: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Union[str, torch.device] = 'cpu',
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to model checkpoint
        model: Model to load state into (optional)
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to map tensors to
        strict: Whether to strictly enforce state dict keys match
        
    Returns:
        Dictionary with loaded checkpoint data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {filepath}")
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def save_results(
    results: Dict[str, Any],
    filepath: Union[str, Path],
    format: str = 'json'
):
    """
    Save experiment results to file.
    
    Args:
        results: Results dictionary
        filepath: Path to save results
        format: File format ('json', 'pickle', 'npz')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        json_results = _convert_arrays_to_lists(results)
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    elif format == 'npz':
        # Save as numpy compressed format
        arrays_dict = _extract_arrays(results)
        np.savez_compressed(filepath, **arrays_dict)
        
        # Save non-array data as JSON
        metadata = _extract_non_arrays(results)
        if metadata:
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load experiment results from file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Results dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            results = json.load(f)
        # Convert lists back to numpy arrays where appropriate
        return _convert_lists_to_arrays(results)
    
    elif filepath.suffix in ['.pkl', '.pickle']:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    elif filepath.suffix == '.npz':
        # Load arrays
        arrays = dict(np.load(filepath))
        
        # Load metadata if exists
        metadata_path = filepath.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            arrays.update(metadata)
        
        return arrays
    
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def save_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save model predictions and targets.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        filepath: Path to save predictions
        metadata: Additional metadata
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'predictions': predictions,
        'targets': targets,
    }
    
    if metadata is not None:
        data['metadata'] = metadata
    
    np.savez_compressed(filepath, **data)


def load_predictions(filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Load saved predictions and targets.
    
    Args:
        filepath: Path to predictions file
        
    Returns:
        Dictionary with predictions and targets
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Predictions file not found: {filepath}")
    
    return dict(np.load(filepath))


def save_embeddings(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray],
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save learned embeddings.
    
    Args:
        embeddings: Learned embeddings
        labels: Corresponding labels (optional)
        filepath: Path to save embeddings
        metadata: Additional metadata
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    data = {'embeddings': embeddings}
    
    if labels is not None:
        data['labels'] = labels
    
    if metadata is not None:
        data['metadata'] = metadata
    
    np.savez_compressed(filepath, **data)


def load_embeddings(filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Load saved embeddings.
    
    Args:
        filepath: Path to embeddings file
        
    Returns:
        Dictionary with embeddings and optional labels
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Embeddings file not found: {filepath}")
    
    return dict(np.load(filepath))


def create_directory_structure(base_dir: Union[str, Path], subdirs: list):
    """
    Create directory structure for experiments.
    
    Args:
        base_dir: Base directory path
        subdirs: List of subdirectories to create
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    for subdir in subdirs:
        (base_path / subdir).mkdir(parents=True, exist_ok=True)


def get_experiment_dirs(experiment_name: str, base_dir: str = './outputs') -> Dict[str, Path]:
    """
    Get standardized directory structure for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        base_dir: Base output directory
        
    Returns:
        Dictionary with directory paths
    """
    base_path = Path(base_dir) / experiment_name
    
    dirs = {
        'base': base_path,
        'models': base_path / 'models',
        'logs': base_path / 'logs',
        'results': base_path / 'results',
        'figures': base_path / 'figures',
        'configs': base_path / 'configs',
    }
    
    # Create directories
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def _convert_arrays_to_lists(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_arrays_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_arrays_to_lists(item) for item in obj]
    else:
        return obj


def _convert_lists_to_arrays(obj):
    """Convert lists back to numpy arrays where appropriate."""
    if isinstance(obj, dict):
        return {key: _convert_lists_to_arrays(value) for key, value in obj.items()}
    elif isinstance(obj, list) and len(obj) > 0:
        # Check if this looks like a numeric array
        if all(isinstance(x, (int, float)) for x in obj):
            return np.array(obj)
        else:
            return [_convert_lists_to_arrays(item) for item in obj]
    else:
        return obj


def _extract_arrays(obj, prefix=''):
    """Extract numpy arrays from nested dictionary."""
    arrays = {}
    
    if isinstance(obj, np.ndarray):
        return {prefix: obj}
    elif isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}_{key}" if prefix else key
            arrays.update(_extract_arrays(value, new_prefix))
    
    return arrays


def _extract_non_arrays(obj, prefix=''):
    """Extract non-array data from nested dictionary."""
    non_arrays = {}
    
    if isinstance(obj, np.ndarray):
        return {}
    elif isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}_{key}" if prefix else key
            if isinstance(value, np.ndarray):
                continue
            elif isinstance(value, dict):
                sub_data = _extract_non_arrays(value, new_prefix)
                non_arrays.update(sub_data)
            else:
                non_arrays[new_prefix] = value
    else:
        non_arrays[prefix] = obj
    
    return non_arrays
