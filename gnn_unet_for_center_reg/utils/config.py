"""
Configuration utilities for TopoGeoNet.

This module contains functions for loading, saving, and managing configurations.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config or {}


def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    
    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    
    for config in configs:
        merged = _deep_merge(merged, config)
    
    return merged


def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def update_config_from_args(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with command line arguments.
    
    Args:
        config: Base configuration
        args: Command line arguments dictionary
        
    Returns:
        Updated configuration
    """
    updated_config = config.copy()
    
    # Map common argument names to config keys
    arg_mapping = {
        'lr': 'optimizer.lr',
        'batch_size': 'data.batch_size',
        'epochs': 'training.num_epochs',
        'seed': 'experiment.seed',
        'device': 'training.device',
    }
    
    for arg_key, value in args.items():
        if value is None:
            continue
            
        # Check if there's a mapping
        config_key = arg_mapping.get(arg_key, arg_key)
        
        # Set nested keys
        _set_nested_key(updated_config, config_key, value)
    
    return updated_config


def _set_nested_key(config: Dict[str, Any], key_path: str, value: Any):
    """
    Set a nested key in configuration dictionary.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'optimizer.lr')
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def validate_config(config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> bool:
    """
    Validate configuration against a schema.
    
    Args:
        config: Configuration to validate
        schema: Validation schema (optional)
        
    Returns:
        True if valid, raises exception otherwise
    """
    if schema is None:
        # Basic validation for required keys
        required_sections = ['model', 'data', 'training']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        return True
    
    # Custom schema validation would go here
    return True


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for TopoGeoNet.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'experiment': {
            'name': 'topogeonet_experiment',
            'seed': 42,
            'output_dir': './outputs',
            'log_level': 'INFO',
        },
        'model': {
            'name': 'TopoGeoNet',
            'input_dim': 64,
            'hidden_dim': 128,
            'output_dim': 64,
            'num_layers': 3,
            'dropout': 0.1,
            'num_attention_heads': 8,
        },
        'data': {
            'dataset_path': './data',
            'batch_size': 32,
            'num_workers': 4,
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'shuffle': True,
        },
        'training': {
            'num_epochs': 100,
            'device': 'auto',
            'grad_clip_norm': 1.0,
            'log_every_n_steps': 10,
            'validate_every_n_epochs': 1,
        },
        'optimizer': {
            'name': 'adam',
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'betas': [0.9, 0.999],
        },
        'scheduler': {
            'name': 'cosine',
            'T_max': 100,
            'eta_min': 1e-6,
        },
        'loss': {
            'name': 'combined',
            'weights': {
                'reconstruction': 1.0,
                'topological': 0.1,
                'geometric': 0.1,
            }
        },
        'evaluation': {
            'metrics': ['mse', 'mae', 'r2'],
            'save_predictions': False,
        },
        'callbacks': {
            'early_stopping': {
                'patience': 20,
                'monitor': 'val_loss',
                'mode': 'min',
            },
            'model_checkpoint': {
                'save_best_only': True,
                'monitor': 'val_loss',
                'mode': 'min',
            },
        },
    }


def create_experiment_config(
    base_config: Optional[Dict[str, Any]] = None,
    experiment_name: Optional[str] = None,
    **overrides
) -> Dict[str, Any]:
    """
    Create configuration for a new experiment.
    
    Args:
        base_config: Base configuration (uses default if None)
        experiment_name: Name of the experiment
        **overrides: Configuration overrides
        
    Returns:
        Experiment configuration
    """
    if base_config is None:
        base_config = get_default_config()
    
    config = base_config.copy()
    
    # Set experiment name
    if experiment_name:
        config['experiment']['name'] = experiment_name
    
    # Apply overrides
    for key, value in overrides.items():
        _set_nested_key(config, key, value)
    
    return config


def load_config_with_overrides(
    config_path: Union[str, Path],
    **overrides
) -> Dict[str, Any]:
    """
    Load configuration from file and apply overrides.
    
    Args:
        config_path: Path to configuration file
        **overrides: Configuration overrides
        
    Returns:
        Configuration with overrides applied
    """
    config = load_config(config_path)
    
    # Apply overrides
    for key, value in overrides.items():
        _set_nested_key(config, key, value)
    
    return config


def print_config(config: Dict[str, Any], indent: int = 0):
    """
    Pretty print configuration.
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


def config_to_string(config: Dict[str, Any]) -> str:
    """
    Convert configuration to string representation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        String representation of configuration
    """
    return yaml.dump(config, default_flow_style=False, indent=2)
