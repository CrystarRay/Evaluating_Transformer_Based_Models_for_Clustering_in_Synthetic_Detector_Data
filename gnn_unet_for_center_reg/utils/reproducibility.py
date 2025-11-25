"""
Reproducibility utilities for TopoGeoNet.

This module contains functions for ensuring reproducible experiments.
"""

import os
import random
import numpy as np
import torch
from typing import Optional, Dict, Any


def set_seed(seed: int = 42):
    """
    Set random seed for reproducible experiments.
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for PYTHONHASHSEED
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'device_properties': [],
    }
    
    if torch.cuda.is_available():
        info['current_device'] = torch.cuda.current_device()
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                'device_id': i,
                'name': props.name,
                'total_memory': props.total_memory,
                'major': props.major,
                'minor': props.minor,
                'multi_processor_count': props.multi_processor_count,
            }
            info['device_properties'].append(device_info)
    
    return info


def get_auto_device() -> torch.device:
    """
    Automatically select the best available device.
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def setup_reproducible_training(
    seed: int = 42,
    device: Optional[str] = None
) -> torch.device:
    """
    Setup reproducible training environment.
    
    Args:
        seed: Random seed
        device: Device specification ('cpu', 'cuda', 'auto', or None for auto)
        
    Returns:
        Selected device
    """
    # Set random seed
    set_seed(seed)
    
    # Select device
    if device is None or device == 'auto':
        selected_device = get_auto_device()
    else:
        selected_device = torch.device(device)
    
    return selected_device


def check_deterministic_operations():
    """
    Check if deterministic operations are enabled.
    
    Returns:
        Dictionary with deterministic settings
    """
    return {
        'torch_deterministic': torch.backends.cudnn.deterministic,
        'torch_benchmark': torch.backends.cudnn.benchmark,
        'pythonhashseed': os.environ.get('PYTHONHASHSEED', 'Not set'),
    }


def enable_deterministic_mode():
    """
    Enable fully deterministic mode for PyTorch.
    
    Note: This may impact performance.
    """
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def disable_deterministic_mode():
    """
    Disable deterministic mode for better performance.
    """
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_environment_info() -> Dict[str, Any]:
    """
    Get comprehensive environment information for reproducibility.
    
    Returns:
        Dictionary with environment information
    """
    import platform
    import sys
    
    info = {
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
        },
        'python': {
            'version': sys.version,
            'executable': sys.executable,
        },
        'torch': {
            'version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        },
        'device_info': get_device_info(),
        'deterministic_settings': check_deterministic_operations(),
    }
    
    # Add package versions
    try:
        import pkg_resources
        installed_packages = [d for d in pkg_resources.working_set]
        
        # Get versions of key packages
        key_packages = ['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn']
        package_versions = {}
        
        for pkg in installed_packages:
            if pkg.project_name.lower() in key_packages:
                package_versions[pkg.project_name] = pkg.version
        
        info['packages'] = package_versions
        
    except ImportError:
        pass
    
    return info


def save_environment_info(filepath: str):
    """
    Save environment information to file for reproducibility.
    
    Args:
        filepath: Path to save environment info
    """
    import json
    
    env_info = get_environment_info()
    
    with open(filepath, 'w') as f:
        json.dump(env_info, f, indent=2, default=str)


def create_reproducibility_report(
    config: Dict[str, Any],
    output_dir: str = './outputs'
) -> str:
    """
    Create a comprehensive reproducibility report.
    
    Args:
        config: Experiment configuration
        output_dir: Output directory
        
    Returns:
        Path to the created report
    """
    import json
    from pathlib import Path
    from datetime import datetime
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_path / f"reproducibility_report_{timestamp}.json"
    
    report = {
        'timestamp': timestamp,
        'config': config,
        'environment': get_environment_info(),
        'git_info': _get_git_info(),
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return str(report_path)


def _get_git_info() -> Dict[str, str]:
    """
    Get git repository information if available.
    
    Returns:
        Dictionary with git information
    """
    try:
        import subprocess
        
        # Get current commit hash
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        # Get current branch
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        # Check if there are uncommitted changes
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        has_uncommitted_changes = len(status) > 0
        
        # Get remote URL
        try:
            remote_url = subprocess.check_output(
                ['git', 'config', '--get', 'remote.origin.url'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
        except subprocess.CalledProcessError:
            remote_url = 'Unknown'
        
        return {
            'commit_hash': commit_hash,
            'branch': branch,
            'has_uncommitted_changes': has_uncommitted_changes,
            'remote_url': remote_url,
        }
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            'error': 'Git information not available'
        }


class ReproducibilityContext:
    """
    Context manager for reproducible experiments.
    
    Usage:
        with ReproducibilityContext(seed=42):
            # Your experiment code here
            pass
    """
    
    def __init__(self, seed: int = 42, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self.original_state = None
    
    def __enter__(self):
        # Save original state
        self.original_state = {
            'torch_deterministic': torch.backends.cudnn.deterministic,
            'torch_benchmark': torch.backends.cudnn.benchmark,
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
        }
        
        if torch.cuda.is_available():
            self.original_state['torch_cuda_random_state'] = torch.cuda.get_rng_state()
        
        # Set reproducible state
        set_seed(self.seed)
        
        if self.deterministic:
            enable_deterministic_mode()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        if self.original_state:
            torch.backends.cudnn.deterministic = self.original_state['torch_deterministic']
            torch.backends.cudnn.benchmark = self.original_state['torch_benchmark']
            
            random.setstate(self.original_state['python_random_state'])
            np.random.set_state(self.original_state['numpy_random_state'])
            torch.set_rng_state(self.original_state['torch_random_state'])
            
            if torch.cuda.is_available() and 'torch_cuda_random_state' in self.original_state:
                torch.cuda.set_rng_state(self.original_state['torch_cuda_random_state'])
