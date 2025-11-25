"""
Compatibility module for h5py with newer NumPy versions.

This module handles the numpy.typeDict compatibility issue that occurs
when using older h5py versions with newer NumPy versions (>1.21).
"""

import warnings
from typing import Optional, Any, Dict, Tuple


def safe_import_h5py():
    """
    Safely import h5py with numpy compatibility handling.
    
    Returns:
        Tuple[h5py_module_or_None, is_available]
    """
    try:
        # Try to import h5py
        import h5py
        return h5py, True
        
    except ImportError:
        warnings.warn(
            "h5py is not installed. HDF5 file support is not available. "
            "Install h5py to enable HDF5 support: pip install h5py>=3.7.0",
            ImportWarning
        )
        return None, False
        
    except AttributeError as e:
        # This typically happens due to numpy.typeDict compatibility issues
        if "typeDict" in str(e):
            warnings.warn(
                "h5py is incompatible with the current NumPy version due to the deprecated "
                "numpy.typeDict attribute. Please upgrade h5py to >=3.7.0 or downgrade "
                "NumPy to <1.21. HDF5 file support is disabled.",
                ImportWarning
            )
        else:
            warnings.warn(
                f"h5py import failed with AttributeError: {e}. "
                "HDF5 file support is disabled.",
                ImportWarning
            )
        return None, False
        
    except Exception as e:
        warnings.warn(
            f"Unexpected error importing h5py: {e}. "
            "HDF5 file support is disabled.",
            ImportWarning
        )
        return None, False


def load_h5_file_safe(file_path):
    """
    Safely load an HDF5 file with compatibility handling.
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        Dict containing the loaded data
        
    Raises:
        RuntimeError: If h5py is not available or file cannot be loaded
    """
    h5py, is_available = safe_import_h5py()
    
    if not is_available:
        raise RuntimeError(
            f"Cannot load HDF5 file '{file_path}': h5py is not available or incompatible. "
            "Please install a compatible version of h5py (>=3.7.0) or convert the file "
            "to a different format (.pt, .pkl, .npy)."
        )
    
    try:
        data = {}
        with h5py.File(file_path, 'r') as f:
            def _recursive_load(group, data_dict):
                """Recursively load HDF5 group data."""
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Dataset):
                        data_dict[key] = item[:]
                    elif isinstance(item, h5py.Group):
                        data_dict[key] = {}
                        _recursive_load(item, data_dict[key])
                    else:
                        # Handle other types as needed
                        try:
                            data_dict[key] = item[:]
                        except Exception:
                            data_dict[key] = str(item)
            
            _recursive_load(f, data)
        
        return data
        
    except Exception as e:
        raise RuntimeError(f"Failed to load HDF5 file '{file_path}': {e}")


def is_h5py_available():
    """
    Check if h5py is available and compatible.
    
    Returns:
        bool: True if h5py can be used, False otherwise
    """
    _, is_available = safe_import_h5py()
    return is_available


def get_h5py_info():
    """
    Get information about h5py availability and version.
    
    Returns:
        Dict with h5py status information
    """
    h5py, is_available = safe_import_h5py()
    
    if is_available and h5py is not None:
        try:
            version = h5py.__version__
            hdf5_version = h5py.version.hdf5_version
        except Exception:
            version = "unknown"
            hdf5_version = "unknown"
            
        return {
            "available": True,
            "h5py_version": version,
            "hdf5_version": hdf5_version,
            "module": h5py
        }
    else:
        return {
            "available": False,
            "h5py_version": None,
            "hdf5_version": None,
            "module": None
        }


def recommend_h5py_fix():
    """
    Provide recommendations for fixing h5py compatibility issues.
    
    Returns:
        String with recommendations
    """
    return """
To fix h5py compatibility issues with newer NumPy versions:

1. Upgrade h5py to a compatible version:
   pip install "h5py>=3.7.0"

2. Or, if you must use an older h5py version, downgrade NumPy:
   pip install "numpy<1.21.0"

3. Alternative: Convert your HDF5 files to other formats:
   - PyTorch format (.pt): torch.save(data, 'file.pt')
   - Pickle format (.pkl): pickle.dump(data, open('file.pkl', 'wb'))
   - NumPy format (.npy): np.save('file.npy', data)

4. For conda users:
   conda install h5py>=3.7.0
"""