"""
Logging utilities for TopoGeoNet.

This module contains logging setup and configuration functions.
"""

import logging
import os
import sys
from typing import Optional, Union
from datetime import datetime
import torch


def setup_logger(
    name: str = "topogeonet",
    level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to log to
        log_format: Custom log format string
        include_timestamp: Whether to include timestamp in logs
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if log_format is None:
        if include_timestamp:
            log_format = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
        else:
            log_format = "%(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "topogeonet") -> logging.Logger:
    """
    Get an existing logger or create a basic one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set up a basic one
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


def create_experiment_logger(
    experiment_name: str,
    log_dir: str = "logs",
    level: Union[str, int] = logging.INFO
) -> logging.Logger:
    """
    Create a logger for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to store log files
        level: Logging level
        
    Returns:
        Experiment logger
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create log file name
    log_filename = f"{experiment_name}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Create logger
    logger = setup_logger(
        name=f"experiment_{experiment_name}",
        level=level,
        log_file=log_path,
        include_timestamp=True
    )
    
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Log file: {log_path}")
    
    return logger


def log_system_info(logger: logging.Logger):
    """
    Log system information.
    
    Args:
        logger: Logger to use for output
    """
    import platform
    import torch
    
    logger.info("System Information:")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("CUDA: Not available")


def log_model_info(model: torch.nn.Module, logger: logging.Logger):
    """
    Log model information.
    
    Args:
        model: PyTorch model
        logger: Logger to use for output
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("Model Information:")
    logger.info(f"Model type: {type(model).__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Model size estimate
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    logger.info(f"Model size: {model_size_mb:.2f} MB")


def log_config(config: dict, logger: logging.Logger):
    """
    Log configuration dictionary.
    
    Args:
        config: Configuration dictionary
        logger: Logger to use for output
    """
    logger.info("Configuration:")
    
    def log_dict(d, prefix=""):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info(f"{prefix}{key}:")
                log_dict(value, prefix + "  ")
            else:
                logger.info(f"{prefix}{key}: {value}")
    
    log_dict(config)


class TensorBoardLogger:
    """
    Simple TensorBoard logger wrapper.
    """
    
    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
            experiment_name: Name of the experiment
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
            self.writer = SummaryWriter(self.log_dir)
            self.available = True
            
        except ImportError:
            self.writer = None
            self.available = False
            print("TensorBoard not available. Install with: pip install tensorboard")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.available:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: dict, step: int):
        """Log multiple scalar values."""
        if self.available:
            self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log a histogram of values."""
        if self.available:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """Log an image."""
        if self.available:
            self.writer.add_image(tag, image, step)
    
    def log_graph(self, model: torch.nn.Module, input_to_model: torch.Tensor):
        """Log model graph."""
        if self.available:
            self.writer.add_graph(model, input_to_model)
    
    def close(self):
        """Close the logger."""
        if self.available:
            self.writer.close()


class WandBLogger:
    """
    Weights & Biases logger wrapper with error handling.
    """
    
    def __init__(
        self, 
        project: str, 
        experiment_name: str,
        config: Optional[dict] = None,
        wandb_dir: Optional[str] = None
    ):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name
            experiment_name: Name of the experiment
            config: Configuration dictionary
            wandb_dir: Directory for wandb files
        """
        try:
            import wandb
            
            # Set wandb directory if specified
            if wandb_dir:
                os.environ['WANDB_DIR'] = wandb_dir
                os.makedirs(wandb_dir, exist_ok=True)
            
            wandb.init(
                project=project,
                name=experiment_name,
                config=config
            )
            self.wandb = wandb
            self.available = True
            
        except ImportError:
            self.wandb = None
            self.available = False
            print("W&B not available. Install with: pip install wandb")
        except Exception as e:
            self.wandb = None
            self.available = False
            print(f"Warning: Failed to initialize W&B logger: {e}")
            print("Training will continue without W&B logging")
    
    def log(self, metrics: dict, step: Optional[int] = None):
        """Log metrics with error handling."""
        if self.available:
            try:
                self.wandb.log(metrics, step=step)
            except Exception as e:
                print(f"Warning: Failed to log metrics to W&B: {e}")
                print("Training will continue without W&B logging")
    
    def watch(self, model: torch.nn.Module):
        """Watch model for gradients and parameters with error handling."""
        if self.available:
            try:
                self.wandb.watch(model)
            except Exception as e:
                print(f"Warning: Failed to watch model in W&B: {e}")
                print("Training will continue without W&B model watching")
    
    def finish(self):
        """Finish the run with error handling."""
        if self.available:
            try:
                self.wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to finish W&B run: {e}")
                print("Training will continue without W&B cleanup")
