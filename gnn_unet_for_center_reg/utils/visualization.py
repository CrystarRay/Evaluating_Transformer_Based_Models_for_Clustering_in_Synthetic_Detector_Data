"""
Visualization utilities for TopoGeoNet.

This module contains functions for plotting and visualizing results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path


def plot_training_curves(
    train_history: List[Dict[str, float]],
    val_history: Optional[List[Dict[str, float]]] = None,
    metrics: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        train_history: List of training metrics per epoch
        val_history: List of validation metrics per epoch (optional)
        metrics: Specific metrics to plot (if None, plot all)
        save_path: Path to save figure (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not train_history:
        raise ValueError("Training history is empty")
    
    # Determine metrics to plot
    if metrics is None:
        metrics = list(train_history[0].keys())
        # Remove non-numeric keys
        metrics = [m for m in metrics if isinstance(train_history[0].get(m), (int, float))]
    
    # Create subplots
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    epochs = range(1, len(train_history) + 1)
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Extract metric values
        train_values = [epoch_data.get(metric, 0) for epoch_data in train_history]
        
        # Plot training curve
        ax.plot(epochs, train_values, label=f'Train {metric}', marker='o', markersize=3)
        
        # Plot validation curve if available
        if val_history and len(val_history) > 0:
            val_values = [epoch_data.get(metric, 0) for epoch_data in val_history]
            ax.plot(epochs, val_values, label=f'Val {metric}', marker='s', markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.title())
        ax.set_title(f'{metric.title()} vs Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8),
    sample_size: Optional[int] = 1000
) -> plt.Figure:
    """
    Plot predictions vs targets.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        save_path: Path to save figure (optional)
        figsize: Figure size
        sample_size: Number of samples to plot (for large datasets)
        
    Returns:
        Matplotlib figure
    """
    # Flatten arrays
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Sample data if too large
    if sample_size and len(pred_flat) > sample_size:
        indices = np.random.choice(len(pred_flat), sample_size, replace=False)
        pred_flat = pred_flat[indices]
        target_flat = target_flat[indices]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(target_flat, pred_flat, alpha=0.6, s=10)
    
    # Perfect prediction line
    min_val = min(target_flat.min(), pred_flat.min())
    max_val = max(target_flat.max(), pred_flat.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Predictions vs True Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    ax2 = axes[1]
    residuals = pred_flat - target_flat
    ax2.scatter(target_flat, residuals, alpha=0.6, s=10)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Residuals (Predicted - True)')
    ax2.set_title('Residuals vs True Values')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_embeddings(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = 'tsne',
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot 2D visualization of embeddings.
    
    Args:
        embeddings: High-dimensional embeddings
        labels: Optional labels for coloring points
        method: Dimensionality reduction method ('tsne', 'pca', 'umap')
        save_path: Path to save figure (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Reduce dimensionality to 2D
    if embeddings.shape[1] > 2:
        if method == 'tsne':
            try:
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
                embeddings_2d = reducer.fit_transform(embeddings)
            except ImportError:
                raise ImportError("scikit-learn required for t-SNE")
        
        elif method == 'pca':
            try:
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2)
                embeddings_2d = reducer.fit_transform(embeddings)
            except ImportError:
                raise ImportError("scikit-learn required for PCA")
        
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                embeddings_2d = reducer.fit_transform(embeddings)
            except ImportError:
                raise ImportError("umap-learn required for UMAP")
        
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        embeddings_2d = embeddings
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot embeddings
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                c=[colors[i]], 
                label=f'Class {label}',
                alpha=0.7,
                s=20
            )
        ax.legend()
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=20)
    
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(f'Embedding Visualization ({method.upper()})')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_graph_structure(
    adjacency_matrix: np.ndarray,
    node_features: Optional[np.ndarray] = None,
    node_labels: Optional[np.ndarray] = None,
    layout: str = 'spring',
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot graph structure.
    
    Args:
        adjacency_matrix: Graph adjacency matrix
        node_features: Node features for sizing/coloring (optional)
        node_labels: Node labels (optional)
        layout: Graph layout algorithm ('spring', 'circular', 'random')
        save_path: Path to save figure (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX required for graph visualization")
    
    # Create graph from adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'random':
        pos = nx.random_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray', ax=ax)
    
    # Draw nodes
    if node_features is not None:
        # Use first feature for node size
        node_sizes = node_features[:, 0] if node_features.ndim > 1 else node_features
        node_sizes = (node_sizes - node_sizes.min()) / (node_sizes.max() - node_sizes.min() + 1e-8)
        node_sizes = 100 + node_sizes * 400  # Scale to reasonable size range
    else:
        node_sizes = 200
    
    if node_labels is not None:
        # Color by labels
        unique_labels = np.unique(node_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        node_colors = [colors[np.where(unique_labels == label)[0][0]] for label in node_labels]
    else:
        node_colors = 'lightblue'
    
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8,
        ax=ax
    )
    
    # Add labels if requested
    if node_labels is not None and len(G.nodes()) <= 50:  # Only for small graphs
        nx.draw_networkx_labels(G, pos, ax=ax)
    
    ax.set_title(f'Graph Structure ({len(G.nodes())} nodes, {len(G.edges())} edges)')
    ax.axis('off')
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_loss_landscape(
    model: 'torch.nn.Module',
    dataloader: 'torch.utils.data.DataLoader',
    criterion: 'torch.nn.Module',
    center_point: Optional[Dict[str, 'torch.Tensor']] = None,
    resolution: int = 25,
    distance: float = 1.0,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot loss landscape around current model parameters.
    
    Args:
        model: PyTorch model
        dataloader: Data loader for computing loss
        criterion: Loss function
        center_point: Center point for landscape (current params if None)
        resolution: Resolution of the grid
        distance: Distance to explore around center
        save_path: Path to save figure (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    import torch
    
    device = next(model.parameters()).device
    
    # Get current parameters as center point
    if center_point is None:
        center_point = {name: param.clone() for name, param in model.named_parameters()}
    
    # Generate random directions
    direction1 = {}
    direction2 = {}
    
    for name, param in model.named_parameters():
        direction1[name] = torch.randn_like(param)
        direction2[name] = torch.randn_like(param)
        
        # Normalize directions
        direction1[name] = direction1[name] / torch.norm(direction1[name])
        direction2[name] = direction2[name] / torch.norm(direction2[name])
    
    # Create grid
    alpha_range = np.linspace(-distance, distance, resolution)
    beta_range = np.linspace(-distance, distance, resolution)
    
    loss_grid = np.zeros((resolution, resolution))
    
    # Evaluate loss at each grid point
    model.eval()
    with torch.no_grad():
        for i, alpha in enumerate(alpha_range):
            for j, beta in enumerate(beta_range):
                # Set model parameters
                for name, param in model.named_parameters():
                    new_param = (
                        center_point[name] + 
                        alpha * direction1[name] + 
                        beta * direction2[name]
                    )
                    param.data.copy_(new_param)
                
                # Compute loss
                total_loss = 0.0
                num_batches = 0
                
                for batch in dataloader:
                    if isinstance(batch, dict):
                        # Move batch to device
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                batch[key] = value.to(device)
                        
                        if 'features' in batch:
                            outputs = model(batch['features'], 
                                           edge_index=batch.get('edge_index'))
                        else:
                            outputs = model(batch['input'])
                        
                        if 'target' in batch:
                            loss = criterion(outputs, batch['target'])
                        else:
                            loss = criterion(outputs, batch['features'])
                    else:
                        inputs, targets = batch
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Limit evaluation for speed
                    if num_batches >= 10:
                        break
                
                loss_grid[i, j] = total_loss / num_batches
    
    # Restore original parameters
    for name, param in model.named_parameters():
        param.data.copy_(center_point[name])
    
    # Create contour plot
    fig, ax = plt.subplots(figsize=figsize)
    
    contour = ax.contour(beta_range, alpha_range, loss_grid, levels=20)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Mark center point
    ax.plot(0, 0, 'r*', markersize=15, label='Current parameters')
    
    ax.set_xlabel('Direction 2')
    ax.set_ylabel('Direction 1')
    ax.set_title('Loss Landscape')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def save_figure(fig: plt.Figure, filepath: Union[str, Path], dpi: int = 300):
    """
    Save matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure
        filepath: Path to save figure
        dpi: Resolution for raster formats
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')


def create_comparison_plot(
    results_dict: Dict[str, Dict[str, float]],
    metric: str,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create comparison plot for multiple models/experiments.
    
    Args:
        results_dict: Dictionary with model names and their metrics
        metric: Metric to compare
        save_path: Path to save figure (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    models = list(results_dict.keys())
    values = [results_dict[model].get(metric, 0) for model in models]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(models, values)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    ax.set_ylabel(metric.title())
    ax.set_title(f'Model Comparison - {metric.title()}')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels if needed
    if len(max(models, key=len)) > 10:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig
