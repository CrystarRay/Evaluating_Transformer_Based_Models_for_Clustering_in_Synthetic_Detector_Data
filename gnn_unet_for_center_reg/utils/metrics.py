"""
Metrics computation utilities for TopoGeoNet.

This module contains functions for computing various evaluation metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    task_type: str = 'regression',
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics based on task type.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        task_type: Type of task ('regression', 'classification', 'multilabel')
        metrics: List of specific metrics to compute
        
    Returns:
        Dictionary of computed metrics
    """
    if task_type == 'regression':
        return regression_metrics(predictions, targets, metrics)
    elif task_type in ['classification', 'binary_classification']:
        return classification_metrics(predictions, targets, metrics)
    elif task_type == 'multilabel':
        return multilabel_metrics(predictions, targets, metrics)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        predictions: Predicted values
        targets: True values
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of regression metrics
    """
    if metrics is None:
        metrics = ['mse', 'rmse', 'mae', 'r2']
    
    results = {}
    
    # Flatten arrays if needed
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    if 'mse' in metrics:
        results['mse'] = mean_squared_error(target_flat, pred_flat)
    
    if 'rmse' in metrics:
        results['rmse'] = np.sqrt(mean_squared_error(target_flat, pred_flat))
    
    if 'mae' in metrics:
        results['mae'] = mean_absolute_error(target_flat, pred_flat)
    
    if 'r2' in metrics:
        results['r2'] = r2_score(target_flat, pred_flat)
    
    if 'mape' in metrics:
        # Mean Absolute Percentage Error
        mask = target_flat != 0
        results['mape'] = np.mean(
            np.abs((target_flat[mask] - pred_flat[mask]) / target_flat[mask])
        ) * 100
    
    if 'smape' in metrics:
        # Symmetric Mean Absolute Percentage Error
        numerator = np.abs(pred_flat - target_flat)
        denominator = (np.abs(pred_flat) + np.abs(target_flat)) / 2
        mask = denominator != 0
        results['smape'] = np.mean(numerator[mask] / denominator[mask]) * 100
    
    return results


def classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted probabilities or classes
        targets: True class labels
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of classification metrics
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    results = {}
    
    # Convert predictions to class labels if probabilities
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        pred_classes = np.argmax(predictions, axis=1)
        pred_probs = predictions
    else:
        # Binary classification with probabilities
        if predictions.max() <= 1.0 and predictions.min() >= 0.0:
            pred_classes = (predictions > 0.5).astype(int)
            pred_probs = predictions
        else:
            pred_classes = predictions.astype(int)
            pred_probs = None
    
    # Flatten if needed
    pred_classes = pred_classes.flatten()
    targets_flat = targets.flatten().astype(int)
    
    if 'accuracy' in metrics:
        results['accuracy'] = accuracy_score(targets_flat, pred_classes)
    
    if 'precision' in metrics:
        results['precision'] = precision_score(
            targets_flat, pred_classes, average='weighted', zero_division=0
        )
    
    if 'recall' in metrics:
        results['recall'] = recall_score(
            targets_flat, pred_classes, average='weighted', zero_division=0
        )
    
    if 'f1' in metrics:
        results['f1'] = f1_score(
            targets_flat, pred_classes, average='weighted', zero_division=0
        )
    
    # Metrics requiring probabilities
    if pred_probs is not None:
        if 'auc' in metrics:
            try:
                if len(np.unique(targets_flat)) == 2:
                    # Binary classification
                    results['auc'] = roc_auc_score(targets_flat, pred_probs.flatten())
                else:
                    # Multi-class classification
                    results['auc'] = roc_auc_score(
                        targets_flat, pred_probs, multi_class='ovr', average='weighted'
                    )
            except ValueError:
                results['auc'] = 0.0
        
        if 'ap' in metrics:
            try:
                results['ap'] = average_precision_score(targets_flat, pred_probs.flatten())
            except ValueError:
                results['ap'] = 0.0
    
    return results


def multilabel_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute multilabel classification metrics.
    
    Args:
        predictions: Predicted probabilities for each label
        targets: True binary labels
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of multilabel metrics
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    results = {}
    
    # Convert probabilities to binary predictions
    pred_binary = (predictions > 0.5).astype(int)
    
    if 'accuracy' in metrics:
        # Subset accuracy (exact match)
        results['subset_accuracy'] = accuracy_score(targets, pred_binary)
        
        # Element-wise accuracy
        results['accuracy'] = np.mean(targets == pred_binary)
    
    if 'precision' in metrics:
        results['precision_micro'] = precision_score(targets, pred_binary, average='micro')
        results['precision_macro'] = precision_score(targets, pred_binary, average='macro')
    
    if 'recall' in metrics:
        results['recall_micro'] = recall_score(targets, pred_binary, average='micro')
        results['recall_macro'] = recall_score(targets, pred_binary, average='macro')
    
    if 'f1' in metrics:
        results['f1_micro'] = f1_score(targets, pred_binary, average='micro')
        results['f1_macro'] = f1_score(targets, pred_binary, average='macro')
    
    if 'auc' in metrics:
        try:
            results['auc_micro'] = roc_auc_score(targets, predictions, average='micro')
            results['auc_macro'] = roc_auc_score(targets, predictions, average='macro')
        except ValueError:
            results['auc_micro'] = 0.0
            results['auc_macro'] = 0.0
    
    return results


def topological_metrics(
    embeddings: np.ndarray,
    original_distances: Optional[np.ndarray] = None,
    original_adjacency: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute topological quality metrics for embeddings.
    
    Args:
        embeddings: Learned embeddings
        original_distances: Original distance matrix (optional)
        original_adjacency: Original adjacency matrix (optional)
        
    Returns:
        Dictionary of topological metrics
    """
    results = {}
    
    # Compute pairwise distances in embedding space
    embedding_distances = np.linalg.norm(
        embeddings[:, None] - embeddings[None, :], axis=2
    )
    
    # Distance preservation (if original distances provided)
    if original_distances is not None:
        # Spearman correlation between distance matrices
        from scipy.stats import spearmanr
        
        # Flatten upper triangular matrices
        mask = np.triu(np.ones_like(original_distances, dtype=bool), k=1)
        orig_flat = original_distances[mask]
        emb_flat = embedding_distances[mask]
        
        correlation, _ = spearmanr(orig_flat, emb_flat)
        results['distance_correlation'] = correlation
        
        # Trustworthiness and Continuity
        results.update(_compute_trustworthiness_continuity(
            original_distances, embedding_distances
        ))
    
    # Neighborhood preservation (if adjacency provided)
    if original_adjacency is not None:
        results.update(_compute_neighborhood_preservation(
            original_adjacency, embedding_distances
        ))
    
    # Intrinsic dimensionality estimation
    results['intrinsic_dim'] = _estimate_intrinsic_dimension(embeddings)
    
    return results


def geometric_metrics(
    predicted_coords: np.ndarray,
    true_coords: np.ndarray
) -> Dict[str, float]:
    """
    Compute geometric quality metrics.
    
    Args:
        predicted_coords: Predicted coordinates
        true_coords: True coordinates
        
    Returns:
        Dictionary of geometric metrics
    """
    results = {}
    
    # Basic distance metrics
    results['coord_mse'] = mean_squared_error(true_coords, predicted_coords)
    results['coord_mae'] = mean_absolute_error(true_coords, predicted_coords)
    
    # Distance matrix preservation
    true_distances = np.linalg.norm(
        true_coords[:, None] - true_coords[None, :], axis=2
    )
    pred_distances = np.linalg.norm(
        predicted_coords[:, None] - predicted_coords[None, :], axis=2
    )
    
    results['distance_mse'] = mean_squared_error(true_distances, pred_distances)
    
    # Procrustes distance (shape similarity)
    try:
        from scipy.spatial.distance import procrustes
        _, _, disparity = procrustes(true_coords, predicted_coords)
        results['procrustes_disparity'] = disparity
    except ImportError:
        pass
    
    return results


def _compute_trustworthiness_continuity(
    original_distances: np.ndarray,
    embedding_distances: np.ndarray,
    k: int = 10
) -> Dict[str, float]:
    """Compute trustworthiness and continuity metrics."""
    n = original_distances.shape[0]
    
    # Get k-nearest neighbors in original space
    orig_neighbors = np.argsort(original_distances, axis=1)[:, 1:k+1]
    
    # Get k-nearest neighbors in embedding space
    emb_neighbors = np.argsort(embedding_distances, axis=1)[:, 1:k+1]
    
    # Trustworthiness
    trustworthiness = 0.0
    for i in range(n):
        for j in emb_neighbors[i]:
            if j not in orig_neighbors[i]:
                rank_in_orig = np.where(np.argsort(original_distances[i]) == j)[0][0]
                trustworthiness += max(0, rank_in_orig - k)
    
    trustworthiness = 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * trustworthiness
    
    # Continuity
    continuity = 0.0
    for i in range(n):
        for j in orig_neighbors[i]:
            if j not in emb_neighbors[i]:
                rank_in_emb = np.where(np.argsort(embedding_distances[i]) == j)[0][0]
                continuity += max(0, rank_in_emb - k)
    
    continuity = 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * continuity
    
    return {
        'trustworthiness': trustworthiness,
        'continuity': continuity
    }


def _compute_neighborhood_preservation(
    adjacency: np.ndarray,
    embedding_distances: np.ndarray
) -> Dict[str, float]:
    """Compute neighborhood preservation metrics."""
    # For each node, check if its neighbors in the graph
    # are also close in the embedding space
    
    n = adjacency.shape[0]
    preservation_scores = []
    
    for i in range(n):
        # Find neighbors in graph
        graph_neighbors = np.where(adjacency[i] > 0)[0]
        
        if len(graph_neighbors) == 0:
            continue
        
        # Get distances to all nodes in embedding
        distances = embedding_distances[i]
        
        # Rank nodes by embedding distance
        distance_ranks = np.argsort(distances)
        
        # Check how many graph neighbors are in top-k closest
        k = len(graph_neighbors)
        closest_k = distance_ranks[:k+1]  # +1 to exclude self
        closest_k = closest_k[closest_k != i]  # Remove self
        
        preserved = len(set(graph_neighbors) & set(closest_k))
        preservation_scores.append(preserved / len(graph_neighbors))
    
    return {
        'neighborhood_preservation': np.mean(preservation_scores)
    }


def _estimate_intrinsic_dimension(embeddings: np.ndarray) -> float:
    """Estimate intrinsic dimensionality of embeddings."""
    try:
        from sklearn.decomposition import PCA
        
        pca = PCA()
        pca.fit(embeddings)
        
        # Find number of components explaining 95% variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = np.sum(cumsum < 0.95) + 1
        
        return float(intrinsic_dim)
    
    except ImportError:
        # Fallback: use embedding dimension
        return float(embeddings.shape[1])
