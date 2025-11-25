import os
import glob
import argparse
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import SAGEConv, GraphUNet
    PYG_AVAILABLE = True
except Exception as e:
    PYG_AVAILABLE = False
    raise ImportError("This training script requires torch-geometric. Please install it first: pip install torch-geometric") from e

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – triggers 3-D support in mpl

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npz_configs(folder: str) -> List[str]:
    """
    Find all configuration npz files produced by the synthetic generator under a split folder.
    Expected filenames like: synthetic_detector_data_<points>pts.npz
    """
    pattern = os.path.join(folder, "*.npz")
    files = [p for p in glob.glob(pattern) if os.path.basename(p).startswith("synthetic_detector_data_")]
    if not files:
        raise FileNotFoundError(f"No npz config files found in {folder}")
    files.sort()
    return files


def build_event_graphs_from_config(npz_path: str, grid_dims: Tuple[int, int, int]) -> Tuple[List[Data], dict]:
    """
    Load a single configuration .npz and build a list of PyG Data graphs (one per event).

    Uses saved proximity graph (edge_index/edge_weight), per-event node features, targets and active mask.
    """
    data = np.load(npz_path, allow_pickle=True)

    # Common across all events for this configuration
    detector_xyz = data['detector_xyz'].astype(np.float32)                   # [N, 3]
    prox_edge_index = data['proximity_edge_index'].astype(np.int64)          # [2, E]
    prox_edge_weight = data['proximity_edge_weight'].astype(np.float32)      # [E]

    X_all_mod = data['X_all_mod'].astype(np.float32)                         # [Evs, N, F]
    node_centres = data['node_centres'].astype(np.float32)                   # [Evs, N, 3]
    active_flags = data['active_flags'].astype(np.int64)                     # [Evs, N]
    k_all = data['k_all'].astype(np.int64) if 'k_all' in data.files else None

    num_events = X_all_mod.shape[0]
    num_nodes = X_all_mod.shape[1]
    feat_dim = X_all_mod.shape[2]

    # Sanity checks
    assert detector_xyz.shape[0] == num_nodes, "detector_xyz and node features must share node count"
    assert prox_edge_index.shape[0] == 2, "edge_index must be [2, E]"
    assert prox_edge_weight.ndim == 1 and prox_edge_weight.shape[0] == prox_edge_index.shape[1], "edge_weight length must match edges"

    graphs: List[Data] = []

    # Precompute global bounds from detector geometry for consistent per-event voxelization
    # detector_xyz columns correspond to (x, y, z)
    min_x, min_y, min_z = detector_xyz.min(axis=0)
    max_x, max_y, max_z = detector_xyz.max(axis=0)
    range_x = max(max_x - min_x, 1e-8)
    range_y = max(max_y - min_y, 1e-8)
    range_z = max(max_z - min_z, 1e-8)

    Dz, Hy, Wx = grid_dims  # per-axis grid sizes

    # Build an event graph per event index
    for ev in range(num_events):
        x = X_all_mod[ev]                         # [N, F]
        y = node_centres[ev]                      # [N, 3]
        mask = active_flags[ev].astype(np.bool_)  # [N]

        # Coordinates come from last three features of x
        coords_from_x = x[:, -3:]

        # Create assignment from coordinates using GLOBAL bounds and per-axis grid sizes
        # Map (x,y,z) -> integer bins [0..Wx-1], [0..Hy-1], [0..Dz-1]
        ax = ((coords_from_x[:, 0] - min_x) / range_x * (Wx - 1)).astype(np.int64)
        ay = ((coords_from_x[:, 1] - min_y) / range_y * (Hy - 1)).astype(np.int64)
        az = ((coords_from_x[:, 2] - min_z) / range_z * (Dz - 1)).astype(np.int64)
        ax = np.clip(ax, 0, Wx - 1)
        ay = np.clip(ay, 0, Hy - 1)
        az = np.clip(az, 0, Dz - 1)
        # Stack to (z, y, x)
        assignment = np.stack([az, ay, ax], axis=1).astype(np.int64)
        
        # Package as PyG Data. We also stash coords explicitly for convenience.
        # Optional k label per event (number of clusters)
        k_ev = int(k_all[ev]) if k_all is not None else -1

        graph = Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(prox_edge_index),
            edge_attr=torch.from_numpy(prox_edge_weight).unsqueeze(1),  # [E, 1]
            y=torch.from_numpy(y),
            active_mask=torch.from_numpy(mask.astype(np.uint8)),
            coords=torch.from_numpy(coords_from_x.astype(np.float32)),
            assignment=torch.from_numpy(assignment),  # Add assignment for spatial processing
            num_nodes=num_nodes,
            k=torch.tensor(k_ev, dtype=torch.int16),
        )
        graphs.append(graph)

    meta = {
        'num_events': num_events,
        'num_nodes': num_nodes,
        'feature_dim': feat_dim,
        'npz_path': npz_path,
        'grid_dims': (Dz, Hy, Wx),
        'k_events': k_all.tolist() if k_all is not None else None,
    }
    return graphs, meta


def split_indices(num_items: int, train_ratio: float, val_ratio: float, seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    indices = list(range(num_items))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_train = int(num_items * train_ratio)
    n_val = int(num_items * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


def stratified_split_indices_by_k(graphs: List[Data], train_ratio: float, val_ratio: float, seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """
    Stratify the event indices by their k (cluster count) so each split keeps per-k proportions.
    Falls back to random split if k is unavailable on any graph.
    """
    if len(graphs) == 0:
        return [], [], []
    # Check availability of k per event
    try:
        k_values = [int(getattr(g, 'k').item() if hasattr(g, 'k') else -1) for g in graphs]
    except Exception:
        k_values = [-1 for _ in graphs]
    if any(kv < 0 for kv in k_values):
        return split_indices(len(graphs), train_ratio, val_ratio, seed)

    # Group indices by k
    from collections import defaultdict
    k_to_indices = defaultdict(list)
    for idx, kv in enumerate(k_values):
        k_to_indices[int(kv)].append(idx)

    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for k_val, idxs in k_to_indices.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        n_total = len(idxs)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        # Ensure we do not exceed and keep at least remaining for test
        n_train = min(n_train, n_total)
        n_val = min(n_val, max(0, n_total - n_train))
        n_test = max(0, n_total - n_train - n_val)
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])

    # Shuffle within splits to avoid ordered-by-k batches downstream
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def _print_k_distribution(split_name: str, indices: List[int], graphs: List[Data]) -> None:
    try:
        if not indices:
            print(f"{split_name}: empty")
            return
        ks = [int(getattr(graphs[i], 'k').item()) for i in indices if hasattr(graphs[i], 'k')]
        if len(ks) == 0:
            print(f"{split_name}: k unavailable")
            return
        unique, counts = np.unique(np.asarray(ks), return_counts=True)
        dist = {int(k): int(c) for k, c in zip(unique, counts)}
        print(f"K distribution per split (events) – {split_name}: {dist}")
    except Exception as _:
        pass


class CenterRegressor(nn.Module):
    """
    Simple 3D point-cloud aware GNN for per-node XYZ regression.
    - Stack of GraphSAGE layers with residuals
    - Optional GraphUNet head (disabled by default for simplicity)
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.1, use_unet: bool = False):
        super().__init__()
        self.use_unet = use_unet
        self.dropout = dropout

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_c = hidden_dim
            out_c = hidden_dim
            self.convs.append(SAGEConv(in_c, out_c, normalize=True))

        if use_unet:
            # Example sizes for GraphUNet pooling; can be tuned
            self.unet = GraphUNet(hidden_dim, hidden_dim, hidden_dim, depth=3, pool_ratios=0.5)

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, getattr(data, 'edge_attr', None)
        h = self.input_proj(x)
        for conv, ln in zip(self.convs, self.norms):
            h_in = h
            h = conv(h, edge_index)
            h = ln(h)
            h = F.silu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_in

        if self.use_unet:
            # GraphUNet expects edge_index; returns node embeddings
            h = self.unet(h, edge_index)

        out = self.out_head(h)
        return out


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    MSE over nodes selected by mask. If mask is None, use all nodes.
    """
    if mask is None:
        diff = pred - target
        return diff.pow(2).mean()
    mask_f = mask.float().view(-1, 1)
    diff = (pred - target) * mask_f
    denom = mask_f.sum().clamp_min(1.0)
    return (diff.pow(2).sum() / denom)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    MAE over nodes selected by mask. If mask is None, use all nodes.
    """
    if mask is None:
        return (pred - target).abs().mean()
    # Compute per-node MAE across xyz, then average over masked nodes
    per_node_mae = (pred - target).abs().mean(dim=-1)  # [N]
    mask_f = mask.float().view(-1)
    denom = mask_f.sum().clamp_min(1.0)
    return (per_node_mae * mask_f).sum() / denom


def masked_cosine_similarity(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    """
    Cosine similarity averaged over masked nodes. If mask is None, use all nodes.
    """
    p = pred
    t = target
    denom = (p.norm(dim=1).clamp_min(eps) * t.norm(dim=1).clamp_min(eps))
    cos = (p * t).sum(dim=1) / denom
    if mask is None:
        return cos.mean()
    mask_f = mask.float()
    return (cos * mask_f).sum() / mask_f.sum().clamp_min(1.0)

def logits_to_mask(logits: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Convert logits to boolean mask using threshold 0 (sigmoid>0.5) by default.
    """
    return (logits > threshold).to(logits.dtype)

def indicator_accuracy_from_logits(logits: Optional[torch.Tensor], target_mask: torch.Tensor) -> Optional[float]:
    """
    Compute node indicator accuracy given logits and GT mask (0/1).
    Returns None if logits are not provided.
    """
    if logits is None:
        return None
    pred_mask = (logits > 0).to(target_mask.dtype)
    correct = (pred_mask.view(-1) == target_mask.view(-1).to(pred_mask.dtype))
    return float(correct.float().mean().item())


def create_dataloaders(graphs: List[Data], train_idx: List[int], val_idx: List[int], test_idx: List[int], batch_size: int = 1, shuffle_train: bool = True) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    train_ds = [graphs[i] for i in train_idx]
    val_ds = [graphs[i] for i in val_idx] if val_idx else []
    test_ds = [graphs[i] for i in test_idx] if test_idx else []

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) if test_ds else None
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    print_interval_events: int = 250,
    loss_reg_type: str = 'mse',
    w_reg: float = 1.0,
    w_bce: float = 1.0,
) -> dict:
    model.train()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_bce_loss = 0.0
    total_mae = 0.0
    total_cos = 0.0
    total_acc = 0.0
    acc_batches = 0
    n_batches = 0
    events_seen = 0
    next_print = print_interval_events
    bce_criterion = nn.BCEWithLogitsLoss()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        if isinstance(pred, tuple):
            pred_centers, logits = pred
        else:
            pred_centers, logits = pred, None
        # Training uses GT mask for regression
        gt_mask = batch.active_mask.float()
        if loss_reg_type == 'mae':
            reg_loss = masked_mae(pred_centers, batch.y, gt_mask)
        else:
            reg_loss = masked_mse(pred_centers, batch.y, gt_mask)
        if logits is not None:
            bce_loss = bce_criterion(logits.view(-1), gt_mask.view(-1))
            loss = w_reg * reg_loss + w_bce * bce_loss
        else:
            bce_loss = torch.tensor(0.0, device=device)
            loss = w_reg * reg_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            mae = masked_mae(pred_centers, batch.y, gt_mask)
            cos = masked_cosine_similarity(pred_centers, batch.y, gt_mask)
            acc = indicator_accuracy_from_logits(logits, gt_mask) if logits is not None else None
        total_loss += loss.item()
        total_reg_loss += reg_loss.item()
        total_bce_loss += bce_loss.item() if isinstance(bce_loss, torch.Tensor) else 0.0
        total_mae += mae.item()
        total_cos += cos.item()
        if acc is not None:
            total_acc += acc
            acc_batches += 1
        n_batches += 1
        # interval logging by events processed
        graphs_in_batch = getattr(batch, 'num_graphs', 1)
        events_seen += int(graphs_in_batch)
        if events_seen >= next_print:
            avg_loss = total_loss / max(n_batches, 1)
            avg_reg = total_reg_loss / max(n_batches, 1)
            avg_bce = total_bce_loss / max(n_batches, 1)
            avg_mae = total_mae / max(n_batches, 1)
            avg_cos = total_cos / max(n_batches, 1)
            if acc_batches > 0:
                avg_acc = total_acc / max(acc_batches, 1)
                print(f"[Train] events={events_seen} | loss={avg_loss:.6f} (reg={avg_reg:.6f}, bce={avg_bce:.6f}) | mae={avg_mae:.6f} | cos={avg_cos:.4f} | acc={avg_acc:.4f}")
            else:
                print(f"[Train] events={events_seen} | loss={avg_loss:.6f} (reg={avg_reg:.6f}, bce={avg_bce:.6f}) | mae={avg_mae:.6f} | cos={avg_cos:.4f}")
            next_print += print_interval_events
    return {
        'loss': total_loss / max(n_batches, 1),
        'reg_loss': total_reg_loss / max(n_batches, 1),
        'bce_loss': total_bce_loss / max(n_batches, 1),
        'mae': total_mae / max(n_batches, 1),
        'cos': total_cos / max(n_batches, 1),
        'acc': (total_acc / max(acc_batches, 1)) if acc_batches > 0 else None,
        'batches': n_batches,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    print_interval_events: int = 250,
    tag: str = 'Val',
    loss_reg_type: str = 'mse',
    w_reg: float = 1.0,
    w_bce: float = 1.0,
) -> dict:
    if loader is None:
        return {}
    model.eval()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_bce_loss = 0.0
    total_mae = 0.0
    total_cos = 0.0
    total_acc = 0.0
    acc_batches = 0
    n_batches = 0
    events_seen = 0
    next_print = print_interval_events
    bce_criterion = nn.BCEWithLogitsLoss()
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        if isinstance(pred, tuple):
            pred_centers, logits = pred
        else:
            pred_centers, logits = pred, None
        # Eval: use predicted mask from logits if available for loss/cos; report MAE on GT-active
        if logits is not None:
            pred_mask = logits_to_mask(logits)
        else:
            pred_mask = batch.active_mask.float()
        if loss_reg_type == 'mae':
            reg_loss = masked_mae(pred_centers, batch.y, pred_mask)
        else:
            reg_loss = masked_mse(pred_centers, batch.y, pred_mask)
        if logits is not None:
            bce_loss = bce_criterion(logits.view(-1), batch.active_mask.float().view(-1))
            loss = w_reg * reg_loss + w_bce * bce_loss
        else:
            bce_loss = torch.tensor(0.0, device=device)
            loss = w_reg * reg_loss
        # Validation MAE: use predicted-active mask; if resulting MAE is 0, fall back to unmasked
        mae = masked_mae(pred_centers, batch.y, pred_mask)
        if mae.item() <= 0:
            mae = masked_mae(pred_centers, batch.y, None)
        # Cosine similarity: use predicted-active mask when available; if empty, fall back to GT-active
        if isinstance(pred_mask, torch.Tensor) and pred_mask.float().sum() <= 0:
            cos = masked_cosine_similarity(pred_centers, batch.y, batch.active_mask.float())
        else:
            cos = masked_cosine_similarity(pred_centers, batch.y, pred_mask)
        acc = indicator_accuracy_from_logits(logits, batch.active_mask.float()) if logits is not None else None
        total_loss += loss.item()
        total_reg_loss += reg_loss.item()
        total_bce_loss += bce_loss.item() if isinstance(bce_loss, torch.Tensor) else 0.0
        total_mae += mae.item()
        total_cos += cos.item()
        if acc is not None:
            total_acc += acc
            acc_batches += 1
        n_batches += 1
        graphs_in_batch = getattr(batch, 'num_graphs', 1)
        events_seen += int(graphs_in_batch)
        if events_seen >= next_print:
            avg_loss = total_loss / max(n_batches, 1)
            avg_reg = total_reg_loss / max(n_batches, 1)
            avg_bce = total_bce_loss / max(n_batches, 1)
            avg_mae = total_mae / max(n_batches, 1)
            avg_cos = total_cos / max(n_batches, 1)
            if acc_batches > 0:
                avg_acc = total_acc / max(acc_batches, 1)
                print(f"[{tag}] events={events_seen} | loss={avg_loss:.6f} (reg={avg_reg:.6f}, bce={avg_bce:.6f}) | mae={avg_mae:.6f} | cos={avg_cos:.4f} | acc={avg_acc:.4f}")
            else:
                print(f"[{tag}] events={events_seen} | loss={avg_loss:.6f} (reg={avg_reg:.6f}, bce={avg_bce:.6f}) | mae={avg_mae:.6f} | cos={avg_cos:.4f}")
            next_print += print_interval_events
    return {
        'val_loss': total_loss / max(n_batches, 1),
        'val_reg_loss': total_reg_loss / max(n_batches, 1),
        'val_bce_loss': total_bce_loss / max(n_batches, 1),
        'val_mae': total_mae / max(n_batches, 1),
        'val_cos': total_cos / max(n_batches, 1),
        'val_acc': (total_acc / max(acc_batches, 1)) if acc_batches > 0 else None,
        'val_batches': n_batches,
    }


def visualise_events_simple(model: nn.Module, dataset: List[Data], device: torch.device, num_events: int = 10, out_dir: str = "event_vis2") -> None:
    """Save GT vs predicted 3-D scatter plots for num_events events.

    Each figure contains two panels:
      - Left: GT centre labels (unique centres from y)
      - Right: predicted centers, colored by their own KMeans cluster assignment.
    The original detector geometry (coords) is drawn in light grey on both panels.
    """
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    cmap = plt.cm.get_cmap('plasma')

    test_indices = list(range(len(dataset)))
    if len(test_indices) == 0:
        print("visualise_events_simple: empty dataset; skipping.")
        return
    selected_indices = np.linspace(0, len(test_indices) - 1, min(num_events, len(test_indices)), dtype=int)
    idxs = [test_indices[i] for i in selected_indices]

    with torch.no_grad():
        for idx in idxs:
            graph: Data = dataset[idx]
            print(f"Visualizing test event {idx}")

            # Predict centres
            pred_out = model(graph.to(device))
            if isinstance(pred_out, tuple):
                pred_centres_t, logits_t = pred_out
                # Build predicted-active mask from logits (sigmoid>0.5 => logit>0)
                pred_mask = (logits_t > 0).detach().cpu().numpy().astype(bool)
            else:
                pred_centres_t = pred_out
                pred_mask = None
            centres_pred = pred_centres_t.detach().cpu().numpy()

            # GT centres and detector xyz
            centres_gt = graph.y.detach().cpu().numpy()
            detector_xyz = graph.coords.detach().cpu().numpy() if hasattr(graph, 'coords') else graph.x.detach().cpu().numpy()[:, -3:]
            # Do not remove inactive nodes from visualization; always show full geometry
            detector_xyz_vis = detector_xyz
            # For clustering/visualizing predicted centers, use predicted-active mask if available
            if pred_mask is not None:
                centres_pred_vis = centres_pred[pred_mask]
            else:
                centres_pred_vis = centres_pred

            fig = plt.figure(figsize=(14, 6))
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')

            # Common style: detector geometry as light grey backdrop
            for ax in (ax1, ax2):
                ax.scatter(detector_xyz_vis[:, 0], detector_xyz_vis[:, 1], detector_xyz_vis[:, 2], c='lightgrey', s=5, alpha=0.3)

            # Left – ground-truth centres (unique over GT-active nodes only)
            gt_mask = graph.active_mask.detach().cpu().numpy().astype(bool) if hasattr(graph, 'active_mask') else None
            if gt_mask is not None and gt_mask.any():
                unique_gt = np.unique(centres_gt[gt_mask], axis=0)
            else:
                unique_gt = np.unique(centres_gt, axis=0)
            ax1.scatter(unique_gt[:, 0], unique_gt[:, 1], unique_gt[:, 2], c='red', marker='x', s=60, linewidths=2, label='GT centers')
            ax1.set_title(f"Event {idx}: GT centers ({unique_gt.shape[0]} unique)")
            ax1.legend(loc='upper left', fontsize=8)

            # Right – predicted centers themselves, colored by their own KMeans cluster assignment (no GT masking)
            from sklearn.cluster import KMeans
            k = max(1, unique_gt.shape[0])
            centres_pred_all = centres_pred_vis
            if centres_pred_all.shape[0] < k:
                print(f"Skipping event {idx}: n_samples={centres_pred_all.shape[0]} < n_clusters={k}")
                plt.close(fig)
                continue
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(centres_pred_all)
            labels_pred = kmeans.labels_
            cmap_kmeans = plt.cm.get_cmap('tab10')
            colors_pred = cmap_kmeans(labels_pred % 10)

            ax2.scatter(centres_pred_all[:, 0], centres_pred_all[:, 1], centres_pred_all[:, 2], c=colors_pred, s=4, alpha=0.8, label='Predicted centers (KMeans, pred-active only)')
            ax2.scatter(unique_gt[:, 0], unique_gt[:, 1], unique_gt[:, 2], c='red', marker='x', s=80, linewidths=3, label='GT centers')
            ax2.set_title("KMeans clusters on predicted centers + GT centers")
            ax2.legend(loc='upper left', fontsize=8)

            for ax in (ax1, ax2):
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xlim(-120, 120)
                ax.set_ylim(-120, 120)
                ax.set_zlim(-120, 120)
                ax.view_init(elev=25, azim=135)

            plt.tight_layout()
            outfile = os.path.join(out_dir, f"event_{idx}.png")
            plt.savefig(outfile, dpi=150)
            plt.close(fig)
            print(f"Saved visualisation to {outfile}")


def visualise_kmeans(model: nn.Module, dataset: List[Data], device: torch.device, num_events: int = 10, out_dir: str = "event_vis_kmeans", event_indices: Optional[List[int]] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    cmap = plt.cm.get_cmap('tab10')

    test_indices = list(range(len(dataset)))
    if len(test_indices) == 0:
        print("visualise_kmeans: empty dataset; skipping.")
        return
    if event_indices is not None and len(event_indices) > 0:
        idxs = [i for i in event_indices if i < len(dataset)]
    else:
        selected_indices = np.linspace(0, len(test_indices) - 1, min(num_events, len(test_indices)), dtype=int)
        idxs = [test_indices[i] for i in selected_indices]

    from sklearn.cluster import KMeans
    with torch.no_grad():
        for idx in idxs:
            if idx >= len(dataset):
                print(f"Skipping event {idx}: index out of range for test set (size={len(dataset)})")
                continue
            graph: Data = dataset[idx]
            print(f"K-means visualizing test event {idx}")

            # Predictions
            pred_out = model(graph.to(device))
            if isinstance(pred_out, tuple):
                pred_centres_t, logits_t = pred_out
                pred_mask = (logits_t > 0).detach().cpu().numpy().astype(bool)
            else:
                pred_centres_t = pred_out
                pred_mask = None
            centres_pred = pred_centres_t.detach().cpu().numpy()

            # GT data
            centres_gt = graph.y.detach().cpu().numpy()
            unique_gt, inv_gt = np.unique(centres_gt, axis=0, return_inverse=True)
            k_gt = max(1, unique_gt.shape[0])
            xyz = graph.coords.detach().cpu().numpy() if hasattr(graph, 'coords') else graph.x.detach().cpu().numpy()[:, -3:]
            # Prepare masks
            gt_mask = graph.active_mask.detach().cpu().numpy().astype(bool) if hasattr(graph, 'active_mask') else None
            if pred_mask is not None and pred_mask.any():
                centres_pred_vis = centres_pred[pred_mask]
            else:
                centres_pred_vis = centres_pred

            # K-means on predicted centres
            kmeans = KMeans(n_clusters=k_gt, n_init=10, random_state=0).fit(centres_pred_vis)
            labels_pred = kmeans.labels_

            fig = plt.figure(figsize=(15, 6))
            ax1 = fig.add_subplot(1, 3, 1, projection='3d')
            ax2 = fig.add_subplot(1, 3, 2, projection='3d')
            ax3 = fig.add_subplot(1, 3, 3, projection='3d')

            # Panel 1: GT clusters – keep all nodes grey, color only GT-active nodes
            ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='lightgrey', s=4, alpha=0.3)
            colors_gt = cmap(inv_gt % 10)
            if gt_mask is not None and gt_mask.any():
                ax1.scatter(xyz[gt_mask, 0], xyz[gt_mask, 1], xyz[gt_mask, 2], c=colors_gt[gt_mask], s=4, alpha=0.9)
            else:
                ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors_gt, s=4, alpha=0.9)
            ax1.set_title(f"Event {idx}: GT clusters (GT-active colored, k={k_gt})")

            # Panel 2: Predicted clusters with (proxy) predicted active flags (using GT active as proxy)
            ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='lightgrey', s=4, alpha=0.3)
            colors_pred = cmap(labels_pred % 10)
            if pred_mask is not None and pred_mask.any():
                ax2.scatter(xyz[pred_mask, 0], xyz[pred_mask, 1], xyz[pred_mask, 2], c=colors_pred, s=4, alpha=0.9)
            else:
                ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors_pred, s=4, alpha=0.9)
            ax2.set_title(f"Event {idx}: Predicted clusters (pred-active colored, k={k_gt})")

            # Panel 3: Predicted clusters with GT active flags (same as panel 2 but explicit)
            ax3.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='lightgrey', s=4, alpha=0.3)
            if pred_mask is not None and pred_mask.any():
                ax3.scatter(xyz[pred_mask, 0], xyz[pred_mask, 1], xyz[pred_mask, 2], c=colors_pred, s=4, alpha=0.9)
            else:
                ax3.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors_pred, s=4, alpha=0.9)
            ax3.set_title(f"Event {idx}: Predicted clusters (pred-active colored, k={k_gt})")

            for ax in (ax1, ax2, ax3):
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xlim(-120, 120)
                ax.set_ylim(-120, 120)
                ax.set_zlim(-120, 120)
                ax.view_init(elev=25, azim=135)

            plt.tight_layout()
            outfile = os.path.join(out_dir, f"event_{idx}.png")
            plt.savefig(outfile, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved k-means visualisation to {outfile}")


def main():
    parser = argparse.ArgumentParser(description='Per-node XYZ center regression with proximity graphs')
    # NOTE: default paths assume data lives in the parent directory's synthetic_events folder:
    #   ../synthetic_events/train, ../synthetic_events/val, ../synthetic_events/test
    # so you can run this script from inside gnn_unet_for_center_reg.
    parser.add_argument('--data_root', type=str, default='../synthetic_events/train', help='Folder with *.npz configs (train split moved here)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train ratio within available events')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Val ratio within available events')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (graphs per batch)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--layers', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--use_unet', action='store_true', help='Use GraphUNet head')
    # Unified UNet stage control for TopoGeoNet backbones
    parser.add_argument('--unet_stage', type=str, default='both', choices=['none', 'first', 'second', 'both'], help='Enable which UNet stages (TopoGeoNet backbones): none|first|second|both')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--reg_loss', type=str, default='mse', choices=['mse', 'mae'], help='Regression loss type')
    parser.add_argument('--w_reg', type=float, default=1.0, help='Weight for regression loss')
    parser.add_argument('--w_bce', type=float, default=10.0, help='Weight for BCE loss')
    # Per-axis grid sizes for 3D UNet voxelization (Dz, Hy, Wx)
    parser.add_argument('--grid_d', type=int, default=32, help='Grid size along Z (depth)')
    parser.add_argument('--grid_h', type=int, default=32, help='Grid size along Y (height)')
    parser.add_argument('--grid_w', type=int, default=32, help='Grid size along X (width)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--backbone', type=str, default='full', choices=['lite', 'sage', 'full', '3d'], help='Model backbone')
    parser.add_argument('--fourier', action='store_true', help='Use Fourier coord encoding (lite backbone)')
    # GNN architecture options for 'full' and '3d' backbones
    parser.add_argument('--layer_type', type=str, default='sage', choices=['sage', 'gcn', 'gin', 'gatv2', 'transformer'], help='GNN layer type for TopoGeoNet variants')
    parser.add_argument('--heads', type=int, default=4, help='Attention heads (for gatv2/transformer)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume or 'latest'")
    # Visualization removed per request
    parser.add_argument('--max_graphs', type=int, default=None, help='Debug: limit total graphs (train+val+test)')
    # Secondary validation/test roots (distribution-shifted data)
    parser.add_argument('--sec_val_root', type=str, default='../synthetic_events/val', help='Folder with *.npz configs for secondary (shifted) validation')
    parser.add_argument('--sec_test_root', type=str, default='../synthetic_events/test', help='Folder with *.npz configs for secondary (shifted) test')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    # Load all configs in the provided folder and concatenate their event lists
    config_files = load_npz_configs(args.data_root)
    all_graphs: List[Data] = []
    meta_agg = []
    for cfg in config_files:
        graphs, meta = build_event_graphs_from_config(cfg, grid_dims=(args.grid_d, args.grid_h, args.grid_w))
        all_graphs.extend(graphs)
        meta_agg.append(meta)

    if not all_graphs:
        raise RuntimeError("No graphs built from npz files.")

    # Optionally limit total graphs for debug runs
    if args.max_graphs is not None and args.max_graphs > 0:
        if len(all_graphs) > args.max_graphs:
            print(f"Debug mode: limiting total graphs from {len(all_graphs)} to {args.max_graphs}")
            all_graphs = all_graphs[:args.max_graphs]

    # Determine feature dimension from first graph
    in_dim = all_graphs[0].x.shape[1]

    # Build splits from the concatenated event list (stratified by k if available)
    train_idx, val_idx, test_idx = stratified_split_indices_by_k(all_graphs, args.train_ratio, args.val_ratio, seed=args.seed)

    train_loader, val_loader, test_loader = create_dataloaders(
        all_graphs, train_idx, val_idx, test_idx, batch_size=args.batch_size, shuffle_train=True
    )

    # Keep a handle to the exact test set used during training for later visualisation
    test_set: List[Data] = [all_graphs[i] for i in test_idx]

    # Load secondary validation and test sets (no internal split)
    sec_val_loader: Optional[DataLoader] = None
    sec_test_loader: Optional[DataLoader] = None
    sec_val_graphs: Optional[List[Data]] = None
    sec_test_graphs: Optional[List[Data]] = None
    # Secondary Validation
    try:
        if args.sec_val_root and os.path.isdir(args.sec_val_root):
            cfgs_val2 = load_npz_configs(args.sec_val_root)
            graphs_val2: List[Data] = []
            for cfg in cfgs_val2:
                g_list, _ = build_event_graphs_from_config(cfg, grid_dims=(args.grid_d, args.grid_h, args.grid_w))
                graphs_val2.extend(g_list)
            if graphs_val2:
                sec_val_graphs = graphs_val2
                sec_val_loader = DataLoader(graphs_val2, batch_size=args.batch_size, shuffle=False)
            else:
                print(f"Secondary Val: no graphs built from {args.sec_val_root}")
        else:
            if args.sec_val_root:
                print(f"Secondary Val: directory not found: {args.sec_val_root}")
    except Exception as e:
        print(f"Failed to load secondary validation from {args.sec_val_root}: {e}")
        sec_val_loader = None
        sec_val_graphs = None
    # Secondary Test
    try:
        if args.sec_test_root and os.path.isdir(args.sec_test_root):
            cfgs_test2 = load_npz_configs(args.sec_test_root)
            graphs_test2: List[Data] = []
            for cfg in cfgs_test2:
                g_list, _ = build_event_graphs_from_config(cfg, grid_dims=(args.grid_d, args.grid_h, args.grid_w))
                graphs_test2.extend(g_list)
            if graphs_test2:
                sec_test_graphs = graphs_test2
                sec_test_loader = DataLoader(graphs_test2, batch_size=args.batch_size, shuffle=False)
            else:
                print(f"Secondary Test: no graphs built from {args.sec_test_root}")
        else:
            if args.sec_test_root:
                print(f"Secondary Test: directory not found: {args.sec_test_root}")
    except Exception as e:
        print(f"Failed to load secondary test from {args.sec_test_root}: {e}")
        sec_test_loader = None
        sec_test_graphs = None

    # Create model
    if args.backbone == 'lite':
        from models.topogeonet_lite import TopoGeoNetLite
        model = TopoGeoNetLite(
            input_dim=in_dim,
            hidden_dim=args.hidden_dim,
            output_dim=3,
            num_layers=args.layers,
            layer_type='sage',
            dropout=args.dropout,
            use_unet=args.use_unet,
            use_fourier_features=args.fourier,
        )
    elif args.backbone == 'full':
        from models.topogeonet_full import TopoGeoNetFull
        # For TopoGeoNetFull, input_dim should be the base feature dimension (without Fourier)
        # The model will handle Fourier encoding internally and adjust the encoder accordingly
        model = TopoGeoNetFull(
            input_dim=in_dim,
            hidden_dim=args.hidden_dim,
            output_dim=3,
            num_layers=args.layers,
            layer_type=args.layer_type,
            dropout=args.dropout,
            heads=args.heads,
            edge_dim=1,  # we provide edge_attr as distance
            use_unet=(args.unet_stage != 'none'),
            use_fourier_features=args.fourier,
        )
        # Directly set UNet stage toggles here
        try:
            if hasattr(model, 'enable_unet_first'):
                model.enable_unet_first = (args.unet_stage in ['first', 'both'])
            if hasattr(model, 'enable_unet_second'):
                model.enable_unet_second = (args.unet_stage in ['second', 'both'])
        except Exception:
            pass
    elif args.backbone == '3d':
        from models.topogeonet import TopoGeoNet
        model = TopoGeoNet(
            input_dim=in_dim,
            hidden_dim=args.hidden_dim,
            output_dim=3,
            num_layers=args.layers,
            layer_type=args.layer_type,
            dropout=max(args.dropout, 0.1), 
            heads=args.heads,
            edge_dim=1,  # we provide edge_attr as distance
            use_unet=(args.unet_stage != 'none'),  # Enable UNet stack if any stage requested
            use_first_unet=(args.unet_stage in ['first', 'both']),  # Control first-stage UNet
            use_fourier_features=args.fourier,
            use_3d=True,  # Enable 3D mode
            config={'converter': {'aggregation_method': 'amax', 'log_init': False}, 'suppress_warnings': True},
        )
        # Enforce exact stage selection by nulling unused UNet modules (clarity, avoids conflicts)
        try:
            if args.unet_stage == 'first' and hasattr(model, 'second_unet'):
                model.second_unet = None
            if args.unet_stage == 'second' and hasattr(model, 'first_unet'):
                model.first_unet = None
        except Exception:
            pass
        # config={'converter': {'aggregation_method': 'mean'}},
    else:
        model = CenterRegressor(in_dim=in_dim, hidden_dim=args.hidden_dim, num_layers=args.layers, dropout=args.dropout, use_unet=args.use_unet)
    model.to(device)

    # Stash the test set on the model to mirror caller usage expectations
    try:
        setattr(model, '_test_set', test_set)
    except Exception:
        pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    # Resume support
    os.makedirs(args.output_dir, exist_ok=True)
    start_epoch = 0
    if args.resume:
        ckpt_path = args.resume
        if ckpt_path == 'latest':
            # find latest epoch_*.pt in output_dir
            cand = []
            for fname in os.listdir(args.output_dir):
                if fname.startswith('epoch_') and fname.endswith('.pt'):
                    try:
                        ep = int(fname.split('_')[1].split('.')[0])
                        cand.append((ep, os.path.join(args.output_dir, fname)))
                    except Exception:
                        pass
            if cand:
                cand.sort(key=lambda x: x[0])
                ckpt_path = cand[-1][1]
            else:
                latest_path = os.path.join(args.output_dir, 'latest.pt')
                ckpt_path = latest_path if os.path.isfile(latest_path) else None
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"Resuming from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = int(checkpoint.get('epoch', 0))
        else:
            print(f"Warning: resume path not found or invalid: {args.resume}")

    def save_checkpoint(epoch: int, train_stats: dict, val_stats: dict):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_stats': train_stats,
            'val_stats': val_stats,
            'config': {
                'hidden_dim': args.hidden_dim,
                'layers': args.layers,
                'dropout': args.dropout,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'backbone': args.backbone,
                'fourier': args.fourier,
            }
        }
        path_epoch = os.path.join(args.output_dir, f'epoch_{epoch:04d}.pt')
        torch.save(ckpt, path_epoch)
        # also update latest
        latest_path = os.path.join(args.output_dir, 'latest.pt')
        torch.save(ckpt, latest_path)
        print(f"Saved checkpoint: {path_epoch}")

    print(f"Loaded {len(all_graphs)} graphs from {len(config_files)} config file(s)")
    print(f"Train/Val/Test sizes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    # Print per-k distributions similar to xRFM/test.py
    _print_k_distribution('Train', train_idx, all_graphs)
    _print_k_distribution('Val', val_idx, all_graphs)
    _print_k_distribution('Test', test_idx, all_graphs)
    # Secondary split summaries
    if sec_val_graphs is not None:
        print(f"Secondary Val size (events): {len(sec_val_graphs)}")
        _print_k_distribution('Val2', list(range(len(sec_val_graphs))), sec_val_graphs)
    if sec_test_graphs is not None:
        print(f"Secondary Test size (events): {len(sec_test_graphs)}")
        _print_k_distribution('Test2', list(range(len(sec_test_graphs))), sec_test_graphs)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    best_val = float('inf')
    for epoch in range(max(1, start_epoch + 1), args.epochs + 1):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device,
            print_interval_events=250,
            loss_reg_type=args.reg_loss,
            w_reg=args.w_reg,
            w_bce=args.w_bce,
        )
        val_stats = evaluate(
            model, val_loader, device,
            print_interval_events=250, tag='Val',
            loss_reg_type=args.reg_loss,
            w_reg=args.w_reg,
            w_bce=args.w_bce,
        )
        # Secondary validation on shifted distribution
        val2_stats = evaluate(
            model, sec_val_loader, device,
            print_interval_events=250, tag='Val2',
            loss_reg_type=args.reg_loss,
            w_reg=args.w_reg,
            w_bce=args.w_bce,
        ) if sec_val_loader is not None else {}

        msg = f"Epoch {epoch:03d} | train_loss {train_stats['loss']:.6f} reg {train_stats['reg_loss']:.6f} bce {train_stats['bce_loss']:.6f} mae {train_stats['mae']:.6f} cos {train_stats['cos']:.4f}"
        if val_stats:
            msg += f" | val_loss {val_stats['val_loss']:.6f} val_reg {val_stats['val_reg_loss']:.6f} val_bce {val_stats['val_bce_loss']:.6f} val_mae {val_stats['val_mae']:.6f} val_cos {val_stats['val_cos']:.4f}"
        if train_stats.get('acc') is not None:
            msg += f" | acc {train_stats['acc']:.4f}"
        if val_stats and val_stats.get('val_acc') is not None:
            msg += f" val_acc {val_stats['val_acc']:.4f}"
            best_val = min(best_val, val_stats['val_loss'])
        print(msg)
        if val2_stats:
            print(f"Epoch {epoch:03d} | val2_loss {val2_stats['val_loss']:.6f} val2_reg {val2_stats['val_reg_loss']:.6f} val2_bce {val2_stats['val_bce_loss']:.6f} val2_mae {val2_stats['val_mae']:.6f} val2_cos {val2_stats['val_cos']:.4f}")

        # Save checkpoint every epoch
        save_checkpoint(epoch, train_stats, val_stats if val_stats else {})

    # Final test evaluation
    test_stats = evaluate(model, test_loader, device)
    if test_stats:
        print(f"Test: loss {test_stats['val_loss']:.6f} mae {test_stats['val_mae']:.6f} cos {test_stats.get('val_cos', 0.0):.6f}")
    # Secondary test evaluation
    test2_stats = evaluate(model, sec_test_loader, device) if sec_test_loader is not None else {}
    if test2_stats:
        print(f"Sec Test: loss {test2_stats['val_loss']:.6f} mae {test2_stats['val_mae']:.6f} cos {test2_stats.get('val_cos', 0.0):.6f}")

    # Visualisation
    if len(test_set) > 0:
        # Use the SAME test set from training
        visualise_events_simple(model, test_set, device, num_events=20, out_dir="event_vis2")
        visualise_kmeans(model, test_set, device, num_events=20, out_dir="event_vis_kmeans")
    else:
        print("No test set available for visualization.")


if __name__ == '__main__':
    main()


