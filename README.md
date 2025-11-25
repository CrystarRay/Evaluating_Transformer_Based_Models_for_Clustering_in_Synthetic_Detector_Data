## Overview

This directory contains three training pipelines for centre and cluster prediction on synthetic detector data:

- **Offset Transformer** (`offset_transformer_train.py`)
- **K‑center SetTransformer** (`K_center_transformer_train.py`)
- **GNN / TopoGeoNet centre regressor** (`gnn_unet_for_center_reg/train.py`)

All three expect the synthetic `.npz` files produced by your data generators under a shared `synthetic_events/` directory.

---

## Quick start

These are the minimal steps for a fresh clone so that **other people can reproduce results** without touching any local paths.

For a high-level overview of the project and results, see the accompanying presentation:  
[Detector clustering transformers and GNNs – presentation](https://drive.google.com/file/d/1ebNVpTzbxWo1XX1tmrEB542-bl2xnC9L/view?usp=sharing)

### 0. Prerequisite

- You need a working **Conda installation** (Anaconda or Miniconda) with `conda` available on your `PATH`.

### 1. Create and activate the Conda environment

From this directory (where `environment_full.yml` lives):

```bash
cd Evaluating_Transformer_Based_Models_for_Clustering_in_Detector_Data

conda env create -f environment_full.yml -n detector_env
conda activate detector_env
```

You can replace `detector_env` with any name you like. The `prefix:` line has been removed from `environment_full.yml`, so the file is portable.

### 2. Get the synthetic datasets (train / val / test)

You have **two options**:

- **Option A – download a pre‑generated dataset archive (fast start)**

  1. Download the archive from Google Drive:  
     [`synthetic_events` archive](https://drive.google.com/file/d/139C7lTBZOvV4dADI3L0u6XyijFQcxrw4/view?usp=sharing).
  2. Place the downloaded file in the **`Evaluating_Transformer_Based_Models_for_Clustering_in_Detector_Data`** directory.
  3. Extract it there so that you end up with a `synthetic_events/` folder next to the training scripts.

- **Option B – generate locally** (slower, but fully reproducible)

  ```bash
  cd Evaluating_Transformer_Based_Models_for_Clustering_in_Detector_Data
  python synthetic_data_dynamic_nodes_new.py
  ```

  This will create the `synthetic_events/` directory with all required `.npz` files.

### 3. Train the three models (dynamic mode)

All commands below assume you are still in `Evaluating_Transformer_Based_Models_for_Clustering_in_Detector_Data` and the Conda env is active.

- **Offset Transformer – dynamic variable‑node mode**

```bash
python offset_transformer_train.py --dynamic
```

- **K‑center SetTransformer – dynamic variable‑node mode**

```bash
python K_center_transformer_train.py --dynamic
```

- **GNN / TopoGeoNet centre regressor**

The GNN script lives in its own subfolder and by default reads data from the parent `synthetic_events/` directory created in step 2:

```bash
cd gnn_unet_for_center_reg
python train.py
```

If you want to override the device, you can optionally add `--device cuda` or `--device cpu` to any of the commands above.

---

## Data layout

- **Shared synthetic datasets**
  - **Directory**: `synthetic_events/`
  - **Static scripts (Transformer, SetTransformer)**:
    - Expect files such as `synthetic_events/synthetic_detector_data.npz`
  - **Dynamic scripts (Transformer, SetTransformer)**:
    - Expect split folders and per‑configuration files:
      - `synthetic_events/train/synthetic_detector_data_<points>pts.npz`
      - `synthetic_events/val/synthetic_detector_data_<points>pts.npz`
      - `synthetic_events/test/synthetic_detector_data_<points>pts.npz`
- **GNN / TopoGeoNet**
  - **Default roots (set in `gnn_unet_for_center_reg/train.py`)**:
    - `../synthetic_events/train` for training
    - `../synthetic_events/val` for shifted validation
    - `../synthetic_events/test` for shifted test
  - Each `.npz` must already contain proximity graphs (`proximity_edge_index`, `proximity_edge_weight`) and per‑event node features, as produced by your preprocessing scripts.

---

## What each training script does

### Offset Transformer (`offset_transformer_train.py`)

- **Goal**
  - Per‑node regression of cluster centres in 3D.
  - Per‑event prediction of the cluster count \(k\).
  - Per‑node binary indicator of whether a detector node is active.
  - Per‑node prediction of a 3D covariance (upper‑triangular part of the inverse covariance matrix).

- **Model**
  - **Backbone**: GPT‑style Transformer encoder (`GPTEncoderModel` in `offset_network.py`).
  - **Heads**:
    - **Offset head**: predicts node‑wise 3D centres.
    - **K head**: predicts logits over \(k = 1 \dots \text{max\_k}\).
    - **Node‑indicator head**: predicts active/inactive for each node.
    - **Covariance head**: predicts 6‑D inverse covariance upper‑triangle per node.

- **Data modes**
  - **Static mode** (default, no `--dynamic`):
    - Loads a single padded dataset `synthetic_events/synthetic_detector_data.npz`.
  - **Dynamic mode** (`--dynamic`):
    - Uses `DynamicCentresDataset` and `MultiConfigDataLoader` to iterate over multiple node‑count configs (e.g. 3000‑point, 4000‑point events) without padding.

- **Important arguments**
  - **`--dynamic`**: switch between static and variable‑nodes training.
  - **`--data_dir`**: base directory containing `train/val/test` splits (default `./synthetic_events`).
  - **`--train_points`, `--val_points`, `--test_points`**: comma‑separated total node counts (e.g. `"2000,3000"`).
  - **`--device`**: e.g. `cuda:0` or `cpu`.
  - **`--epochs`, `--batch_size`**: standard training controls.

- **Outputs**
  - **Checkpoints**:
    - Static: saved under `checkpoints/` (one file per epoch plus `final_model.pth`).
    - Dynamic: if you pass `--checkpoint_dir`, it saves into that directory.
  - **Metrics and plots**:
    - Prints MAE/L2/cosine similarity, k‑accuracy, node‑indicator accuracy, and covariance metrics for train/val/test.
    - Saves training/validation loss curves as PNG images in the working directory.

### K‑center SetTransformer (`K_center_transformer_train.py`)

- **Goal**
  - Predict a *set* of event‑level cluster summaries, one row per cluster:
    - First 3 values: cluster centre coordinates.
    - Next 6 values: per‑cluster covariance (upper‑triangular).
  - Predict the number of clusters \(k\) per event.
  - Predict per‑node activity indicators to support visualisation and auxiliary supervision.

- **Model**
  - **Backbone**: SetTransformer‑style encoder (`SetTransformer` in `k_center_model.py`).
  - **Heads**:
    - **Centre head**: outputs a fixed maximum number of cluster centres (`Kmax`).
    - **Covariance head**: outputs per‑cluster 6‑D covariance parameters.
    - **K head**: predicts \(k\) (1..Kmax).
    - **Activity head**: per‑node logits for active/inactive.
  - **Matching**:
    - Uses the Hungarian algorithm (linear sum assignment) to match predicted centres to ground truth before computing losses.

- **Data modes**
  - **Static mode**:
    - Uses `SyntheticSet2GraphDataset` with `synthetic_events/synthetic_detector_data.npz`.
  - **Dynamic mode** (`--dynamic`):
    - Uses `DynamicSummaryDataset` + `MultiConfigDataLoader` over `synthetic_events/{train,val,test}/synthetic_detector_data_<points>pts.npz`.

- **Important arguments**
  - **`--dynamic`**: enable dynamic variable‑node batches.
  - **`--data_dir`**, `--train_points`, `--val_points`, `--test_points`: same semantics as in the offset script.
  - **`--epochs`, `--batch_size`**: training length and batch size.
  - **`--device`**: CUDA device or CPU.
  - **Static‑only options**:
    - Learning‑rate scheduling (`--fixed_lr`, `--lr`, `--max_lr`).
    - Resuming from checkpoints (`--resume_from`, `--resume_epoch`).

- **Outputs**
  - **Static mode**:
    - Checkpoints in `checkpoint/` (one per epoch plus `model_latest.pth`).
    - A CSV‑style log (`training_metrics_with_val.png` and DataFrame plots) summarising metrics over epochs.
    - Visualisations of GT vs predicted centres in `event_vis_pool/`.
  - **Dynamic mode**:
    - Checkpoints in `checkpoint_dynamic/`.
    - Per‑configuration metrics (per total‑points config) printed for train/val/test.
    - Visualisations in `event_vis_dynamic_*` folders, including by‑k plots.

### GNN / TopoGeoNet centre regressor (`gnn_unet_for_center_reg/train.py`)

- **Goal**
  - Perform per‑node 3D centre regression on graphs built from the detector geometry.
  - Optionally predict per‑node activity indicators via a BCE head (depending on backbone).
  - Support more geometric backbones (TopoGeoNet variants) with voxelised 3D UNet components.

- **Data flow**
  - Each `.npz` configuration file (e.g. `synthetic_detector_data_3000pts.npz`) is converted into a list of `torch_geometric.data.Data` graphs by:
    - `build_event_graphs_from_config` in `train.py`.
  - Each `Data` object contains:
    - **`x`**: node features (including energies and coordinates).
    - **`edge_index` / `edge_attr`**: proximity graph structure and edge weights.
    - **`y`**: target centre coordinates per node.
    - **`active_mask`**: binary active/inactive flags.
    - **`coords`** and **`assignment`**: 3D positions and voxel grid indices for UNet backbones.

- **Model options**
  - **`--backbone sage`**:
    - Uses the simple `CenterRegressor` stack of SAGEConv layers (plus optional GraphUNet).
  - **`--backbone lite`**:
    - Uses `TopoGeoNetLite` from `models/topogeonet_lite.py` (supports Fourier features).
  - **`--backbone full`** (default):
    - Uses `TopoGeoNetFull` from `models/topogeonet_full.py`.
  - **`--backbone 3d`**:
    - Uses full 3D `TopoGeoNet` with voxelisation and UNet stages (`models/topogeonet.py`).

- **Important arguments**
  - **`--data_root`**: directory of training `.npz` configs (default `../synthetic_events/train`).
  - **`--sec_val_root`**, **`--sec_test_root`**: optional shifted validation/test roots (defaults are the corresponding `../synthetic_events/val` and `../synthetic_events/test`).
  - **`--batch_size`**: number of graphs per batch.
  - **`--epochs`**, **`--lr`**, **`--weight_decay`**: standard training hyper‑parameters.
  - **`--reg_loss`**: `mse` or `mae` for the regression term.
  - **`--w_reg`**, **`--w_bce`**: weights for regression vs activity BCE loss.
  - **`--grid_d`, `--grid_h`, `--grid_w`**: voxel grid resolution for 3D UNet backbones.
  - **`--unet_stage`**: controls which UNet stages are active in 3D/full backbones (`none`, `first`, `second`, `both`).
  - **`--resume`**: resume from a previous checkpoint (`path` or `'latest'`).

- **Outputs**
  - **Checkpoints**:
    - Saved under `gnn_unet_for_center_reg/checkpoints/` as `epoch_XXXX.pt` plus `latest.pt`.
  - **Metrics**:
    - Printed per epoch for train, in‑distribution val, and optional shifted val/test.
    - Includes loss, MAE, cosine similarity and (when available) node‑indicator accuracy.
  - **Visualisations**:
    - GT vs predicted centres: `event_vis2/`.
    - K‑means cluster visualisations on predicted centres: `event_vis_kmeans/`.

---

## Typical workflow

1. **Set up environment**
   - **Create env**: `conda env create -f environment_full.yml -n detector_env`
   - **Activate**: `conda activate detector_env`
2. **Generate synthetic data** by running `python synthetic_data_dynamic_nodes_new.py` from `Evaluating_Transformer_Based_Models_for_Clustering_in_Detector_Data/` (this populates `synthetic_events/`).
3. **Train one or more models**
   - **Offset Transformer** (node‑level centres, k, indicators, covariances).
   - **K‑center SetTransformer** (event‑level cluster summaries).
   - **GNN / TopoGeoNet** (graph‑based centre regression).
4. **Inspect checkpoints, metrics and visualisations** to compare model families and choose the best one for downstream analysis.


