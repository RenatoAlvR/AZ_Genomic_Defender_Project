import os
import scanpy as sc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from pathlib import Path
from typing import Dict, Any, Union, Tuple
import scipy.sparse
from scipy.io import mmread
import gzip


def preprocess_detect(
    data_dir: str,
    config: Dict[str, Any],
    training_dataset_name: str
) -> Tuple[sc.AnnData, Union[np.ndarray, Data], None]:
    """Preprocess a suspicious 10x Genomics dataset for anomaly detection.

    Applies the IDENTICAL transformation used during training:
        1. Load raw counts
        2. Subset to training HVGs (by gene name)
        3. Normalize → log1p
        4. Apply training-set scaler stats (mean/std) — NOT recomputed
        5. Clip to [-10, 10] (matches max_value=10 from training scale step)
        6. Zero-pad any genes missing from the suspicious dataset

    This guarantees the suspicious data lands in the same coordinate space
    as the training data — which is required for anomaly scores to be
    meaningful. Recomputing scaling from the suspicious data would introduce
    a distribution shift that creates false positives on clean cells and
    masks subtle attacks.

    Args:
        data_dir:               Directory with matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz.
        config:                 Config dict from YAML (model, input_dim, k_neighbors, device).
        training_dataset_name:  Name of the training dataset directory
                                (e.g. 'final_data' if trained on data/GSE161529/final_data).
                                Used to locate hvg_genes.txt and scaler_stats.npy.

    Returns:
        Tuple of:
            adata:      AnnData object (subset to present HVGs, pre-scaling) for reporting.
            X or Data:  np.ndarray (n_cells, input_dim) for CAE/VAE/DDPM,
                        torch_geometric Data for GNN-AE.
            None:       Placeholder — PCA object no longer used (HVG pipeline).
    """
    model       = config.get('model', '').lower()
    input_dim   = config.get('input_dim', 10000)
    k_neighbors = config.get('k_neighbors', 5)
    device      = config.get('device', 'cpu')

    assert model in ['cae', 'vae', 'gnn_ae', 'ddpm'], f"Unsupported model: {model}"
    assert input_dim > 0,   "input_dim must be positive"
    assert k_neighbors > 0, "k_neighbors must be positive"
    assert device in ['cpu', 'cuda'], "device must be 'cpu' or 'cuda'"

    data_dir = Path(data_dir)
    assert data_dir.is_dir(), f"Dataset directory does not exist: {data_dir}"

    # ── Locate training artifacts ─────────────────────────────────────────────
    training_preprocess_dir = Path('preprocessing') / training_dataset_name
    hvg_path   = training_preprocess_dir / 'hvg_genes.txt'
    stats_path = training_preprocess_dir / 'scaler_stats.npy'

    if not hvg_path.exists():
        raise FileNotFoundError(
            f"HVG gene list not found at {hvg_path}.\n"
            f"Train the model on '{training_dataset_name}' first to generate this file."
        )
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Scaler stats not found at {stats_path}.\n"
            f"Retrain with the updated preprocess_train.py to generate this file."
        )

    # ── Load training gene list and scaler stats ──────────────────────────────
    hvg_genes    = pd.read_csv(hvg_path, header=None).values.flatten().tolist()
    scaler_stats = np.load(stats_path, allow_pickle=True).item()
    train_means  = scaler_stats['mean']   # shape: (input_dim,)
    train_stds   = scaler_stats['std']    # shape: (input_dim,)

    print(f"Loaded {len(hvg_genes)} training HVGs from {hvg_path}")
    print(f"Loaded scaler stats from {stats_path}")

    if len(hvg_genes) != input_dim:
        raise ValueError(
            f"HVG list has {len(hvg_genes)} genes but config expects input_dim={input_dim}. "
            f"Make sure you're using the config that matches this training run."
        )

    # ── Step 1: Load raw counts ───────────────────────────────────────────────
    print(f"Loading suspicious dataset from {data_dir}...")
    print("Loading matrix manually (non-standard orientation)...")

    with gzip.open(data_dir / 'matrix.mtx.gz', 'rb') as f:
        X = mmread(f).tocsr()  # reads as (428024 × 33538) — cells × genes

    # Our combine_data.py wrote cells as rows, so no transpose needed here.
    # X is already (n_cells, n_genes).

    barcodes = pd.read_csv(
        data_dir / 'barcodes.tsv.gz', compression='gzip', header=None
    ).values.flatten()

    features = pd.read_csv(
        data_dir / 'features.tsv.gz', compression='gzip',
        sep='\t', header=None
    )
    gene_names = features[0].values  # column 0 = gene symbols

    adata = sc.AnnData(
        X=X,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=gene_names)
    )
    print(f"Loaded: {adata.shape}  ({adata.n_obs} cells × {adata.n_vars} genes)")


    # ── Step 2: Align to training gene space ──────────────────────────────────
    # Find which training HVGs are present in this dataset
    suspicious_genes = set(adata.var_names)
    present_genes    = [g for g in hvg_genes if g in suspicious_genes]
    missing_genes    = [g for g in hvg_genes if g not in suspicious_genes]

    if missing_genes:
        print(f"WARNING: {len(missing_genes)} training HVGs absent in suspicious data "
              f"({len(missing_genes)/len(hvg_genes)*100:.1f}%) — will be zero-padded.")
        if len(missing_genes) > input_dim * 0.1:
            print(f"WARNING: >10% of training genes are missing. This suspicious dataset may be "
                  f"from a different tissue type or sequencing platform than the training data. "
                  f"Detection results may be unreliable.")

    # Subset to present genes only (preserves order of hvg_genes)
    adata_detect = adata[:, present_genes].copy()
    print(f"After gene alignment: {adata_detect.shape}")

    # ── Step 3: Normalize → log1p (same as training pipeline) ────────────────
    print("Normalizing and log-transforming...")
    sc.pp.normalize_total(adata_detect, target_sum=1e4)
    sc.pp.log1p(adata_detect)

    # ── Step 4: Densify ───────────────────────────────────────────────────────
    if scipy.sparse.issparse(adata_detect.X):
        print("Densifying sparse matrix...")
        adata_detect.X = adata_detect.X.toarray()

    X = np.array(adata_detect.X, dtype=np.float32)

    # ── Step 5: Apply TRAINING scaler stats ───────────────────────────────────
    # Use the means/stds computed from the training dataset — NOT recomputed here.
    # This maps the suspicious data into the same coordinate space the model learned.
    #
    # We only have stats for present_genes (subset of all training HVGs).
    # Build an index map: for each gene in present_genes, find its position
    # in the full hvg_genes list to grab the correct mean/std.
    hvg_index_map = {gene: i for i, gene in enumerate(hvg_genes)}
    present_idx   = np.array([hvg_index_map[g] for g in present_genes])

    means = train_means[present_idx]   # scaler stats for the genes we have
    stds  = train_stds[present_idx]
    stds  = np.where(stds == 0, 1.0, stds)  # avoid division by zero for zero-variance genes

    print("Applying training scaler stats...")
    X = (X - means) / stds
    X = np.clip(X, -10, 10)   # matches max_value=10 from training sc.pp.scale()

    # ── Step 6: Zero-pad missing genes ───────────────────────────────────────
    # Missing genes are appended as zero columns.
    # Zero in scaled space = mean expression from training — a neutral, non-anomalous value.
    # This is conservative: it won't create false positives from missing genes.
    if missing_genes:
        print(f"Zero-padding {len(missing_genes)} missing genes...")
        pad = np.zeros((X.shape[0], len(missing_genes)), dtype=np.float32)
        X   = np.hstack([X, pad])

        # Reorder columns so they match hvg_genes order exactly
        # Build final array in the correct gene order
        X_ordered              = np.zeros((X.shape[0], input_dim), dtype=np.float32)
        present_positions      = [hvg_index_map[g] for g in present_genes]
        X_ordered[:, present_positions] = X[:, :len(present_genes)]
        X = X_ordered

    print(f"Final detection data: {X.shape} | mean={X.mean():.4f} | std={X.std():.4f}")

    if np.isnan(X).any():
        n_nan = np.isnan(X).sum()
        raise ValueError(
            f"{n_nan} NaNs detected in preprocessed detection data. "
            f"Check the suspicious dataset for empty cells or corrupt entries."
        )

    # ── Generate UMAP for visual inspection (optional but useful for reports) ─
    dataset_name = data_dir.name
    output_dir   = Path('preprocessing') / f"{dataset_name}_detect"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_umap = min(20000, X.shape[0])
    X_u    = X[np.random.choice(X.shape[0], n_umap, replace=False)] if X.shape[0] > n_umap else X

    print("Generating UMAP for suspicious dataset...")
    reducer   = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X_u)

    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.5, c='steelblue')
    plt.title(f"UMAP of Suspicious Dataset - {dataset_name}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(output_dir / "umap_suspicious.png", dpi=300)
    plt.close()
    print(f"Saved suspicious UMAP to {output_dir / 'umap_suspicious.png'}")

    # ── Build graph or return array ───────────────────────────────────────────
    if model == 'gnn_ae':
        X_torch    = torch.tensor(X, dtype=torch.float32)
        edge_index = _build_knn_graph_detect(X_torch, k_neighbors, device)
        return adata_detect, Data(x=X_torch, edge_index=edge_index), None

    return adata_detect, X, None


def _build_knn_graph_detect(X_torch: torch.Tensor, k: int, device: str) -> torch.Tensor:
    """Build kNN graph for detection with GPU attempt and CPU fallback.

    Detection graphs are NOT cached (suspicious data is per-run).

    Args:
        X_torch: Node feature tensor (n_cells, input_dim).
        k:       Number of neighbours per cell.
        device:  Preferred device.

    Returns:
        edge_index tensor (2, n_edges) on CPU.
    """
    if device == 'cuda' and torch.cuda.is_available():
        try:
            print(f"Building detection kNN graph on GPU (k={k})...")
            X_gpu      = X_torch.to('cuda')
            edge_index = knn_graph(X_gpu, k=k, loop=False)
            edge_index = edge_index.cpu()
            del X_gpu
            torch.cuda.empty_cache()
            return edge_index

        except torch.cuda.OutOfMemoryError:
            print("GPU OOM during kNN graph — falling back to CPU.")
            torch.cuda.empty_cache()

    print(f"Building detection kNN graph on CPU (k={k}, workers={os.cpu_count()})...")
    return knn_graph(X_torch, k=k, loop=False, num_workers=os.cpu_count())