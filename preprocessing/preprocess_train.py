import time
import os
import scanpy as sc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from pathlib import Path
from typing import Dict, Any, Union
import scipy.sparse


def save_and_visualize(adata: sc.AnnData, X: np.ndarray, data_dir: Path, dataset_name: str):
    """Save preprocessed data and generate violin plot & UMAP visualization.

    Args:
        adata:        AnnData object with cell and gene names (post-HVG selection).
        X:            Processed dense array (n_cells, input_dim).
        data_dir:     Original dataset directory.
        dataset_name: Name of the dataset (e.g., 'final_data').
    """
    output_dir = Path("preprocessing") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save processed matrix, cell names, gene names
    np.save(output_dir / "data.npy", X)
    pd.Series(adata.obs_names).to_csv(output_dir / "cells.txt", index=False, header=False)
    pd.Series(adata.var_names).to_csv(output_dir / "genes.txt", index=False, header=False)
    print(f"Saved preprocessed data to {output_dir / 'data.npy'}")

    # ── Violin plot (subsample to avoid memory explosion) ─────────────────────
    n_violin = 5000
    if X.shape[0] > n_violin:
        idx = np.random.choice(X.shape[0], n_violin, replace=False)
        X_v = X[idx]
        adata_v = adata[idx]
    else:
        X_v, adata_v = X, adata

    variances     = np.var(X_v, axis=0)
    top_idx       = np.argsort(variances)[-10:]
    top_genes     = adata_v.var_names[top_idx]
    df            = pd.DataFrame(X_v[:, top_idx], columns=top_genes)
    df_melt       = df.melt(var_name="Gene", value_name="Expression")

    plt.figure(figsize=(12, 6))
    sns.violinplot(x="Gene", y="Expression", data=df_melt, inner="quartile")
    plt.title(f"Gene Expression Distributions - {dataset_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "gene_distributions.png", dpi=300)
    plt.close()
    print(f"Saved violin plot to {output_dir / 'gene_distributions.png'}")

    # ── UMAP (subsample for speed) ────────────────────────────────────────────
    n_umap = 20000
    X_u    = X[np.random.choice(X.shape[0], n_umap, replace=False)] if X.shape[0] > n_umap else X

    reducer       = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding     = reducer.fit_transform(X_u)

    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.5)
    plt.title(f"UMAP of {dataset_name}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(output_dir / "umap.png", dpi=300)
    plt.close()
    print(f"Saved UMAP to {output_dir / 'umap.png'}")


def preprocess_train(data_dir: str, config: Dict[str, Any]) -> Union[DataLoader, Data]:
    """Preprocess 10x Genomics data for training.

    Pipeline (Option A — seurat_v3 on raw counts):
        1. Load raw counts
        2. HVG selection on raw counts  (seurat_v3 requires this)
        3. Subset to HVGs
        4. Normalize → log1p
        5. Scale (zero mean, unit variance, capped at max_value=10)
        6. Save dense array + HVG gene list

    Args:
        data_dir: Directory containing matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz.
        config:   Config dict from YAML.

    Returns:
        DataLoader for CAE / VAE / DDPM.
        torch_geometric Data object for GNN-AE.
    """
    model            = config.get('model', '').lower()
    device           = config.get('device', 'cpu')
    input_dim        = config.get('input_dim', 10000)
    k_neighbors      = config.get('k_neighbors', 5)
    train_batch_size = config.get('train_batch_size', 512)
    raw              = config.get('raw', False)

    assert model in ['cae', 'vae', 'gnn_ae', 'ddpm'], f"Unsupported model: {model}"
    assert input_dim > 0,    "input_dim must be positive"
    assert k_neighbors > 0,  "k_neighbors must be positive"
    assert device in ['cpu', 'cuda'], "device must be 'cpu' or 'cuda'"

    data_dir      = Path(data_dir)
    dataset_name  = data_dir.name
    output_dir    = Path('preprocessing') / dataset_name
    data_path     = output_dir / 'data.npy'
    hvg_path      = output_dir / 'hvg_genes.txt'
    graph_path    = output_dir / 'edge_index.pt'

    # ── Cache hit: skip preprocessing if outputs already exist ────────────────
    if data_path.exists() and hvg_path.exists() and not raw:
        print(f"Loading cached preprocessed data from {data_path}")
        X = np.load(data_path)
        if X.shape[1] != input_dim:
            raise ValueError(
                f"Cached data has {X.shape[1]} features but config expects {input_dim}. "
                f"Delete preprocessing/{dataset_name}/ and rerun."
            )
        print(f"Loaded cached data: {X.shape}")

    else:
        # ── Step 1: Load raw counts ───────────────────────────────────────────
        print("Loading 10x data (raw counts)...")
        adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=False)
        # Combined matrix was written cells×genes instead of standard genes×cells.
        # Scanpy reads it transposed — detect and correct.
        if adata.n_obs < adata.n_vars:
            print(f"Matrix appears transposed ({adata.shape}) — correcting to (cells × genes)...")
            adata = adata.T
            print(f"Corrected shape: {adata.shape}")
            
        print(f"Loaded raw data: {adata.shape}  ({adata.n_obs} cells × {adata.n_vars} genes)")

        # ── Step 2: HVG selection ON RAW COUNTS ──────────────────────────────
        # seurat_v3 uses a raw-count variance estimator (Poisson model).
        # Running it after normalization/log gives incorrect variance estimates
        # and selects the wrong genes — kept before normalize intentionally.
        print(f"Selecting top {input_dim} highly variable genes (seurat_v3 on raw counts)...")
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=input_dim,
            flavor='seurat_v3',
            span=1.0           # span=1.0 avoids edge-gene exclusion in small datasets
        )
        adata = adata[:, adata.var.highly_variable].copy()
        print(f"After HVG selection: {adata.shape}")

        # Save HVG gene list immediately — detection pipeline MUST use this exact set
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.Series(adata.var_names.tolist()).to_csv(hvg_path, index=False, header=False)
        print(f"Saved HVG gene list to {hvg_path}")

        # ── Step 3: Normalize → log1p ─────────────────────────────────────────
        print("Normalizing and log-transforming...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # ── Step 4: Densify BEFORE scale ─────────────────────────────────────
        # sc.pp.scale() densifies internally regardless, but doing it explicitly
        # here gives us control over WHEN the ~16 GB spike happens and lets us
        # log it clearly. (428k cells × 10k genes × 4 bytes ≈ 16 GB)
        if scipy.sparse.issparse(adata.X):
            print("Densifying sparse matrix before scaling (~16 GB RAM spike expected)...")
            adata.X = adata.X.toarray()
            print("Densification complete.")

        # ── Step 5: Scale (zero mean, unit variance, capped at 10) ───────────
        # max_value=10 clips extreme outliers (e.g. marker genes with very high
        # expression in rare cell types) that would otherwise dominate the
        # anomaly score distribution and mask subtle attacks.
        print("Scaling (zero mean, unit variance, max_value=10)...")
        sc.pp.scale(adata, max_value=10)

        # Save per-gene scaling parameters so detection uses identical transformation
        scaler_stats = {
            'mean': adata.var['mean'].values,      # scanpy stores these in adata.var after scale
            'std':  adata.var['std'].values
        }
        np.save(output_dir / 'scaler_stats.npy', scaler_stats)
        print(f"Saved scaler stats to {output_dir / 'scaler_stats.npy'}")

        # ── Step 6: Extract final dense array ────────────────────────────────
        X = adata.X if isinstance(adata.X, np.ndarray) else np.array(adata.X)
        print(f"Final data shape: {X.shape} | mean={X.mean():.4f} | std={X.std():.4f}")

        if np.isnan(X).any():
            raise ValueError("NaNs detected in preprocessed data — check input matrix for empty cells.")

        # Save and generate visualizations
        save_and_visualize(adata, X, data_dir, dataset_name)

    # ── Build graph or DataLoader ─────────────────────────────────────────────
    X_torch = torch.tensor(X, dtype=torch.float32)

    if model == 'gnn_ae':
        if graph_path.exists():
            print(f"Loading cached kNN graph from {graph_path}")
            edge_index = torch.load(graph_path)
        else:
            edge_index = _build_knn_graph(X_torch, k_neighbors, device, graph_path)
        return Data(x=X_torch, edge_index=edge_index)

    dataset     = TensorDataset(X_torch)
    data_loader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,   # Parallel CPU data loading for RTX 4090
        pin_memory=True  # Faster CPU → GPU transfers
    )
    print(f"DataLoader ready: {len(data_loader)} batches × {train_batch_size} cells")
    return data_loader


def _build_knn_graph(X_torch: torch.Tensor, k: int, device: str, save_path: Path) -> torch.Tensor:
    """Build kNN graph with GPU attempt and CPU fallback.

    Args:
        X_torch:   Node feature tensor (n_cells, input_dim).
        k:         Number of neighbours per cell.
        device:    Preferred device ('cuda' or 'cpu').
        save_path: Where to cache the edge_index tensor.

    Returns:
        edge_index tensor (2, n_edges) on CPU.
    """
    if device == 'cuda' and torch.cuda.is_available():
        try:
            print(f"Building kNN graph on GPU (k={k})...")
            t0 = time.time()
            X_gpu      = X_torch.to('cuda')
            edge_index = knn_graph(X_gpu, k=k, loop=False)
            edge_index = edge_index.cpu()
            del X_gpu
            torch.cuda.empty_cache()
            print(f"GPU kNN graph built in {time.time() - t0:.1f}s")

        except torch.cuda.OutOfMemoryError:
            # 428k × 10k on GPU requires ~17 GB VRAM — may exceed 4090's 24 GB
            # when other processes are running. CPU fallback is slower but safe.
            print("GPU OOM during kNN graph construction — falling back to CPU.")
            print("This may take 10–30 minutes for 428k cells. Consider k_neighbors=3 to speed up.")
            torch.cuda.empty_cache()
            t0         = time.time()
            num_cores  = os.cpu_count()
            edge_index = knn_graph(X_torch, k=k, loop=False, num_workers=num_cores)
            print(f"CPU kNN graph built in {time.time() - t0:.1f}s")
    else:
        print(f"Building kNN graph on CPU (k={k}, workers={os.cpu_count()})...")
        t0         = time.time()
        edge_index = knn_graph(X_torch, k=k, loop=False, num_workers=os.cpu_count())
        print(f"CPU kNN graph built in {time.time() - t0:.1f}s")

    torch.save(edge_index, save_path)
    print(f"kNN graph cached to {save_path}")
    return edge_index