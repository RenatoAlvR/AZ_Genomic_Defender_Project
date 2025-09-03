import scanpy as sc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from pathlib import Path
from typing import Dict, Any, Union
from preprocessing.preprocess_train import save_and_visualize

def preprocess_detect(data_dir: str, config: Dict[str, Any]) -> tuple[sc.AnnData, Union[np.ndarray, Data], Any]:
    """Preprocess 10x Genomics data for anomaly detection.

    Args:
        data_dir (str): Directory containing matrix.mtx, barcodes.tsv, and features.tsv.
        config (Dict): Configuration dictionary with model, input_dim, batch_size, k_neighbors.

    Returns:
        tuple: (AnnData object, processed data (np.ndarray for CAE/VAE/DDPM, Data for GNN-AE), PCA object)
    """
    # Extract config parameters
    model = config.get('model', '').lower()
    input_dim = config.get('input_dim', 3000)
    batch_size = config.get('batch_size', 10000)
    k_neighbors = config.get('k_neighbors', 5)

    # Validate inputs
    data_dir = Path(data_dir)
    assert data_dir.is_dir(), f"Data directory {data_dir} does not exist"
    assert model in ['cae', 'vae', 'gnn_ae', 'ddpm'], f"Unsupported model: {model}"
    assert input_dim > 0, "input_dim must be positive"
    assert batch_size > 0, "batch_size must be positive"
    assert k_neighbors > 0, "k_neighbors must be positive"

    # Load 10x Genomics data
    adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=False)
    print(f"Initial AnnData shape: {adata.shape}")

    # Generate UMAP for raw data
    dataset_name = data_dir.name
    output_dir = Path('preprocessing') / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    umap_embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(adata.X)
    plt.figure(figsize=(10, 8))
    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s=10, alpha=0.5)
    plt.title(f'UMAP of Raw Data - {dataset_name}')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(output_dir / 'raw_umap.png', dpi=300)
    plt.close()
    print(f"Saved raw UMAP plot to {output_dir / 'raw_umap.png'}")

    # Check if dimensions are transposed
    expected_cells = 69032
    expected_genes = 33538
    if adata.shape == (expected_genes, expected_cells):
        print("Transposing AnnData to correct shape (cells × genes)")
        adata.X = adata.X.T
        obs_names = adata.var_names
        var_names = adata.obs_names
        adata.obs_names = obs_names
        adata.var_names = var_names
        adata.obs = adata.obs.reindex(adata.obs_names)
        adata.var = adata.var.reindex(adata.var_names)
        print(f"Corrected AnnData shape: {adata.shape}")

    # Process in batches if dataset is large
    n_cells = adata.n_obs
    batch_size = min(batch_size, n_cells)
    processed_data = []
    pca = None  # Initialize PCA object

    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        batch = adata[start:end].copy()

        # Preprocessing steps
        sc.pp.normalize_total(batch, target_sum=1e4)
        sc.pp.log1p(batch)
        sc.pp.pca(batch, n_comps=input_dim, svd_solver='arpack')

        # Store PCA object from the first batch (assuming consistent PCA across batches)
        if pca is None:
            pca = batch.uns['pca']

        # Extract PCA features
        batch_data = batch.obsm['X_pca']
        processed_data.append(batch_data)

    # Combine batches
    X = np.concatenate(processed_data, axis=0)

    # Save preprocessed data and visualize
    save_and_visualize(adata, X, data_dir, dataset_name)

    if model == 'gnn_ae':
        X_torch = torch.tensor(X, dtype=torch.float32)
        edge_index = knn_graph(X_torch, k=k_neighbors, loop=False)
        return adata, Data(x=X_torch, edge_index=edge_index), pca
    
    return adata, X, pca