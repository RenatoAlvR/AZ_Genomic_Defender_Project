import scanpy as sc
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from sklearn.neighbors import kneighbors_graph
from pathlib import Path
from typing import Dict, Any, Union

def preprocess_train(data_dir: str, config: Dict[str, Any]) -> Union[np.ndarray, Data]:
    """Preprocess 10x Genomics data for training.

    Args:
        data_dir (str): Directory containing matrix.mtx, barcodes.tsv, and features.tsv.
        config (Dict): Configuration dictionary with model, input_dim, batch_size, k_neighbors.

    Returns:
        np.ndarray: PCA-reduced data (n_cells, input_dim) for CAE, VAE, DDPM.
        Data: PyTorch Geometric Data object with x and edge_index for GNN-AE.
    """
    # Extract config parameters
    model = config.get('model', '').lower()
    input_dim = config.get('input_dim', 3000)
    batch_size = config.get('batch_size', 10000)  # For memory-efficient processing
    k_neighbors = config.get('k_neighbors', 5)  # For GNN-AE graph construction

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

    # Check if dimensions are transposed (genes as obs, cells as vars)
    expected_cells = 69032  # From barcodes.tsv.gz
    expected_genes = 33538  # From features.tsv.gz
    if adata.shape == (expected_genes, expected_cells):
        print("Transposing AnnData to correct shape (cells × genes)")
        adata.X = adata.X.T
        # Swap obs and var
        obs_names = adata.var_names
        var_names = adata.obs_names
        adata.obs_names = adata.obs_names
        adata.var_names = var_names
        # Update obs and var to avoid empty DataFrames
        adata.obs = adata.obs.reindex(adata.obs_names)
        adata.var = adata.var.reindex(adata.var_names)
        print(f"Corrected AnnData shape: {adata.shape}")

    # Process in batches if dataset is large
    n_cells = adata.n_obs
    batch_size = min(batch_size, n_cells)
    processed_data = []

    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        batch = adata[start:end].copy()

        # Preprocessing steps
        sc.pp.normalize_total(batch, target_sum=1e4)
        sc.pp.log1p(batch)
        sc.pp.pca(batch, n_comps=input_dim, svd_solver='arpack')

        # Extract PCA features
        batch_data = batch.obsm['X_pca']
        processed_data.append(batch_data)

    # Combine batches
    X = np.concatenate(processed_data, axis=0)
    
    # Convert to PyTorch tensor
    X_torch = torch.tensor(X, dtype=torch.float32)

    if model == 'gnn_ae':
        # Construct k-NN graph
        edge_index = knn_graph(X_torch, k=k_neighbors, loop=False)
        return Data(x=X_torch, edge_index=edge_index)
    
    # Create DataLoader for other models
    dataset = TensorDataset(X_torch)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: x[0])
    
    print(f"DataLoader batch example: {next(iter(data_loader))}")

    return data_loader
    
