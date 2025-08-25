import scanpy as sc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from sklearn.neighbors import kneighbors_graph
from pathlib import Path
from typing import Dict, Any, Union

def save_and_visualize(adata: sc.AnnData, X_pca: np.ndarray, data_dir: Path, dataset_name: str):
    """Save preprocessed PCA data and generate violin plot for gene distributions.
    
    Args:
        adata (sc.AnnData): AnnData object with cell and gene names.
        X_pca (np.ndarray): PCA-reduced data (n_cells, input_dim).
        data_dir (Path): Original dataset directory.
        dataset_name (str): Name of the dataset (e.g., 'end_data').
    """
    # Create output directory
    output_dir = Path('preprocessing') / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PCA data, cell names, and gene names
    np.save(output_dir / 'data.npy', X_pca)
    pd.Series(adata.obs_names).to_csv(output_dir / 'cells.txt', index=False, header=False)
    pd.Series(adata.var_names[:X_pca.shape[1]]).to_csv(output_dir / 'genes.txt', index=False, header=False)
    print(f"Saved preprocessed data to {output_dir / 'data.npy'}")
    
    # Generate violin plot for top 10 variable genes
    variances = np.var(X_pca, axis=0)
    top_genes_idx = np.argsort(variances)[-10:]
    top_genes = adata.var_names[top_genes_idx]
    
    # Convert to DataFrame for visualization
    df = pd.DataFrame(X_pca[:, top_genes_idx], columns=top_genes)
    df_melt = df.melt(var_name='Gene', value_name='Expression')
    
    # Create violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Gene', y='Expression', data=df_melt, inner='quartile')
    plt.title(f'Gene Expression Distributions - {dataset_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'gene_distributions.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved gene distribution plot to {plot_path}")
    
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
    train_batch_size = config.get('train_batch_size', 32)  # For DataLoader

    # Validate inputs
    data_dir = Path(data_dir)
    assert data_dir.is_dir(), f"Data directory {data_dir} does not exist"
    assert model in ['cae', 'vae', 'gnn_ae', 'ddpm'], f"Unsupported model: {model}"
    assert input_dim > 0, "input_dim must be positive"
    assert batch_size > 0, "batch_size must be positive"
    assert k_neighbors > 0, "k_neighbors must be positive"

    # Check for preprocessed data
    dataset_name = data_dir.name
    output_dir = Path('preprocessing') / dataset_name
    data_path = output_dir / 'data.npy'

    if data_path.exists():
        print(f"Loading preprocessed data from {data_path}")
        X = np.load(data_path)
        # Verify shape matches expected input_dim
        if X.shape[1] != input_dim:
            raise ValueError(f"Loaded data has {X.shape[1]} features, expected {input_dim}")

    else:
        # Load 10x Genomics data
        adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=False)
        print(f"Initial AnnData shape: {adata.shape}")

        # Check if dimensions are transposed (genes as obs, cells as vars)
        expected_cells = 69032  # From barcodes.tsv.gz
        expected_genes = 33538  # From features.tsv.gz
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
        
        # Save preprocessed data and visualize
        save_and_visualize(adata, X, data_dir, dataset_name)
    
    # Convert to PyTorch tensor
    X_torch = torch.tensor(X, dtype=torch.float32)

    if model == 'gnn_ae':
        # Construct k-NN graph
        print(f"Entered GNN if")
        edge_index = knn_graph(X_torch, k=k_neighbors, loop=False)
        print(f"Generated edge_index")
        return Data(x=X_torch, edge_index=edge_index)
    
    # Create DataLoader for other models
    dataset = TensorDataset(X_torch)
    data_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=True, collate_fn=lambda batch: torch.stack([x[0] for x in batch]))
    
    # Debug DataLoader output
    batch_example = next(iter(data_loader))
    print(f"Preprocess DataLoader batch example: shape: {batch_example.shape}, type: {type(batch_example)}")
    return data_loader
    
