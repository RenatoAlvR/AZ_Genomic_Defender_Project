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
from sklearn.neighbors import kneighbors_graph
from pathlib import Path
from typing import Dict, Any, Union
import scipy.sparse

def save_and_visualize(adata: sc.AnnData, X_pca: np.ndarray, data_dir: Path, dataset_name: str):
    """Save preprocessed PCA data and generate violin plot & UMAP visualization
    without running into memory issues.

    Args:
        adata (sc.AnnData): AnnData object with cell and gene names.
        X_pca (np.ndarray or sparse matrix): PCA-reduced data (n_cells, input_dim).
        data_dir (Path): Original dataset directory.
        dataset_name (str): Name of the dataset (e.g., 'end_data').
    """
    # Create output directory
    output_dir = Path("preprocessing") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save PCA data, cell names, and gene names
    np.save(output_dir / "data.npy", X_pca if not scipy.sparse.issparse(X_pca) else X_pca.toarray())
    pd.Series(adata.obs_names).to_csv(output_dir / "cells.txt", index=False, header=False)
    pd.Series(adata.var_names[:X_pca.shape[1]]).to_csv(output_dir / "genes.txt", index=False, header=False)
    print(f"Saved preprocessed data to {output_dir / 'data.npy'}")

    # --- Violin Plot ---
    # Use a subset of cells for plotting (avoid memory explosion)
    n_violin_samples = 5000
    if X_pca.shape[0] > n_violin_samples:
        idx_violin = np.random.choice(X_pca.shape[0], n_violin_samples, replace=False)
        X_violin = X_pca[idx_violin]
        adata_violin = adata[idx_violin]
    else:
        X_violin = X_pca
        adata_violin = adata

    if scipy.sparse.issparse(X_violin):
        X_violin = X_violin.toarray()

    # Select top variable genes
    variances = np.var(X_violin, axis=0)
    top_genes_idx = np.argsort(variances)[-10:]
    top_genes = adata_violin.var_names[top_genes_idx]

    # Convert to DataFrame for violin plot
    df = pd.DataFrame(X_violin[:, top_genes_idx], columns=top_genes)
    df_melt = df.melt(var_name="Gene", value_name="Expression")

    # Create violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x="Gene", y="Expression", data=df_melt, inner="quartile")
    plt.title(f"Gene Expression Distributions - {dataset_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "gene_distributions.png", dpi=300)
    plt.close()
    print(f"Saved gene distribution plot to {output_dir / 'gene_distributions.png'}")

    # --- UMAP ---
    # Subsample for UMAP (visualization only)
    n_umap_samples = 20000
    if X_pca.shape[0] > n_umap_samples:
        idx_umap = np.random.choice(X_pca.shape[0], n_umap_samples, replace=False)
        X_umap = X_pca[idx_umap]
    else:
        X_umap = X_pca

    if scipy.sparse.issparse(X_umap):
        X_umap = X_umap.toarray()

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_embedding = reducer.fit_transform(X_umap)

    plt.figure(figsize=(10, 8))
    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s=10, alpha=0.5)
    plt.title(f"UMAP of {dataset_name}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(output_dir / "umap.png", dpi=300)
    plt.close()
    print(f"Saved UMAP plot to {output_dir / 'umap.png'}")

    
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
    device = config.get('device', 'cpu')
    input_dim = config.get('input_dim', 3000)
    batch_size = config.get('batch_size', 10000)  # For memory-efficient processing
    k_neighbors = config.get('k_neighbors', 5)  # For GNN-AE graph construction
    train_batch_size = config.get('train_batch_size', 32)  # For DataLoader
    raw = config.get('raw', False)  # Use or not the raw data

    # Validate inputs
    data_dir = Path(data_dir)
    assert data_dir.is_dir(), f"Data directory {data_dir} does not exist"
    assert model in ['cae', 'vae', 'gnn_ae', 'ddpm'], f"Unsupported model: {model}"
    assert input_dim > 0, "input_dim must be positive"
    assert batch_size > 0, "batch_size must be positive"
    assert k_neighbors > 0, "k_neighbors must be positive"
    assert device in ['cpu', 'cuda'], f"No device available"

    # Check for preprocessed data
    dataset_name = data_dir.name
    output_dir = Path('preprocessing') / dataset_name
    data_path = output_dir / 'data.npy'
    graph_path = output_dir / 'edge_index.pt'
    labels_path = output_dir / 'labels.txt'
    labels = None

    if data_path.exists() and not raw:
        print(f"Loading preprocessed data from {data_path}")
        X = np.load(data_path)
        # Verify shape matches expected input_dim
        if X.shape[1] != input_dim:
            raise ValueError(f"Loaded data has {X.shape[1]} features, expected {input_dim}")

        if labels_path.exists():
            print(f"Loading labels from {labels_path}")
            labels = np.loadtxt(labels_path)
            if labels.shape[0] != X.shape[0]:
                raise ValueError(f"Labels shape {labels.shape[0]} does not match data shape {X.shape[0]}")

    else:
        # Load 10x Genomics data
        adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=False)
        print(f"Initial AnnData shape: {adata.shape}")

        '''
        # Generate UMAP for raw data
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
        '''

        '''
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
        '''

        if raw:
            print(f"Going to pass raw data to the model...")
            # Use raw data, limit to input_dim features if necessary
            X = adata.X[:, :input_dim]   # keep as sparse if already sparse
        else:
            print(f"Processsing data for the model...")
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
    
    if scipy.sparse.issparse(X):
        X = X.toarray()

    # Convert to PyTorch tensor
    X_torch = torch.tensor(X, dtype=torch.float32)
    labels_torch = None
    if labels is not None:
        labels_torch = torch.tensor(labels, dtype=torch.long)

    if model == 'gnn_ae':
        if graph_path.exists():
            edge_index = torch.load(graph_path)
            return Data(x=X_torch, edge_index=edge_index)
        else:
            # Construct k-NN graph
            print(f"Constructing k-NN graph... (could take a lot of time)")
            start_time = time.time()

            if(device == 'cuda' and torch.cuda.is_available()):
                # Use GPU to process the graph (caution with VRAM)
                print(f"Using GPU for graph construction")
                X_torch = X_torch.to('cuda')
                edge_index = knn_graph(X_torch, k=k_neighbors, loop=False)  # No num_workers needed on GPU
                X_torch = X_torch.cpu()
                edge_index = edge_index.to('cpu')
            else:
                num_cores = os.cpu_count()
                edge_index = knn_graph(X_torch, k=k_neighbors, loop=False, num_workers= num_cores)

            torch.save(edge_index, graph_path)
            print(f"knn_graph took {time.time() - start_time:.2f} seconds")

        # Create Data object
        data = Data(x=X_torch, edge_index=edge_index)
        if labels_torch is not None:
            data.labels = labels_torch
        return data  # Return single Data object for GNN-AE
    
    # Create DataLoader for other models
    if labels_torch is not None:
        dataset = TensorDataset(X_torch, labels_torch)
    else:
        dataset = TensorDataset(X_torch)
    data_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
    
    # Debug DataLoader output
    batch_example = next(iter(data_loader))
    print(f"Preprocess DataLoader batch example: shape: {batch_example[0].shape}, type: {type(batch_example)}")
    if len(batch_example) > 1:
        print(f"Labels shape: {batch_example[1].shape}")
    return data_loader
    
