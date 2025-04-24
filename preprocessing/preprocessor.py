import os
import argparse
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix, issparse
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import yaml
import torch
from torch_geometric.data import Data
from typing import Tuple, Dict, Optional

# Suppress some common warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class GenomicDataPreprocessor:
    """
    Preprocessor for single-cell RNA-seq data that handles:
    - Loading from 10x Genomics format (mtx + features + barcodes)
    - Quality control filtering
    - Normalization and scaling
    - Dimensionality reduction
    - Cell similarity graph construction for GNN
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Loaded configuration dictionary (None will use defaults)
        """
        self.config = config.get('preprocessing', {}) if config else {}
        
        # Set defaults if not in config
        self.min_genes_per_cell = self.config.get('min_genes_per_cell', 200)
        self.min_cells_per_gene = self.config.get('min_cells_per_gene', 3)
        self.n_pca_components = self.config.get('n_pca_components', 50)
        self.n_neighbors = self.config.get('n_neighbors', 15)
        self.normalize = self.config.get('normalize', 'log1p')
        self.scale = self.config.get('scale', True)
        self.random_state = self.config.get('random_state', 42)
        
        # Validate critical parameters
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if not isinstance(self.n_pca_components, int) or self.n_pca_components <= 0:
            raise ValueError("n_pca_components must be a positive integer")
        # Add other validations as needed
        
    def load_data(self, matrix_path: str, features_path: str, barcodes_path: str) -> Tuple[csr_matrix, pd.DataFrame, pd.DataFrame]:
        """
        Load data from 10x Genomics format files.
        
        Args:
            matrix_path: Path to .mtx file containing counts matrix
            features_path: Path to features.tsv file containing gene information
            barcodes_path: Path to barcodes.tsv file containing cell barcodes
            
        Returns:
            Tuple of (counts_matrix, features_df, barcodes_df)
        """
        # Load sparse matrix
        counts_matrix = sp.io.mmread(matrix_path).tocsr()
        
        # Load features (genes)
        features_df = pd.read_csv(
            features_path, 
            sep='\t', 
            header=None,
            names=['gene_id', 'gene_symbol', 'feature_type'] if os.path.getsize(features_path) > 0 else ['gene_id']
        )
        
        # Load barcodes (cells)
        barcodes_df = pd.read_csv(
            barcodes_path, 
            sep='\t', 
            header=None,
            names=['barcode']
        )
        
        # Validate shapes
        if counts_matrix.shape[1] != len(features_df):
            raise ValueError(f"Matrix columns ({counts_matrix.shape[1]}) don't match number of features ({len(features_df)})")
        if counts_matrix.shape[0] != len(barcodes_df):
            raise ValueError(f"Matrix rows ({counts_matrix.shape[0]}) don't match number of barcodes ({len(barcodes_df)})")
            
        return counts_matrix, features_df, barcodes_df
    
    def quality_control(self, counts: csr_matrix, features: pd.DataFrame, barcodes: pd.DataFrame) -> Tuple[csr_matrix, pd.DataFrame, pd.DataFrame]:
        """
        Perform basic quality control filtering.
        
        Args:
            counts: Sparse counts matrix (cells x genes)
            features: DataFrame with gene information
            barcodes: DataFrame with cell barcodes
            
        Returns:
            Filtered (counts, features, barcodes)
        """
        # Filter cells with too few genes
        n_genes_per_cell = counts.getnnz(axis=1)
        cell_mask = n_genes_per_cell >= self.min_genes_per_cell
        counts = counts[cell_mask, :]
        barcodes = barcodes.iloc[cell_mask].reset_index(drop=True)
        
        # Filter genes detected in too few cells
        n_cells_per_gene = counts.getnnz(axis=0)
        gene_mask = np.asarray(n_cells_per_gene >= self.min_cells_per_gene).ravel()
        counts = counts[:, gene_mask]
        features = features.iloc[gene_mask].reset_index(drop=True)
        
        return counts, features, barcodes
    
    def normalize_data(self, counts: csr_matrix) -> np.ndarray:
        """
        Normalize and scale the counts matrix.
        
        Args:
            counts: Sparse counts matrix (cells x genes)
            
        Returns:
            Dense normalized and scaled matrix
        """
        # Convert to counts per million (CPM) if specified
        if self.normalize == 'cpm':
            norm_counts = counts.copy()
            norm_counts = norm_counts.multiply(1e6 / norm_counts.sum(axis=1))
        # Log1p normalization (log(1 + x))
        elif self.normalize == 'log1p':
            norm_counts = counts.log1p()
        else:
            norm_counts = counts.copy()
            
        # Convert to dense array if small enough, otherwise keep sparse
        if norm_counts.shape[0] * norm_counts.shape[1] < 1e7:  # ~10M elements
            norm_counts = norm_counts.toarray()
            
        # Scale features to zero mean and unit variance
        if self.scale:
            scaler = StandardScaler(with_mean=issparse(norm_counts))
            norm_counts = scaler.fit_transform(norm_counts)
            
        return norm_counts
    
    def reduce_dimensions(self, data: np.ndarray) -> np.ndarray:
        """
        Perform PCA dimensionality reduction.
        
        Args:
            data: Normalized expression matrix (cells x genes)
            
        Returns:
            PCA-reduced matrix (cells x n_components)
        """
        pca = PCA(n_components=self.n_pca_components, random_state=self.random_state)
        reduced_data = pca.fit_transform(data)
        return reduced_data
    
    def build_cell_graph(self, data: np.ndarray) -> Data:
        """
        Construct k-nearest neighbor graph from cell data for GNN.
        
        Args:
            data: Processed cell data (PCA reduced or normalized)
            
        Returns:
            PyG Data object with graph structure
        """
        # Compute k-nearest neighbors graph
        adj = kneighbors_graph(
            data, 
            n_neighbors=self.n_neighbors, 
            mode='connectivity',
            include_self=False
        )
        
        # Convert to edge index format expected by PyTorch Geometric
        adj = adj.tocoo()
        edge_index = np.vstack([adj.row, adj.col])
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        # Create PyG Data object
        x = torch.tensor(data, dtype=torch.float)
        graph_data = Data(x=x, edge_index=edge_index)
        
        return graph_data
    
    def preprocess_pipeline(
        self, 
        matrix_path: str, 
        features_path: str, 
        barcodes_path: str
    ) -> Dict[str, object]:
        """
        Complete preprocessing pipeline from raw files to processed outputs.
        
        Args:
            matrix_path: Path to .mtx file
            features_path: Path to features.tsv
            barcodes_path: Path to barcodes.tsv
            
        Returns:
            Dictionary containing all processed data and intermediates
        """
        # Load raw data
        counts, features, barcodes = self.load_data(matrix_path, features_path, barcodes_path)
        
        # Quality control filtering
        counts, features, barcodes = self.quality_control(counts, features, barcodes)
        
        # Normalize and scale
        norm_data = self.normalize_data(counts)
        
        # Dimensionality reduction
        pca_data = self.reduce_dimensions(norm_data)
        
        # Build cell similarity graph for GNN
        cell_graph = self.build_cell_graph(pca_data)
        
        return {
            'raw_counts': counts,
            'features': features,
            'barcodes': barcodes,
            'normalized_data': norm_data,
            'pca_data': pca_data,
            'cell_graph': cell_graph
        }

def parse_args():
    """Parse command line arguments for preprocessing."""
    parser = argparse.ArgumentParser(description='Preprocess single-cell RNA-seq data for poisoning detection.')
    parser.add_argument('--matrix_path', type=str, required=True, help='Path to .mtx matrix file')
    parser.add_argument('--features_path', type=str, required=True, help='Path to features.tsv file')
    parser.add_argument('--barcodes_path', type=str, required=True, help='Path to barcodes.tsv file')
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Path to config YAML file')
    parser.add_argument('--output_path', type=str, default='preprocessed_data.pkl', help='Path to save preprocessed data')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Initialize and run preprocessor
    preprocessor = GenomicDataPreprocessor(config_path=args.config_path)
    processed_data = preprocessor.preprocess_pipeline(
        matrix_path=args.matrix_path,
        features_path=args.features_path,
        barcodes_path=args.barcodes_path
    )
    
    # Save processed data
    import pickle
    with open(args.output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Preprocessing complete. Results saved to {args.output_path}")