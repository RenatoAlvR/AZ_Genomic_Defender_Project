# preprocessor.py
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix, issparse
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
import yaml
import torch
from torch_geometric.data import Data
from typing import Tuple, Dict, Optional, Any

# Attempt to import scanpy, optional for some basic functionality but preferred for HVG
try:
    import scanpy as sc
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    warnings.warn("Scanpy not installed. HVG selection will use a basic variance method if enabled, or fall back. "
                  "Install scanpy for robust HVG selection ('pip install scanpy').")

# Suppress some common warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning) # Be cautious with this one

logger = logging.getLogger(__name__)

class GenomicDataPreprocessor:
    def __init__(self, config: Dict[str, Any]): # Expect the full config
        self.config = config.get('preprocessing', {})
        if not self.config:
            logger.warning("Preprocessing configuration section not found. Using defaults.")

        self.min_genes_per_cell = self.config.get('min_genes_per_cell', 100)
        self.min_cells_per_gene = self.config.get('min_cells_per_gene', 3)
        self.n_top_genes = self.config.get('n_top_genes', 3000) # For HVG selection
        self.n_pca_components = self.config.get('n_pca_components', 50)
        self.n_neighbors_graph = self.config.get('n_neighbors_graph', 15)
        self.normalize_method = self.config.get('normalize_method', 'log1p')
        self.scale_data = self.config.get('scale_data', True)
        self.split_ratio = self.config.get('split_ratio', 0.8)
        self.random_state = config.get('seed', 42) # Use global seed

        self._validate_config()
        logger.info(f"Preprocessor initialized with n_top_genes={self.n_top_genes}, n_pca_components={self.n_pca_components}")

    def _validate_config(self):
        if not (0 < self.n_top_genes <= 50000): # Max sensible genes
            raise ValueError("n_top_genes must be a positive integer, typically a few thousands.")
        if not (0 < self.n_pca_components < self.n_top_genes):
            raise ValueError("n_pca_components must be positive and less than n_top_genes.")
        if not (0 < self.split_ratio < 1):
            raise ValueError("split_ratio must be between 0 and 1.")

    def load_data(self, matrix_path: str, features_path: str, barcodes_path: str) -> Tuple[csr_matrix, pd.DataFrame, pd.DataFrame]:
        logger.info(f"Loading data from: {matrix_path}, {features_path}, {barcodes_path}")
        counts_matrix = sp.io.mmread(matrix_path).tocsr().transpose() # Transpose to get cells x genes

        features_df = pd.read_csv(features_path, sep='\t', header=None,
                                  names=['gene_id', 'gene_symbol', 'feature_type'])
        barcodes_df = pd.read_csv(barcodes_path, sep='\t', header=None, names=['barcode'])

        if counts_matrix.shape[1] != len(features_df):
            raise ValueError(f"Matrix columns ({counts_matrix.shape[1]}) don't match features ({len(features_df)}) after transpose.")
        if counts_matrix.shape[0] != len(barcodes_df):
            raise ValueError(f"Matrix rows ({counts_matrix.shape[0]}) don't match barcodes ({len(barcodes_df)}).")
        logger.info(f"Loaded data: {counts_matrix.shape[0]} cells, {counts_matrix.shape[1]} genes.")
        return counts_matrix, features_df, barcodes_df

    def quality_control(self, counts: csr_matrix, features: pd.DataFrame, barcodes: pd.DataFrame) -> Tuple[csr_matrix, pd.DataFrame, pd.DataFrame]:
        logger.info("Performing quality control...")
        # Filter cells
        genes_per_cell = np.array(counts.getnnz(axis=1)).ravel()
        cell_mask = genes_per_cell >= self.min_genes_per_cell
        counts = counts[cell_mask, :]
        barcodes = barcodes[cell_mask].reset_index(drop=True)
        logger.info(f"Cells after gene count filter: {counts.shape[0]}")

        # Filter genes
        cells_per_gene = np.array(counts.getnnz(axis=0)).ravel()
        gene_mask = cells_per_gene >= self.min_cells_per_gene
        counts = counts[:, gene_mask]
        features = features[gene_mask].reset_index(drop=True)
        logger.info(f"Genes after cell count filter: {counts.shape[1]}")
        return counts, features, barcodes

    def select_highly_variable_genes(self, counts: csr_matrix, features: pd.DataFrame) -> Tuple[csr_matrix, pd.DataFrame]:
        logger.info(f"Selecting top {self.n_top_genes} highly variable genes...")
        if counts.shape[1] <= self.n_top_genes:
            logger.info("Number of genes is already less than or equal to n_top_genes. Skipping HVG selection.")
            return counts, features

        if SCANPY_AVAILABLE:
            adata = sc.AnnData(X=counts.copy()) # Use a copy
            adata.var_names = features['gene_id'].astype(str) # Ensure var_names are strings and unique
            adata.var_names_make_unique()
            sc.pp.highly_variable_genes(adata, n_top_genes=self.n_top_genes, flavor='seurat_v3', inplace=True)
            hvg_mask = adata.var['highly_variable'].values
        else:
            logger.warning("Scanpy not available. Using basic variance-based HVG selection (less robust).")
            if issparse(counts):
                gene_vars = np.array(counts.power(2).mean(axis=0) - (counts.mean(axis=0))**2).ravel()
            else:
                gene_vars = np.var(counts, axis=0)
            hvg_indices = np.argsort(gene_vars)[-self.n_top_genes:]
            hvg_mask = np.zeros(counts.shape[1], dtype=bool)
            hvg_mask[hvg_indices] = True
            if np.sum(hvg_mask) == 0 and counts.shape[1] > 0: # Fallback if all variances are zero
                logger.warning("All gene variances are zero or HVG selection failed; selecting first n_top_genes or all if fewer.")
                hvg_mask[:min(self.n_top_genes, counts.shape[1])] = True


        counts_hvg = counts[:, hvg_mask]
        features_hvg = features[hvg_mask].reset_index(drop=True)
        logger.info(f"Selected {counts_hvg.shape[1]} HVGs.")
        return counts_hvg, features_hvg

    def normalize_and_scale_data(self, counts: csr_matrix) -> np.ndarray:
        logger.info(f"Normalizing data using '{self.normalize_method}' and scaling (if enabled)...")
        
        # Ensure data is float for normalization precision
        processed_counts = counts.astype(np.float32)

        if self.normalize_method == 'log1p':
            if SCANPY_AVAILABLE:
                adata = sc.AnnData(X=processed_counts)
                sc.pp.normalize_total(adata, target_sum=1e4) # CPM-like before log
                sc.pp.log1p(adata)
                norm_data = adata.X
            else: # Manual log1p after library size normalization
                row_sums = np.array(processed_counts.sum(axis=1)).ravel()
                # Avoid division by zero for empty cells (should be filtered by QC)
                row_sums[row_sums == 0] = 1 
                if issparse(processed_counts):
                    processed_counts = processed_counts.multiply(1e4 / row_sums[:, np.newaxis])
                    norm_data = processed_counts.log1p()
                else:
                    processed_counts = processed_counts * (1e4 / row_sums[:, np.newaxis])
                    norm_data = np.log1p(processed_counts)

        elif self.normalize_method == 'cpm':
            row_sums = np.array(processed_counts.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1
            if issparse(processed_counts):
                norm_data = processed_counts.multiply(1e6 / row_sums[:, np.newaxis])
            else:
                norm_data = processed_counts * (1e6 / row_sums[:, np.newaxis])
        else:
            logger.info("No normalization method applied or unknown method specified.")
            norm_data = processed_counts # Keep as is

        # Convert to dense for scaling and PCA if it's not already
        # Be mindful of memory for large HVG sets.
        if issparse(norm_data):
            # If n_top_genes is large (e.g., >5k) and many cells, this could be large
            # For 20k cells x 3k genes ~ 230MB as float32 dense. Manageable.
            logger.debug("Converting normalized data to dense for scaling/PCA.")
            norm_data = norm_data.toarray()
        
        if self.scale_data:
            logger.info("Scaling data to zero mean and unit variance.")
            scaler = StandardScaler(with_mean=True) # with_mean=True as it's dense now
            norm_data = scaler.fit_transform(norm_data)
        
        return norm_data

    def reduce_dimensions_pca(self, data: np.ndarray) -> np.ndarray:
        if data.shape[1] <= self.n_pca_components:
            logger.warning(f"Number of features ({data.shape[1]}) is <= n_pca_components ({self.n_pca_components}). Skipping PCA.")
            return data
        logger.info(f"Performing PCA to {self.n_pca_components} components...")
        pca = PCA(n_components=self.n_pca_components, random_state=self.random_state)
        reduced_data = pca.fit_transform(data)
        logger.info(f"PCA completed. Reduced data shape: {reduced_data.shape}")
        return reduced_data

    def build_cell_graph(self, data_for_graph: np.ndarray, node_features: np.ndarray) -> Data:
        logger.info(f"Building k-NN graph with k={self.n_neighbors_graph} using data of shape {data_for_graph.shape} for structure...")
        # Graph structure built on `data_for_graph` (e.g. PCA data)
        adj = kneighbors_graph(data_for_graph, n_neighbors=self.n_neighbors_graph,
                               mode='connectivity', include_self=False, n_jobs=-1)
        adj = adj.tocoo()
        edge_index = np.vstack([adj.row, adj.col])
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)

        # Node features for the graph are `node_features` (e.g. PCA data)
        x_tensor = torch.tensor(node_features, dtype=torch.float)
        graph = Data(x=x_tensor, edge_index=edge_index_tensor)
        logger.info(f"Cell graph built: {graph.num_nodes} nodes, {graph.num_edges} edges.")
        return graph

    def split_data(self, feature_matrix: np.ndarray, full_graph: Optional[Data] = None,
                   # Optional: Add labels here if you inject synthetic cells for supervised training
                   # syn_labels: Optional[np.ndarray] = None
                   ) -> Dict[str, Any]:
        logger.info(f"Splitting data with ratio {self.split_ratio} for training/validation...")
        num_samples = feature_matrix.shape[0]
        indices = np.arange(num_samples)

        train_indices, val_indices = train_test_split(
            indices, train_size=self.split_ratio, random_state=self.random_state,
            # stratify=syn_labels if syn_labels is not None else None # Stratify if labels exist
        )
        
        # Split feature matrix (for CAE, VAE, DDPM)
        fm_train = torch.tensor(feature_matrix[train_indices], dtype=torch.float)
        fm_val = torch.tensor(feature_matrix[val_indices], dtype=torch.float)

        output_data = {
            'feature_matrix_train': fm_train,
            'feature_matrix_val': fm_val,
        }
        
        # Handle graph data: For GNNs using NeighborLoader, you pass the full graph
        # and the loader uses train_idx/val_idx masks or specific input_nodes.
        # For now, we'll assume the trainer handles how to use indices with the full graph.
        # If full graph training (not sampling) was intended with split, graph needs to be subsetted.
        # Given NeighborLoader, passing the full graph and train/val indices to the loader is common.
        if full_graph:
            output_data['gnn_full_graph'] = full_graph # Pass the whole graph
            output_data['train_indices'] = torch.tensor(train_indices, dtype=torch.long)
            output_data['val_indices'] = torch.tensor(val_indices, dtype=torch.long)
            # The GNN trainer's NeighborLoader will use these indices for train/val seed nodes.

        # If synthetic labels were generated, split them too
        # if syn_labels is not None:
        #     output_data['syn_labels_train'] = torch.tensor(syn_labels[train_indices], dtype=torch.float)
        #     output_data['syn_labels_val'] = torch.tensor(syn_labels[val_indices], dtype=torch.float)
        #     # And add to graph object if needed
        #     if full_graph:
        #         full_graph.syn_labels = torch.tensor(syn_labels, dtype=torch.float)


        logger.info(f"Data split: {len(train_indices)} train samples, {len(val_indices)} val samples.")
        return output_data


    def preprocess_pipeline(self, matrix_path: str, features_path: str, barcodes_path: str,
                            # Optional: pass paths to pre-generated synthetic labels for supervised training
                            # syn_labels_path: Optional[str] = None 
                           ) -> Dict[str, Any]:
        logger.info("Starting preprocessing pipeline...")
        counts, features, barcodes = self.load_data(matrix_path, features_path, barcodes_path)
        counts, features, barcodes = self.quality_control(counts, features, barcodes)
        
        # Store original (filtered) counts and features if needed later
        # original_filtered_counts = counts.copy()
        # original_filtered_features = features.copy()

        counts_hvg, features_hvg = self.select_highly_variable_genes(counts, features)
        
        # Normalization and scaling on HVGs
        processed_hvg_data = self.normalize_and_scale_data(counts_hvg)
        
        # PCA on normalized, scaled HVGs
        pca_data = self.reduce_dimensions_pca(processed_hvg_data)
        
        # Build cell graph using PCA data for structure and features
        # The GNN models will take pca_data (n_pca_components) as input_dim
        cell_graph = self.build_cell_graph(data_for_graph=pca_data, node_features=pca_data)

        # Optional: Load synthetic labels if path provided
        # syn_labels_all = None
        # if syn_labels_path and os.path.exists(syn_labels_path):
        #    syn_labels_df = pd.read_csv(syn_labels_path, index_col=0) # Assuming barcodes are index
        #    # Align labels with current barcodes (after QC)
        #    # This step is complex: ensure barcodes from labels match current barcodes
        #    # For simplicity, assuming syn_labels_all is a numpy array aligned with `barcodes`
        #    # syn_labels_all = syn_labels_df.loc[barcodes['barcode']].values.astype(float) 
        #    logger.info(f"Loaded synthetic labels from {syn_labels_path}")


        # Split all relevant data (PCA data for CAE, graph + indices for GNN)
        # If syn_labels_all exists, pass it for stratified splitting and inclusion
        final_data_splits = self.split_data(feature_matrix=pca_data, full_graph=cell_graph, 
                                            # syn_labels=syn_labels_all
                                            )
        
        # Add some raw/intermediate data for potential reference, not directly for model input usually
        # final_data_splits['raw_counts_qc'] = counts 
        # final_data_splits['features_qc'] = features
        # final_data_splits['barcodes_qc'] = barcodes
        # final_data_splits['hvg_normalized_scaled_data'] = processed_hvg_data
        
        logger.info("Preprocessing pipeline finished.")
        return final_data_splits


if __name__ == '__main__':
    # This is an example of how to run the preprocessor standalone
    # In your tool, this would likely be called from main.py
    parser = argparse.ArgumentParser(description='Preprocess single-cell RNA-seq data.')
    parser.add_argument('--matrix_path', type=str, required=True, help='Path to .mtx matrix file')
    parser.add_argument('--features_path', type=str, required=True, help='Path to features.tsv file')
    parser.add_argument('--barcodes_path', type=str, required=True, help='Path to barcodes.tsv file')
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Path to config YAML file')
    parser.add_argument('--output_pickle_path', type=str, default='outputs/preprocessed_data.pkl', 
                        help='Path to save preprocessed data dictionary as a pickle file')
    
    cli_args = parser.parse_args()

    # Setup basic logging for standalone run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    try:
        with open(cli_args.config_path, 'r') as f:
            full_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {cli_args.config_path}. Exiting.")
        exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config file {cli_args.config_path}: {e}. Exiting.")
        exit(1)

    preprocessor = GenomicDataPreprocessor(config=full_config) # Pass the full config
    
    processed_output_dict = preprocessor.preprocess_pipeline(
        matrix_path=cli_args.matrix_path,
        features_path=cli_args.features_path,
        barcodes_path=cli_args.barcodes_path
    )

    # Ensure output directory exists for pickle
    output_dir = Path(cli_args.output_pickle_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    import pickle
    with open(cli_args.output_pickle_path, 'wb') as f:
        pickle.dump(processed_output_dict, f)
    
    logger.info(f"Preprocessing complete. Results saved to {cli_args.output_pickle_path}")
    logger.info(f"Output dictionary keys: {processed_output_dict.keys()}")