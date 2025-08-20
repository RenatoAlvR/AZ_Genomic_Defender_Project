import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import logging
import sys
from typing import List, Tuple
import gzip
import tempfile
import shutil

def setup_logging(log_level: str = 'INFO', log_file: str = 'logs/genome_defender.log'):
    """Set up logging to file and console."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_patient_data(patient_dir: Path) -> Tuple[sc.AnnData, str]:
    """
    Load scRNA-seq data from a patient directory, handling gzipped matrix.mtx.gz, barcodes.tsv.gz, and features.tsv.gz.

    Args:
        patient_dir: Path to patient directory containing matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz

    Returns:
        Tuple of AnnData object and patient ID
    """
    patient_id = patient_dir.name
    matrix_file = patient_dir / 'matrix.mtx.gz'
    barcodes_file = patient_dir / 'barcodes.tsv.gz'
    features_file = patient_dir / 'features.tsv.gz'

    # Validate files
    if not matrix_file.is_file():
        logging.error(f"No matrix.mtx.gz found in {patient_dir}")
        raise ValueError(f"No matrix.mtx.gz found in {patient_dir}")
    if not barcodes_file.is_file():
        logging.error(f"No barcodes.tsv.gz found in {patient_dir}")
        raise ValueError(f"No barcodes.tsv.gz found in {patient_dir}")
    if not features_file.is_file():
        logging.error(f"No features.tsv.gz found in {patient_dir}")
        raise ValueError(f"No features.tsv.gz found in {patient_dir}")

    logging.info(f"Loading data from {patient_dir} (matrix: matrix.mtx.gz, barcodes: barcodes.tsv.gz, features: features.tsv.gz)")
    try:
        adata = sc.read_10x_mtx(patient_dir, var_names='gene_symbols', cache=True)
        logging.info(f"Loaded {patient_id} with {adata.shape[0]} cells, {adata.shape[1]} genes, {adata.X.nnz} non-zero entries")
        adata.obs_names = [f"{patient_id}_{barcode}" for barcode in adata.obs_names]
        return adata, patient_id
    except Exception as e:
        logging.error(f"Failed to load data from {patient_dir}: {e}")
        raise

def merge_patient_data(input_dir: str, output_dir: str, log_level: str = 'INFO', batch_size: int = 10) -> None:
    """
    Merge scRNA-seq data from multiple patient folders into a single dataset in batches.

    Args:
        input_dir: Directory containing patient subfolders (e.g., data/patient_1).
        output_dir: Directory to save merged dataset (e.g., final_data).
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        batch_size: Number of patients to process per batch.
    """
    setup_logging(log_level)
    logging.info(f"Starting data merge from {input_dir} to {output_dir}")

    # Validate input directory
    input_path = Path(input_dir)
    if not input_path.is_dir():
        logging.error(f"Input directory {input_dir} does not exist")
        raise ValueError(f"Input directory {input_dir} does not exist")

    # Get list of patient subfolders
    patient_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    if not patient_dirs:
        logging.error(f"No patient subfolders found in {input_dir}")
        raise ValueError(f"No patient subfolders found in {input_dir}")

    logging.info(f"Found {len(patient_dirs)} patient folders: {[d.name for d in patient_dirs]}")

    # Initialize temporary directory for batch files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        all_genes = set()
        batch_adatas = []
        batch_barcodes = []
        batch_count = 0
        total_nnz = 0

        # Process patients in batches
        for i, patient_dir in enumerate(patient_dirs):
            try:
                adata, patient_id = load_patient_data(patient_dir)
                all_genes.update(adata.var_names)
                batch_adatas.append(adata)
                batch_barcodes.extend(adata.obs_names)
                total_nnz += adata.X.nnz
                logging.info(f"Processing patient {patient_id} in batch {batch_count + 1}, nnz: {adata.X.nnz}")

                # Save batch if batch_size is reached or last patient
                if len(batch_adatas) >= batch_size or i == len(patient_dirs) - 1:
                    all_genes_batch = sorted(list(all_genes))
                    merged_matrices = []
                    merged_barcodes = []

                    for adata in batch_adatas:
                        gene_indices = [all_genes_batch.index(gene) if gene in all_genes_batch else -1 for gene in adata.var_names]
                        n_cells = adata.shape[0]
                        n_genes = len(all_genes_batch)
                        new_matrix = sparse.csr_matrix((n_cells, n_genes), dtype=np.float32)
                        for j, gene_idx in enumerate(gene_indices):
                            if gene_idx != -1:
                                new_matrix[:, gene_idx] = adata.X[:, j]
                        merged_matrices.append(new_matrix)
                        merged_barcodes.extend(adata.obs_names)

                    batch_matrix = sparse.vstack(merged_matrices)
                    batch_barcodes_df = pd.Series(merged_barcodes, name='barcode')
                    batch_file = tmp_path / f"batch_{batch_count + 1}"
                    batch_file.mkdir()
                    with open(batch_file / 'matrix.mtx', 'w') as f:
                        f.write("%%MatrixMarket matrix coordinate integer general\n")
                        f.write(f"{batch_matrix.shape[1]} {batch_matrix.shape[0]} {batch_matrix.nnz}\n")
                        for i, j, v in zip(*sparse.find(batch_matrix)):
                            f.write(f"{i+1} {j+1} {int(v)}\n")
                        f.flush()  # Ensure buffer is written
                    batch_barcodes_df.to_csv(batch_file / 'barcodes.tsv', index=False, header=False)
                    pd.DataFrame(all_genes_batch, columns=['gene']).to_csv(batch_file / 'features.tsv', index=False, header=False)
                    logging.info(f"Batch {batch_count + 1} saved to {batch_file} with {batch_matrix.shape[0]} cells, {batch_matrix.nnz} nnz")
                    batch_adatas = []
                    batch_barcodes = []
                    batch_count += 1
            except Exception as e:
                logging.warning(f"Skipping {patient_dir.name}: {e}")
                continue

        if not batch_count:
            logging.error("No valid patient data loaded")
            raise ValueError("No valid patient data loaded")

        logging.info(f"Total non-zero entries across patients: {total_nnz}")
        logging.info(f"Merging {batch_count} batches to {output_dir}")
        all_genes = sorted(list(all_genes))
        merged_matrices = []
        merged_barcodes = []
        final_nnz = 0

        # Merge batches
        for i in range(1, batch_count + 1):
            batch_file = tmp_path / f"batch_{i}"
            logging.info(f"Merging matrix from {batch_file}")
            batch_matrix = sparse.load_npz(batch_file / 'matrix.npz') if (batch_file / 'matrix.npz').exists() else sparse.csr_matrix(sparse.load_npz(batch_file / 'matrix.npz'))
            batch_barcodes = pd.read_csv(batch_file / 'barcodes.tsv', header=None, names=['barcode'])['barcode'].tolist()
            merged_matrices.append(batch_matrix)
            merged_barcodes.extend(batch_barcodes)
            final_nnz += batch_matrix.nnz
            logging.info(f"Batch {i} nnz: {batch_matrix.nnz}")

        merged_matrix = sparse.vstack(merged_matrices)
        merged_barcodes = pd.Series(merged_barcodes, name='barcode')
        logging.info(f"Final matrix nnz: {final_nnz}, expected: {total_nnz}")

        if final_nnz != total_nnz:
            logging.error(f"Non-zero entry mismatch: final {final_nnz}, expected {total_nnz}")
            raise ValueError(f"Non-zero entry mismatch: final {final_nnz}, expected {total_nnz}")

        # Save final dataset
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving merged dataset to {output_dir}")
        with open(output_path / 'matrix.mtx', 'w') as f:
            f.write("%%MatrixMarket matrix coordinate integer general\n")
            f.write(f"{merged_matrix.shape[1]} {merged_matrix.shape[0]} {merged_matrix.nnz}\n")
            for i, j, v in zip(*sparse.find(merged_matrix)):
                f.write(f"{i+1} {j+1} {int(v)}\n")
            f.flush()  # Ensure buffer is written
        merged_barcodes.to_csv(output_path / 'barcodes.tsv', index=False, header=False)
        pd.DataFrame(all_genes, columns=['gene']).to_csv(output_path / 'features.tsv', index=False, header=False)
        logging.info(f"Combined dataset saved to {output_dir} with {merged_matrix.shape[0]} cells and {merged_matrix.shape[1]} genes")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Merge scRNA-seq patient datasets into a single dataset')
    parser.add_argument('--input_dir', type=str, default='data', help='Directory containing patient subfolders')
    parser.add_argument('--output_dir', type=str, default='final_data', help='Directory to save merged dataset')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of patients per batch')
    args = parser.parse_args()

    merge_patient_data(args.input_dir, args.output_dir, args.log_level, args.batch_size)