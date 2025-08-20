import scanpy as sc
import numpy as np
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import vstack
from pathlib import Path
import logging
from typing import List, Tuple
import argparse
import gzip
import shutil
import tempfile

def setup_logging(log_level: str = 'INFO', log_file: str = 'logs/combine_data.log'):
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

def get_10x_path(directory: Path, base_name: str) -> Path:
    """Find the path to a 10x file, handling possible .gz extension."""
    p = directory / base_name
    if p.exists():
        return p
    p_gz = directory / (base_name + '.gz')
    if p_gz.exists():
        return p_gz
    raise FileNotFoundError(f"Could not find {base_name} or {base_name}.gz in {directory}")

def open_maybe_gz(path: Path):
    """Return an open function based on whether the file is gzipped."""
    if path.suffix == '.gz':
        return gzip.open
    else:
        return open

def validate_features(patient_dirs: List[Path]) -> str:
    """
    Validate that all patients have the same gene features by comparing file contents.

    Args:
        patient_dirs: List of patient directory paths

    Returns:
        The features content from the first patient
    """
    first_features = None
    for patient_dir in patient_dirs:
        features_path = get_10x_path(patient_dir, 'features.tsv')
        opener = open_maybe_gz(features_path)
        with opener(features_path, 'rt') as f:
            features = f.read()
        if first_features is None:
            first_features = features
        elif features != first_features:
            logging.error(f"Feature mismatch in {patient_dir}: features differ from first patient")
            raise ValueError(f"Feature mismatch in {patient_dir}")
    logging.info("All patients have consistent gene features")
    return first_features

def read_matrix_dimensions(mtx_path: Path, opener) -> Tuple[int, int, int]:
    """Read dimensions from a matrix.mtx file, skipping comment lines."""
    with opener(mtx_path, 'rt') as f:
        # Read header
        header = f.readline().strip()
        if not header.startswith('%%MatrixMarket'):
            raise ValueError(f"Invalid MatrixMarket header in {mtx_path}")
        # Skip comment lines starting with '%'
        while True:
            dim_line = f.readline().strip()
            if not dim_line.startswith('%'):
                break
        try:
            genes, cells, nnz = map(int, dim_line.split())
            return genes, cells, nnz
        except ValueError as e:
            logging.error(f"Invalid dimension line in {mtx_path}: {dim_line}")
            raise ValueError(f"Invalid dimension line in {mtx_path}: {e}")

def process_batch(patient_dirs: List[Path], patient_ids: List[str], batch_idx: int, temp_dir: Path, genes: int) -> Tuple[int, int]:
    """
    Process a batch of patients, saving to a temporary folder.

    Args:
        patient_dirs: List of patient directory paths in the batch
        patient_ids: List of patient IDs in the batch
        batch_idx: Index of the batch
        temp_dir: Temporary directory for batch output
        genes: Number of genes (consistent across patients)

    Returns:
        Tuple of total cells and total non-zero entries in the batch
    """
    batch_dir = temp_dir / f"batch_{batch_idx}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    total_cells = 0
    total_nnz = 0
    
    # Save features.tsv from first patient
    first_features_path = get_10x_path(patient_dirs[0], 'features.tsv')
    shutil.copy(first_features_path, batch_dir / 'features.tsv')
    
    # Process barcodes and matrices
    with open(batch_dir / 'barcodes.tsv', 'w') as barc_f, open(batch_dir / 'matrix.mtx', 'w') as mtx_f:
        mtx_f.write('%%MatrixMarket matrix coordinate integer general\n')
        # We'll update dimensions later
        mtx_f.write(f"{genes} 0 0\n")
        
        offset = 0
        for patient_dir, patient_id in zip(patient_dirs, patient_ids):
            logging.info(f"Processing patient {patient_id} in batch {batch_idx}")
            
            # Read matrix dimensions
            mtx_path = get_10x_path(patient_dir, 'matrix.mtx')
            opener = open_maybe_gz(mtx_path)
            p_genes, p_cells, p_nnz = read_matrix_dimensions(mtx_path, opener)
            if p_genes != genes:
                raise ValueError(f"Gene count mismatch in {patient_dir}: expected {genes}, got {p_genes}")
            
            total_cells += p_cells
            total_nnz += p_nnz
            
            # Process barcodes
            barcodes_path = get_10x_path(patient_dir, 'barcodes.tsv')
            with opener(barcodes_path, 'rt') as b_in:
                for line in b_in:
                    barcode = line.strip()
                    barc_f.write(f"{patient_id}_{barcode}\n")
            
            # Process matrix
            with opener(mtx_path, 'rt') as m_in:
                m_in.readline()  # header
                while True:  # skip comments
                    line = m_in.readline().strip()
                    if not line.startswith('%'):
                        break
                for line in m_in:
                    row, col, val = line.split()
                    col = int(col) + offset
                    mtx_f.write(f"{row} {col} {val}\n")
            offset += p_cells
        
        # Update matrix dimensions
        mtx_f.seek(0)
        mtx_f.write(f"%%MatrixMarket matrix coordinate integer general\n{genes} {total_cells} {total_nnz}\n")
    
    logging.info(f"Batch {batch_idx} saved to {batch_dir} with {total_cells} cells")
    return total_cells, total_nnz

def combine_patient_data(data_dir: str, output_dir: str, batch_size: int = 7) -> None:
    """
    Combine scRNA-seq data from multiple patient directories into a single dataset using batching.

    Args:
        data_dir: Directory containing patient subfolders
        output_dir: Directory to save combined dataset
        batch_size: Number of patients per batch (default: 7)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Get sorted list of patient directories
    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not patient_dirs:
        logging.error(f"No patient directories found in {data_dir}")
        raise ValueError(f"No patient directories found in {data_dir}")
    
    patient_ids = [d.name for d in patient_dirs]
    logging.info(f"Found {len(patient_dirs)} patient directories in {data_dir}")
    
    # Validate features
    first_features = validate_features(patient_dirs)
    
    # Get gene count from first patient
    mtx_path = get_10x_path(patient_dirs[0], 'matrix.mtx')
    opener = open_maybe_gz(mtx_path)
    genes, _, _ = read_matrix_dimensions(mtx_path, opener)
    
    # Process patients in batches
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        batch_dirs = []
        total_cells = 0
        total_nnz = 0
        
        for i in range(0, len(patient_dirs), batch_size):
            batch_dirs_subset = patient_dirs[i:i + batch_size]
            batch_ids = patient_ids[i:i + batch_size]
            batch_idx = i // batch_size + 1
            cells, nnz = process_batch(batch_dirs_subset, batch_ids, batch_idx, temp_dir, genes)
            batch_dirs.append(temp_dir / f"batch_{batch_idx}")
            total_cells += cells
            total_nnz += nnz
        
        # Merge batches
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Merging {len(batch_dirs)} batches to {output_dir}")
        
        # Copy features.tsv from first batch
        shutil.copy(batch_dirs[0] / 'features.tsv', output_dir / 'features.tsv')
        
        # Merge barcodes
        with open(output_dir / 'barcodes.tsv', 'w') as barc_f:
            for batch_dir in batch_dirs:
                with open(batch_dir / 'barcodes.tsv', 'r') as b_in:
                    for line in b_in:
                        barc_f.write(line)
        
        # Merge matrices
        with open(output_dir / 'matrix.mtx', 'w') as mtx_f:
            mtx_f.write('%%MatrixMarket matrix coordinate integer general\n')
            mtx_f.write(f"{genes} {total_cells} {total_nnz}\n")
            offset = 0
            for batch_dir in batch_dirs:
                logging.info(f"Merging matrix from {batch_dir}")
                with open(batch_dir / 'matrix.mtx', 'r') as m_in:
                    m_in.readline()  # header
                    m_in.readline()  # dimensions
                    for line in m_in:
                        row, col, val = line.split()
                        col = int(col) + offset
                        mtx_f.write(f"{row} {col} {val}\n")
                # Update offset based on batch cell count
                with open(batch_dir / 'matrix.mtx', 'r') as m_in:
                    m_in.readline()
                    _, cells, _ = map(int, m_in.readline().split())
                    offset += cells
        
        logging.info(f"Combined dataset saved to {output_dir} with {total_cells} cells and {genes} genes")

def main():
    parser = argparse.ArgumentParser(description='Combine scRNA-seq patient data into a single dataset')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing patient subfolders (default: data)')
    parser.add_argument('--output_dir', type=str, default='final_data',
                        help='Directory to save combined dataset (default: final_data)')
    parser.add_argument('--batch_size', type=int, default=7,
                        help='Number of patients per batch (default: 7)')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    args = parser.parse_args()

    setup_logging(args.log_level)
    combine_patient_data(args.data_dir, args.output_dir, args.batch_size)

if __name__ == '__main__':
    main()