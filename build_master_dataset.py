#!/usr/bin/env python3
"""
Build Master Dataset - Memory-Efficient Data Fusion for GenomeDefender

This script combines multiple scRNA-seq samples from different categories
(e.g., healthy, cancer, neoplastic) into a single "master" dataset for training.

Memory Efficiency:
- Streams data sample-by-sample without loading everything into memory
- Writes directly to output files during processing
- Uses temporary files for intermediate batching only when necessary

Usage:
    python build_master_dataset.py \
        --input_dirs data/healthy data/cancer data/neoplastic \
        --output_dir master_data \
        --batch_size 5
"""

import gzip
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import shutil


def setup_logging(log_level: str = 'INFO', log_file: str = 'logs/build_master.log'):
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


def get_file_path(directory: Path, base_name: str) -> Path:
    """Find file path, handling optional .gz extension."""
    p = directory / base_name
    if p.exists():
        return p
    p_gz = directory / (base_name + '.gz')
    if p_gz.exists():
        return p_gz
    raise FileNotFoundError(f"Could not find {base_name} or {base_name}.gz in {directory}")


def open_file(path: Path, mode: str = 'rt'):
    """Open file handling .gz compression."""
    if path.suffix == '.gz':
        return gzip.open(path, mode)
    return open(path, mode)


def read_matrix_header(mtx_path: Path) -> Tuple[int, int, int]:
    """Read MatrixMarket header to get dimensions."""
    with open_file(mtx_path) as f:
        # Skip header line
        header = f.readline().strip()
        if not header.startswith('%%MatrixMarket'):
            raise ValueError(f"Invalid MatrixMarket header in {mtx_path}")
        # Skip comment lines
        while True:
            line = f.readline().strip()
            if not line.startswith('%'):
                break
        genes, cells, nnz = map(int, line.split())
        return genes, cells, nnz


def validate_features(sample_dirs: List[Path]) -> Tuple[str, int]:
    """Validate all samples have consistent features. Returns features content and gene count."""
    reference_content = None
    gene_count = None
    
    for sample_dir in sample_dirs:
        features_path = get_file_path(sample_dir, 'features.tsv')
        with open_file(features_path) as f:
            content = f.read()
        
        if reference_content is None:
            reference_content = content
            gene_count = len(content.strip().split('\n'))
        elif content != reference_content:
            logging.warning(f"Feature mismatch in {sample_dir.name} - will use reference features")
    
    logging.info(f"Validated features: {gene_count} genes across {len(sample_dirs)} samples")
    return reference_content, gene_count


def stream_sample_to_output(
    sample_dir: Path,
    sample_id: str,
    barcodes_out,
    matrix_out,
    cell_offset: int,
    gene_count: int
) -> Tuple[int, int]:
    """
    Stream a single sample's data directly to output files.
    
    Returns:
        Tuple of (cells_added, nnz_added)
    """
    # Get file paths
    mtx_path = get_file_path(sample_dir, 'matrix.mtx')
    barcodes_path = get_file_path(sample_dir, 'barcodes.tsv')
    
    # Read dimensions
    sample_genes, sample_cells, sample_nnz = read_matrix_header(mtx_path)
    
    if sample_genes != gene_count:
        raise ValueError(f"Gene count mismatch in {sample_dir}: expected {gene_count}, got {sample_genes}")
    
    # Stream barcodes with sample prefix
    with open_file(barcodes_path) as barc_in:
        for line in barc_in:
            barcode = line.strip()
            barcodes_out.write(f"{sample_id}_{barcode}\n")
    
    # Stream matrix entries with column offset
    with open_file(mtx_path) as mtx_in:
        # Skip header and comment lines
        mtx_in.readline()  # MatrixMarket header
        while True:
            line = mtx_in.readline().strip()
            if not line.startswith('%'):
                break  # This was the dimension line, skip it
        
        # Stream data lines
        for line in mtx_in:
            parts = line.strip().split()
            if len(parts) >= 3:
                row, col, val = parts[0], int(parts[1]), parts[2]
                new_col = col + cell_offset
                matrix_out.write(f"{row} {new_col} {val}\n")
    
    logging.info(f"  Added {sample_id}: {sample_cells} cells, {sample_nnz} non-zero entries")
    return sample_cells, sample_nnz


def build_master_dataset(input_dirs: List[str], output_dir: str) -> None:
    """
    Build a master dataset by streaming samples from multiple input directories.
    
    Args:
        input_dirs: List of directories containing sample subfolders
        output_dir: Directory to save combined master dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all sample directories from all input directories
    all_samples = []
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.is_dir():
            logging.warning(f"Input directory not found: {input_dir}")
            continue
        
        samples = sorted([d for d in input_path.iterdir() if d.is_dir()])
        category = input_path.name
        for sample in samples:
            all_samples.append((sample, f"{category}_{sample.name}"))
        logging.info(f"Found {len(samples)} samples in {input_dir}")
    
    if not all_samples:
        raise ValueError("No valid samples found in any input directory")
    
    logging.info(f"Total: {len(all_samples)} samples to combine")
    
    # Validate features consistency
    sample_dirs = [s[0] for s in all_samples]
    features_content, gene_count = validate_features(sample_dirs)
    
    # Save features.tsv (from first sample)
    first_features_path = get_file_path(sample_dirs[0], 'features.tsv')
    if first_features_path.suffix == '.gz':
        with open_file(first_features_path) as f_in:
            with open(output_path / 'features.tsv', 'w') as f_out:
                f_out.write(f_in.read())
    else:
        shutil.copy(first_features_path, output_path / 'features.tsv')
    
    # Stream all samples to output
    temp_matrix = output_path / 'matrix_temp.mtx'
    total_cells = 0
    total_nnz = 0
    
    with open(output_path / 'barcodes.tsv', 'w') as barcodes_out, \
         open(temp_matrix, 'w') as matrix_out:
        
        for sample_dir, sample_id in all_samples:
            try:
                cells, nnz = stream_sample_to_output(
                    sample_dir=sample_dir,
                    sample_id=sample_id,
                    barcodes_out=barcodes_out,
                    matrix_out=matrix_out,
                    cell_offset=total_cells,
                    gene_count=gene_count
                )
                total_cells += cells
                total_nnz += nnz
            except Exception as e:
                logging.error(f"Error processing {sample_dir}: {e}")
                continue
    
    # Write final matrix with correct header
    final_matrix = output_path / 'matrix.mtx'
    with open(final_matrix, 'w') as f_out:
        f_out.write('%%MatrixMarket matrix coordinate integer general\n')
        f_out.write(f'{gene_count} {total_cells} {total_nnz}\n')
        with open(temp_matrix, 'r') as f_in:
            for line in f_in:
                f_out.write(line)
    
    # Remove temp file
    temp_matrix.unlink()
    
    logging.info(f"Master dataset saved to {output_dir}")
    logging.info(f"  Total cells: {total_cells:,}")
    logging.info(f"  Total genes: {gene_count:,}")
    logging.info(f"  Non-zero entries: {total_nnz:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Build master scRNA-seq dataset from multiple sample categories'
    )
    parser.add_argument(
        '--input_dirs', type=str, nargs='+', required=True,
        help='Directories containing sample subfolders (e.g., data/healthy data/cancer data/neoplastic)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='master_data',
        help='Directory to save combined master dataset (default: master_data)'
    )
    parser.add_argument(
        '--log_level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logging.info("=" * 60)
    logging.info("GenomeDefender - Building Master Dataset")
    logging.info("=" * 60)
    build_master_dataset(args.input_dirs, args.output_dir)
    logging.info("Done!")


if __name__ == '__main__':
    main()
