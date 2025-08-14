import scanpy as sc
import numpy as np
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import vstack
from pathlib import Path
import logging
from typing import List, Tuple
import argparse

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

def load_patient_data(patient_dir: Path) -> Tuple[sc.AnnData, str]:
    """
    Load scRNA-seq data from a patient directory.

    Args:
        patient_dir: Path to patient directory containing matrix.mtx, barcodes.tsv, features.tsv

    Returns:
        Tuple of AnnData object and patient ID
    """
    try:
        logging.info(f"Loading data from {patient_dir}")
        adata = sc.read_10x_mtx(patient_dir, var_names='gene_symbols', cache=True)
        patient_id = patient_dir.name
        # Prefix barcodes with patient ID to ensure uniqueness
        adata.obs_names = [f"{patient_id}_{barcode}" for barcode in adata.obs_names]
        return adata, patient_id
    except Exception as e:
        logging.error(f"Failed to load data from {patient_dir}: {e}")
        raise

def validate_features(adatas: List[sc.AnnData], patient_ids: List[str]) -> pd.DataFrame:
    """
    Validate that all patients have the same gene features.

    Args:
        adatas: List of AnnData objects
        patient_ids: List of patient IDs

    Returns:
        Common features DataFrame
    """
    feature_dfs = [adata.var for adata in adatas]
    # Check if all feature DataFrames are identical
    first_features = feature_dfs[0]
    for i, (features, patient_id) in enumerate(zip(feature_dfs[1:], patient_ids[1:])):
        if not first_features.equals(features):
            logging.error(f"Feature mismatch in {patient_id}: features differ from {patient_ids[0]}")
            raise ValueError(f"Feature mismatch in {patient_id}")
    logging.info("All patients have consistent gene features")
    return first_features

def combine_patient_data(data_dir: str, output_dir: str) -> None:
    """
    Combine scRNA-seq data from multiple patient directories into a single dataset.

    Args:
        data_dir: Directory containing patient subfolders (e.g., data/patient_1)
        output_dir: Directory to save combined dataset (e.g., final_data)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Get list of patient directories
    patient_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if not patient_dirs:
        logging.error(f"No patient directories found in {data_dir}")
        raise ValueError(f"No patient directories found in {data_dir}")
    
    logging.info(f"Found {len(patient_dirs)} patient directories in {data_dir}")
    
    # Load data for each patient
    adatas = []
    patient_ids = []
    for patient_dir in patient_dirs:
        adata, patient_id = load_patient_data(patient_dir)
        adatas.append(adata)
        patient_ids.append(patient_id)
    
    # Validate features
    common_features = validate_features(adatas, patient_ids)
    
    # Combine AnnData objects
    logging.info("Combining patient data")
    combined_adata = adatas[0].concatenate(
        adatas[1:],
        join='inner',
        batch_key='patient_id',
        batch_categories=patient_ids
    )
    
    # Save combined dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving combined dataset to {output_dir}")
    
    # Save matrix.mtx
    mmwrite(output_dir / 'matrix.mtx', combined_adata.X)
    
    # Save barcodes.tsv
    combined_adata.obs_names.to_series().to_csv(
        output_dir / 'barcodes.tsv', index=False, header=False
    )
    
    # Save features.tsv
    common_features.to_csv(output_dir / 'features.tsv', sep='\t', header=False)
    
    logging.info(f"Combined dataset saved to {output_dir} with {combined_adata.n_obs} cells and {combined_adata.n_vars} genes")

def main():
    parser = argparse.ArgumentParser(description='Combine scRNA-seq patient data into a single dataset')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing patient subfolders (default: data)')
    parser.add_argument('--output_dir', type=str, default='final_data',
                        help='Directory to save combined dataset (default: final_data)')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    args = parser.parse_args()

    setup_logging(args.log_level)
    combine_patient_data(args.data_dir, args.output_dir)

if __name__ == '__main__':
    main()