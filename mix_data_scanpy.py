import scanpy as sc
import pandas as pd
import anndata as ad
from scipy.io import mmwrite
from pathlib import Path
import logging
from typing import List, Tuple
import argparse
import gc
import gzip
import shutil
import io

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
    """
    try:
        logging.info(f"Loading data from {patient_dir.name}...")
        adata = sc.read_10x_mtx(patient_dir, var_names='gene_symbols', cache=True)
        patient_id = patient_dir.name
        
        # Prefix barcodes to ensure uniqueness across the merged dataset
        adata.obs_names = [f"{patient_id}_{barcode}" for barcode in adata.obs_names]
        
        # Manually add the batch label here to avoid concat overhead
        adata.obs['patient_id'] = patient_id
        
        return adata, patient_id
    except Exception as e:
        logging.error(f"Failed to load data from {patient_dir}: {e}")
        raise

def validate_features(adatas: List[sc.AnnData], patient_ids: List[str]) -> pd.DataFrame:
    """
    Validate that all patients have the same gene features.
    """
    feature_dfs = [adata.var for adata in adatas]
    first_features = feature_dfs[0]
    
    for i, (features, patient_id) in enumerate(zip(feature_dfs[1:], patient_ids[1:])):
        if not first_features.equals(features):
            logging.error(f"Feature mismatch in {patient_id}.")
            raise ValueError(f"Feature mismatch in {patient_id}")
            
    logging.info("Validation passed: All patients have consistent gene features.")
    return first_features

def combine_patient_data(data_dir: str, output_dir: str) -> None:
    """
    Combine scRNA-seq data memory-efficiently.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    patient_dirs = [d for d in data_dir.glob('*/*') if d.is_dir()]
    if not patient_dirs:
        raise ValueError(f"No patient directories found in {data_dir}")

    logging.info(f"Found {len(patient_dirs)} patient directories. Beginning load sequence...")

    adatas = []
    patient_ids = []
    
    # Load all datasets into memory
    for patient_dir in patient_dirs:
        adata, patient_id = load_patient_data(patient_dir)
        adatas.append(adata)
        patient_ids.append(patient_id)

    # Validate genes align perfectly before merging
    common_features = validate_features(adatas, patient_ids)

    logging.info("Merging matrices... this may cause a temporary spike in RAM.")
    
    # Use modern anndata.concat (highly optimized for scipy.sparse matrices)
    combined_adata = ad.concat(adatas, join='inner')
    
    # CRITICAL MEMORY OPTIMIZATION: 
    # Delete the list of original objects and force RAM clearance immediately
    logging.info("Merge complete. Purging individual matrices from memory to free up RAM...")
    del adatas 
    gc.collect()

    # Save combined dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving compressed dataset to {output_dir}...")

    # Write sparse matrix to an in-memory buffer, then gzip it to disk
    # (mmwrite doesn't support writing directly to gzip streams)
    logging.info("Writing matrix.mtx.gz...")
    mtx_buffer = io.BytesIO()
    mmwrite(mtx_buffer, combined_adata.X)
    with gzip.open(output_dir / 'matrix.mtx.gz', 'wb') as f_gz:
        f_gz.write(mtx_buffer.getvalue())
    del mtx_buffer

    # Write barcodes.tsv.gz
    logging.info("Writing barcodes.tsv.gz...")
    with gzip.open(output_dir / 'barcodes.tsv.gz', 'wt') as f_gz:
        combined_adata.obs_names.to_series().to_csv(f_gz, index=False, header=False)

    # Write features.tsv.gz
    logging.info("Writing features.tsv.gz...")
    with gzip.open(output_dir / 'features.tsv.gz', 'wt') as f_gz:
        common_features.to_csv(f_gz, sep='\t', header=False)

    logging.info(f"Success! Dataset saved with {combined_adata.n_obs} cells and {combined_adata.n_vars} genes.")

def main():
    parser = argparse.ArgumentParser(description='Memory-optimized scRNA-seq concatenator.')
    parser.add_argument('--data_dir', type=str, default='data', help='Input directory')
    parser.add_argument('--output_dir', type=str, default='final_data', help='Output directory')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')
    args = parser.parse_args()

    setup_logging(args.log_level)
    combine_patient_data(args.data_dir, args.output_dir)

if __name__ == '__main__':
    main()