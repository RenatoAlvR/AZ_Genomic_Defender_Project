#!/usr/bin/env python3
"""
GenomeDefender - CLI Tool for scRNA-seq Data Poisoning Detection

Usage:
    python main.py --mode train --model ddpm --dataset data/master --config configs/ddpm_config.yaml --output weights/ddpm.pt
    python main.py --mode detect --model ddpm --dataset data/suspicious --config configs/ddpm_config.yaml --output results/detection --weights weights/ddpm.pt
    python main.py --mode generate --model ddpm --config configs/ddpm_config.yaml --output data/synthetic --weights weights/ddpm.pt --num_samples 1000
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from training.trainer import train
from detection.detector import detect
from utils.logger import setup_logging, get_logger, AuditTrail


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    logger = get_logger()
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


def generate_synthetic(config_path: str, weights_path: str, output_path: str,
                       num_samples: int, poison_factor: float, gene_names_path: str = None) -> None:
    """Generate synthetic cells using trained DDPM model.
    
    Args:
        config_path: Path to DDPM config
        weights_path: Path to trained DDPM weights
        output_path: Output directory for generated data
        num_samples: Number of cells to generate
        poison_factor: Poisoning intensity (0.0=clean, 1.0=heavily perturbed)
        gene_names_path: Optional path to genes.txt for labeling
    """
    from models.ddpm_model import DenoisingDiffusionPM
    
    logger = get_logger()
    
    # Load model
    logger.info(f"Loading DDPM model from {weights_path}")
    model = DenoisingDiffusionPM.load(weights_path)
    
    # Generate
    logger.info(f"Generating {num_samples} synthetic cells (poison_factor={poison_factor})")
    synthetic = model.generate(
        num_samples=num_samples,
        poison_factor=poison_factor,
        batch_size=min(500, num_samples),
        seed=42
    ).numpy()
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data
    np.save(output_dir / 'data.npy', synthetic)
    logger.info(f"Saved synthetic data to {output_dir / 'data.npy'}")
    
    # Save cell names
    cell_names = [f'SYN_CELL_{i+1}' for i in range(num_samples)]
    pd.Series(cell_names).to_csv(output_dir / 'cells.txt', index=False, header=False)
    
    # Copy gene names if provided
    if gene_names_path and Path(gene_names_path).exists():
        import shutil
        shutil.copy(gene_names_path, output_dir / 'genes.txt')
        logger.info(f"Copied gene names from {gene_names_path}")
    
    logger.info(f"Generated {synthetic.shape[0]} cells with {synthetic.shape[1]} features")
    logger.info(f"Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='GenomeDefender: scRNA-seq anomaly detection')
    parser.add_argument('--mode', type=str, choices=['train', 'detect', 'generate'], required=True,
                        help='Mode: train, detect, or generate')
    parser.add_argument('--model', type=str, choices=['cae', 'vae', 'gnn_ae', 'ddpm'], required=True,
                        help='Model to use')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset directory (required for train/detect)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model config YAML')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path (weights for train, results for detect, directory for generate)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Model weights path (required for detect/generate)')
    parser.add_argument('--incremental', action='store_true',
                        help='Continue training from weights')
    parser.add_argument('--base_weights', type=str, default=None,
                        help='Source weights to fine-tune from (used with --incremental). '
                        'If not set, loads from --output path.')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Detection threshold (0-1 quantile)')
    parser.add_argument('--no-report', action='store_true',
                        help='Skip report generation')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to generate (generate mode only)')
    parser.add_argument('--poison_factor', type=float, default=0.0,
                        help='Poisoning intensity 0.0-1.0 (generate mode only)')
    parser.add_argument('--gene_names', type=str, default=None,
                        help='Path to genes.txt for labeling generated data')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = get_logger()
    audit = AuditTrail()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    config['model'] = args.model

    # Mode: TRAIN
    if args.mode == 'train':
        if not args.dataset:
            raise ValueError("--dataset is required for train mode")
        dataset_dir = Path(args.dataset)
        if not dataset_dir.is_dir():
            raise ValueError(f"Dataset directory {dataset_dir} does not exist")
        
        logger.info(f"Starting training for {args.model.upper()} on {args.dataset}")
        train(args.config, args.dataset, args.model, args.output, args.incremental, args.base_weights)
        logger.info("Training completed successfully")
        audit.log_training(args.model, args.dataset, config.get('epochs', 100), 0.0)

    # Mode: DETECT
    elif args.mode == 'detect':
        if not args.dataset:
            raise ValueError("--dataset is required for detect mode")
        if not args.weights:
            raise ValueError("--weights is required for detect mode")
        
        dataset_dir = Path(args.dataset)
        if not dataset_dir.is_dir():
            raise ValueError(f"Dataset directory {dataset_dir} does not exist")
        if not Path(args.weights).is_file():
            raise ValueError(f"Weights file {args.weights} does not exist")
        
        logger.info(f"Starting detection for {args.model.upper()}")
        detect(
            config_path=args.config,
            dataset_path=args.dataset,
            model_name=args.model,
            output_path=args.output,
            weights_path=args.weights,
            threshold=args.threshold,
            generate_report=not getattr(args, 'no_report', False)
        )
        logger.info("Detection completed successfully")

    # Mode: GENERATE
    elif args.mode == 'generate':
        if args.model != 'ddpm':
            raise ValueError("Generate mode only supports DDPM model")
        if not args.weights:
            raise ValueError("--weights is required for generate mode")
        if not Path(args.weights).is_file():
            raise ValueError(f"Weights file {args.weights} does not exist")
        
        generate_synthetic(
            config_path=args.config,
            weights_path=args.weights,
            output_path=args.output,
            num_samples=args.num_samples,
            poison_factor=args.poison_factor,
            gene_names_path=args.gene_names
        )


if __name__ == '__main__':
    main()