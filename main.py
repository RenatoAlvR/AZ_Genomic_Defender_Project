#!/usr/bin/env python3
"""
GenomeDefender - CLI Tool for scRNA-seq Data Poisoning Detection

Usage:
    python main.py --mode train --model ddpm --dataset data/master --config configs/ddpm_config.yaml --output weights/ddpm.pt
    python main.py --mode detect --model ddpm --dataset data/suspicious --config configs/ddpm_config.yaml --output results/detection --weights weights/ddpm.pt
"""

import argparse
import yaml
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


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='GenomeDefender: scRNA-seq anomaly detection')
    parser.add_argument('--mode', type=str, choices=['train', 'detect'], required=True,
                        help='Mode: train or detect')
    parser.add_argument('--model', type=str, choices=['cae', 'vae', 'gnn_ae', 'ddpm'], required=True,
                        help='Model to use: cae, vae, gnn_ae, or ddpm')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Directory containing 10x Genomics data (matrix.mtx, barcodes.tsv, features.tsv)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model configuration YAML file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save model weights (train) or anomaly scores (detect)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model weights file for detection')
    parser.add_argument('--incremental', action='store_true',
                        help='Enable incremental training (load pre-trained weights)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Detection threshold (0-1 quantile). Default: 0.95')
    parser.add_argument('--no-report', action='store_true',
                        help='Skip generation of reports')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    args = parser.parse_args()

    # Setup logging with colored console and JSON file logging
    setup_logging(args.log_level)
    logger = get_logger()
    audit = AuditTrail()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    config['model'] = args.model

    # Validate dataset directory
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_dir():
        logger.error(f"Dataset directory {dataset_dir} does not exist")
        raise ValueError(f"Dataset directory {dataset_dir} does not exist")

    # Validate weights file for detection mode
    if args.mode == 'detect' and args.weights:
        weights_path = Path(args.weights)
        if not weights_path.is_file():
            logger.error(f"Weights file {weights_path} does not exist")
            raise ValueError(f"Weights file {weights_path} does not exist")
    elif args.mode == 'detect' and not args.weights:
        # Set default weights paths if not provided
        default_weights = {
            'cae': 'weights/cae_synthetic_cells.pt',
            'vae': 'weights/vae_gene_scaling.pt',
            'gnn_ae': 'weights/gnn_ae_label_flips.pt',
            'ddpm': 'weights/ddpm_noise.pt'
        }
        args.weights = default_weights.get(args.model, None)
        if not args.weights or not Path(args.weights).is_file():
            logger.error(f"Default weights file {args.weights} for model {args.model} does not exist")
            raise ValueError(f"Default weights file {args.weights} for model {args.model} does not exist")

    # Run mode
    if args.mode == 'train':
        if args.weights:
            logger.warning("The --weights flag is ignored in train mode")
        logger.info(f"Starting training for {args.model.upper()} on {args.dataset}")
        train(args.config, args.dataset, args.model, args.output, args.incremental)
        logger.info("Training completed successfully")
        audit.log_training(args.model, args.dataset, config.get('epochs', 100), 0.0)
        
    elif args.mode == 'detect':
        logger.info(f"Starting detection for {args.model.upper()} on {args.dataset}")
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


if __name__ == '__main__':
    main()