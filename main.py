#!/usr/bin/env python3
import argparse
import yaml
import logging
from pathlib import Path
from training.trainer import train
from detection.detector import detect
from preprocessing.preprocess_train import preprocess_train
from preprocessing.preprocess_detect import preprocess_detect

def setup_logging(log_level: str, log_file: str = 'logs/genome_defender.log'):
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

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
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
                        help='Path to model weights file for detection (e.g., weights/cae_finetuned_june.pt). Used only in detect mode.')
    parser.add_argument('--incremental', action='store_true',
                        help='Enable incremental training (load pre-trained weights)')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load configuration
    logging.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    config['model'] = args.model  # Ensure model is set in config

    # Validate dataset directory
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_dir():
        logging.error(f"Dataset directory {dataset_dir} does not exist")
        raise ValueError(f"Dataset directory {dataset_dir} does not exist")

    # Validate weights file for detection mode
    if args.mode == 'detect' and args.weights:
        weights_path = Path(args.weights)
        if not weights_path.is_file():
            logging.error(f"Weights file {weights_path} does not exist")
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
            logging.error(f"Default weights file {args.weights} for model {args.model} does not exist")
            raise ValueError(f"Default weights file {args.weights} for model {args.model} does not exist")

    # Run mode
    if args.mode == 'train':
        if args.weights:
            logging.warning("The --weights flag is ignored in train mode")
        logging.info(f"Starting training for {args.model} on {args.dataset}")
        train(args.config, args.dataset, args.model, args.output, args.incremental)
        logging.info("Training completed successfully")
    elif args.mode == 'detect':
        logging.info(f"Starting detection for {args.model} on {args.dataset} using weights {args.weights}")
        detect(args.config, args.dataset, args.model, args.output, args.weights)
        logging.info("Detection completed successfully")

if __name__ == '__main__':
    main()