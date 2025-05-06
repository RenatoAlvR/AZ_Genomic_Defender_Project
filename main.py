#!/usr/bin/env python3
import argparse
import yaml
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from genomic_defender_v1.preprocessing.preprocessor import GenomicDataPreprocessor
from genomic_defender_v1.training.trainer import ModelTrainer
from genomic_defender_v1.detection.detector import PoisonDetector
from genomic_defender_v1.utils.logger import GenomeGuardianLogger
from genomic_defender_v1.utils.logger import _setup_logger

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GenomeGuardian: Data Poisoning Detection for scRNA-seq Data')
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', 
                          help='Train models on input data')
    mode_group.add_argument('--detect', action='store_true',
                          help='Run detection using trained models')
    
    # Data input arguments
    parser.add_argument('--matrix_path', type=str, required=True, 
                       help='Path to .mtx matrix file')
    parser.add_argument('--features_path', type=str, required=True, 
                       help='Path to features.tsv file')
    parser.add_argument('--barcodes_path', type=str, required=True, 
                       help='Path to barcodes.tsv file')
    
    # Configuration arguments
    parser.add_argument('--config_path', type=str, default='config.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    parser.add_argument('--force_retrain', action='store_true',
                      help='Force full retrain of all models (including non-EIF)')
    
    return parser.parse_args()

def save_results(results: Dict[str, Any]) -> None:
    """Save detection results to fixed 'results' directory"""
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)  # Auto-create if missing
    
    with open(output_dir / 'detection_scores.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    with open(output_dir / 'report.txt', 'w') as f:
        for model_name, scores in results['individual_scores'].items():
            f.write(f"{model_name} scores:\n{scores}\n\n")
        f.write(f"Final fused scores:\n{results['final_scores']}\n")
    
    logging.info(f"Results saved to {output_dir}")

def main():
    # Parse arguments and setup
    args = parse_arguments()
    try:
        config = load_config(args.config_path)
    except Exception as e:
        print(f"Failed to load config: {e}")
        config = {}  # Fallback empty config
    
    # Initialize logger WITH the loaded config
    logger = GenomeGuardianLogger(config=config.get('logging', {}))
    _setup_logger(args.log_level)
    
    # Preprocess data
    logging.info("Starting data preprocessing...")
    preprocessor = GenomicDataPreprocessor(config=config)
    processed_data = preprocessor.preprocess_pipeline(
        matrix_path=args.matrix_path,
        features_path=args.features_path,
        barcodes_path=args.barcodes_path
    )
    logging.info("Data preprocessing completed.")
    
    if args.train:
        # Training mode
        logging.info("Initializing model trainer...")
        trainer = ModelTrainer(config=config)
        trainer.train_incrementally(
            processed_data=processed_data,
            force_retrain=args.force_retrain
        )
        logging.info("Training completed successfully.")
        
    elif args.detect:
        # Detection mode
        logging.info("Initializing poison detector...")
        detector = PoisonDetector(config_path=args.config_path)
        results = detector.run(processed_data)
        save_results(results)  # Removed output_dir parameter
        logging.info("Detection completed successfully.")
        
    else:
        logging.error("No valid mode specified (use --train or --detect)")
        raise ValueError("Must specify either --train or --detect")

if __name__ == '__main__':
    main()