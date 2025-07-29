import argparse
import yaml
import numpy as np
from pathlib import Path
import torch
from torch_geometric.data import Data
from preprocessing.preprocess_detect import preprocess_detect
from models.cae_model import ContrastiveAutoencoder
from models.vae_model import VariationalAutoencoder
from models.gnn_model import GNNAutoencoder
from models.ddpm_model import DenoisingDiffusionPM

def detect(config_path: str, dataset_path: str, model_name: str, output_path: str) -> None:
    """Run anomaly detection using the specified model and dataset."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load preprocessed data
    data = preprocess_detect(dataset_path, config)
    
    # Initialize model and load weights
    model_name = model_name.lower()
    if model_name == 'cae':
        model = ContrastiveAutoencoder.load(f'weights/cae_synthetic_cells.pt', device=config['device'])
    elif model_name == 'vae':
        model = VariationalAutoencoder.load(f'weights/vae_gene_scaling.pt', device=config['device'])
    elif model_name == 'gnn_ae':
        model = GNNAutoencoder.load(f'weights/gnn_ae_label_flips.pt', device=config['device'])
    elif model_name == 'ddpm':
        model = DenoisingDiffusionPM.load(f'weights/ddpm_noise.pt', device=config['device'])
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose 'cae', 'vae', 'gnn_ae', or 'ddpm'.")

    # Run detection
    model.eval()
    anomaly_scores = model.detect(data)

    # Compute metrics
    threshold = config.get('detection_threshold', 0.95)  # Default: flag top 5%
    anomaly_threshold = np.quantile(anomaly_scores, threshold)
    is_anomalous = anomaly_scores > anomaly_threshold
    anomaly_percentage = 100 * np.mean(is_anomalous)
    suspected_indices = np.where(is_anomalous)[0]

    # Output metrics
    target = 'cells' if model_name in ['cae', 'gnn_ae', 'ddpm'] else 'genes'
    print(f"Detection completed, {anomaly_percentage:.2f}% of the dataset could be poisoned, "
          f"suspected {target}: {suspected_indices.tolist()}")

    # Save anomaly scores
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, anomaly_scores, delimiter=',')
    print(f"Anomaly scores saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run anomaly detection for scRNA-seq data.')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['cae', 'vae', 'gnn_ae', 'ddpm'], 
                        help='Model to use for detection')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Path to input dataset (CSV or preprocessed)')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to model configuration YAML file')
    parser.add_argument('--output', type=str, required=True, 
                        help='Path to save anomaly scores')
    args = parser.parse_args()

    detect(args.config, args.dataset, args.model, args.output)