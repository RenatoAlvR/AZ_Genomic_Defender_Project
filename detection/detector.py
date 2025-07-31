import torch
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Dict
from preprocessing.preprocess_detect import preprocess_detect
from models.cae_model import ContrastiveAutoencoder
from models.vae_model import VariationalAutoencoder
from models.gnn_model import GNNAutoencoder
from models.ddpm_model import DenoisingDiffusionPM

def detect(config_path: str, dataset_path: str, model_name: str, output_path: str, weights_path: str) -> None:
    """Detect anomalies in scRNA-seq data using a trained model."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded config: {config}")

    # Preprocess data
    logging.info(f"Preprocessing data from {dataset_path}")
    data = preprocess_detect(dataset_path, config)
    
    # Initialize model and load weights
    model_name = model_name.lower()
    if model_name == 'cae':
        model = ContrastiveAutoencoder.load(weights_path, device=config['device'])
    elif model_name == 'vae':
        model = VariationalAutoencoder.load(weights_path, device=config['device'])
    elif model_name == 'gnn_ae':
        model = GNNAutoencoder.load(weights_path, device=config['device'])
    elif model_name == 'ddpm':
        model = DenoisingDiffusionPM.load(weights_path, device=config['device'])
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose 'cae', 'vae', 'gnn_ae', or 'ddpm'.")
    
    # Detect anomalies
    logging.info(f"Detecting anomalies with {model_name}")
    if model_name == 'gnn_ae':
        anomaly_scores = model.detect(data)  # GNN-AE uses Data object
    else:
        anomaly_scores = model.detect(data)  # Other models use numpy array
    
    # Threshold for anomalies
    threshold = config.get('detection_threshold', 0.95)
    anomaly_indices = np.where(anomaly_scores >= np.quantile(anomaly_scores, threshold))[0]
    anomaly_percentage = (len(anomaly_indices) / len(anomaly_scores)) * 100
    
    logging.info(f"Detection completed, {anomaly_percentage:.2f}% of the dataset could be poisoned, "
                 f"suspected cells: {anomaly_indices.tolist()[:10]}{'...' if len(anomaly_indices) > 10 else ''}")
    print(f"Detection completed, {anomaly_percentage:.2f}% of the dataset could be poisoned, "
          f"suspected cells: {anomaly_indices.tolist()[:10]}{'...' if len(anomaly_indices) > 10 else ''}")
          
    # Save anomaly scores
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, anomaly_scores, delimiter=',')
    logging.info(f"Anomaly scores saved to {output_path}")


