import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Union
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import logging

from preprocessing.preprocess_train import preprocess_train
from models.cae_model import ContrastiveAutoencoder
from models.vae_model import VariationalAutoencoder
from models.gnn_model import GNNAutoencoder
from models.ddpm_model import DenoisingDiffusionPM

def train(config_path: str, dataset_path: str, model_name: str, output_path: str, incremental: bool = False) -> None:
    """Train a model for anomaly detection in scRNA-seq data."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded config: {config}")

    # Preprocess data
    logging.info(f"Preprocessing data from {dataset_path}")
    data = preprocess_train(dataset_path, config)
    
    # Convert to tensor for DataLoader
    if model_name == 'gnn_ae':
        # GNN-AE expects a Data object with x, edge_index
        x = torch.tensor(data.x, dtype=torch.float32)
        edge_index = data.edge_index
        dataset = TensorDataset(x)  # edge_index handled separately in GNN-AE
    else:
        # Other models expect a numpy array of shape (n_cells, input_dim)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        dataset = TensorDataset(data_tensor)
    
    # Create DataLoader with train_batch_size from config
    batch_size = config.get('train_batch_size', 32)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Initialize model
    logging.info(f"Initializing {model_name} model")
    if incremental and Path(output_path).exists():
        logging.info(f"Loading pre-trained weights from {output_path}")
        if model_name == 'cae':
            model = ContrastiveAutoencoder.load(output_path, device=config['device'])
        elif model_name == 'vae':
            model = VariationalAutoencoder.load(output_path, device=config['device'])
        elif model_name == 'gnn_ae':
            model = GNNAutoencoder.load(output_path, device=config['device'])
        elif model_name == 'ddpm':
            model = DenoisingDiffusionPM.load(output_path, device=config['device'])
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    else:
        if model_name == 'cae':
            model = ContrastiveAutoencoder(config)
        elif model_name == 'vae':
            model = VariationalAutoencoder(config)
        elif model_name == 'gnn_ae':
            model = GNNAutoencoder(config)
        elif model_name == 'ddpm':
            model = DenoisingDiffusionPM(config)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    
    # Train model
    print(f"Training {'incrementally' if incremental else 'from scratch'} with {model_name.upper()} on {dataset_path}")
    if model_name == 'gnn_ae':
        # Pass edge_index for GNN-AE
        model.fit(data_loader, optimizer, epochs=config.get('epochs', 100), patience=config.get('patience', 20), edge_index=edge_index)
    else:
        model.fit(data_loader, optimizer, epochs=config.get('epochs', 100), patience=config.get('patience', 20))
    
    # Save model
    model.save(output_path)
    logging.info(f"Model saved to {output_path}")