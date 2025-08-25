
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Union
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
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
    
    # Handle data based on model type
    if model_name == 'gnn_ae':
        print(f"Entered GNN if of trainer file")
        # GNN-AE expects a Data object with x, edge_index
        if not isinstance(data, Data):
            raise ValueError(f"Expected torch_geometric.data.Data for gnn_ae, got {type(data)}")
        # Create a DataLoader for the single Data object (batch_size=1 for GNN)
        data_loader = DataLoader([data], batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    else:
        # Other models (e.g., CAE, VAE, DDPM) expect a DataLoader
        if not isinstance(data, DataLoader):
            raise ValueError(f"Expected DataLoader for {model_name}, got {type(data)}")
        data_loader = data
    
    #Verify data loader output
    batch_example = next(iter(data_loader))
    print(f"DataLoader batch example: {batch_example}, type: {type(batch_example)}")
    
    #Initialize model
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
