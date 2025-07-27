import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import numpy as np
from pathlib import Path
from preprocessing.preprocess_train import preprocess_train
from models.CAE import ContrastiveAutoencoder
from models.vae_model import VariationalAutoencoder
from models.gnn_model import GNNAutoencoder
from models.ddpm_model import DenoisingDiffusionPM

def train(config_path: str, dataset_path: str, model_name: str, output_path: str, incremental: bool = False) -> None:
    """Train the specified model on the given dataset with batch and incremental training support."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set batch size and epochs
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 100)
    learning_rate = config.get('learning_rate', 0.0001 if incremental else 0.001)

    # Load preprocessed data
    data = preprocess_train(dataset_path, config)
    
    # Initialize model
    model_name = model_name.lower()
    if model_name == 'cae':
        model = ContrastiveAutoencoder(config) if not incremental else \
                ContrastiveAutoencoder.load(f'weights/cae_synthetic_cells.pt', device=config['device'])
        data_loader = DataLoader(torch.tensor(data, dtype=torch.float32), batch_size=batch_size, shuffle=True)
    elif model_name == 'vae':
        model = VariationalAutoencoder(config) if not incremental else \
                VariationalAutoencoder.load(f'weights/vae_gene_scaling.pt', device=config['device'])
        data_loader = DataLoader(torch.tensor(data, dtype=torch.float32), batch_size=batch_size, shuffle=True)
    elif model_name == 'gnn_ae':
        model = GNNAutoencoder(config) if not incremental else \
                GNNAutoencoder.load(f'weights/gnn_ae_label_flips.pt', device=config['device'])
        data_loader = data  # GNN-AE uses Data object directly (no batching for simplicity)
    elif model_name == 'ddpm':
        model = DenoisingDiffusionPM(config) if not incremental else \
                DenoisingDiffusionPM.load(f'weights/ddpm_noise.pt', device=config['device'])
        data_loader = DataLoader(torch.tensor(data, dtype=torch.float32), batch_size=batch_size, shuffle=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose 'cae', 'vae', 'gnn_ae', or 'ddpm'.")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    print(f"Training {'incrementally' if incremental else 'from scratch'} with {model_name.upper()} on {dataset_path}")
    if model_name == 'gnn_ae':
        model.fit(data_loader, optimizer, epochs=epochs, patience=config.get('patience', 20))
    else:
        model.fit(data_loader, optimizer, epochs=epochs, patience=config.get('patience', 20))

    # Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Model saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model for scRNA-seq anomaly detection.')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['cae', 'vae', 'gnn_ae', 'ddpm'], 
                        help='Model to train')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Path to input dataset (CSV or preprocessed)')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to model configuration YAML file')
    parser.add_argument('--output', type=str, required=True, 
                        help='Path to save trained model weights')
    parser.add_argument('--incremental', action='store_true', 
                        help='Enable incremental training (load pre-trained weights)')
    args = parser.parse_args()

    train(args.config, args.dataset, args.model, args.output, args.incremental)