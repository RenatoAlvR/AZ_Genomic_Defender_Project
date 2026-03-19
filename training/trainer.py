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

    # ── Load configuration ────────────────────────────────────────────────────
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logging.info(f"Loaded config: {config}")

    # Store checkpoint path in config so fit() can save best weights mid-training
    best_weights_path = str(Path(output_path).parent / f"{Path(output_path).stem}_best.pt")
    config['checkpoint_path'] = best_weights_path

    # ── Preprocess data ───────────────────────────────────────────────────────
    logging.info(f"Preprocessing data from {dataset_path}")
    data = preprocess_train(dataset_path, config)

    # ── Build DataLoader ──────────────────────────────────────────────────────
    if model_name == 'gnn_ae':
        if not isinstance(data, Data):
            raise ValueError(f"Expected torch_geometric.data.Data for gnn_ae, got {type(data)}")
        # GNN-AE works on a single graph object
        data_loader = DataLoader([data], batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    else:
        if not isinstance(data, DataLoader):
            raise ValueError(f"Expected DataLoader for {model_name}, got {type(data)}")
        data_loader = data

    # Sanity-check DataLoader output
    batch_example = next(iter(data_loader))
    if isinstance(batch_example, (tuple, list)):
        print(f"DataLoader batch shape: {batch_example[0].shape}, type: {type(batch_example)}")
    else:
        print(f"DataLoader batch shape: {batch_example.shape}, type: {type(batch_example)}")

    # ── Initialize model ──────────────────────────────────────────────────────
    logging.info(f"Initializing {model_name} model")
    if incremental and Path(output_path).exists():
        logging.info(f"Loading pre-trained weights from {output_path} for incremental training")
        model_map = {
            'cae':    lambda: ContrastiveAutoencoder.load(output_path, device=config['device']),
            'vae':    lambda: VariationalAutoencoder.load(output_path, device=config['device']),
            'gnn_ae': lambda: GNNAutoencoder.load(output_path, device=config['device']),
            'ddpm':   lambda: DenoisingDiffusionPM.load(output_path, device=config['device']),
        }
    else:
        model_map = {
            'cae':    lambda: ContrastiveAutoencoder(config),
            'vae':    lambda: VariationalAutoencoder(config),
            'gnn_ae': lambda: GNNAutoencoder(config),
            'ddpm':   lambda: DenoisingDiffusionPM(config),
        }

    if model_name not in model_map:
        raise ValueError(f"Unsupported model: {model_name}")
    model = model_map[model_name]()

    # ── Optimizer: AdamW beats Adam for generalization ────────────────────────
    # weight_decay acts as L2 regularization, preventing the model from
    # memorizing training noise — critical for detecting subtle anomalies.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 0.0003),
        weight_decay=1e-4
    )

    # ── LR Scheduler: cosine annealing ───────────────────────────────────────
    # Smoothly decays LR to near-zero over training. Avoids oscillating
    # around the loss minimum in the final epochs, giving a sharper
    # anomaly score distribution (better separation between clean/poisoned).
    epochs = config.get('epochs', 200)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nTraining {'incrementally' if incremental else 'from scratch'} "
          f"| Model: {model_name.upper()} | Dataset: {dataset_path}")
    print(f"Optimizer: AdamW (lr={config.get('learning_rate', 0.0003)}, wd=1e-4) "
          f"| Scheduler: CosineAnnealingLR (T_max={epochs}, eta_min=1e-6)\n")

    model.fit(
        data_loader,
        optimizer,
        epochs=epochs,
        patience=config.get('patience', 20),
        scheduler=scheduler
    )

    # ── Save final model ──────────────────────────────────────────────────────
    # Best weights were already saved mid-training by fit() via checkpoint_path.
    # This saves the final-epoch weights as well for reference.
    model.save(output_path)
    logging.info(f"Final model saved to {output_path}")
    logging.info(f"Best checkpoint saved to {best_weights_path}")
    print(f"\nDone. Final weights → {output_path}")
    print(f"      Best weights  → {best_weights_path}")