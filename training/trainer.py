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


def train(config_path: str, dataset_path: str, model_name: str, output_path: str,
          incremental: bool = False, base_weights: str = None) -> None:
    """Train a model for anomaly detection in scRNA-seq data.

    Args:
        config_path:   Path to YAML config file.
        dataset_path:  Path to dataset directory (10x format).
        model_name:    One of: cae, vae, gnn_ae, ddpm.
        output_path:   Where to save the final weights (.pt file).
        incremental:   If True, fine-tune from existing weights instead of
                       training from scratch.
        base_weights:  Path to the weights file to load when --incremental is
                       set. If None, falls back to output_path (original
                       behaviour). Use this to fine-tune from a master
                       checkpoint while saving to a new file, e.g.:
                           --incremental
                           --base_weights weights/vae_master_best.pt
                           --output       weights/vae_semisupervised.pt
    """

    # ── Load configuration ────────────────────────────────────────────────────
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logging.info(f"Loaded config: {config}")

    # Store checkpoint path so fit() can save best weights mid-training
    best_weights_path = str(
        Path(output_path).parent / f"{Path(output_path).stem}_best.pt"
    )
    config['checkpoint_path'] = best_weights_path

    # ── Preprocess data ───────────────────────────────────────────────────────
    logging.info(f"Preprocessing data from {dataset_path}")
    data = preprocess_train(dataset_path, config)

    # ── Build DataLoader ──────────────────────────────────────────────────────
    if model_name == 'gnn_ae':
        if not isinstance(data, Data):
            raise ValueError(
                f"Expected torch_geometric.data.Data for gnn_ae, got {type(data)}"
            )
        data_loader = DataLoader(
            [data], batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
        )
    else:
        if not isinstance(data, DataLoader):
            raise ValueError(
                f"Expected DataLoader for {model_name}, got {type(data)}"
            )
        data_loader = data

    # Sanity-check DataLoader output
    batch_example = next(iter(data_loader))
    if isinstance(batch_example, (tuple, list)):
        print(f"DataLoader batch shape: {batch_example[0].shape}, "
              f"type: {type(batch_example)}")
    elif hasattr(batch_example, 'x'):
        print(f"DataLoader batch shape: nodes={batch_example.x.shape}, "
              f"edges={batch_example.edge_index.shape}, type: {type(batch_example)}")
    else:
        print(f"DataLoader batch shape: {batch_example.shape}, "
              f"type: {type(batch_example)}")

    # ── Initialize model ──────────────────────────────────────────────────────
    logging.info(f"Initializing {model_name} model")

    if incremental:
        # Resolve which weights file to load from:
        #   1. --base_weights  (explicit source, recommended)
        #   2. --output        (legacy fallback — same file used for load + save)
        load_path = base_weights if base_weights else output_path

        if not Path(load_path).exists():
            raise FileNotFoundError(
                f"--incremental was set but weights file not found at '{load_path}'.\n"
                f"  If fine-tuning from a master checkpoint, pass:\n"
                f"    --base_weights weights/{model_name}_master_best.pt\n"
                f"    --output       weights/{model_name}_semisupervised.pt\n"
                f"  The base_weights file is the SOURCE and output is the DESTINATION.\n"
                f"  They can be different files — the master checkpoint is never overwritten."
            )

        logging.info(f"Loading pre-trained weights from '{load_path}' "
                     f"(will save to '{output_path}')")
        model_map = {
            'cae':    lambda: ContrastiveAutoencoder.load(load_path,
                                                          device=config['device']),
            'vae':    lambda: VariationalAutoencoder.load(load_path,
                                                          device=config['device']),
            'gnn_ae': lambda: GNNAutoencoder.load(load_path,
                                                   device=config['device']),
            'ddpm':   lambda: DenoisingDiffusionPM.load(load_path,
                                                         device=config['device']),
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
    model.config['checkpoint_path'] = best_weights_path

    # ── Optimizer: AdamW ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 0.0003),
        weight_decay=1e-4
    )

    # ── LR Scheduler: cosine annealing ───────────────────────────────────────
    epochs = config.get('epochs', 200)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    mode_label = 'incrementally' if incremental else 'from scratch'
    if incremental and base_weights:
        mode_label = f"incrementally (base: {base_weights})"

    print(f"\nTraining {mode_label} "
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
    model.save(output_path)
    logging.info(f"Final model saved to {output_path}")
    logging.info(f"Best checkpoint saved to {best_weights_path}")
    print(f"\nDone. Final weights → {output_path}")
    print(f"      Best weights  → {best_weights_path}")