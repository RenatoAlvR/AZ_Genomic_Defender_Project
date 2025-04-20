import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from ..utils.logger import logger
from ..utils.metrics import AnomalyMetrics

class ModelTrainer:
    def __init__(self, config_path: str):
        self.models_dir = Path("training/trained_models")  # Fixed path
        self.models_dir.mkdir(parents=True, exist_ok=True)  # Auto-create
        self.config = load_config(config_path)
        
    def train_incrementally(self, processed_data: Dict, force_retrain: bool = False):
        """Models will be saved to training/trained_models/"""
        for model_name in ['eif', 'vae', 'gnn', 'contrastive_ae']:
            model = self._load_or_init_model(model_name)
            # ... training logic ...
            model.save(self.models_dir / f"{model_name}.pt")  # Saves to trained_models

    def _load_or_init_model(self, model_name: str):
        """Smart model loader"""
        model_file = self.models_dir / f"{model_name}.pt"
        model_class = {
            'eif': ExtendedIsolationForest,
            'vae': VariationalAutoencoder,
            'gnn': GNNAutoencoder,
            'contrastive_ae': ContrastiveAutoencoder
        }[model_name]

        if model_file.exists():
            logger.info(f"Loading existing {model_name} model")
            return model_class.load(model_file)
        else:
            logger.info(f"Initializing new {model_name} model")
            return model_class(**self.config['models'][model_name])