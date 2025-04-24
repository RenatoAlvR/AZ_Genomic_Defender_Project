import logging
from pathlib import Path
from typing import Dict, Any
import pickle
import torch
import numpy as np
import yaml

from fusion.decision_fusion import DecisionFusion
from models.eif_model import ExtendedIsolationForest
from models.vae_model import VariationalAutoencoder
from models.gnn_model import GNNAutoencoder
from models.contrastive_ae_model import ContrastiveAutoencoder
from models.ocsvm_model import OneClassSVM
from models.ddpm_model import DDPMDetector
from utils.logger import logger

class PoisonDetector:
    """
    Detection orchestrator that:
    1. Loads all trained models from disk
    2. Runs detection using each model
    3. Fuses scores for final decision
    
    Requires:
    - All models to be pre-trained and saved in training/trained_models/
    - Preprocessed input data (from GenomicDataPreprocessor)
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize detector with configuration.
        
        Args:
            config_path: Path to YAML config file
        """
        self.config = config.get('detection', {}) if config else {}
        self.models_dir = Path("training/trained_models")
        self._validate_model_dir()
        self.fusion = DecisionFusion(config=self.config)
        device_config = self.config.get('hardware', {}).get('device', 'auto')
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and device_config in ['auto', 'cuda'] else 'cpu'
        )
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        
    def _validate_model_dir(self) -> None:
        required_models = self.config.get('required_models', [
        'eif.pt', 'vae.pt', 'gnn.pt', 'contrastive_ae.pt', 'ddpm.pt'
        ])

        """Verify model directory exists with required models."""
        if not self.models_dir.exists():
            raise FileNotFoundError(
                f"Model directory {self.models_dir} not found. "
                "Please train models first using --train mode."
            )
    
        missing = [m for m in required_models if not (self.models_dir / m).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing trained models: {', '.join(missing)}. "
                "Please complete training first."
            )
    
    def _load_model(self, model_name: str) -> Any:
        """Load a single trained model."""
        model_file = self.models_dir / f"{model_name}.pt"
        
        model_classes = {
            'eif': ExtendedIsolationForest,
            'vae': VariationalAutoencoder,
            'gnn': GNNAutoencoder,
            'contrastive_ae': ContrastiveAutoencoder,
            'ddpm': DDPMDetector
        }
        
        if model_name == 'eif':
            with open(model_file, 'rb') as f:
                return pickle.load(f)
        else:
            return torch.load(model_file)
    
    def run(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute full detection pipeline.
        
        Args:
            processed_data: Output from GenomicDataPreprocessor containing:
                - pca_data: PCA-reduced features
                - cell_graph: PyG graph object (for GNN)
                
        Returns:
            Dictionary with:
            - individual_scores: Dict of per-model anomaly scores
            - final_score: Fused anomaly scores
        """
        scores = {}
        
        for model_name in ['eif', 'vae', 'gnn', 'contrastive_ae', 'ddpm']:
            logger.info(f"Running {model_name} detection...")
            model = self._load_model(model_name)
            
            if model_name == 'gnn':
                scores[model_name] = model.detect(processed_data['cell_graph'])
            else:
                scores[model_name] = model.detect(processed_data['pca_data'])
        
        return {
            'individual_scores': scores,
            'final_score': self.fusion.fuse(scores)
        }