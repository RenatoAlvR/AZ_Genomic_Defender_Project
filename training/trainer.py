# trainer.py (General Trainer)
import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Model Imports (adjust paths as per your project structure)
from ..models.gnn_ae_model import GNNAutoencoder
from ..models.vae_model import VariationalAutoencoder
from ..models.contrastive_ae_model import ContrastiveAutoencoder # Assuming you have this
from ..models.ddpm_model import DenoisingDiffusionPM       # Assuming you have this

# Specific Trainer Imports
from .gnn_ae_trainer import GNNAETrainer
# Placeholder for other specific trainers - you'll need to create these
# from .vae_trainer import VAETrainer
# from .cae_trainer import CAETrainer
# from .ddpm_trainer import DDPMTrainer

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config: Dict[str, Any], output_dir_base: str = "training"):
        self.main_config = config
        self.output_dir = Path(output_dir_base) # e.g., "training"
        self.models_storage_dir = self.output_dir / "trained_models" # Final models go here
        self.models_storage_dir.mkdir(parents=True, exist_ok=True)

        # Mapping model names (keys from config) to their classes and specific trainers
        self.model_registry = {
            'GNN': { # Key used in config for GNN settings
                'model_class': GNNAutoencoder,
                'trainer_class': GNNAETrainer,
                'data_key_train': 'gnn_graph_data_train', # Expected key in processed_data for GNN train data
                'data_key_val': 'gnn_graph_data_val'    # Expected key in processed_data for GNN val data
            },
            'VAE': {
                'model_class': VariationalAutoencoder,
                'trainer_class': None, # Replace with VAETrainer once created
                'data_key_train': 'feature_matrix_train', # Example key
                'data_key_val': 'feature_matrix_val'
            },
            'CAE': { # Contrastive AE
                'model_class': ContrastiveAutoencoder,
                'trainer_class': None, # Replace with CAETrainer once created
                'data_key_train': 'feature_matrix_train', # Example key
                'data_key_val': 'feature_matrix_val'
            },
            'DDPM': {
                'model_class': DenoisingDiffusionPM,
                'trainer_class': None, # Replace with DDPMTrainer once created
                'data_key_train': 'feature_matrix_train', # Example key for DDPM (needs clean data)
                'data_key_val': 'feature_matrix_val'      # Needs clean data for validation loss
            }
            # Add other models (CAE, DDPM) here similarly once their trainers exist
        }
        logger.info(f"ModelTrainer initialized. Trained models will be stored in: {self.models_storage_dir}")
        logger.info("Note: Models will be trained sequentially due to typical resource constraints.")

    def _load_or_init_model(self, model_config_key: str, model_class: type):
        """
        Loads a model if it exists, otherwise initializes a new one.
        model_config_key: The key in the main_config for this model's parameters (e.g., 'GNN', 'VAE').
        """
        model_filename = f"{model_class.__name__.lower()}_model.pth"
        model_path = self.models_storage_dir / model_filename

        model_params = self.main_config.get(model_config_key, {}).get('params', {})
        if not model_params:
             logger.warning(f"No 'params' found in config for model key '{model_config_key}'. Initializing with default model params.")


        if model_path.exists() and not self.force_retrain_flags.get(model_config_key, False):
            logger.info(f"Loading existing {model_class.__name__} model from {model_path}")
            try:
                # The model's own .load() method should handle config loading internally if needed
                # Pass device from config if model.load supports it.
                device_to_load = self.main_config.get(model_config_key, {}).get('device', 'cpu')
                return model_class.load(model_path, device=device_to_load)
            except Exception as e:
                logger.error(f"Failed to load {model_class.__name__} from {model_path}: {e}. Initializing new model.", exc_info=True)
                # Fall through to initialize new model

        logger.info(f"Initializing new {model_class.__name__} model.")
        # Pass the specific model's config section for its initialization
        return model_class(config=self.main_config.get(model_config_key, {}))


    def train_all_models(self, processed_data: Dict[str, Any], force_retrain_flags: Optional[Dict[str, bool]] = None):
        """
        Trains all registered models sequentially.

        Args:
            processed_data: A dictionary where keys map to specific data structures
                            needed by different models (e.g., PyG Data for GNN, tensors for others).
            force_retrain_flags: A dictionary mapping model config keys (e.g., 'GNN') to boolean
                                 indicating whether to force retraining for that model.
        """
        self.force_retrain_flags = force_retrain_flags if force_retrain_flags is not None else {}

        for model_config_key, details in self.model_registry.items():
            model_class = details['model_class']
            TrainerClass = details['trainer_class']
            data_key_train = details['data_key_train']
            data_key_val = details['data_key_val']

            if not TrainerClass:
                logger.warning(f"No specific trainer class defined for {model_config_key}. Skipping training.")
                continue

            logger.info(f"--- Preparing to train {model_config_key} ({model_class.__name__}) ---")

            # 1. Load or initialize the model instance
            model_instance = self._load_or_init_model(model_config_key, model_class)
            if not model_instance:
                 logger.error(f"Failed to load or initialize model {model_config_key}. Skipping.")
                 continue

            # 2. Prepare data for this specific model
            # Ensure the preprocessor creates these specific keys in `processed_data`
            current_train_data = processed_data.get(data_key_train)
            current_val_data = processed_data.get(data_key_val) # Can be None

            if current_train_data is None:
                logger.error(f"Training data ('{data_key_train}') not found in processed_data for {model_config_key}. Skipping.")
                continue
            
            # 3. Instantiate its specific trainer
            # The specific trainer will pull its own sub-config from main_config
            specific_trainer = TrainerClass(
                main_config=self.main_config,
                model=model_instance,
                output_dir=str(self.output_dir) # Pass base "training" dir
            )

            # 4. Run the training
            try:
                logger.info(f"Starting training for {model_config_key}...")
                specific_trainer.train(current_train_data, current_val_data)
                logger.info(f"Training completed for {model_config_key}.")
            except Exception as e:
                logger.error(f"Error during training of {model_config_key}: {e}", exc_info=True)
            
            logger.info(f"--- Finished training attempt for {model_config_key} ---")
        
        logger.info("All model training processes concluded.")

    def train_incrementally(self, processed_data: Dict, force_retrain: bool = False):
        """
        Wrapper for training all models. 'Incremental' aspect is handled by
        loading existing models if `force_retrain` is False for that model.
        Actual fine-tuning logic would need more specific handling if desired beyond this.
        """
        logger.info("Starting incremental training process for all models...")
        
        # Create a dictionary for force_retrain_flags based on the single boolean
        # This means 'force_retrain' applies to all models if True.
        # For per-model force_retrain, main.py would need to pass a dict.
        force_retrain_all = {key: force_retrain for key in self.model_registry.keys()}

        self.train_all_models(processed_data, force_retrain_flags=force_retrain_all)