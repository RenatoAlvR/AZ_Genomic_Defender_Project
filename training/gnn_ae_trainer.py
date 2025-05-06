import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
# Ensure GNNAutoencoder is importable, adjust path if needed
# from ..models.gnn_ae_model import GNNAutoencoder
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import logging
import os # Import os for path manipulation

# Get logger for this module
logger = logging.getLogger(__name__) # Use module's name for logger

class GNNAETrainer:
    """
    Trainer for the GNN-Autoencoder model.

    Handles training loops, validation, early stopping, and saving the final model.

    Args:
        config: Dictionary containing overall configuration.
        model: Initialized GNNAutoencoder instance.
        output_dir: Base directory for saving outputs (e.g., 'training').
    """
    def __init__(self, config: Dict[str, Any], model, output_dir: str = 'training'):
        # Ensure model is the GNNAutoencoder class (or duck-typed equivalent)
        # assert isinstance(model, GNNAutoencoder), "Model must be an instance of GNNAutoencoder"

        self.config = config.get('GNN', {}) # Extract GNN-specific config, default to empty dict if not found
        if not self.config:
             logger.warning("GNN configuration not found in main config.")

        self.model = model
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

        # Training parameters from config with defaults
        self.epochs = self.config.get('epochs', 100) # Default epochs
        self.patience = self.config.get('patience', 20) # Default patience
        lr = self.config.get('learning_rate', 1e-3) # Default LR
        weight_decay = self.config.get('weight_decay', 1e-5) # Default weight decay

        # Check if required model loss weights exist
        if not hasattr(model, 'loss_weights') or not model.loss_weights:
             logger.error("Model instance does not have 'loss_weights' attribute or it's empty.")
             raise ValueError("Model loss_weights are required for training.")
        self.loss_weights = model.loss_weights # Use weights defined in the model

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Directories
        self.base_output_dir = Path(output_dir)
        self.checkpoint_dir = self.base_output_dir / 'checkpoints' # Subdir for temp checkpoints
        self.final_model_dir = self.base_output_dir / 'trained_models' # The required final dir
        self.final_model_path = self.final_model_dir / 'gnn_ae_model.pth' # Specific final model file

        # Ensure directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.final_model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"GNN-AE Trainer initialized. Device: {self.device}")
        logger.info(f"Checkpoints will be saved in: {self.checkpoint_dir}")
        logger.info(f"Final model will be saved to: {self.final_model_path}")


    def train(self, train_data: Data, val_data: Optional[Data] = None):
        """
        Full training loop with validation and early stopping.

        Args:
            train_data: PyG Data object for training (assumed single graph).
            val_data: Optional PyG Data object for validation (assumed single graph).
        """
        best_val_loss = np.inf
        epochs_no_improve = 0
        best_epoch = 0

        # Create DataLoaders (handles full-batch case correctly)
        train_loader = self._create_dataloader(train_data)
        val_loader = self._create_dataloader(val_data) if val_data else None

        logger.info("Starting GNN-AE training...")
        for epoch in range(1, self.epochs + 1):
            # Training epoch
            train_loss_dict = self._run_epoch(train_loader, is_training=True)
            avg_train_loss = train_loss_dict['total_loss']

            # Validation epoch
            val_loss_str = ""
            current_loss_for_stopping = avg_train_loss # Use train loss if no validation
            if val_loader:
                val_loss_dict = self._run_epoch(val_loader, is_training=False)
                avg_val_loss = val_loss_dict['total_loss']
                current_loss_for_stopping = avg_val_loss # Use val loss if available
                val_loss_str = f" | Val Loss {avg_val_loss:.4f}"
                # Log detailed validation losses
                val_detail_str = ", ".join([f"Val_{k}: {v:.4f}" for k, v in val_loss_dict.items() if k != 'total_loss'])
                if val_detail_str: val_loss_str += f" ({val_detail_str})"


            # Log epoch results
            train_detail_str = ", ".join([f"Train_{k}: {v:.4f}" for k, v in train_loss_dict.items() if k != 'total_loss'])
            logger.info(f"Epoch {epoch}/{self.epochs}: Train Loss {avg_train_loss:.4f} ({train_detail_str}){val_loss_str}")

            # Checkpoint saving and Early stopping
            if current_loss_for_stopping < best_val_loss:
                best_val_loss = current_loss_for_stopping
                epochs_no_improve = 0
                best_epoch = epoch
                self._save_checkpoint(epoch, 'best_gnnae.pt')
                logger.info(f"New best model saved at epoch {epoch} with loss {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} after {self.patience} epochs without improvement.")
                    break

            # Optionally save a 'latest' checkpoint periodically or always
            # self._save_checkpoint(epoch, 'latest_gnnae.pt')

        logger.info(f"Training finished. Best epoch: {best_epoch}, Best validation loss: {best_val_loss:.4f}")

        # Load the best performing model state from checkpoint
        try:
            self._load_checkpoint('best_gnnae.pt')
            logger.info("Loaded best model state from checkpoint.")

            # Save the final best model to the required directory
            self.model.save(self.final_model_path) # Use the model's own save method
            logger.info(f"Final best GNN-AE model saved successfully to {self.final_model_path}")

        except FileNotFoundError:
             logger.error(f"Failed to load best checkpoint 'best_gnnae.pt'. Final model not saved.")
        except Exception as e:
             logger.error(f"An error occurred during final model saving: {e}", exc_info=True)


    def _calculate_combined_loss(self, outputs: Dict[str, torch.Tensor], batch: Data) -> Dict[str, torch.Tensor]:
        """Calculates individual and combined weighted loss components."""
        losses = {}

        # Reconstruction Loss (always present)
        losses['recon'] = F.mse_loss(outputs['recon'], batch.x)

        # Supervised Losses - Use BCEWithLogitsLoss for raw logits output
        if 'synth' in outputs and hasattr(batch, 'synth_labels'):
            # Assuming SyntheticHead outputs sigmoid probabilities (as per original GNN code)
            losses['synth'] = F.binary_cross_entropy(outputs['synth'].squeeze(-1), batch.synth_labels.float())
        else:
             losses['synth'] = torch.tensor(0.0, device=self.device) # Placeholder if not active or labels missing

        if 'flip' in outputs and hasattr(batch, 'flip_labels'):
            # Assuming LabelFlipHead outputs raw logits (as per original GNN code)
             losses['flip'] = F.binary_cross_entropy_with_logits(outputs['flip'].squeeze(-1), batch.flip_labels.float())
        else:
             losses['flip'] = torch.tensor(0.0, device=self.device)

        if 'noise' in outputs and hasattr(batch, 'noise_labels'):
             # Assuming NoiseHead outputs sigmoid probabilities (as per original GNN code)
             # Need to handle potential shape mismatch if noise output is feature-wise
             noise_pred = outputs['noise']
             if noise_pred.dim() > 1 and noise_pred.shape[1] > 1:
                 # Example: if predicting per-feature noise prob, maybe average or use appropriate target shape
                 noise_pred = noise_pred.mean(dim=1) # Or adapt based on label format
             losses['noise'] = F.binary_cross_entropy(noise_pred.squeeze(-1), batch.noise_labels.float())
        else:
             losses['noise'] = torch.tensor(0.0, device=self.device)

        # Calculate total weighted loss
        total_loss = torch.tensor(0.0, device=self.device)
        active_losses_found = False
        for loss_name, loss_value in losses.items():
             weight = self.loss_weights.get(loss_name, 0) # Get weight from model's config
             if weight > 0 and not torch.isnan(loss_value): # Only add if weight > 0 and loss is valid
                 total_loss += weight * loss_value
                 active_losses_found = True

        if not active_losses_found:
             logger.warning("No active loss components found with weight > 0. Total loss is 0.")

        losses['total_loss'] = total_loss
        return losses


    def _run_epoch(self, data_loader: DataLoader, is_training: bool) -> Dict[str, float]:
        """Runs a single epoch of training or validation."""
        if is_training:
            self.model.train()
            context = torch.enable_grad() # Enable gradients for training
            mode_desc = "Training"
        else:
            self.model.eval()
            context = torch.no_grad() # Disable gradients for validation
            mode_desc = "Validating"

        total_losses_accum = { 'recon': 0.0, 'synth': 0.0, 'flip': 0.0, 'noise': 0.0, 'total_loss': 0.0 }
        num_batches = 0

        with context:
            for batch in tqdm(data_loader, desc=f"{mode_desc} Epoch", leave=False):
                batch = batch.to(self.device)
                if is_training:
                    self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch.x, batch.edge_index)

                # Calculate losses (using the corrected combined loss function)
                # Ensure batch has necessary label attributes for the active heads
                loss_dict = self._calculate_combined_loss(outputs, batch)
                epoch_loss = loss_dict['total_loss']

                if is_training:
                    if torch.isnan(epoch_loss):
                        logger.warning(f"NaN loss detected during training. Skipping batch. Loss components: {loss_dict}")
                        continue # Skip backprop if loss is NaN
                    epoch_loss.backward()
                    # Optional: Gradient clipping
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                # Accumulate losses for reporting average
                for k, v in loss_dict.items():
                     if not torch.isnan(v): # Don't accumulate NaN values
                         total_losses_accum[k] += v.item()
                num_batches += 1

        # Calculate average losses for the epoch
        avg_losses = {k: v / num_batches if num_batches > 0 else 0.0 for k, v in total_losses_accum.items()}
        return avg_losses


    def _create_dataloader(self, data: Optional[Data]) -> Optional[DataLoader]:
        """Create DataLoader for a single PyG Data object (full-batch)."""
        if data is None:
            return None

        # Note: shuffle=True has no effect for a dataset of size 1
        # Batch size from config, assuming full-batch if not specified or 1.
        batch_size = self.config.get('batch_size', 1)
        if batch_size != 1:
            logger.warning(f"Batch size is {batch_size} but input is a single Data object. Effective batch size is 1 (full graph).")
            batch_size = 1 # Force batch size 1 for single Data object

        return DataLoader(
            [data], # DataLoader expects a dataset (list of Data objects)
            batch_size=batch_size, # Effectively always 1 for this setup
            shuffle=False # Shuffle is meaningless here
        )

    def _save_checkpoint(self, epoch: int, filename: str):
        """Save model and optimizer state to the checkpoint directory."""
        save_path = self.checkpoint_dir / filename
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.model.config, # Save model's internal config
            'loss_weights': self.loss_weights # Save weights used during training
        }
        try:
            torch.save(checkpoint, save_path)
            # logger.debug(f"Checkpoint saved to {save_path}") # Use debug level for frequent saves
        except Exception as e:
             logger.error(f"Failed to save checkpoint {save_path}: {e}")


    def _load_checkpoint(self, filename: str):
        """Load model and optimizer state from the checkpoint directory."""
        load_path = self.checkpoint_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded checkpoint from {load_path} (Epoch {checkpoint.get('epoch', 'N/A')})")
        except Exception as e:
             logger.error(f"Failed to load checkpoint {load_path}: {e}", exc_info=True)
             raise # Re-raise error after logging