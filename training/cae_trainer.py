# cae_trainer.py
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import logging
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)

class CAETrainer:
    """
    Trainer for a Contrastive Autoencoder (CAE) or standard Autoencoder model.
    Focuses on reconstruction and optionally a supervised head for synthetic cell detection.

    Args:
        main_config: Dictionary containing the overall configuration.
        model: Initialized CAE/AE model instance.
        output_dir: Base directory for saving outputs (e.g., 'training').
    """
    def __init__(self, main_config: Dict[str, Any], model, output_dir: str = 'training'):
        self.main_config = main_config
        self.cae_config = main_config.get('CAE', {}) # Extract CAE-specific config
        if not self.cae_config:
            logger.warning("CAE configuration section not found in main config. Using defaults.")

        self.model = model
        self.device = torch.device(
            self.cae_config.get('device',
                                main_config.get('global_model_config', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        )
        self.model.to(self.device)

        # Training parameters
        self.epochs = self.cae_config.get('epochs', 50)
        self.patience = self.cae_config.get('patience', 10)
        lr = self.cae_config.get('learning_rate', 1e-3)
        weight_decay = self.cae_config.get('weight_decay', 1e-5)
        self.loader_batch_size = self.cae_config.get('batch_size', 128)

        # Loss weights from model's own config or defaults for CAE
        # The model's internal config should have the definitive loss_weights
        if hasattr(model, 'config') and 'loss_weights' in model.config:
            self.loss_weights = model.config['loss_weights']
        elif 'loss_weights' in self.cae_config: # Fallback to trainer's CAE config
            self.loss_weights = self.cae_config['loss_weights']
        else:
            logger.warning("Loss weights not found in model config or CAE trainer config. Defaulting for AE.")
            self.loss_weights = {'recon': 1.0, 'synth': 0.0} # Default: only reconstruction

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Mixed Precision Scaler
        self.use_amp = self.cae_config.get('use_mixed_precision',
                                           main_config.get('global_model_config', {}).get('use_mixed_precision', True)) \
                       and self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            logger.info("Mixed precision training (AMP) is ENABLED for CAE on GPU.")
        elif self.device.type == 'cuda':
            logger.info("Mixed precision training (AMP) is DISABLED for CAE on GPU.")

        # Directories
        self.base_output_dir = Path(output_dir)
        self.checkpoint_dir = self.base_output_dir / 'checkpoints'
        self.final_model_dir = self.base_output_dir / 'trained_models'
        # Use model class name for the saved file to be generic
        self.final_model_path = self.final_model_dir / f"{self.model.__class__.__name__.lower()}_model.pth"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.final_model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"{self.model.__class__.__name__} Trainer initialized. Device: {self.device}, Output Path: {self.final_model_path}")
        logger.info(f"Loss weights being used: {self.loss_weights}")


    def train(self,
              train_feature_matrix: torch.Tensor,
              val_feature_matrix: Optional[torch.Tensor] = None,
              train_labels_synth: Optional[torch.Tensor] = None, # For supervised SynCell head
              val_labels_synth: Optional[torch.Tensor] = None):   # For supervised SynCell head
        best_val_loss = np.inf
        epochs_no_improve = 0
        best_epoch = 0

        train_loader = self._create_dataloader(train_feature_matrix, train_labels_synth, is_training=True)
        val_loader = self._create_dataloader(val_feature_matrix, val_labels_synth, is_training=False) if val_feature_matrix is not None else None

        if not train_loader:
            logger.error("Failed to create training DataLoader for CAE. Aborting training.")
            return

        logger.info(f"Starting {self.model.__class__.__name__} training for {self.epochs} epochs...")
        for epoch in range(1, self.epochs + 1):
            train_loss_dict = self._run_epoch(train_loader, is_training=True, epoch_num=epoch)
            avg_train_loss = train_loss_dict['total_loss']

            val_loss_str = ""
            current_loss_for_stopping = avg_train_loss
            if val_loader:
                val_loss_dict = self._run_epoch(val_loader, is_training=False, epoch_num=epoch)
                avg_val_loss = val_loss_dict['total_loss']
                current_loss_for_stopping = avg_val_loss
                val_loss_str = f" | Val Loss {avg_val_loss:.4f}"
                val_detail_str = ", ".join([f"Val_{k}: {v:.4f}" for k, v in val_loss_dict.items() if k != 'total_loss'])
                if val_detail_str: val_loss_str += f" ({val_detail_str})"
            
            train_detail_str = ", ".join([f"Train_{k}: {v:.4f}" for k, v in train_loss_dict.items() if k != 'total_loss'])
            logger.info(f"Epoch {epoch}/{self.epochs}: Train Loss {avg_train_loss:.4f} ({train_detail_str}){val_loss_str}")

            if current_loss_for_stopping < best_val_loss:
                best_val_loss = current_loss_for_stopping
                epochs_no_improve = 0
                best_epoch = epoch
                self._save_checkpoint(epoch, f'best_{self.model.__class__.__name__.lower()}.pt')
                logger.info(f"New best CAE model saved at epoch {epoch} with loss {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"CAE early stopping at epoch {epoch} after {self.patience} epochs.")
                    break
        
        logger.info(f"CAE training finished. Best epoch: {best_epoch}, Best loss: {best_val_loss:.4f}")

        try:
            self._load_checkpoint(f'best_{self.model.__class__.__name__.lower()}.pt')
            logger.info("Loaded best CAE model state from checkpoint.")
            self.model.save(self.final_model_path) # Use the model's own save method
            logger.info(f"Final best CAE model saved successfully to {self.final_model_path}")
        except FileNotFoundError:
            logger.error(f"Failed to load best CAE checkpoint. Final model NOT saved to {self.final_model_path}")
        except Exception as e:
            logger.error(f"Error during final CAE model saving: {e}", exc_info=True)


    def _calculate_combined_loss(self,
                                 outputs: Dict[str, torch.Tensor],
                                 target_features: torch.Tensor,
                                 target_labels_synth: Optional[torch.Tensor] = None
                                 ) -> Dict[str, torch.Tensor]:
        losses = {}
        current_batch_size = target_features.shape[0]

        # 1. Reconstruction Loss (Standard for AE)
        recon_loss_weight = self.loss_weights.get('recon', 1.0) # Default to 1.0 if not specified
        if recon_loss_weight > 0 and 'recon' in outputs:
            losses['recon'] = F.mse_loss(outputs['recon'], target_features)
        else:
            losses['recon'] = torch.tensor(0.0, device=self.device)

        # 2. Supervised Synthetic Cell Detection Loss (Optional)
        # This assumes the model's forward pass returns 'synth' if a head is active
        synth_loss_weight = self.loss_weights.get('synth', 0.0) # Default to 0.0
        if synth_loss_weight > 0 and 'synth' in outputs and target_labels_synth is not None:
            # Ensure labels have the correct shape (Batch, 1) or (Batch,)
            synth_pred = outputs['synth'].squeeze()
            synth_labels = target_labels_synth.squeeze().float()
            if synth_pred.shape == synth_labels.shape:
                 losses['synth'] = F.binary_cross_entropy_with_logits(synth_pred, synth_labels)
            else:
                logger.warning(f"Shape mismatch for synth loss: pred {synth_pred.shape}, labels {synth_labels.shape}. Skipping.")
                losses['synth'] = torch.tensor(0.0, device=self.device)
        else:
            losses['synth'] = torch.tensor(0.0, device=self.device)
        
        # 3. Contrastive Loss (If this were a true Contrastive AE)
        # contrastive_loss_weight = self.loss_weights.get('contrastive', 0.0)
        # if contrastive_loss_weight > 0 and 'projection1' in outputs and 'projection2' in outputs:
        #     # Example: losses['contrastive'] = self.model.contrastive_loss_fn(outputs['projection1'], outputs['projection2'])
        #     pass # Placeholder for actual contrastive loss calculation

        # Calculate total weighted loss
        total_loss = torch.tensor(0.0, device=self.device)
        active_losses_found = False
        for loss_name, loss_value in losses.items():
            weight = self.loss_weights.get(loss_name, 0.0)
            if weight > 0 and not torch.isnan(loss_value) and loss_value.numel() > 0:
                total_loss += weight * loss_value
                active_losses_found = True
        
        if not active_losses_found and sum(w for w in self.loss_weights.values() if isinstance(w, (float, int)) and w > 0) > 0:
             logger.debug("No active loss components contributed to total_loss for CAE, or all relevant weights are zero.")

        losses['total_loss'] = total_loss
        return losses

    def _run_epoch(self, data_loader: DataLoader, is_training: bool, epoch_num: int) -> Dict[str, float]:
        self.model.train(is_training)
        mode_desc = "Training" if is_training else "Validating"

        # Initialize accumulators for all potential loss components
        total_losses_accum = { 'recon': 0.0, 'synth': 0.0, 'contrastive':0.0, 'total_loss': 0.0 }
        num_samples_processed = 0

        pbar_desc = f"CAE {mode_desc} Epoch {epoch_num}"
        pbar = tqdm(data_loader, desc=pbar_desc, leave=False)

        for batch_idx, batch_data in enumerate(pbar):
            if len(batch_data) == 2: # Features and labels
                features, labels_synth = batch_data
                features = features.to(self.device)
                labels_synth = labels_synth.to(self.device)
            elif len(batch_data) == 1: # Only features
                features = batch_data[0].to(self.device)
                labels_synth = None
            else:
                logger.warning(f"Unexpected batch data format with length {len(batch_data)}. Skipping batch.")
                continue
            
            current_batch_size = features.shape[0]

            if is_training:
                self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(features) # Model's forward pass
                loss_dict = self._calculate_combined_loss(outputs, features, labels_synth)
                current_loss = loss_dict['total_loss']
            
            if is_training:
                if torch.isnan(current_loss) or current_loss.numel() == 0:
                    logger.warning(f"NaN or empty loss detected during CAE training. Skipping batch. Loss: {current_loss}")
                    continue
                self.scaler.scale(current_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            for k, v_tensor in loss_dict.items(): # v_tensor is a torch.Tensor
                 if not torch.isnan(v_tensor) and v_tensor.numel() > 0:
                     # Ensure k exists in accumulator, if not, it's an unexpected loss component
                     if k in total_losses_accum:
                         total_losses_accum[k] += v_tensor.item() * current_batch_size # Weight by batch size for averaging
                     else:
                         logger.warning(f"Unexpected loss component '{k}' from _calculate_combined_loss for CAE.")
            num_samples_processed += current_batch_size
            
            if num_samples_processed > 0:
                pbar.set_postfix({k: f"{v / num_samples_processed:.4f}" for k,v in total_losses_accum.items() if k in losses and num_samples_processed > 0})


        avg_losses = {k: v / num_samples_processed if num_samples_processed > 0 else 0.0 for k, v in total_losses_accum.items()}
        return avg_losses

    def _create_dataloader(self,
                           feature_matrix: Optional[torch.Tensor],
                           labels_synth: Optional[torch.Tensor] = None,
                           is_training: bool = True) -> Optional[DataLoader]:
        if feature_matrix is None:
            return None

        if labels_synth is not None:
            if len(feature_matrix) != len(labels_synth):
                logger.error(f"Feature matrix ({len(feature_matrix)}) and synth labels ({len(labels_synth)}) have different lengths. Cannot create DataLoader.")
                return None
            dataset = TensorDataset(feature_matrix, labels_synth.unsqueeze(-1) if labels_synth.ndim == 1 else labels_synth)
        else:
            dataset = TensorDataset(feature_matrix)

        num_workers_cfg = self.cae_config.get('num_workers', self.main_config.get('global_model_config', {}).get('num_workers', 0))

        return DataLoader(
            dataset,
            batch_size=self.loader_batch_size,
            shuffle=is_training,
            num_workers=num_workers_cfg,
            pin_memory=self.device.type == 'cuda' # Pin memory if on CUDA
        )

    def _save_checkpoint(self, epoch: int, filename: str):
        save_path = self.checkpoint_dir / filename
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'model_config': self.model.config if hasattr(self.model, 'config') else self.cae_config, # Prefer model's own config
            'loss_weights': self.loss_weights
        }
        try:
            torch.save(checkpoint, save_path)
        except Exception as e:
            logger.error(f"Failed to save CAE checkpoint {save_path}: {e}")

    def _load_checkpoint(self, filename: str):
        load_path = self.checkpoint_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"CAE Checkpoint file not found: {load_path}")
        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.use_amp and checkpoint.get('scaler_state_dict') is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info(f"Loaded CAE checkpoint from {load_path} (Epoch {checkpoint.get('epoch', 'N/A')})")
        except Exception as e:
            logger.error(f"Failed to load CAE checkpoint {load_path}: {e}", exc_info=True)
            raise