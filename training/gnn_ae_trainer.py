# gnn_ae_trainer.py
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphDataLoader # Full graph loader
from torch_geometric.loader import NeighborLoader # Graph sampling loader
from tqdm import tqdm
import logging
import os
from torch.cuda.amp import autocast, GradScaler

# Get logger for this module
logger = logging.getLogger(__name__)

class GNNAETrainer:
    """
    Trainer for the GNN-Autoencoder model.

    Handles training loops, validation, early stopping, graph sampling for large graphs,
    mixed-precision training, and saving the final model.

    Args:
        config: Dictionary containing overall configuration.
        model: Initialized GNNAutoencoder instance.
        output_dir: Base directory for saving outputs (e.g., 'training').
    """
    def __init__(self, main_config: Dict[str, Any], model, output_dir: str = 'training'):
        self.main_config = main_config
        self.gnn_config = main_config.get('GNN', {})
        if not self.gnn_config:
            logger.warning("GNN configuration section not found in main config. Using defaults.")

        self.model = model
        self.device = torch.device(self.gnn_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

        # Training parameters
        self.epochs = self.gnn_config.get('epochs', 50) # Reduced default for potentially longer epochs with sampling
        self.patience = self.gnn_config.get('patience', 10)
        lr = self.gnn_config.get('learning_rate', 1e-3)
        weight_decay = self.gnn_config.get('weight_decay', 1e-5) # Make weight_decay configurable

        # Graph Sampling Config
        self.use_graph_sampling = self.gnn_config.get('use_graph_sampling', True) # Default to True for low-resource
        self.loader_batch_size = self.gnn_config.get('batch_size', 64) # For seed nodes if sampling, else ignored
        self.num_neighbors = self.gnn_config.get('num_neighbors', [10, 5]) # Fan-out for NeighborLoader

        if not hasattr(model, 'loss_weights') or not model.loss_weights:
            logger.error("Model instance does not have 'loss_weights' attribute or it's empty.")
            raise ValueError("Model loss_weights are required for training.")
        self.loss_weights = model.loss_weights

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Mixed Precision Scaler (only if CUDA is used and enabled)
        self.use_amp = self.gnn_config.get('use_mixed_precision', True) and self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            logger.info("Mixed precision training (AMP) is ENABLED for GNN-AE on GPU.")
        elif self.device.type == 'cuda':
             logger.info("Mixed precision training (AMP) is DISABLED for GNN-AE on GPU.")


        self.base_output_dir = Path(output_dir)
        self.checkpoint_dir = self.base_output_dir / 'checkpoints'
        self.final_model_dir = self.base_output_dir / 'trained_models'
        self.final_model_path = self.final_model_dir / f"{self.model.__class__.__name__.lower()}_model.pth" # Use model class name

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.final_model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"{self.model.__class__.__name__} Trainer initialized. Device: {self.device}, Output Path: {self.final_model_path}")

    def train(self, train_graph_data: Data, val_graph_data: Optional[Data] = None):
        best_val_loss = np.inf
        epochs_no_improve = 0
        best_epoch = 0

        # Create DataLoaders based on config (full graph or sampling)
        train_loader = self._create_dataloader(train_graph_data, is_training=True)
        val_loader = self._create_dataloader(val_graph_data, is_training=False) if val_graph_data else None

        if not train_loader:
            logger.error("Failed to create training DataLoader. Aborting GNN-AE training.")
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
                logger.info(f"New best GNN-AE model saved at epoch {epoch} with loss {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"GNN-AE early stopping at epoch {epoch} after {self.patience} epochs.")
                    break
        
        logger.info(f"GNN-AE training finished. Best epoch: {best_epoch}, Best loss: {best_val_loss:.4f}")

        try:
            self._load_checkpoint(f'best_{self.model.__class__.__name__.lower()}.pt')
            logger.info("Loaded best GNN-AE model state from checkpoint.")
            self.model.save(self.final_model_path)
            logger.info(f"Final best GNN-AE model saved successfully to {self.final_model_path}")
        except FileNotFoundError:
            logger.error(f"Failed to load best GNN-AE checkpoint. Final model NOT saved to {self.final_model_path}")
        except Exception as e:
            logger.error(f"Error during final GNN-AE model saving: {e}", exc_info=True)

    def _calculate_combined_loss(self, outputs: Dict[str, torch.Tensor], batch: Union[Data, Any]) -> Dict[str, torch.Tensor]:
        losses = {}
        
        # Determine target features (batch.x for full graph, batch.x[:batch.batch_size] for NeighborLoader)
        target_x = batch.x
        if hasattr(batch, 'batch_size') and self.use_graph_sampling : # NeighborLoader puts seed nodes first
             target_x = batch.x[:batch.batch_size]
             # Also, model output 'recon' might be for all nodes in the subgraph, need to slice it for loss.
             outputs['recon'] = outputs['recon'][:batch.batch_size]


        losses['recon'] = F.mse_loss(outputs['recon'], target_x)

        # Helper for supervised losses
        def get_supervised_loss(task_name: str, output_key: str, label_attr: str, use_logits: bool):
            if output_key in outputs and hasattr(batch, label_attr):
                pred = outputs[output_key]
                labels = getattr(batch, label_attr)
                
                # Slice predictions and labels if using NeighborLoader (only for seed nodes)
                if hasattr(batch, 'batch_size') and self.use_graph_sampling:
                    pred = pred[:batch.batch_size]
                    labels = labels[:batch.batch_size] # Assuming labels on Data obj are for all nodes

                if pred.dim() > 1 and pred.shape[1] > 1 and task_name == 'noise': # Specific for NoiseHead output
                    pred = pred.mean(dim=1)
                
                pred = pred.squeeze(-1) if pred.dim() > 1 and pred.shape[-1] == 1 else pred

                if labels.numel() == 0 or pred.numel() == 0 or labels.shape[0] != pred.shape[0]:
                    # logger.warning(f"Label/prediction shape mismatch or empty for {task_name}. Pred: {pred.shape}, Label: {labels.shape}")
                    return torch.tensor(0.0, device=self.device)

                if use_logits:
                    return F.binary_cross_entropy_with_logits(pred, labels.float())
                else:
                    return F.binary_cross_entropy(pred, labels.float())
            return torch.tensor(0.0, device=self.device)

        losses['synth'] = get_supervised_loss('synth', 'synth', 'synth_labels', use_logits=False) # SyntheticHead outputs sigmoid
        losses['flip']  = get_supervised_loss('flip',  'flip',  'flip_labels',  use_logits=True)  # LabelFlipHead outputs logits
        losses['noise'] = get_supervised_loss('noise', 'noise', 'noise_labels', use_logits=False) # NoiseHead outputs sigmoid

        total_loss = torch.tensor(0.0, device=self.device)
        active_losses_found = False
        for loss_name, loss_value in losses.items():
            weight = self.loss_weights.get(loss_name, 0)
            if weight > 0 and not torch.isnan(loss_value) and loss_value.numel() > 0 :
                total_loss += weight * loss_value
                active_losses_found = True
        
        if not active_losses_found and sum(self.loss_weights.values()) > 0:
            logger.debug("No active loss components contributed to total_loss, or all weights are zero.")
        
        losses['total_loss'] = total_loss
        return losses

    def _run_epoch(self, data_loader: Union[GraphDataLoader, NeighborLoader], is_training: bool, epoch_num: int) -> Dict[str, float]:
        self.model.train(is_training)
        # context = torch.enable_grad() if is_training else torch.no_grad() # Redundant with model.train/eval for grads
        mode_desc = "Training" if is_training else "Validating"

        total_losses_accum = { 'recon': 0.0, 'synth': 0.0, 'flip': 0.0, 'noise': 0.0, 'total_loss': 0.0 }
        num_samples_processed = 0 # Count actual samples used for loss calculation

        # Use tqdm for progress bar
        pbar_desc = f"{mode_desc} Epoch {epoch_num}"
        pbar = tqdm(data_loader, desc=pbar_desc, leave=False)

        for batch_idx, batch_data in enumerate(pbar):
            batch_data = batch_data.to(self.device)
            
            if is_training:
                self.optimizer.zero_grad(set_to_none=True) # More memory efficient

            with autocast(enabled=self.use_amp): # AMP
                # For NeighborLoader, model needs to handle subgraph (batch_data) vs full graph.
                # Assuming GNNAutoencoder's forward method takes (x, edge_index) from the batch.
                outputs = self.model(batch_data.x, batch_data.edge_index)
                loss_dict = self._calculate_combined_loss(outputs, batch_data)
                current_loss = loss_dict['total_loss']
            
            current_batch_size = batch_data.batch_size if hasattr(batch_data, 'batch_size') and self.use_graph_sampling else batch_data.num_nodes

            if is_training:
                if torch.isnan(current_loss) or current_loss.numel() == 0:
                    logger.warning(f"NaN or empty loss detected during training. Skipping batch. Loss: {current_loss}, Components: {loss_dict}")
                    continue
                self.scaler.scale(current_loss).backward() # AMP
                self.scaler.step(self.optimizer) # AMP
                self.scaler.update() # AMP
            
            for k, v in loss_dict.items():
                if not torch.isnan(v) and v.numel() > 0:
                    total_losses_accum[k] += v.item() * current_batch_size # Weight by batch size
            num_samples_processed += current_batch_size
            
            # Update tqdm progress bar
            if num_samples_processed > 0:
                 pbar.set_postfix({k: f"{v / num_samples_processed:.4f}" for k,v in total_losses_accum.items()})


        avg_losses = {k: v / num_samples_processed if num_samples_processed > 0 else 0.0 for k, v in total_losses_accum.items()}
        return avg_losses

    def _create_dataloader(self, graph_data: Optional[Data], is_training: bool) -> Optional[Union[GraphDataLoader, NeighborLoader]]:
        if graph_data is None:
            return None
        
        if not isinstance(graph_data, Data):
            logger.error(f"Expected torch_geometric.data.Data object for GNN training, got {type(graph_data)}. Cannot create DataLoader.")
            return None
        if graph_data.x is None or graph_data.edge_index is None:
             logger.error("Input graph_data is missing node features (x) or edge index (edge_index).")
             return None


        if self.use_graph_sampling:
            logger.info(f"Using NeighborLoader for GNN data. Batch size (seed nodes): {self.loader_batch_size}, Num neighbors: {self.num_neighbors}")
            return NeighborLoader(
                graph_data,
                num_neighbors=self.num_neighbors,
                batch_size=self.loader_batch_size,
                shuffle=is_training, # Shuffle only for training
                num_workers=self.gnn_config.get('num_workers', 0), # Configurable num_workers
                pin_memory=self.device.type == 'cuda' # Pin memory if on CUDA
            )
        else:
            logger.info("Using full graph DataLoader for GNN data (batch_size=1).")
            # Full graph training, batch_size is effectively 1 (one graph)
            return GraphDataLoader([graph_data], batch_size=1, shuffle=False) # No shuffle for single graph

    def _save_checkpoint(self, epoch: int, filename: str):
        save_path = self.checkpoint_dir / filename
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None, # Save scaler state
            'model_config': self.model.config,
            'loss_weights': self.loss_weights
        }
        try:
            torch.save(checkpoint, save_path)
        except Exception as e:
            logger.error(f"Failed to save GNN-AE checkpoint {save_path}: {e}")

    def _load_checkpoint(self, filename: str):
        load_path = self.checkpoint_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"GNN-AE Checkpoint file not found: {load_path}")
        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.use_amp and checkpoint.get('scaler_state_dict') is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info(f"Loaded GNN-AE checkpoint from {load_path} (Epoch {checkpoint.get('epoch', 'N/A')})")
        except Exception as e:
            logger.error(f"Failed to load GNN-AE checkpoint {load_path}: {e}", exc_info=True)
            raise