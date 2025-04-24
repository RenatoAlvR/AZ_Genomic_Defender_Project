import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import logging

class GNNAETrainer:
    """
    Trainer for the GNN-Autoencoder model with:
    - Synthetic cell detection
    - Label flip detection
    - Noise detection
    
    Args:
        config: Dictionary containing all configuration parameters
        model: Initialized GNNAutoencoder instance
    """
    def __init__(self, config: Dict[str, Any], model):
        self.config = config['GNN']  # Extract GNN-specific config
        self.model = model
        self.device = torch.device(self.config['device'])
        self.model.to(self.device)
        
        # Training parameters
        self.epochs = self.config['epochs']
        self.patience = self.config['patience']
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=1e-5
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('GNN-AE Trainer')
        
    def train(self, train_data: Data, val_data: Data = None):
        """
        Full training loop with early stopping
        
        Args:
            train_data: PyG Data object with features, edges, and labels
            val_data: Optional validation data
        """
        best_loss = np.inf
        epochs_no_improve = 0
        
        # Convert to DataLoader for batching if needed
        train_loader = self._create_dataloader(train_data)
        val_loader = self._create_dataloader(val_data) if val_data else None
        
        for epoch in range(1, self.epochs + 1):
            # Training epoch
            train_loss = self._train_epoch(train_loader)
            
            # Validation
            val_loss = np.nan
            if val_loader:
                val_loss = self._validate(val_loader)
                self.logger.info(f'Epoch {epoch}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}')
            else:
                self.logger.info(f'Epoch {epoch}: Train Loss {train_loss:.4f}')
            
            # Early stopping
            current_loss = val_loss if val_loader else train_loss
            if current_loss < best_loss:
                best_loss = current_loss
                epochs_no_improve = 0
                self._save_checkpoint(epoch, 'best')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    self.logger.info(f'Early stopping at epoch {epoch}')
                    break
                    
        # Load best model
        self._load_best_checkpoint()
        
    def _train_epoch(self, data_loader: DataLoader) -> float:
        """Single training epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(data_loader, desc="Training"):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch.x, batch.edge_index)
            
            # Calculate losses
            recon_loss = F.mse_loss(outputs['recon'], batch.x)
            
            synth_loss = F.binary_cross_entropy(
                outputs['synth'].squeeze(),
                batch.synth_labels.float()
            ) if 'synth' in outputs else 0
                
            flip_loss = F.binary_cross_entropy(
                outputs['flip'].squeeze(),
                batch.flip_labels.float()
            ) if 'flip' in outputs else 0
                
            noise_loss = F.binary_cross_entropy(
                outputs['noise'].squeeze(),
                batch.noise_labels.float()
            ) if 'noise' in outputs else 0
            
            # Weighted loss
            loss = (
                self.model.loss_weights['recon'] * recon_loss +
                self.model.loss_weights['synth'] * synth_loss +
                self.model.loss_weights['flip'] * flip_loss +
                self.model.loss_weights['noise'] * noise_loss
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(data_loader)
    
    def _validate(self, data_loader: DataLoader) -> float:
        """Validation epoch"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Validating"):
                batch = batch.to(self.device)
                outputs = self.model(batch.x, batch.edge_index)
                
                # Validation only uses reconstruction loss
                loss = F.mse_loss(outputs['recon'], batch.x)
                total_loss += loss.item()
                
        return total_loss / len(data_loader)
    
    def _create_dataloader(self, data: Data) -> DataLoader:
        """Create DataLoader with configured batch size"""
        return DataLoader(
            [data],
            batch_size=self.config.get('batch_size', 1),
            shuffle=True
        )
    
    def _save_checkpoint(self, epoch: int, name: str):
        """Save model and trainer state"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config
        }
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(checkpoint, f"checkpoints/{name}_gnnae.pt")
        
    def _load_best_checkpoint(self):
        """Load best performing model"""
        checkpoint = torch.load("checkpoints/best_gnnae.pt")
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])