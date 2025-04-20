import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import yaml
import logging
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

class ContrastiveAutoencoder(nn.Module):
    """
    Contrastive Autoencoder for genomic anomaly detection.
    Combines reconstruction loss with contrastive loss to learn robust representations.
    
    Args:
        input_dim: Dimension of input features
        hidden_dims: List of hidden layer dimensions
        latent_dim: Dimension of latent space
        temperature: Temperature for contrastive loss
        alpha: Weight for contrastive loss
        dropout: Dropout rate
        learning_rate: Learning rate for optimizer
        batch_size: Training batch size
        epochs: Number of training epochs
        patience: Patience for early stopping
        device: Device to run on ('cuda' or 'cpu')
    """
    
    def __init__(self,
                 input_dim: int = 50,
                 hidden_dims: List[int] = [128, 64],
                 latent_dim: int = 32,
                 temperature: float = 0.1,
                 alpha: float = 0.5,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 64,
                 epochs: int = 150,
                 patience: int = 15,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 **kwargs):
        super().__init__()
        
        # Initialize parameters
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.alpha = alpha
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.scaler = StandardScaler()
        
        # Handle additional configuration
        self._configure_from_kwargs(kwargs)
        
        # Build encoder and decoder
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Move to device
        self.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Early stopping
        self.best_loss = np.inf
        self.epochs_without_improvement = 0
        
    def _configure_from_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Handle additional configuration parameters."""
        if 'config_path' in kwargs:
            with open(kwargs['config_path'], 'r') as f:
                config = yaml.safe_load(f).get('models', {}).get('contrastive_ae', {})
            self.input_dim = config.get('input_dim', self.input_dim)
            self.hidden_dims = config.get('hidden_dims', self.hidden_dims)
            self.latent_dim = config.get('latent_dim', self.latent_dim)
            self.temperature = config.get('temperature', self.temperature)
            self.alpha = config.get('alpha', self.alpha)
            self.dropout = config.get('dropout', self.dropout)
            self.learning_rate = config.get('learning_rate', self.learning_rate)
            self.batch_size = config.get('batch_size', self.batch_size)
            self.epochs = config.get('epochs', self.epochs)
            self.patience = config.get('patience', self.patience)
            
    def _build_encoder(self) -> nn.Module:
        """Build the encoder network."""
        layers = []
        prev_dim = self.input_dim
        
        for dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(self.dropout)
            ])
            prev_dim = dim
            
        # Final layer to latent space
        layers.append(nn.Linear(prev_dim, self.latent_dim))
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Module:
        """Build the decoder network."""
        layers = []
        reversed_dims = list(reversed(self.hidden_dims))
        prev_dim = self.latent_dim
        
        for dim in reversed_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(self.dropout)
            ])
            prev_dim = dim
            
        # Final reconstruction layer
        layers.append(nn.Linear(prev_dim, self.input_dim))
        
        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)
    
    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Project latent representation for contrastive learning."""
        return self.projection(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass."""
        z = self.encode(x)
        z_proj = self.project(z)
        x_recon = self.decode(z)
        return x_recon, z, z_proj
    
    def _contrastive_loss(self, z_proj: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Compute normalized temperature-scaled cross entropy loss."""
        # Compute similarity matrix
        sim_matrix = torch.mm(z_proj, z_proj.T) / self.temperature
        sim_matrix.fill_diagonal_(-float('inf'))
        
        # Positive samples are the augmented versions (same index)
        # Negative samples are all other examples in batch
        labels = torch.arange(batch_size).to(self.device)
        
        return F.cross_entropy(sim_matrix, labels)
    
    def _reconstruction_loss(self, x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute mean squared error reconstruction loss."""
        return F.mse_loss(x_recon, x)
    
    def loss_function(self, 
                     x_recon: torch.Tensor, 
                     x: torch.Tensor, 
                     z_proj: torch.Tensor,
                     batch_size: int) -> torch.Tensor:
        """Combine reconstruction and contrastive losses."""
        recon_loss = self._reconstruction_loss(x_recon, x)
        contrast_loss = self._contrastive_loss(z_proj, batch_size)
        return recon_loss + self.alpha * contrast_loss
    
    def fit(self, X: np.ndarray) -> None:
        """Train the contrastive autoencoder."""
        # Scale and convert to tensor
        X = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, 
                               batch_size=self.batch_size, 
                               shuffle=True,
                               drop_last=True)  # Important for contrastive loss
        
        logging.info(f"Training Contrastive AE with input_dim={self.input_dim}, "
                    f"latent_dim={self.latent_dim}, temperature={self.temperature}, "
                    f"alpha={self.alpha}")
        
        # Training loop
        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0.0
            
            for batch in dataloader:
                x = batch[0]
                batch_size = x.size(0)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                x_recon, _, z_proj = self(x)
                
                # Compute loss
                loss = self.loss_function(x_recon, x, z_proj, batch_size)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Average loss for the epoch
            avg_loss = epoch_loss / len(dataloader)
            
            # Early stopping check
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logging.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
    def detect(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for input data.
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Anomaly scores (n_samples,)
        """
        if not hasattr(self, 'scaler'):
            self.scaler = StandardScaler().fit(X)
        
        # Scale and convert to tensor
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Set to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Get reconstruction and latent projections
            x_recon, z, z_proj = self(X_tensor)
            
            # Calculate reconstruction error
            recon_error = torch.sum((x_recon - X_tensor) ** 2, dim=1)
            
            # Calculate contrastive anomaly score
            # Compare each sample to mean of batch
            mean_proj = z_proj.mean(dim=0, keepdim=True)
            contrast_scores = 1 - F.cosine_similarity(z_proj, mean_proj, dim=1)
            
            # Combine scores
            scores = recon_error.cpu().numpy() + contrast_scores.cpu().numpy()
            
            # Normalize scores to [0, 1] range
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            
        logging.info(f"Contrastive AE detection completed. Score range: {scores.min():.3f}-{scores.max():.3f}")
        
        return scores
    
    def save(self, path: str) -> None:
        """Save model to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'scaler': self.scaler,
            'config': {
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'latent_dim': self.latent_dim,
                'temperature': self.temperature,
                'alpha': self.alpha,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> 'ContrastiveAutoencoder':
        """Load model from file."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_dim=checkpoint['config']['input_dim'],
            hidden_dims=checkpoint['config']['hidden_dims'],
            latent_dim=checkpoint['config']['latent_dim'],
            temperature=checkpoint['config']['temperature'],
            alpha=checkpoint['config']['alpha'],
            dropout=checkpoint['config']['dropout'],
            learning_rate=checkpoint['config']['learning_rate'],
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.scaler = checkpoint['scaler']
        return model
    
    @classmethod
    def from_config(cls, config_path: str = 'config.yaml', device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> 'ContrastiveAutoencoder':
        """
        Create instance from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            device: Device to run the model on
            
        Returns:
            Configured ContrastiveAutoencoder instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f).get('models', {}).get('contrastive_ae', {})
        
        return cls(
            input_dim=config.get('input_dim', 50),
            hidden_dims=config.get('hidden_dims', [128, 64]),
            latent_dim=config.get('latent_dim', 32),
            temperature=config.get('temperature', 0.1),
            alpha=config.get('alpha', 0.5),
            dropout=config.get('dropout', 0.2),
            learning_rate=config.get('learning_rate', 0.001),
            batch_size=config.get('batch_size', 64),
            epochs=config.get('epochs', 150),
            patience=config.get('patience', 15),
            device=device
        )