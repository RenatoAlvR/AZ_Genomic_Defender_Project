import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Optional
import yaml
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for genomic anomaly detection.
    Configurable architecture with support for both linear and convolutional layers.
    """
    
    def __init__(self,
                 input_dim: int = 50,
                 latent_dim: int = 10,
                 hidden_dims: Optional[list] = None,
                 conv_dims: Optional[list] = None,
                 kernel_size: int = 3,
                 learning_rate: float = 1e-3,
                 dropout: float = 0.2,
                 batch_size: int = 32,
                 epochs: int = 100,
                 patience: int = 10,
                 reconstruction_loss_weight: float = 1.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 **kwargs):
        super().__init__()
        
        # Initialize parameters
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [32, 16]
        self.conv_dims = conv_dims
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.device = device
        self.scaler = StandardScaler()
        
        # Handle additional configuration
        self._configure_from_kwargs(kwargs)
        
        # Build encoder and decoder
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        # Move to device
        self.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Early stopping
        self.best_loss = np.inf
        self.epochs_without_improvement = 0
        
    def _configure_from_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Handle additional configuration parameters."""
        if 'config_path' in kwargs:
            with open(kwargs['config_path'], 'r') as f:
                config = yaml.safe_load(f).get('models', {}).get('vae', {})
            self.input_dim = config.get('input_dim', self.input_dim)
            self.latent_dim = config.get('latent_dim', self.latent_dim)
            self.hidden_dims = config.get('hidden_dims', self.hidden_dims)
            self.conv_dims = config.get('conv_dims', self.conv_dims)
            self.kernel_size = config.get('kernel_size', self.kernel_size)
            self.learning_rate = config.get('learning_rate', self.learning_rate)
            self.dropout = config.get('dropout', self.dropout)
            self.batch_size = config.get('batch_size', self.batch_size)
            self.epochs = config.get('epochs', self.epochs)
            self.patience = config.get('patience', self.patience)
            self.reconstruction_loss_weight = config.get('reconstruction_loss_weight', 
                                                     self.reconstruction_loss_weight)
            
    def _build_encoder(self) -> nn.Module:
        """Construct the encoder network."""
        layers = []
        prev_dim = self.input_dim
        
        # Add convolutional layers if specified
        if self.conv_dims:
            for i, dim in enumerate(self.conv_dims):
                layers.extend([
                    nn.Conv1d(1 if i == 0 else self.conv_dims[i-1], 
                             dim, 
                             kernel_size=self.kernel_size,
                             padding=self.kernel_size//2),
                    nn.BatchNorm1d(dim),
                    nn.LeakyReLU(0.2),
                    nn.MaxPool1d(2),
                    nn.Dropout(self.dropout)
                ])
            prev_dim = (prev_dim // (2 ** len(self.conv_dims))) * self.conv_dims[-1]
        
        # Add linear layers
        for dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(self.dropout)
            ])
            prev_dim = dim
        
        # Final layers for mean and log variance
        self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
        self.fc_var = nn.Linear(prev_dim, self.latent_dim)
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Module:
        """Construct the decoder network."""
        layers = []
        prev_dim = self.latent_dim
        
        # Reverse hidden dims for decoder
        reversed_hidden_dims = list(reversed(self.hidden_dims))
        
        for dim in reversed_hidden_dims[1:]:
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
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input into latent distribution parameters."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent samples."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass through the VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, 
                     recon_x: torch.Tensor, 
                     x: torch.Tensor, 
                     mu: torch.Tensor, 
                     logvar: torch.Tensor) -> torch.Tensor:
        """Compute VAE loss (reconstruction + KL divergence)."""
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return self.reconstruction_loss_weight * recon_loss + kl_loss
    
    def fit(self, X: np.ndarray) -> None:
        """Train the VAE on the input data."""
        # Scale and convert to tensor
        X = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=self.batch_size, 
                                              shuffle=True)
        
        logging.info(f"Training VAE with input_dim={self.input_dim}, latent_dim={self.latent_dim}, "
                   f"hidden_dims={self.hidden_dims}, conv_dims={self.conv_dims}")
        
        # Training loop
        for epoch in range(self.epochs):
            self.train()
            train_loss = 0
            
            for batch in dataloader:
                x = batch[0]
                self.optimizer.zero_grad()
                
                # Forward pass
                recon_x, mu, logvar = self(x)
                
                # Compute loss
                loss = self.loss_function(recon_x, x, mu, logvar)
                
                # Backward pass
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
            
            # Average loss for the epoch
            avg_loss = train_loss / len(dataloader.dataset)
            
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
            # Get reconstruction and latent space
            recon_x, mu, logvar = self(X_tensor)
            
            # Calculate reconstruction error
            recon_error = torch.sum((recon_x - X_tensor) ** 2, dim=1)
            
            # Calculate KL divergence term
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            
            # Combine into anomaly score
            scores = recon_error.cpu().numpy() + kl_div.cpu().numpy()
            
            # Normalize scores to [0, 1] range
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            
        logging.info(f"VAE detection completed. Score range: {scores.min():.3f}-{scores.max():.3f}")
        
        return scores
    
    def save(self, path: str) -> None:
        """Save model to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'scaler': self.scaler,
            'config': {
                'input_dim': self.input_dim,
                'latent_dim': self.latent_dim,
                'hidden_dims': self.hidden_dims,
                'conv_dims': self.conv_dims,
                'kernel_size': self.kernel_size,
                'learning_rate': self.learning_rate,
                'dropout': self.dropout
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> 'VariationalAutoencoder':
        """Load model from file."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_dim=checkpoint['config']['input_dim'],
            latent_dim=checkpoint['config']['latent_dim'],
            hidden_dims=checkpoint['config']['hidden_dims'],
            conv_dims=checkpoint['config']['conv_dims'],
            kernel_size=checkpoint['config']['kernel_size'],
            learning_rate=checkpoint['config']['learning_rate'],
            dropout=checkpoint['config']['dropout'],
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.scaler = checkpoint['scaler']
        return model
    
    @classmethod
    def from_config(cls, config_path: str = 'config.yaml', device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> 'VariationalAutoencoder':
        """
        Create instance from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            device: Device to run the model on
            
        Returns:
            Configured VariationalAutoencoder instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f).get('models', {}).get('vae', {})
        
        return cls(
            input_dim=config.get('input_dim', 50),
            latent_dim=config.get('latent_dim', 10),
            hidden_dims=config.get('hidden_dims'),
            conv_dims=config.get('conv_dims'),
            kernel_size=config.get('kernel_size', 3),
            learning_rate=config.get('learning_rate', 1e-3),
            dropout=config.get('dropout', 0.2),
            batch_size=config.get('batch_size', 32),
            epochs=config.get('epochs', 100),
            patience=config.get('patience', 10),
            reconstruction_loss_weight=config.get('reconstruction_loss_weight', 1.0),
            device=device
        )