import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
from typing import Optional, Dict, Any, List
import yaml
import logging
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class GNNAutoencoder(nn.Module):
    """
    Graph Neural Network Autoencoder for detecting anomalies in genomic data.
    Processes cell-to-cell similarity graphs to detect structural anomalies.
    
    Args:
        input_dim: Dimension of input node features
        hidden_dims: List of hidden layer dimensions
        latent_dim: Dimension of latent space
        dropout: Dropout rate
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        patience: Patience for early stopping
        device: Device to run on ('cuda' or 'cpu')
    """
    
    def __init__(self,
                 input_dim: int = 50,
                 hidden_dims: List[int] = [64, 32],
                 latent_dim: int = 16,
                 dropout: float = 0.3,
                 learning_rate: float = 0.001,
                 epochs: int = 200,
                 patience: int = 20,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 **kwargs):
        super().__init__()
        
        # Initialize parameters
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.scaler = StandardScaler()
        
        # Handle additional configuration
        self._configure_from_kwargs(kwargs)
        
        # Build encoder and decoder
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        # Move to device
        self.to(self.device)
        
        # Setup optimizer and loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Early stopping
        self.best_loss = np.inf
        self.epochs_without_improvement = 0
        
    def _configure_from_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Handle additional configuration parameters."""
        if 'config_path' in kwargs:
            with open(kwargs['config_path'], 'r') as f:
                config = yaml.safe_load(f).get('models', {}).get('gnn', {})
            self.input_dim = config.get('input_dim', self.input_dim)
            self.hidden_dims = config.get('hidden_dims', self.hidden_dims)
            self.latent_dim = config.get('latent_dim', self.latent_dim)
            self.dropout = config.get('dropout', self.dropout)
            self.learning_rate = config.get('learning_rate', self.learning_rate)
            self.epochs = config.get('epochs', self.epochs)
            self.patience = config.get('patience', self.patience)
            
    def _build_encoder(self) -> nn.Module:
        """Build the GNN encoder."""
        layers = nn.ModuleList()
        prev_dim = self.input_dim
        
        # Add hidden layers
        for dim in self.hidden_dims:
            layers.append(GCNConv(prev_dim, dim))
            prev_dim = dim
            
        # Final layer to latent space
        layers.append(GCNConv(prev_dim, self.latent_dim))
        
        return layers
    
    def _build_decoder(self) -> nn.Module:
        """Build the MLP decoder."""
        layers = nn.ModuleList()
        reversed_dims = list(reversed(self.hidden_dims))
        prev_dim = self.latent_dim
        
        # First decoder layer
        layers.append(nn.Linear(prev_dim, reversed_dims[0]))
        prev_dim = reversed_dims[0]
        
        # Additional hidden layers
        for dim in reversed_dims[1:]:
            layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
            
        # Final reconstruction layer
        layers.append(nn.Linear(prev_dim, self.input_dim))
        
        return layers
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode graph data into latent space."""
        h = x
        for i, layer in enumerate(self.encoder):
            h = layer(h, edge_index)
            if i < len(self.encoder) - 1:  # No activation after last layer
                h = F.leaky_relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to feature space."""
        h = z
        for i, layer in enumerate(self.decoder):
            h = layer(h)
            if i < len(self.decoder) - 1:  # No activation after last layer
                h = F.leaky_relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass through the autoencoder."""
        z = self.encode(x, edge_index)
        x_recon = self.decode(z)
        return x_recon, z
    
    def fit(self, graph_data: Data) -> None:
        """Train the GNN autoencoder."""
        # Prepare data
        x = graph_data.x.to(self.device)
        edge_index = graph_data.edge_index.to(self.device)
        
        logging.info(f"Training GNN Autoencoder with input_dim={self.input_dim}, "
                   f"hidden_dims={self.hidden_dims}, latent_dim={self.latent_dim}")
        
        # Training loop
        for epoch in range(self.epochs):
            self.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            x_recon, _ = self(x, edge_index)
            
            # Compute loss
            loss = self.criterion(x_recon, x)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Early stopping check
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logging.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")
    
    def detect(self, graph_data: Data) -> np.ndarray:
        """
        Compute anomaly scores for graph data.
        
        Args:
            graph_data: PyG Data object with x and edge_index
            
        Returns:
            Anomaly scores for each node (n_samples,)
        """
        self.eval()
        x = graph_data.x.to(self.device)
        edge_index = graph_data.edge_index.to(self.device)
        
        with torch.no_grad():
            # Get reconstruction
            x_recon, z = self(x, edge_index)
            
            # Calculate reconstruction error per node
            recon_error = torch.sum((x_recon - x) ** 2, dim=1)
            
            # Calculate neighborhood consistency score
            src, dst = edge_index
            neighbor_consistency = torch.zeros_like(recon_error)
            
            # For each node, compare its latent rep with neighbors
            for i in range(x.size(0)):
                neighbors = dst[src == i]
                if len(neighbors) > 0:
                    # Distance to mean of neighbors
                    neighbor_mean = z[neighbors].mean(dim=0)
                    neighbor_consistency[i] = torch.norm(z[i] - neighbor_mean, p=2)
            
            # Combine scores
            scores = recon_error.cpu().numpy() + neighbor_consistency.cpu().numpy()
            
            # Normalize to [0, 1] range
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            
        logging.info(f"GNN detection completed. Score range: {scores.min():.3f}-{scores.max():.3f}")
        
        return scores
    
    def save(self, path: str) -> None:
        """Save model to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'latent_dim': self.latent_dim,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> 'GNNAutoencoder':
        """Load model from file."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_dim=checkpoint['config']['input_dim'],
            hidden_dims=checkpoint['config']['hidden_dims'],
            latent_dim=checkpoint['config']['latent_dim'],
            dropout=checkpoint['config']['dropout'],
            learning_rate=checkpoint['config']['learning_rate'],
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    @classmethod
    def from_config(cls, config_path: str = 'config.yaml', device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> 'GNNAutoencoder':
        """
        Create instance from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            device: Device to run the model on
            
        Returns:
            Configured GNNAutoencoder instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f).get('models', {}).get('gnn', {})
        
        return cls(
            input_dim=config.get('input_dim', 50),
            hidden_dims=config.get('hidden_dims', [64, 32]),
            latent_dim=config.get('latent_dim', 16),
            dropout=config.get('dropout', 0.3),
            learning_rate=config.get('learning_rate', 0.001),
            epochs=config.get('epochs', 200),
            patience=config.get('patience', 20),
            device=device
        )