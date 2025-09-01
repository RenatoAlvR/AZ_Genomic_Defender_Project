import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge
from torch.utils.data import DataLoader
from tqdm import tqdm

class GNNAutoencoder(nn.Module):
    """Graph Neural Network Autoencoder for detecting label flip attacks in scRNA-seq data."""
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()

        # Default configuration
        default_config = {
            'input_dim': 20000,  # Typical for scRNA-seq
            'hidden_dim': 256,   # Single hidden layer for three-layer encoder/decoder
            'latent_dim': 64,    # Latent space dimension
            'conv_type': 'GCN',  # GNN layer type
            'dropout': 0.3,      # Node dropout probability
            'edge_dropout': 0.2, # Edge dropout probability
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'loss_weight_recon': 1.0  # Reconstruction loss weight
        }

        # Merge user config with defaults
        self.config = {**default_config, **(config or {})}

        # Extract parameters
        self.input_dim = self.config['input_dim']
        self.hidden_dim = self.config['hidden_dim']
        self.latent_dim = self.config['latent_dim']
        self.conv_type = self.config['conv_type']
        self.dropout = self.config['dropout']
        self.edge_dropout = self.config['edge_dropout']
        self.device = self.config['device']
        self.loss_weight_recon = self.config['loss_weight_recon']

        self._validate_config()

        # Encoder and decoder
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.classifier = nn.Linear(self.latent_dim, 1)  # Binary classification head

        self.to(self.device)

    def _validate_config(self):
        """Validate configuration parameters."""
        required_keys = {'input_dim', 'hidden_dim', 'latent_dim', 'conv_type', 'dropout', 'edge_dropout', 'device', 'loss_weight_recon'}
        assert required_keys.issubset(self.config.keys()), \
            f"Missing required config keys: {required_keys - set(self.config.keys())}"

        assert isinstance(self.input_dim, int) and self.input_dim > 0, "input_dim must be positive integer"
        assert isinstance(self.hidden_dim, int) and self.hidden_dim > 0, "hidden_dim must be positive integer"
        assert self.latent_dim > 0 and self.latent_dim <= self.hidden_dim, \
            f"latent_dim must be ≤ {self.hidden_dim}"
        assert self.conv_type in ['GCN'], "conv_type must be 'GCN'"  # Simplified to GCN for stability
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert 0 <= self.edge_dropout < 1, "edge_dropout must be in [0, 1)"
        assert self.device in ['cpu', 'cuda'], "Device must be 'cpu' or 'cuda'"
        assert isinstance(self.loss_weight_recon, (int, float)) and self.loss_weight_recon > 0, \
            "loss_weight_recon must be positive"

    def _build_encoder(self):
        """Build three-layer encoder (input -> hidden -> latent)."""
        return nn.Sequential(
            GCNConv(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.latent_dim)
        )

    def _build_decoder(self):
        """Build three-layer decoder (latent -> hidden -> input)."""
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.input_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        if self.training and self.edge_dropout > 0:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout, force_undirected=True)

        # Encoder
        h = x
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, GCNConv):
                h = layer(h, edge_index)
            else:
                h = layer(h)
        
        # Decoder
        recon = self.decoder(h)
        return {'recon': recon, 'latent': h}

    def compute_loss(self, x: torch.Tensor, edge_index: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute reconstruction and optional classification loss for label flip detection."""
        outputs = self(x, edge_index)
        recon_loss = F.mse_loss(outputs['recon'], x, reduction='sum') / x.size(0)
        loss = self.loss_weight_recon * recon_loss
        if labels is not None:
            labels = labels.to(self.device)
            logits = self.classifier(outputs['z'].mean(dim=1))  # Mean pooling for graph-level classification
            cls_loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
            loss += cls_loss  # Combine reconstruction and classification loss
        return loss

    def fit(self, data_loader, optimizer: torch.optim.Optimizer, epochs: int, patience: int = 20):
        """Train the GNN-AE model."""
        self.train()
        best_loss = float('inf')
        no_improve = 0

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            for graph_data in tqdm(data_loader, desc=f"Epoch {epoch} Training"):
                x = graph_data.x.to(self.device)
                edge_index = graph_data.edge_index.to(self.device)
                labels = graph_data.labels.to(self.device) if hasattr(graph_data, 'labels') else None

                optimizer.zero_grad()
                loss = self.compute_loss(x, edge_index, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch}: Avg Train Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

    def detect(self, graph_data) -> np.ndarray:
        """Detect label flips using reconstruction errors."""
        self.eval()
        x = graph_data.x.to(self.device)
        edge_index = graph_data.edge_index.to(self.device)

        with torch.no_grad():
            outputs = self(x, edge_index)
            recon_error = torch.mean((outputs['recon'] - x) ** 2, dim=1)
        return recon_error.cpu().numpy()

    def save(self, path: str) -> None:
        """Save model state and configuration."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'version': '1.0.0',
            'state_dict': self.state_dict(),
            'config': self.config
        }, path)
        print(f"GNN-AE model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = None) -> 'GNNAutoencoder':
        """Load model state and configuration."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file {path} not found")
        map_location = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(checkpoint.get('config'))
        model.load_state_dict(checkpoint['state_dict'])
        model.to(map_location)
        print(f"GNN-AE model loaded from {path} to device {map_location}")
        return model