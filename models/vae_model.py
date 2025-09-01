import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for detecting gene scaling attacks in scRNA-seq data."""
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()

        # Default configuration
        default_config = {
            'input_dim': 20000,  # Typical for scRNA-seq
            'hidden_dim': 512,   # Single hidden layer for three-layer encoder/decoder
            'latent_dim': 64,    # Latent space dimension
            'dropout': 0.2,      # Dropout probability
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'loss_weights': {
                'recon': 0.8,    # Reconstruction loss weight
                'kl': 0.2,       # KL-divergence loss weight
                'cls': 1.0       # Classification loss weight for fine-tuning
            }
        }

        # Merge user config with defaults
        self.config = {**default_config, **(config or {})}
        self.loss_weights = self.config.get('loss_weights', default_config['loss_weights'])

        # Extract parameters
        self.input_dim = self.config['input_dim']
        self.hidden_dim = self.config['hidden_dim']
        self.latent_dim = self.config['latent_dim']
        self.dropout = self.config['dropout']
        self.device = self.config['device']

        self._validate_config()

        # Encoder and decoder
        self.encoder = self._build_encoder()
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(self.hidden_dim, self.latent_dim)
        self.decoder = self._build_decoder()
        self.classifier = nn.Linear(self.latent_dim, 1)  # Binary classification head

        self.to(self.device)

    def _validate_config(self):
        """Validate configuration parameters."""
        required_keys = {'input_dim', 'hidden_dim', 'latent_dim', 'dropout', 'loss_weights'}
        assert required_keys.issubset(self.config.keys()), \
            f"Missing required config keys: {required_keys - set(self.config.keys())}"

        assert isinstance(self.input_dim, int) and self.input_dim > 0, "input_dim must be positive integer"
        assert isinstance(self.hidden_dim, int) and self.hidden_dim > 0, "hidden_dim must be positive integer"
        assert self.latent_dim > 0 and self.latent_dim <= self.hidden_dim, \
            f"latent_dim must be ≤ {self.hidden_dim}"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.device in ['cpu', 'cuda'], "Device must be 'cpu' or 'cuda'"

        assert 'recon' in self.loss_weights and 'kl' in self.loss_weights, \
            "loss_weights must include 'recon' and 'kl'"
        assert all(0 <= v for v in self.loss_weights.values()), "Loss weights must be non-negative"
        # Removed strict sum-to-1 constraint to allow optional cls weight
        assert abs(sum([self.loss_weights[k] for k in ['recon', 'kl']]) - 1.0) < 1e-5, \
            f"Reconstruction and KL loss weights sum to {sum([self.loss_weights[k] for k in ['recon', 'kl']]):.4f}, must sum to 1.0"

    def _build_encoder(self):
        """Build three-layer encoder (input -> hidden -> latent)."""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

    def _build_decoder(self):
        """Build three-layer decoder (latent -> hidden -> input)."""
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.input_dim)
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * epsilon."""
        if not self.training:
            return mu  # Use mean during inference
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to mu and log_var."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return {'recon': recon_x, 'mu': mu, 'log_var': log_var, 'latent_z': z}

    def _vae_loss(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate VAE reconstruction and KL-divergence loss."""
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        return recon_loss, kl_loss

    def compute_loss(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute total loss (reconstruction + KL-divergence + optional classification)."""
        outputs = self(x)
        recon_loss, kl_loss = self._vae_loss(outputs['recon'], x, outputs['mu'], outputs['log_var'])
        loss = self.loss_weights['recon'] * recon_loss + self.loss_weights['kl'] * kl_loss
        if labels is not None:
            labels = labels.to(self.device)
            logits = self.classifier(outputs['mu'])  # Use mean for classification
            cls_loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
            loss += self.loss_weights.get('cls', 1.0) * cls_loss  # Add classification loss
        return loss

    def fit(self, data_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, epochs: int, patience: int = 20, validation_loader: Optional[torch.utils.data.DataLoader] = None):
        """Train the VAE model."""
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.train()
            train_loss_accum = 0.0
            recon_loss_accum = 0.0
            kl_loss_accum = 0.0
            cls_loss_accum = 0.0
            num_batches = 0

            for batch_data in tqdm(data_loader, desc=f"Epoch {epoch} Training"):
                # Handle tuple or list output from DataLoader
                if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                    x, labels = batch_data
                else:
                    x, labels = batch_data[0], None
                
                x = x.to(self.device)
                if labels is not None:
                    labels = labels.to(self.device)

                optimizer.zero_grad()
                total_loss = self.compute_loss(x, labels)
                total_loss.backward()
                optimizer.step()

                outputs = self(x)
                recon_loss, kl_loss = self._vae_loss(outputs['recon'], x, outputs['mu'], outputs['log_var'])
                train_loss_accum += total_loss.item()
                recon_loss_accum += recon_loss.item()
                kl_loss_accum += kl_loss.item()
                if labels is not None:
                    logits = self.classifier(outputs['mu'])
                    cls_loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
                    cls_loss_accum += cls_loss.item()
                num_batches += 1

            avg_train_loss = train_loss_accum / num_batches
            avg_recon_loss = recon_loss_accum / num_batches
            avg_kl_loss = kl_loss_accum / num_batches
            avg_cls_loss = cls_loss_accum / num_batches if labels is not None else 0.0
            print(f"Epoch {epoch}: Avg Train Loss: {avg_train_loss:.4f} "
                  f"(Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}, Cls: {avg_cls_loss:.4f})")

            if validation_loader:
                self.eval()
                val_loss_accum = 0.0
                val_batches = 0
                with torch.no_grad():
                    for batch_data in validation_loader:
                        if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                            x_val, labels_val = batch_data
                        else:
                            x_val, labels_val = batch_data[0], None
                        x_val = x_val.to(self.device)
                        if labels_val is not None:
                            labels_val = labels_val.to(self.device)
                        total_loss = self.compute_loss(x_val, labels_val)
                        val_loss_accum += total_loss.item()
                        val_batches += 1

                avg_val_loss = val_loss_accum / val_batches
                print(f"Epoch {epoch}: Avg Validation Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        break
            else:
                if avg_train_loss < best_val_loss:
                    best_val_loss = avg_train_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        break

    def detect(self, x: np.ndarray) -> np.ndarray:
        """Detect gene scaling attacks using reconstruction errors."""
        self.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            outputs = self(x_tensor)
            recon_error = torch.mean((outputs['recon'] - x_tensor) ** 2, dim=1)
        return recon_error.cpu().numpy()

    def save(self, path: str) -> None:
        """Save model state and configuration."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'version': '1.0.0',
            'state_dict': self.state_dict(),
            'config': self.config,
            'loss_weights': self.loss_weights
        }, path)
        print(f"VAE model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = None) -> 'VariationalAutoencoder':
        """Load model state and configuration."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file {path} not found")
        map_location = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(checkpoint.get('config'))
        model.loss_weights = checkpoint.get('loss_weights', model.loss_weights)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(map_location)
        print(f"VAE model loaded from {path} to device {map_location}")
        return model