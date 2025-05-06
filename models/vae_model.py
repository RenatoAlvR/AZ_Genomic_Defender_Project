# vae_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np

# Task-specific Head Implementations for VAE
class LabelFlipHeadVAE(nn.Module):
    """Detects potential label flips based on VAE latent space."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        latent_dim = config['latent_dim']
        # Simple MLP head
        self.detector = nn.Sequential(
            nn.Linear(latent_dim, max(32, latent_dim // 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(config.get('head_dropout', 0.2)), # Add dropout if desired
            nn.Linear(max(32, latent_dim // 2), 1)  # Output a single score/logit
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Output raw logits; BCEWithLogitsLoss is numerically stable
        return self.detector(z)

class GeneScaleHeadVAE(nn.Module):
    """Detects potential gene scaling based on VAE latent space."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        latent_dim = config['latent_dim']
        # Another simple MLP head
        self.detector = nn.Sequential(
            nn.Linear(latent_dim, max(64, latent_dim // 2)),
            nn.ReLU(),
            nn.Dropout(config.get('head_dropout', 0.2)),
            nn.Linear(max(64, latent_dim // 2), 1) # Output a single score/logit
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Output raw logits
        return self.detector(z)

class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) for genomic data anomaly detection.
    Configurable detection heads for:
    - Label Flips
    - Gene Scalation

    Assumes input data `x` is a tensor of shape [num_samples, input_dim].
    During training (`fit`), expects data object with `x`, `flip_labels`, `scale_labels`.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()

        # Default configuration
        default_config = {
            # Detection objectives
            'LabelFlip': False,
            'GeneScalation': False,

            # Architecture
            'input_dim': 20000,      # Matches GNN-AE for consistency
            'hidden_dims': [512, 256], # Example MLP hidden layers
            'latent_dim': 64,
            'dropout': 0.2,
            'head_dropout': 0.2,     # Dropout specific to task heads
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }

        self.config = {**default_config, **(config or {})}

        # Default loss weights - KL weight often tuned carefully
        default_loss_weights = {
            'recon': 0.6,
            'kl': 0.1,
            'flip': 0.2,
            'scale': 0.1
        }
        # Merge with user config loss weights
        self.loss_weights = {**default_loss_weights, **(self.config.get('loss_weights', {}))}

        self._validate_config()

        # Encoder layers (outputs mu and log_var)
        self.encoder = self._build_encoder()
        self.fc_mu = nn.Linear(self.config['hidden_dims'][-1], self.config['latent_dim'])
        self.fc_log_var = nn.Linear(self.config['hidden_dims'][-1], self.config['latent_dim'])

        # Decoder layers
        self.decoder = self._build_decoder()

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        if self.config['LabelFlip']:
            self.task_heads['flip'] = LabelFlipHeadVAE(self.config)
        if self.config['GeneScalation']:
            self.task_heads['scale'] = GeneScaleHeadVAE(self.config)

        self.to(self.config['device'])

    def _validate_config(self):
        assert isinstance(self.config['LabelFlip'], bool), "LabelFlip must be a boolean"
        assert isinstance(self.config['GeneScalation'], bool), "GeneScalation must be a boolean"
        assert isinstance(self.config['input_dim'], int) and self.config['input_dim'] > 0
        assert isinstance(self.config['hidden_dims'], list) and len(self.config['hidden_dims']) > 0
        assert all(isinstance(d, int) and d > 0 for d in self.config['hidden_dims'])
        assert isinstance(self.config['latent_dim'], int) and self.config['latent_dim'] > 0
        assert 0 <= self.config['dropout'] < 1
        assert 0 <= self.config['head_dropout'] < 1

        if any([self.config['LabelFlip'], self.config['GeneScalation']]):
            self._validate_loss_weights()
        else:
             # If no task heads, ensure recon and KL weights sum to ~1
             assert 'loss_weights' in self.config, "loss_weights must be defined"
             assert 'recon' in self.config['loss_weights']
             assert 'kl' in self.config['loss_weights']
             required_weights = ['recon', 'kl']
             active_weights = {k: v for k, v in self.config['loss_weights'].items() if k in required_weights}
             weight_sum = sum(active_weights.values())
             assert abs(weight_sum - 1.0) < 1e-5, \
                 f"Recon and KL loss weights sum to {weight_sum:.4f}, should sum to 1.0 when no task heads are active."


    def _validate_loss_weights(self):
        assert 'loss_weights' in self.config, "loss_weights must be defined"
        required_weights = ['recon', 'kl']
        if self.config['LabelFlip']:
            required_weights.append('flip')
        if self.config['GeneScalation']:
            required_weights.append('scale')

        missing = [w for w in required_weights if w not in self.loss_weights]
        assert not missing, f"Missing loss weights for: {missing}"

        for name, value in self.loss_weights.items():
            assert isinstance(value, (int, float)), f"{name} weight must be numeric"
            assert 0 <= value <= 1, f"{name} weight must be between 0 and 1"

        active_weights = {k: v for k, v in self.loss_weights.items() if k in required_weights}
        weight_sum = sum(active_weights.values())
        assert abs(weight_sum - 1.0) < 1e-5, \
            f"Active loss weights sum to {weight_sum:.4f}, should sum to 1.0. Active weights: {active_weights}"

    def _build_encoder(self) -> nn.Sequential:
        layers = []
        in_dim = self.config['input_dim']
        for hidden_dim in self.config['hidden_dims']:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config['dropout']))
            in_dim = hidden_dim
        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Sequential:
        layers = []
        # Reversed hidden dims + latent dim
        decoder_dims = [self.config['latent_dim']] + self.config['hidden_dims'][::-1]
        for i in range(len(decoder_dims) - 1):
            layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
            layers.append(nn.ReLU())
            # Optional: Add dropout to decoder too
            # layers.append(nn.Dropout(self.config['dropout']))

        layers.append(nn.Linear(decoder_dims[-1], self.config['input_dim']))
        # Sigmoid activation if input data is normalized to [0, 1]
        # layers.append(nn.Sigmoid()) # Or leave as linear output
        return nn.Sequential(*layers)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * epsilon"""
        if not self.training:
            return mu # Use mean during evaluation/inference for stability
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes input x into mu and log_var."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent vector z into reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)

        outputs = {
            'recon': recon_x,
            'mu': mu,
            'log_var': log_var,
            'latent_z': z
        }

        # Pass latent sample z to task heads
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(z)

        return outputs

    def _vae_loss(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates VAE reconstruction and KL divergence loss."""
        # Reconstruction Loss (Mean Squared Error)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') # Use sum for consistency with KL term scale

        # KL Divergence Loss (compared to standard normal N(0,I))
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Return per-sample average losses for easier weighting/interpretation
        batch_size = x.size(0)
        return recon_loss / batch_size, kl_loss / batch_size


    def fit(self, data_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, epochs: int, patience: int = 20, validation_loader: Optional[torch.utils.data.DataLoader] = None):
        """Trains the VAE model."""

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.train()
            train_loss_accum = 0.0
            recon_loss_accum = 0.0
            kl_loss_accum = 0.0
            flip_loss_accum = 0.0
            scale_loss_accum = 0.0
            num_batches = 0

            for batch_data in data_loader:
                # Assuming batch_data contains 'x' and potentially labels
                # Adjust access based on your DataLoader structure
                if isinstance(batch_data, torch.Tensor):
                    x = batch_data
                    # Handle case where labels might not be in every batch/loader
                    flip_labels = None
                    scale_labels = None
                elif isinstance(batch_data, dict):
                    x = batch_data['x']
                    flip_labels = batch_data.get('flip_labels')
                    scale_labels = batch_data.get('scale_labels')
                elif hasattr(batch_data, 'x'): # For PyG Data objects if used
                     x = batch_data.x
                     flip_labels = getattr(batch_data, 'flip_labels', None)
                     scale_labels = getattr(batch_data, 'scale_labels', None)
                else:
                    raise ValueError("Unsupported data batch type in VAE fit")

                x = x.to(self.config['device'])
                if flip_labels is not None: flip_labels = flip_labels.to(self.config['device']).float().unsqueeze(-1)
                if scale_labels is not None: scale_labels = scale_labels.to(self.config['device']).float().unsqueeze(-1)

                optimizer.zero_grad()
                outputs = self(x)

                # VAE Core Losses
                recon_loss, kl_loss = self._vae_loss(outputs['recon'], x, outputs['mu'], outputs['log_var'])

                # Supervised Losses (using BCEWithLogitsLoss for stability)
                flip_loss = 0
                if 'flip' in outputs and flip_labels is not None:
                    flip_loss = F.binary_cross_entropy_with_logits(outputs['flip'], flip_labels)

                scale_loss = 0
                if 'scale' in outputs and scale_labels is not None:
                     scale_loss = F.binary_cross_entropy_with_logits(outputs['scale'], scale_labels)

                # Combine losses using weights
                total_loss = (
                    self.loss_weights['recon'] * recon_loss +
                    self.loss_weights['kl'] * kl_loss +
                    self.loss_weights.get('flip', 0) * flip_loss + # Use .get with default 0
                    self.loss_weights.get('scale', 0) * scale_loss
                )

                total_loss.backward()
                optimizer.step()

                train_loss_accum += total_loss.item()
                recon_loss_accum += recon_loss.item()
                kl_loss_accum += kl_loss.item()
                flip_loss_accum += flip_loss.item() if isinstance(flip_loss, torch.Tensor) else flip_loss
                scale_loss_accum += scale_loss.item() if isinstance(scale_loss, torch.Tensor) else scale_loss
                num_batches += 1

            avg_train_loss = train_loss_accum / num_batches
            print(f"Epoch {epoch}: Avg Train Loss: {avg_train_loss:.4f} "
                  f"(Recon: {recon_loss_accum / num_batches:.4f}, KL: {kl_loss_accum / num_batches:.4f}, "
                  f"Flip: {flip_loss_accum / num_batches:.4f}, Scale: {scale_loss_accum / num_batches:.4f})")

            # Validation and Early Stopping
            if validation_loader:
                self.eval()
                val_loss_accum = 0.0
                val_batches = 0
                with torch.no_grad():
                    for batch_data in validation_loader:
                       # Similar data handling as in training loop
                        if isinstance(batch_data, torch.Tensor): x_val = batch_data
                        elif isinstance(batch_data, dict): x_val = batch_data['x']
                        elif hasattr(batch_data, 'x'): x_val = batch_data.x
                        else: continue # Skip unsupported type

                        x_val = x_val.to(self.config['device'])
                        # Get labels if they exist for validation loss calculation
                        flip_labels_val = batch_data.get('flip_labels') if isinstance(batch_data, dict) else getattr(batch_data, 'flip_labels', None)
                        scale_labels_val = batch_data.get('scale_labels') if isinstance(batch_data, dict) else getattr(batch_data, 'scale_labels', None)
                        if flip_labels_val is not None: flip_labels_val = flip_labels_val.to(self.config['device']).float().unsqueeze(-1)
                        if scale_labels_val is not None: scale_labels_val = scale_labels_val.to(self.config['device']).float().unsqueeze(-1)

                        outputs_val = self(x_val)
                        recon_loss_val, kl_loss_val = self._vae_loss(outputs_val['recon'], x_val, outputs_val['mu'], outputs_val['log_var'])

                        flip_loss_val = 0
                        if 'flip' in outputs_val and flip_labels_val is not None:
                            flip_loss_val = F.binary_cross_entropy_with_logits(outputs_val['flip'], flip_labels_val)

                        scale_loss_val = 0
                        if 'scale' in outputs_val and scale_labels_val is not None:
                            scale_loss_val = F.binary_cross_entropy_with_logits(outputs_val['scale'], scale_labels_val)

                        total_val_loss = (
                            self.loss_weights['recon'] * recon_loss_val +
                            self.loss_weights['kl'] * kl_loss_val +
                            self.loss_weights.get('flip', 0) * flip_loss_val +
                            self.loss_weights.get('scale', 0) * scale_loss_val
                        )
                        val_loss_accum += total_val_loss.item()
                        val_batches += 1

                avg_val_loss = val_loss_accum / val_batches
                print(f"Epoch {epoch}: Avg Validation Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    # Optionally save best model checkpoint here
                    # self.save(f"best_vae_model_epoch_{epoch}.pth")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        break
            else:
                # Basic early stopping based on training loss if no validation set
                 if avg_train_loss < best_val_loss: # Use train loss as proxy
                     best_val_loss = avg_train_loss
                     epochs_no_improve = 0
                 else:
                     epochs_no_improve += 1
                     if epochs_no_improve >= patience:
                         print(f"Early stopping triggered after {epoch + 1} epochs based on training loss.")
                         break


    def detect(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Compute anomaly scores for input data x.

        Args:
            x: Input tensor of shape [num_samples, input_dim]

        Returns:
            Dictionary of numpy arrays with detection scores per sample.
            Includes 'recon_error', 'kl_divergence', and task-specific scores ('flip', 'scale').
        """
        self.eval()
        x = x.to(self.config['device'])

        with torch.no_grad():
            outputs = self(x)

            # Anomaly score based on reconstruction error per sample
            recon_error = torch.sum((outputs['recon'] - x) ** 2, dim=1).cpu().numpy()

            # Anomaly score based on KL divergence per sample
            # KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var), dim=1)
            kl_div_per_sample = -0.5 * torch.sum(1 + outputs['log_var'] - outputs['mu'].pow(2) - outputs['log_var'].exp(), dim=1)
            kl_div_score = kl_div_per_sample.cpu().numpy()

            scores = {
                'recon_error': recon_error,
                'kl_divergence': kl_div_score
            }

            # Get task-specific scores (apply sigmoid since heads output logits)
            if 'flip' in outputs:
                scores['flip'] = torch.sigmoid(outputs['flip']).squeeze().cpu().numpy()
            if 'scale' in outputs:
                 scores['scale'] = torch.sigmoid(outputs['scale']).squeeze().cpu().numpy()

            # Example combined score (can be refined in fusion logic)
            # Higher recon error or KL divergence suggests anomaly
            # Higher task scores suggest specific attacks
            # Simple weighted sum example - weights might differ from training weights
            combined_score = (scores['recon_error'] * 0.5 +
                              scores['kl_divergence'] * 0.2 +
                              scores.get('flip', 0) * 0.15 + # Use get for safety
                              scores.get('scale', 0) * 0.15)
            scores['combined_vae'] = combined_score


        return scores

    def save(self, path: str) -> None:
        """Save model state and configuration."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'version': '1.0.0', # Add versioning if needed
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

        # Determine map location based on device preference and availability
        if device:
            map_location = torch.device(device)
        else:
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(path, map_location=map_location)

        # Ensure config compatibility if versions change later
        config = checkpoint.get('config')
        loss_weights = checkpoint.get('loss_weights', {}) # Load loss weights too

        model = cls(config=config) # Re-initialize with saved config
        model.loss_weights = loss_weights # Restore loss weights
        model.load_state_dict(checkpoint['state_dict'])
        model.to(map_location) # Ensure model is on the correct device
        print(f"VAE model loaded from {path} to device {map_location}")
        return model