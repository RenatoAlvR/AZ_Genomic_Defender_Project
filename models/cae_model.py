import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

class ContrastiveAutoencoder(nn.Module):
    """Contrastive Autoencoder for detecting synthetic cell injections in scRNA-seq data."""
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()

        # Default configuration
        default_config = {
            'input_dim': 20000,  # Typical for scRNA-seq
            'hidden_dim': 512,   # Single hidden layer for three-layer encoder/decoder
            'latent_dim': 128,   # Latent space dimension
            'temperature': 0.5,  # Contrastive loss temperature
            'dropout': 0.2,      # Dropout probability
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'loss_weights': {
                'alpha': 0.7,       # Contrastive loss weight
                'beta': 0.3,        # Reconstruction loss weight
                'cls': 1.0          # Classification loss weight (for fine-tuning)
            }
        }

        # Merge user config with defaults
        self.config = {**default_config, **(config or {})}
        self.loss_weights = self.config.get('loss_weights', default_config['loss_weights'])

        # Extract parameters
        self.input_dim = self.config['input_dim']
        self.hidden_dim = self.config['hidden_dim']
        self.latent_dim = self.config['latent_dim']
        self.temperature = self.config['temperature']
        self.dropout = self.config['dropout']
        self.device = self.config['device']

        # Extract loss weights
        self.alpha = self.loss_weights.get('alpha', 0.7)
        self.beta = self.loss_weights.get('beta', 0.3)

        self._validate_config()

        # Model components
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.projection_head = self._build_projection_head()
        self.classifier = nn.Linear(self.latent_dim, 1) # Binary Classification Head

        self.to(self.device)

    def _validate_config(self):
        """Validate configuration parameters."""
        required_keys = {'input_dim', 'hidden_dim', 'latent_dim', 'temperature', 'dropout', 'loss_weights'}
        assert required_keys.issubset(self.config.keys()), \
            f"Missing required config keys: {required_keys - set(self.config.keys())}"

        assert isinstance(self.input_dim, int) and self.input_dim > 0, "input_dim must be positive integer"
        assert isinstance(self.hidden_dim, int) and self.hidden_dim > 0, "hidden_dim must be positive integer"
        assert self.latent_dim > 0 and self.latent_dim <= self.hidden_dim, \
            f"latent_dim must be ≤ {self.hidden_dim}"
        assert 0 < self.temperature < 1, "temperature must be between 0 and 1"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.device in ['cpu', 'cuda'], "Device must be 'cpu' or 'cuda'"
        assert 'alpha' in self.loss_weights and 'beta' in self.loss_weights, \
            "loss_weights must include 'alpha' and 'beta'"
        assert self.loss_weights['alpha'] + self.loss_weights['beta'] > 0, \
            "Contrastive and reconstruction loss weights sum must be positive"
        assert all(0 <= v for v in self.loss_weights.values()), "Loss weights must be non-negative"

    def _build_encoder(self):
        """Build three-layer encoder (input -> hidden -> latent)."""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.latent_dim)
        )

    def _build_decoder(self):
        """Build three-layer decoder (latent -> hidden -> input)."""
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.input_dim)
        )

    def _build_projection_head(self):
        """Build projection head for contrastive learning."""
        head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.BatchNorm1d(self.latent_dim * 2),
            nn.SiLU(),
            nn.Linear(self.latent_dim * 2, 64),
            nn.BatchNorm1d(64, affine=False)
        )
        # Xavier initialization
        for layer in head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        return head

    def contrastive_loss(self, projections, positive_pairs, negative_pairs):
        """
        Compute InfoNCE contrastive loss for synthetic cell detection.

        Args:
            projections: Normalized outputs from projection head (shape: [batch_size, 64])
            positive_pairs: Tensor of shape [n_pairs, 2] with indices of positive pairs
            negative_pairs: Tensor of shape [n_pairs, 2] with indices of negative pairs

        Returns:
            Contrastive loss value
        """
        # Compute similarity matrix
        sim_matrix = torch.mm(projections, projections.t()) / self.temperature
        sim_matrix = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0]  # Stability

        # Positive and negative similarities
        pos_sim = sim_matrix[positive_pairs[:, 0], positive_pairs[:, 1]]
        neg_sim = sim_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]

        # InfoNCE loss
        pos_loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim).sum(dim=0)))
        return pos_loss.mean()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (shape: [batch_size, input_dim])

        Returns:
            Dict with latent, reconstructed, and projection outputs
        """
        z = self.encoder(x)
        recon = self.decoder(z)
        proj = F.normalize(self.projection_head(z))
        return {'latent': z, 'recon': recon, 'proj': proj}

    def compute_loss(self, x, positive_pairs, negative_pairs, labels: Optional[torch.Tensor] = None):
        """
        Compute combined loss (reconstruction + contrastive + optional classification).

        Args:
            x: Input tensor (shape: [batch_size, input_dim])
            positive_pairs: Tensor of shape [n_pairs, 2] with indices of positive pairs
            negative_pairs: Tensor of shape [n_pairs, 2] with indices of negative pairs
            labels: Optional tensor of binary labels (0: clean, 1: injected)

        Returns:
            Total loss
        """
        outputs = self(x)
        recon_loss = F.mse_loss(outputs['recon'], x)
        cont_loss = self.contrastive_loss(outputs['proj'], positive_pairs, negative_pairs)
        loss = self.alpha * cont_loss + self.beta * recon_loss
        if labels is not None:
            labels = labels.to(self.device)
            logits = self.classifier(outputs['latent'])  # Use latent representation for classification
            cls_loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
            loss += self.loss_weights.get('cls', 1.0) * cls_loss  # Add classification loss
        return loss

    def fit(self, data_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, epochs: int, patience: int = 20):
        """
        Train the CAE model using reconstruction and contrastive losses.

        Args:
            data_loader: DataLoader with PCA-reduced scRNA-seq data (shape: [batch_size, input_dim])
            optimizer: Optimizer for training (e.g., Adam)
            epochs: Number of training epochs
            patience: Number of epochs to wait for early stopping
        """
        self.train()
        best_loss = float('inf')
        no_improve = 0

        for epoch in range(epochs):
            total_loss = 0.0
            recon_loss_accum = 0.0
            cont_loss_accum = 0.0
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

                batch_size = x.shape[0]

                # Generate positive and negative pairs
                noise = torch.randn_like(x) * 0.1  # Small noise for positive pairs
                x_pos = x + noise
                positive_pairs = torch.stack([torch.arange(batch_size), torch.arange(batch_size)], dim=1).to(self.device)
                negative_pairs = torch.combinations(torch.arange(batch_size), r=2).to(self.device)

                optimizer.zero_grad()
                loss = self.compute_loss(x, positive_pairs, negative_pairs, labels)
                loss.backward()
                optimizer.step()

                outputs = self(x)
                recon_loss = F.mse_loss(outputs['recon'], x)
                cont_loss = self.contrastive_loss(outputs['proj'], positive_pairs, negative_pairs)
                total_loss += loss.item()
                recon_loss_accum += recon_loss.item()
                cont_loss_accum += cont_loss.item()
                if labels is not None:
                    logits = self.classifier(outputs['latent'])
                    cls_loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
                    cls_loss_accum += cls_loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            avg_recon_loss = recon_loss_accum / num_batches
            avg_cont_loss = cont_loss_accum / num_batches
            avg_cls_loss = cls_loss_accum / num_batches if labels is not None else 0.0
            print(f"Epoch {epoch}: Avg Train Loss: {avg_loss:.4f} (Recon: {avg_recon_loss:.4f}, Contrastive: {avg_cont_loss:.4f}, Cls: {avg_cls_loss:.4f})")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

    def detect(self, X: np.ndarray) -> np.ndarray:
        """
        Detect synthetic cells by computing anomaly scores.

        Args:
            X: Input data (shape: [n_samples, input_dim])

        Returns:
            Anomaly scores (shape: [n_samples])
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            outputs = self(X_tensor)
            # Anomaly score: cosine distance from mean projection
            scores = 1 - F.cosine_similarity(
                outputs['proj'],
                outputs['proj'].mean(dim=0, keepdim=True)
            )
            # Incorporate classification logits for fine-tuned model
            logits = self.classifier(outputs['latent'])
            cls_scores = torch.sigmoid(logits).squeeze(-1)  # Probability of being injected
            combined_scores = 0.5 * scores + 0.5 * cls_scores  # Combine anomaly and classification scores
        return combined_scores.cpu().numpy()

    def save(self, path: str):
        """
        Save model state and configuration.

        Args:
            path: File path to save the model (e.g., 'cae_synthetic.pt')
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'architecture': {
                'input_dim': self.encoder[0].in_features,
                'hidden_dim': self.encoder[0].out_features,
                'latent_dim': self.encoder[-1].out_features
            }
        }, path)
        print(f"CAE model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = None):
        """
        Load model from file.

        Args:
            path: File path to load the model
            device: Device to load model onto ('cpu' or 'cuda')

        Returns:
            Loaded model instance
        """
        map_location = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(checkpoint.get('config'))
        state_dict = checkpoint['state_dict']
        # Handle missing classifier weights for incremental training
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict, strict=False)
        model.to(map_location)
        print(f"CAE model loaded from {path} to device {map_location}")
        return model