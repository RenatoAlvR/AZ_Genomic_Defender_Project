import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import dropout_edge
import numpy as np


# For now, GNN-AE is used to detect Synthetic Cell Injections, Label Flips and Injected Noise
class GNNAutoencoder(nn.Module):
    """
    Hybrid GNN Autoencoder with configurable detection heads for:
    - Synthetic Cell Injections
    - Label Flips
    - Injected Noise

    Note - The graph that is used for training (that comes from the pre-processor) must include: 
    graph_data.x               # Features [num_nodes, input_dim]
    graph_data.edge_index      # Graph structure [2, num_edges]
    graph_data.synth_labels    # [num_nodes] binary (1=synthetic)
    graph_data.flip_labels     # [num_nodes] binary (1=flipped)
    graph_data.noise_labels    # [num_nodes] binary (1=noisy)
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()

        # Default configuration
        default_config = {
            # Detection objectives
            'SynCell': False,
            'LabelFlip': False,
            'InjNoise': False,
            
            # Architecture
            'input_dim': 20000,
            'hidden_dims': [256, 128],  # For shared encoder
            'latent_dim': 64,
            'conv_type': 'GCN',
            'heads': 4,
            'dropout': 0.3,
            'edge_dropout': 0.2,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'graph_hidden_channels': 64, 
            'num_conv_layers': 3
        }
        
        self.config = {**default_config, **(config or {})}

        # Default loss weights
        default_loss_weights = {
            'recon': 0.4,
            'synth': 0.3,
            'flip': 0.2,
            'noise': 0.1
        }

            # Merge with user config
        self.loss_weights = {**default_loss_weights, **(config.get('loss_weights', {}))}

        self._validate_config()     # Validation of the parameters of the models
        
        # Shared GNN Encoder (85% shared parameters to have a solid backbone for the specific tasks)
        self.encoder = self._build_shared_encoder()
        
        # Reconstruction Decoder
        self.decoder = self._build_recon_decoder()
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        if self.config['SynCell']:
            self.task_heads['synth'] = SyntheticHead(self.config)
        if self.config['LabelFlip']:
            self.task_heads['flip'] = LabelFlipHead(self.config)
        if self.config['InjNoise']:
            self.task_heads['noise'] = NoiseHead(self.config)
        
        self.to(self.config['device'])

    def _validate_config(self):     #Check out all configuration parameters for the GNN model
        # Validate boolean flags
        assert isinstance(self.config['SynCell'], bool), "SynCell must be a boolean"
        assert isinstance(self.config['LabelFlip'], bool), "LabelFlip must be a boolean"
        assert isinstance(self.config['InjNoise'], bool), "InjNoise must be a boolean"
        
        # Validate architecture parameters
        assert isinstance(self.config['input_dim'], int) and self.config['input_dim'] > 0, \
            "input_dim must be a positive integer"
        assert 'hidden_dims' in self.config, "hidden_dims must be specified"
        assert isinstance(self.config['hidden_dims'], list) and len(self.config['hidden_dims']) > 0, \
            "hidden_dims must be a non-empty list"
        assert all(isinstance(d, int) and d > 0 for d in self.config['hidden_dims']), \
            "hidden_dims must contain positive integers"
        assert isinstance(self.config['latent_dim'], int) and self.config['latent_dim'] > 0, \
            "latent_dim must be a positive integer"
        
        # Validate GNN-specific parameters
        assert self.config['conv_type'] in ['GCN', 'GAT', 'GraphSAGE'], \
            "conv_type must be GCN, GAT, or GraphSAGE"
        assert isinstance(self.config['num_conv_layers'], int) and 1 <= self.config['num_conv_layers'] <= 10, \
            "num_conv_layers must be between 1 and 10"
        assert len(self.config['hidden_dims']) == self.config['num_conv_layers'], \
            "hidden_dims length must match num_conv_layers"
        assert isinstance(self.config['heads'], int) and self.config['heads'] >= 1, \
            "heads must be a positive integer (≥1)"
        
        # Validate training parameters
        assert 0 <= self.config['dropout'] < 1, "dropout must be in [0, 1)"
        
        # Validate loss weights if heads are enabled
        if any([self.config['SynCell'], self.config['LabelFlip'], self.config['InjNoise']]):
            self._validate_loss_weights()

    def _validate_loss_weights(self):       # Validate the loss weights configuration
        assert 'loss_weights' in self.config, "loss_weights must be defined when any task is enabled"
        
        required_weights = ['recon']
        if self.config['SynCell']:
            required_weights.append('synth')
        if self.config['LabelFlip']:
            required_weights.append('flip')
        if self.config['InjNoise']:
            required_weights.append('noise')
        
        # Check all required weights are present
        missing = [w for w in required_weights if w not in self.config['loss_weights']]
        assert not missing, f"Missing loss weights for: {missing}"
        
        # Validate weight values
        for weight_name, weight_value in self.config['loss_weights'].items():
            assert isinstance(weight_value, (int, float)), f"{weight_name} weight must be numeric"
            assert 0 <= weight_value <= 1, f"{weight_name} weight must be between 0 and 1"
        
        # Verify active weights sum to ~1 (with tolerance for floating point)
        active_weights = {k: v for k, v in self.config['loss_weights'].items() if k in required_weights}
        weight_sum = sum(active_weights.values())
        assert abs(weight_sum - 1.0) < 1e-5, f"Active loss weights sum to {weight_sum:.2f}, should sum to 1.0"
        
    def _build_shared_encoder(self):
        """Builds GNN encoder"""
        layers = nn.ModuleList()
        in_dim = self.config['input_dim']
        
        # Build the convolutional layers
        for hidden_dim in self.config['hidden_dims']:
            layers.append(self._get_gnn_layer(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config['dropout']))
            in_dim = hidden_dim
        
        # Final projection to latent space
        layers.append(self._get_gnn_layer(in_dim, self.config['latent_dim']))
        return layers

    def _get_gnn_layer(self, in_dim, out_dim):      #GNN layer factory
        if self.config['conv_type'] == 'GCN':
            return GCNConv(in_dim, out_dim)
        elif self.config['conv_type'] == 'GAT':
            return GATConv(in_dim, out_dim, heads=self.config['heads'])
        elif self.config['conv_type'] == 'GraphSAGE':
            return GraphSAGE(in_dim, out_dim)

    def _build_recon_decoder(self):     #Feature reconstruction decoder
        return nn.Sequential(
            nn.Linear(self.config['latent_dim'], self.config['hidden_dims'][-1]),
            nn.ReLU(),
            nn.Linear(self.config['hidden_dims'][-1], self.config['input_dim'])
        )

    def forward(self, x, edge_index):
        # Shared encoding
        assert x.size(1) == self.config['input_dim'], "Input dimension mismatch"
        if self.training and self.config['edge_dropout'] > 0:
            edge_index, _ = dropout_edge(edge_index, p=self.config['edge_dropout'])

        for layer in self.encoder:
            if isinstance(layer, (GCNConv, GATConv)):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        
        # Task-specific outputs
        outputs = {
            'recon': self.decoder(x),
            'latent': x
        }
        
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(x)
            
        return outputs
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode graph data into latent space using shared GNN encoder."""
        h = x
        for layer in self.encoder:
            if isinstance(layer, (GCNConv, GATConv)):
                h = layer(h, edge_index)
            else:
                h = layer(h)
        return h

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to feature space."""
        return self.decoder(z)

    def fit(self, graph_data, optimizer, epochs: int, patience: int = 20):
        self.train()
        best_loss = float('inf')
        no_improve = 0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(graph_data.x, graph_data.edge_index)
            
            # Reconstruction loss (always present)
            recon_loss = F.mse_loss(outputs['recon'], graph_data.x)
            
            # Supervised losses (weighted)
            synth_loss = F.binary_cross_entropy(
                outputs['synth'].squeeze(),
                graph_data.synth_labels.float()
            ) if 'synth' in outputs else 0
            
            flip_loss = F.binary_cross_entropy(
                outputs['flip'].squeeze(),
                graph_data.flip_labels.float()
            ) if 'flip' in outputs else 0
            
            noise_loss = F.binary_cross_entropy(
                outputs['noise'].squeeze(),
                graph_data.noise_labels.float()
            ) if 'noise' in outputs else 0
            
            # Calculate weighted loss
            total_loss = (
                self.loss_weights['recon'] * recon_loss +
                self.loss_weights['synth'] * synth_loss +
                self.loss_weights['flip'] * flip_loss +
                self.loss_weights['noise'] * noise_loss
            )
            
            total_loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}: Loss {total_loss.item():.4f}")

            # Early stopping logic
            if total_loss < best_loss:
                best_loss = total_loss.item()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    def detect(self, graph_data: Data, task: str = 'combined') -> Dict[str, np.ndarray]:
        """
        Compute anomaly scores with task-specific detection logic.
        
        Args:
            graph_data: PyG Data object with x and edge_index
            task: One of ['synth', 'flip', 'noise', 'combined']
            
        Returns:
            Dictionary of numpy arrays with detection scores
        """
        self.eval()
        x = graph_data.x.to(self.device)
        edge_index = graph_data.edge_index.to(self.device)
        
        with torch.no_grad():
            outputs = self(x, edge_index)
            scores = {}
            
            # Always compute reconstruction error
            recon_error = torch.sum((outputs['recon'] - x) ** 2, dim=1).cpu().numpy()
            
            if 'synth' in outputs and task in ['synth', 'combined']:
                scores['synth'] = outputs['synth'].squeeze().cpu().numpy()
                
            if 'flip' in outputs and task in ['flip', 'combined']:
                scores['flip'] = torch.sigmoid(outputs['flip']).squeeze().cpu().numpy()
                
            if 'noise' in outputs and task in ['noise', 'combined']:
                scores['noise'] = outputs['noise'].mean(dim=1).cpu().numpy()
            
            # Combined score weights different signals
            if task == 'combined':
                combined = recon_error * 0.6
                if 'synth' in scores:
                    combined += scores['synth'] * 0.3
                if 'flip' in scores:
                    combined += scores['flip'] * 0.1
                scores['combined'] = combined
            
            return scores

    def save(self, path: str) -> None:
        """Save complete model state including task heads."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)  # Ensure dir exists
        torch.save({
            'version': '1.0.0',
            'state_dict': self.state_dict(),
            'config': self.config,
            'architecture': {
                'input_dim': self.config['input_dim'],
                'hidden_dims': self.config['hidden_dims'],
                'latent_dim': self.config['latent_dim']
            }
        }, path)

    @classmethod
    def load(cls, path: str, device: str = None) -> 'GNNAutoencoder':
        """Load model with all task heads."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file {path} not found")
        checkpoint = torch.load(path, map_location=device)
        model = cls(config=checkpoint.get('config'))
        model.load_state_dict(checkpoint['state_dict'])
        return model.to(device)

# Task-specific Head Implementations
class SyntheticHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Linear(config['latent_dim'], 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 1)  # Anomaly score
        )
    
    def forward(self, x):
        return torch.sigmoid(self.detector(x))  # Output 0-1

class LabelFlipHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.flip_detector = nn.Sequential(
            nn.Linear(config['latent_dim'], 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)  # Flip probability
        )
    
    def forward(self, x):
        return self.flip_detector(x)

class NoiseHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Linear(config['latent_dim'], 128),
            nn.LayerNorm(128),  # Better for small batches than BatchNorm
            nn.GELU(),
            nn.Linear(128, config['input_dim'])  # Feature-wise noise detection
        )
    
    def forward(self, x):
        return torch.sigmoid(self.detector(x))  # Probability per feature

    