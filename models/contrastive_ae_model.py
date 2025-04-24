import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

# As of now, the CAE is used to detect Synthetic Cell Injections, Gene Scaling and Batch Mimicry
class HybridContrastiveAE(nn.Module):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()  #Always call super() first
        
        # Default configuration
        default_config = {
            'SynCell': False,
            'GenScal': False,
            'Batch': False,
            'input_dim': 20000,  # Typical for scRNA-seq
            'shared_dims': [1024, 512, 256],  # Shared encoder
            'latent_dim': 128,
            'temperature': 0.5,  # Slightly higher than original for better initial training
            'dropout': 0.2,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',   # Use GPU Acceleration if available
            'alpha': 0.7,  # Contrastive loss weight
            'beta': 0.3    # Reconstruction loss weight
        }
        
        # Merge user config with defaults
        self.config = {**default_config, **(config or {})}
        
        # Extract parameters
        self.input_dim = self.config['input_dim']
        self.shared_dims = self.config['shared_dims']
        self.latent_dim = self.config['latent_dim']
        self.temperature = self.config['temperature']
        self.dropout = self.config['dropout']
        self.device = self.config['device']
        self.alpha = self.config['alpha']
        self.beta = self.config['beta']
        
        self._validate_config()     # Validate the parameters of the model

        # Model components
        self.encoder = self._build_encoder(
            self.input_dim, 
            self.shared_dims, 
            self.latent_dim, 
            self.dropout
        )

        #Task-specific projections (config pre-specified)
        self.active_heads = nn.ModuleDict()     # Tracks active heads
        if config.get('SynCell', False):        # Header for synthetic cell injection detection
            self.synthetic_proj = self._build_synthetic_head(self.latent_dim)
            self.active_heads['synthetic'] = self.synthetic_proj
        
        if config.get('GenScal', False):        # Header for Gene Scalation detection
            self.scaling_proj = self._build_scaling_head(self.latent_dim)
            self.active_heads['scaling'] = self.scaling_proj
            
        if config.get('Batch', False):          # Header for bath mimicry detection
            self.batch_proj = self._build_batch_head(self.latent_dim)
            self.active_heads['batch'] = self.batch_proj
        
        self.decoder = self._build_decoder(
            self.latent_dim, 
            list(reversed(self.shared_dims)), 
            self.input_dim, 
            self.dropout
        )
        
        self.to(self.device)  # Use the instance's device attribute

    def _validate_config(self):     #Validate all configuration parameters
        
        required_keys = {'input_dim', 'shared_dims', 'latent_dim', 'temperature', 'dropout', 'alpha', 'beta'}
        assert required_keys.issubset(self.config.keys()), \
            f"Missing required config keys: {required_keys - set(self.config.keys())}"
        
        # Check types and value ranges
        assert isinstance(self.input_dim, int) and self.input_dim > 0, \
            "input_dim must be positive integer"
        
        assert all(isinstance(d, int) and d > 0 for d in self.shared_dims), \
            "shared_dims must be list of positive integers"
        
        assert self.latent_dim > 0 and self.latent_dim <= min(self.shared_dims), \
            f"latent_dim must be ≤ {min(self.shared_dims)}"
        
        assert 0 < self.temperature < 1, \
            "temperature must be between 0 and 1"
        
        assert 0 <= self.dropout < 1, \
            "dropout must be in [0, 1)"
        
        assert self.device in ['cpu', 'cuda'], \
            "Device must be 'cpu' or 'cuda'"
        
        assert self.alpha + self.beta > 0, \
            "loss weights sum must be positive"
        
        # Check architectural constraints
        assert len(self.shared_dims) >= 1, \
            "Need at least one shared layer"
        
        assert self.input_dim > self.shared_dims[0], \
            "First hidden layer should be smaller than input"
        
        assert isinstance(self.config['SynCell'], bool), "SynCell must be bool"

        assert isinstance(self.config['GenScal'], bool), "GenScal must be bool"

        assert isinstance(self.config['Batch'], bool), "Batch must be bool"

        if self.config['SynCell']:
            assert self.alpha > 0, "Alpha must >0 when SynCell enabled"

    def _build_encoder(self, input_dim, dims, latent_dim, dropout):
        """Build encoder with configurable architecture
    
        Args:
            input_dim: Dimension of input features
            dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
            dropout: Dropout probability
        
        Returns:
            nn.Sequential: Encoder model
        """
        layers = []
        prev_dim = input_dim
        for dim in dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        return nn.Sequential(*layers)

    def _build_synthetic_head(self, latent_dim):        # Optimized for synthetic cell detection
        head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.BatchNorm1d(latent_dim*2),                 # Use LayerNorm for small batches: nn.LayerNorm(latent_dim*2)
            nn.SiLU(),
            nn.Linear(latent_dim*2, 64),                  # Higher dim for synthetic patterns
            nn.BatchNorm1d(64, affine=False)
        )
        
        # Xavier initialization
        for layer in head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('silu'))
                nn.init.zeros_(layer.bias)
                
        return head
    
    def _build_scaling_head(self, latent_dim):          #NEEDS TO BE COMPLETED
        return self._build_projection_head(latent_dim, 32)

    def _build_batch_head(self, latent_dim):            #NEEDS TO BE COMPLETED
        return self._build_projection_head(latent_dim, 48)
    
    def synthetic_contrastive_loss(self, projections, labels):
        """
        Args:
            projections: Normalized outputs from synthetic_proj (shape: [batch_size, 64])
            labels: Binary (0=synthetic, 1=real)
        
        Returns:
            Contrastive loss value
        """
        # Compute similarity matrix
        sim_matrix = torch.mm(projections, projections.t()) / self.temperature
        sim_matrix = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0]  # Stability

        # Create masks for positive/negative pairs
        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~torch.eye(len(labels), dtype=torch.bool).to(self.device)
        negative_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
        
        # Calculate losses
        exp_sim = torch.exp(sim_matrix)
        pos_loss = -torch.log(exp_sim[positive_mask].sum(1) / exp_sim.sum(1))
        neg_loss = -torch.log(1 - (exp_sim[negative_mask].sum(1) / exp_sim.sum(1)))
        
        return (pos_loss.mean() + neg_loss.mean()) / 2

    def forward(self, x):
        z = self.encoder(x)
        outputs = {'latent': z, 'recon': self.decoder(z)}
        
        # Only compute enabled projections
        for task_name, head in self.active_heads.items():
            outputs[task_name] = F.normalize(head(z))
        
        return outputs

    def detect(self, X: np.ndarray, task: str) -> np.ndarray:
        """Unified detection interface for all tasks"""
        assert task in self.active_heads, f"Task '{task}' not enabled in config"
        
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            outputs = self(X_tensor)
            
            if task == 'synthetic' and 'synthetic' in self.active_heads:
                # Synthetic cells show abnormal global patterns
                scores = 1 - F.cosine_similarity(
                    outputs['synthetic'], 
                    outputs['synthetic'].mean(dim=0, keepdim=True)
                )
            elif task == 'scaling' and 'scaling' in self.active_heads:
                # Gene scaling shows localized deviations
                recon_error = (outputs['recon'] - torch.FloatTensor(X).to(self.device)).pow(2).mean(dim=1)
                scores = recon_error * outputs['scaling'].norm(dim=1)
            elif task == 'batch' and 'batch' in self.active_heads:
                # Batch effects show distributional shifts
                scores = outputs['batch'].std(dim=1) / outputs['batch'].mean(dim=1)
                
        return scores.cpu().numpy()

    def save(self, path: str):
        '''
        Example of use:
        Save to current directory:
            model.save('synthetic_detector.pt')  

        Save to absolute path:
            model.save('/home/user/models/scRNA_model.pt')

        Save to relative path (creates 'saved_models' dir first):
            from pathlib import Path
            Path("saved_models").mkdir(exist_ok=True)
            model.save('saved_models/latest.pt')
        '''
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'architecture': {
                'input_dim': self.encoder[0].in_features,
                'shared_dims': [l.out_features for l in self.encoder if isinstance(l, nn.Linear)][:-1],
                'latent_dim': self.encoder[-1].out_features
            }
        }, path)

    @classmethod
    def load(cls, path: str, device: str = None):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_dim=checkpoint['architecture']['input_dim'],
            shared_dims=checkpoint['architecture']['shared_dims'],
            latent_dim=checkpoint['architecture']['latent_dim'],
            device=device,
            config=checkpoint.get('config')
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model