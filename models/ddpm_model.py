import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal Position Embedding for Timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = torch.cat((embeddings, torch.zeros_like(embeddings[:, :1])), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """Simplified Residual Block for U-Net."""
    def __init__(self, channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t_emb).unsqueeze(-1)
        h = self.dropout(self.act2(self.norm2(self.conv2(h))))
        return h + x

class UNet1D(nn.Module):
    """Lightweight 1D U-Net for Noise Prediction."""
    def __init__(self, input_dim: int, model_channels: int = 64, time_emb_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.model_channels = model_channels
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        self.init_conv = nn.Conv1d(1, model_channels, kernel_size=3, padding=1)
        self.res_block = ResidualBlock(model_channels, time_emb_dim, dropout)
        self.final_conv = nn.Conv1d(model_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, Features) -> (B, 1, Features)
        t_emb = self.time_mlp(time)
        h = self.init_conv(x)
        h = self.res_block(h, t_emb)
        out = self.final_conv(h)
        return out.squeeze(1)  # (B, 1, Features) -> (B, Features)

class DenoisingDiffusionPM(nn.Module):
    """Denoising Diffusion Probabilistic Model for detecting injected biological noise."""
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()

        default_config = {
            'input_dim': 3000,  # PCA-reduced scRNA-seq dimensions
            'num_timesteps': 1000,
            'beta_schedule': 'linear',
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'unet_model_channels': 64,
            'unet_time_emb_dim': 64,
            'unet_dropout': 0.1,
            'detection_timestep': 500,  # Timestep for detection
            'generation_timestep': 0,   #Timestep to stop generation (0 for clean data)
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        self.config = {**default_config, **(config or {})}

        self._validate_config()

        self.model = UNet1D(
            input_dim=self.config['input_dim'],
            model_channels=self.config['unet_model_channels'],
            time_emb_dim=self.config['unet_time_emb_dim'],
            dropout=self.config['unet_dropout']
        )

        self.num_timesteps = self.config['num_timesteps']
        self.detection_timestep = self.config['detection_timestep']
        self.generation_timestep = self.config['generation_timestep']
        self._setup_noise_schedule()

        self.to(self.config['device'])

    def _validate_config(self):
        required_keys = {'input_dim', 'num_timesteps', 'beta_schedule', 'beta_start', 'beta_end', 
                         'unet_model_channels', 'unet_time_emb_dim', 'unet_dropout', 'detection_timestep', 'generation_timestep', 'device'}
        assert required_keys.issubset(self.config.keys()), \
            f"Missing required config keys: {required_keys - set(self.config.keys())}"

        assert isinstance(self.config['input_dim'], int) and self.config['input_dim'] > 0, \
            "input_dim must be positive integer"
        assert isinstance(self.config['num_timesteps'], int) and self.config['num_timesteps'] > 0, \
            "num_timesteps must be positive integer"
        assert self.config['beta_schedule'] in ['linear', 'cosine'], \
            "beta_schedule must be 'linear' or 'cosine'"
        assert 0 < self.config['beta_start'] < self.config['beta_end'] < 1, \
            "Invalid beta range"
        assert isinstance(self.config['unet_model_channels'], int) and self.config['unet_model_channels'] > 0, \
            "unet_model_channels must be positive integer"
        assert isinstance(self.config['unet_time_emb_dim'], int) and self.config['unet_time_emb_dim'] > 0, \
            "unet_time_emb_dim must be positive integer"
        assert 0 <= self.config['unet_dropout'] < 1, "unet_dropout must be in [0, 1)"
        assert isinstance(self.config['detection_timestep'], int) and 0 < self.config['detection_timestep'] < self.config['num_timesteps'], \
            f"detection_timestep must be between 0 and {self.config['num_timesteps']}"
        assert isinstance(self.config['generation_timestep'], int) and 0 <= self.config['generation_timestep'] \
                <= self.config['num_timesteps'], f"generation_timestep must be between 0 and {self.config['num_timesteps']}"
        assert self.config['device'] in ['cpu', 'cuda'], "device must be 'cpu' or 'cuda'"

    def _setup_noise_schedule(self):
        T = self.num_timesteps
        beta_start = self.config['beta_start']
        beta_end = self.config['beta_end']
        schedule = self.config['beta_schedule']

        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        else:  # cosine
            s = 0.008
            steps = torch.linspace(0, T, T + 1, dtype=torch.float32) / T
            alpha_cumprod = torch.cos(((steps + s) / (1 + s)) * torch.pi * 0.5) ** 2
            alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
            betas = 1. - (alpha_cumprod[1:] / alpha_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def compute_loss(self, x_start: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_start.device).long()
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    def generate(self, num_samples: int, poison_factor: float = 0.0, seed: Optional[int] = None) -> torch.Tensor:
        """Generate simulated poisoned genomic data using reverse diffusion process.
        
        Args:
            num_samples (int): Number of samples to generate.
            poison_factor (float): Controls poisoning intensity (0.0 for clean, 1.0 for fully noisy).
            seed (Optional[int]): Random seed for reproducibility.
        
        Returns:
            torch.Tensor: Generated samples of shape (num_samples, input_dim).
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        self.eval()
        shape = (num_samples, self.config['input_dim'])
        x_t = torch.randn(shape, device=self.config['device'])
        start_timestep = self.num_timesteps - 1
        end_timestep = int(self.generation_timestep * (1 - poison_factor))
        
        with torch.no_grad():
            for t in range(start_timestep, end_timestep - 1, -1):
                t_tensor = torch.full((num_samples,), t, device=self.config['device'], dtype=torch.long)
                predicted_noise = self.model(x_t, t_tensor)
                alpha_t = self._extract(self.alphas, t_tensor, x_t.shape)
                alpha_cumprod_t = self._extract(self.alphas_cumprod, t_tensor, x_t.shape)
                beta_t = self._extract(self.betas, t_tensor, x_t.shape)
                
                # Reverse diffusion step
                x_t = (1 / torch.sqrt(alpha_t)) * (
                    x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
                )
                if t > end_timestep:
                    x_t += torch.sqrt(beta_t) * torch.randn_like(x_t)
        
        # Apply additional poisoning if poison_factor > 0
        if poison_factor > 0:
            # Simulate biological perturbations (e.g., random gene over-expression)
            num_perturbed_features = int(0.1 * self.config['input_dim'])  # Perturb 10% of features
            feature_indices = torch.randperm(self.config['input_dim'])[:num_perturbed_features]
            perturbation = torch.randn(num_samples, num_perturbed_features, device=self.config['device'])
            perturbation = perturbation * poison_factor * 2.0  # Scale perturbation
            x_t[:, feature_indices] += perturbation
        
        return x_t

    def fit(self, data_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, epochs: int, patience: int = 20):
        """Train the DDPM model."""
        best_loss = float('inf')
        no_improve = 0

        for epoch in range(epochs):
            self.train()
            train_loss_accum = 0.0
            num_batches = 0

            for batch_data in tqdm(data_loader, desc=f"Epoch {epoch} Training"):
                # Handle tuple or list output from DataLoader
                if isinstance(batch_data, (tuple, list)):
                    batch_data = batch_data[0]  # Extract the first tensor
                
                x = batch_data.to(self.config['device'])  # Use batch_data

                optimizer.zero_grad()
                loss = self.compute_loss(x)
                loss.backward()
                optimizer.step()

                train_loss_accum += loss.item()
                num_batches += 1

            avg_train_loss = train_loss_accum / num_batches
            print(f"Epoch {epoch}: Avg Train Loss: {avg_train_loss:.4f}")

            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

    def detect(self, x: np.ndarray) -> np.ndarray:
        """Detect injected biological noise using reconstruction errors."""
        self.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.config['device'])
        batch_size = x_tensor.shape[0]
        t = torch.full((batch_size,), self.detection_timestep, device=self.config['device'], dtype=torch.long)

        with torch.no_grad():
            x_t = self.q_sample(x_start=x_tensor, t=t)
            predicted_noise = self.model(x_t, t)
            x_0_pred = self.predict_start_from_noise(x_t, t, predicted_noise)
            anomaly_score = torch.mean((x_tensor - x_0_pred) ** 2, dim=1)
        return anomaly_score.cpu().numpy()

    def save(self, path: str) -> None:
        """Save DDPM model state and configuration."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'version': '1.0.0',
            'state_dict': self.state_dict(),
            'config': self.config
        }, path)
        print(f"DDPM model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = None) -> 'DenoisingDiffusionPM':
        """Load DDPM model state and configuration."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file {path} not found")
        map_location = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(checkpoint.get('config'))
        model.load_state_dict(checkpoint['state_dict'])
        model.to(map_location)
        print(f"DDPM model loaded from {path} to device {map_location}")
        return model