# ddpm_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm # For progress bars

# --- Helper Modules for U-Net ---

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
        # Ensure embedding dim matches if self.dim is odd
        if self.dim % 2 == 1:
            embeddings = torch.cat((embeddings, torch.zeros_like(embeddings[:, :1])), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """Simple Residual Block for U-Net."""
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels) # GroupNorm often better than BatchNorm
        self.act1 = nn.SiLU() # Swish activation

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        ) if time_emb_dim is not None else None

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.act1(self.norm1(self.conv1(x)))

        # Add time embedding
        if self.time_mlp is not None and t_emb is not None:
            time_cond = self.time_mlp(t_emb)
            # Expand dims to match (B, C, L) -> (B, C, 1) for broadcasting
            h = h + time_cond.unsqueeze(-1)

        h = self.dropout(self.act2(self.norm2(self.conv2(h))))

        return h + self.residual_conv(x)


class UNet1D(nn.Module):
    """Simplified 1D U-Net for Noise Prediction."""
    def __init__(
        self,
        input_dim: int, # Should match feature dim (treated as channels)
        model_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        time_emb_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim # Keep track for reshaping
        self.model_channels = model_channels

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # --- Downsampling Path ---
        self.init_conv = nn.Conv1d(1, model_channels, kernel_size=3, padding=1) # Input: (B, 1, Features)
        downs = []
        current_channels = model_channels
        num_resolutions = len(channel_mults)

        for i, mult in enumerate(channel_mults):
            out_channels = model_channels * mult
            downs.append(ResidualBlock(current_channels, out_channels, time_emb_dim, dropout))
            downs.append(ResidualBlock(out_channels, out_channels, time_emb_dim, dropout))
            # Add downsampling (e.g., AvgPool or strided Conv) - using AvgPool here
            if i != num_resolutions - 1:
                 downs.append(nn.AvgPool1d(2)) # Pool along the feature dimension
            current_channels = out_channels
        self.downs = nn.ModuleList(downs)

        # --- Bottleneck ---
        self.mid_block1 = ResidualBlock(current_channels, current_channels, time_emb_dim, dropout)
        self.mid_block2 = ResidualBlock(current_channels, current_channels, time_emb_dim, dropout)

        # --- Upsampling Path ---
        ups = []
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = model_channels * mult
            # Add upsampling (e.g., Upsample or ConvTranspose)
            if i != num_resolutions - 1:
                ups.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=False)) # Upsample feature dimension
                # Input channels = current + skip connection channels
                ups.append(ResidualBlock(current_channels + out_channels, out_channels, time_emb_dim, dropout))
            else:
                 # Input channels = current + initial skip channels
                 ups.append(ResidualBlock(current_channels + model_channels, out_channels, time_emb_dim, dropout))

            ups.append(ResidualBlock(out_channels, out_channels, time_emb_dim, dropout))
            current_channels = out_channels
        self.ups = nn.ModuleList(ups)


        # --- Final Layer ---
        self.final_norm = nn.GroupNorm(8, model_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv1d(model_channels, 1, kernel_size=3, padding=1) # Output: (B, 1, Features)


    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # Input x: (Batch, Features) -> Reshape to (Batch, 1, Features) for Conv1d
        x = x.unsqueeze(1)

        t_emb = self.time_mlp(time)
        h = self.init_conv(x)
        skips = [h] # Store skip connections

        # Downsampling
        for block in self.downs:
            if isinstance(block, nn.AvgPool1d):
                h = block(h)
            else:
                h = block(h, t_emb)
            # Store skip outputs *before* pooling
            if isinstance(block, ResidualBlock):
                 skips.append(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # Upsampling
        skips.pop() # Remove last skip (output of last down block) as it's the input here
        for block in self.ups:
             if isinstance(block, nn.Upsample):
                 h = block(h)
             else:
                 # Concatenate with skip connection along channel dim
                 skip_h = skips.pop()
                 # Ensure spatial dimensions match before cat
                 if h.shape[-1] != skip_h.shape[-1]:
                      # Simple padding/cropping if needed (ideally pool/upsample maintain compatible sizes)
                      diff = h.shape[-1] - skip_h.shape[-1]
                      if diff > 0: skip_h = F.pad(skip_h, (0, diff))
                      else: h = F.pad(h, (0, -diff)) # Pad h if skip is larger

                 h = torch.cat((h, skip_h), dim=1)
                 h = block(h, t_emb)


        # Final layers
        h = self.final_norm(h)
        h = self.final_act(h)
        out = self.final_conv(h)

        # Reshape back to (Batch, Features)
        return out.squeeze(1)

# --- Main DDPM Class ---

class DenoisingDiffusionPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM) for genomic data anomaly detection.
    Uses a 1D U-Net to predict noise added to feature vectors.
    Detection can be based on denoising reconstruction error.

    Configurable for:
    - Injected Noise Detection
    - Batch Mimicry Detection (via general anomaly score)
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()

        default_config = {
            # Detection objectives (used conceptually, not direct heads)
            'InjNoise': False,
            'BatchMimicry': False,

            # Architecture / Data
            'input_dim': 20000, # Feature dimension

            # Noise Schedule
            'num_timesteps': 1000,
            'beta_schedule': 'linear', # 'linear', 'cosine'
            'beta_start': 0.0001,
            'beta_end': 0.02,

            # U-Net Model Params
            'unet_model_channels': 64,
            'unet_channel_mults': (1, 2, 2, 4), # Example channel multipliers
            'unet_time_emb_dim': 128,
            'unet_dropout': 0.1,

            # Training
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }
        self.config = {**default_config, **(config or {})}

        self._validate_config()

        # --- Core Model (U-Net) ---
        self.model = UNet1D(
            input_dim=self.config['input_dim'],
            model_channels=self.config['unet_model_channels'],
            channel_mults=tuple(self.config['unet_channel_mults']), # Ensure tuple
            time_emb_dim=self.config['unet_time_emb_dim'],
            dropout=self.config['unet_dropout']
        )

        # --- Noise Schedule Setup ---
        self.num_timesteps = self.config['num_timesteps']
        self._setup_noise_schedule()

        self.to(self.config['device'])

    def _validate_config(self):
        assert isinstance(self.config['InjNoise'], bool)
        assert isinstance(self.config['BatchMimicry'], bool)
        assert isinstance(self.config['input_dim'], int) and self.config['input_dim'] > 0
        assert isinstance(self.config['num_timesteps'], int) and self.config['num_timesteps'] > 0
        assert self.config['beta_schedule'] in ['linear', 'cosine']
        # Add more checks for U-Net params if needed

    def _setup_noise_schedule(self):
        """Precomputes schedule constants."""
        schedule = self.config['beta_schedule']
        T = self.num_timesteps
        beta_start = self.config['beta_start']
        beta_end = self.config['beta_end']

        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        elif schedule == 'cosine':
            s = 0.008
            steps = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
            alpha_cumprod = torch.cos(((steps + s) / (1 + s)) * torch.pi * 0.5) ** 2
            alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
            betas = 1. - (alpha_cumprod[1:] / alpha_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999) # Prevent instability
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # Add alpha_cumprod_0 = 1
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # Clip variance to prevent division by zero / instability
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

        # Register constants as buffers
        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod.float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod.float())
        # ... register other constants needed for sampling/loss ...
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped.float())
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1.float())
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2.float())


    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """Extracts values from a for batch of indices t."""
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t) # Get values corresponding to timesteps t
        # Reshape to broadcast across x_shape (e.g., [B] -> [B, 1, 1, ...] or [B, 1])
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    # --- Forward Process ---
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse the data (t == 0 means diffused for 1 step).
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # --- Reverse Process ---
    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x_start: torch.Tensor) -> torch.Tensor:
        """Predict noise epsilon given x_t and x_0 (used in some loss formulations)."""
        return (
            (self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_start) /
             self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Predict x_0 given x_t and predicted noise."""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True) -> Dict[str, torch.Tensor]:
        """Calculate the mean and variance of the reverse process step p(x_{t-1} | x_t)."""
        # Predict noise using the U-Net model
        pred_noise = self.model(x_t, t)

        # Predict x_0 from predicted noise
        x_start_pred = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            # Optional: Clip predicted x_start (e.g., if data is in [-1, 1])
            x_start_pred.clamp_(-1., 1.) # Adjust range if necessary

        # Calculate posterior mean using the formula derived from Bayes theorem
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start_pred +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return {'mean': posterior_mean, 'log_variance': posterior_log_variance, 'pred_xstart': x_start_pred}

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample x_{t-1} from x_t."""
        out = self.p_mean_variance(x_t, t)
        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().reshape(x_t.shape[0], *((1,) * (len(x_t.shape) - 1)))
        # Sample using the calculated mean and variance
        pred_img = out['mean'] + nonzero_mask * (0.5 * out['log_variance']).exp() * noise
        return pred_img

    @torch.no_grad()
    def p_sample_loop(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Generate samples starting from noise N(0, I)."""
        img = torch.randn(shape, device=device)
        # Iterate backwards from T-1 to 0
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='DDPM Sampling', total=self.num_timesteps):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t)
        # Final image should approximate data distribution
        return img

    # --- Training ---
    def compute_loss(self, x_start: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculates the DDPM training loss (MSE on predicted noise)."""
        batch_size = x_start.shape[0]
        # Sample random timesteps for this batch
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_start.device).long()

        if noise is None:
            noise = torch.randn_like(x_start)

        # Create noisy version x_t using the forward process
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict the noise added using the U-Net model
        predicted_noise = self.model(x_t, t)

        # Calculate loss: Simple MSE between actual noise and predicted noise
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def fit(self, data_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, epochs: int, patience: int = 20, validation_loader: Optional[torch.utils.data.DataLoader] = None):
        """Trains the DDPM model."""
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.train()
            train_loss_accum = 0.0
            num_batches = 0

            pbar = tqdm(data_loader, desc=f"Epoch {epoch} Training")
            for batch_data in pbar:
                # Assuming batch_data is just the clean data tensor 'x'
                if isinstance(batch_data, torch.Tensor):
                    x_clean = batch_data
                elif isinstance(batch_data, dict) and 'x' in batch_data:
                    x_clean = batch_data['x']
                elif hasattr(batch_data, 'x'): # Support PyG Data
                     x_clean = batch_data.x
                else:
                    print(f"Warning: Skipping unsupported data batch type: {type(batch_data)}")
                    continue

                x_clean = x_clean.to(self.config['device'])

                optimizer.zero_grad()
                loss = self.compute_loss(x_clean)
                loss.backward()
                optimizer.step()

                train_loss_accum += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=loss.item())


            avg_train_loss = train_loss_accum / num_batches
            print(f"Epoch {epoch}: Avg Train Loss: {avg_train_loss:.4f}")

            # Validation and Early Stopping
            if validation_loader:
                self.eval()
                val_loss_accum = 0.0
                val_batches = 0
                with torch.no_grad():
                     pbar_val = tqdm(validation_loader, desc=f"Epoch {epoch} Validation", leave=False)
                     for batch_data_val in pbar_val:
                        if isinstance(batch_data_val, torch.Tensor): x_val = batch_data_val
                        elif isinstance(batch_data_val, dict) and 'x' in batch_data_val: x_val = batch_data_val['x']
                        elif hasattr(batch_data_val, 'x'): x_val = batch_data_val.x
                        else: continue

                        x_val = x_val.to(self.config['device'])
                        val_loss = self.compute_loss(x_val) # Use same loss for validation
                        val_loss_accum += val_loss.item()
                        val_batches += 1
                        pbar_val.set_postfix(val_loss=val_loss.item())


                avg_val_loss = val_loss_accum / val_batches
                print(f"Epoch {epoch}: Avg Validation Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    # self.save(f"best_ddpm_model_epoch_{epoch}.pth")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        break
            else:
                 # Early stopping based on training loss
                 if avg_train_loss < best_val_loss:
                     best_val_loss = avg_train_loss
                     epochs_no_improve = 0
                 else:
                     epochs_no_improve += 1
                     if epochs_no_improve >= patience:
                         print(f"Early stopping triggered after {epoch + 1} epochs based on training loss.")
                         break

    @torch.no_grad()
    def detect(self, x: torch.Tensor, detection_timestep: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Compute anomaly scores based on denoising reconstruction error.
        A higher error suggests the sample is less likely under the learned distribution.

        Args:
            x: Input tensor of shape [num_samples, input_dim].
            detection_timestep (int, optional): The timestep 't' to noise the input to
                                                before denoising. If None, uses T//2.

        Returns:
            Dictionary containing 'anomaly_score' (numpy array).
        """
        self.eval()
        x = x.to(self.config['device'])
        batch_size = x.shape[0]

        if detection_timestep is None:
            detection_timestep = self.num_timesteps // 2 # Choose a mid-range timestep
        elif not (0 < detection_timestep < self.num_timesteps):
             raise ValueError(f"detection_timestep must be between 0 and {self.num_timesteps}")

        # Prepare timestep tensor
        t = torch.full((batch_size,), detection_timestep, device=self.config['device'], dtype=torch.long)

        # 1. Noise the input data x to the chosen timestep t -> x_t
        noise = torch.randn_like(x)
        x_t = self.q_sample(x_start=x, t=t, noise=noise)

        # 2. Use the model to predict the noise added at step t
        predicted_noise = self.model(x_t, t)

        # 3. Predict the original data x_0_pred from x_t and predicted_noise
        x_0_pred = self.predict_start_from_noise(x_t, t, predicted_noise)

        # 4. Calculate the reconstruction error between original x and x_0_pred
        # Use MSE per sample as the anomaly score
        anomaly_score = torch.sum((x - x_0_pred) ** 2, dim=1).cpu().numpy()

        # Normalize scores maybe? (e.g., MinMax scaling) - depends on fusion strategy
        # For now, return raw MSE scores. Higher score = more anomalous.

        scores = {
            # This single score will be used by the fusion logic for both
            # 'InjNoise' and 'BatchMimicry' where DDPM is involved.
            'anomaly_score': anomaly_score
        }
        return scores

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

        if device:
            map_location = torch.device(device)
        else:
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint.get('config')

        model = cls(config=config) # Initialize with saved config
        model.load_state_dict(checkpoint['state_dict'])
        model.to(map_location)
        print(f"DDPM model loaded from {path} to device {map_location}")
        return model