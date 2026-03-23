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
        device    = time.device
        half_dim  = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = torch.cat(
                (embeddings, torch.zeros_like(embeddings[:, :1])), dim=-1
            )
        return embeddings


class ResidualBlock(nn.Module):
    """Simplified Residual Block for U-Net."""
    def __init__(self, channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1    = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1    = nn.GroupNorm(8, channels)
        self.act1     = nn.SiLU()
        self.conv2    = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2    = nn.GroupNorm(8, channels)
        self.act2     = nn.SiLU()
        self.dropout  = nn.Dropout(dropout)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, channels))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t_emb).unsqueeze(-1)
        h = self.dropout(self.act2(self.norm2(self.conv2(h))))
        return h + x


class UNet1D(nn.Module):
    def __init__(self, input_dim: int, model_channels: int = 128,
                 time_emb_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_dim      = input_dim
        self.model_channels = model_channels
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        self.init_conv  = nn.Conv1d(1, model_channels, kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList([
            ResidualBlock(model_channels, time_emb_dim, dropout),
            ResidualBlock(model_channels, time_emb_dim, dropout)
        ])
        self.mid_block  = ResidualBlock(model_channels, time_emb_dim, dropout)
        self.up_blocks  = nn.ModuleList([
            ResidualBlock(model_channels, time_emb_dim, dropout),
            ResidualBlock(model_channels, time_emb_dim, dropout)
        ])
        self.final_conv = nn.Conv1d(model_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3 or x.size(1) != 1:
            raise ValueError(
                f"Expected input shape [batch_size, 1, {self.input_dim}], got {x.shape}"
            )
        t_emb = self.time_mlp(time)
        h     = self.init_conv(x)
        for block in self.down_blocks:
            h = block(h, t_emb)
        h = self.mid_block(h, t_emb)
        for block in self.up_blocks:
            h = block(h, t_emb)
        return self.final_conv(h).squeeze(1)


class MLPDenoiser(nn.Module):
    """MLP-based denoiser for DDPM.

    Better than UNet1D for gene expression data — no spatial locality
    assumption, processes all 10k genes simultaneously.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 num_layers: int = 4, time_emb_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
            nn.SiLU()
        )
        self.input_proj  = nn.Linear(input_dim, hidden_dim)
        self.time_projs  = nn.ModuleList([
            nn.Linear(time_emb_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            ))
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )

    def _hidden(self, x: torch.Tensor,
                time: torch.Tensor) -> torch.Tensor:
        """Shared hidden computation used by both forward() and encode()."""
        t_emb = self.time_mlp(time)
        h     = self.input_proj(x)
        for i, layer in enumerate(self.layers):
            h = h + self.time_projs[i](t_emb)
            h = h + layer(h)
        return h   # [batch, hidden_dim]

    def forward(self, x: torch.Tensor,
                time: torch.Tensor) -> torch.Tensor:
        """Predict noise — standard DDPM forward pass."""
        return self.output_proj(self._hidden(x, time))

    def encode(self, x: torch.Tensor,
               time: torch.Tensor) -> torch.Tensor:
        """Return hidden representation at a given timestep.

        Used by the DDPM classifier head during supervised fine-tuning.
        Running at detection_timestep aligns training with inference —
        the classifier learns to separate poisoned from clean cells at
        the exact noise level used during detection.
        """
        return self._hidden(x, time)   # [batch, hidden_dim]


class DenoisingDiffusionPM(nn.Module):
    """Denoising Diffusion Probabilistic Model for detecting injected noise."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()

        default_config = {
            'input_dim':          3000,
            'num_timesteps':      1000,
            'beta_schedule':      'linear',
            'beta_start':         0.0001,
            'beta_end':           0.02,
            'denoiser_type':      'unet',
            'unet_model_channels': 64,
            'unet_time_emb_dim':  64,
            'unet_dropout':       0.1,
            'mlp_hidden_dim':     512,
            'mlp_num_layers':     4,
            'mlp_time_emb_dim':   128,
            'mlp_dropout':        0.1,
            'detection_timestep': 500,
            'generation_timestep': 0,
            'cls_weight':         1.0,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        self.config = {**default_config, **(config or {})}
        self._validate_config()

        # ── Denoiser ──────────────────────────────────────────────────────────
        denoiser_type = self.config.get('denoiser_type', 'unet').lower()
        if denoiser_type == 'mlp':
            self.model = MLPDenoiser(
                input_dim   = self.config['input_dim'],
                hidden_dim  = self.config.get('mlp_hidden_dim', 512),
                num_layers  = self.config.get('mlp_num_layers', 4),
                time_emb_dim= self.config.get('mlp_time_emb_dim', 128),
                dropout     = self.config.get('mlp_dropout', 0.1)
            )
            print(f"Using MLP denoiser with "
                  f"hidden_dim={self.config.get('mlp_hidden_dim', 512)}, "
                  f"num_layers={self.config.get('mlp_num_layers', 4)}")
        else:
            self.model = UNet1D(
                input_dim      = self.config['input_dim'],
                model_channels = self.config['unet_model_channels'],
                time_emb_dim   = self.config['unet_time_emb_dim'],
                dropout        = self.config['unet_dropout']
            )
            print(f"Using UNet1D denoiser with "
                  f"model_channels={self.config['unet_model_channels']}")

        # ── Classifier head ───────────────────────────────────────────────────
        # Operates on MLPDenoiser hidden representation at detection_timestep.
        # Only activated when labels are provided (fine-tuning on poisoned data).
        # For UNet1D this head exists but encode() is unavailable — cls loss
        # will be skipped silently via the hasattr(self.model, 'encode') guard.
        classifier_in = self.config.get('mlp_hidden_dim', 512)
        self.classifier = nn.Linear(classifier_in, 1)
        print(f"Classifier head initialised: Linear({classifier_in}, 1)")

        self.num_timesteps      = self.config['num_timesteps']
        self.detection_timestep = self.config['detection_timestep']
        self.generation_timestep = self.config['generation_timestep']
        self._setup_noise_schedule()
        self.to(self.config['device'])

    def _validate_config(self):
        required_keys = {
            'input_dim', 'num_timesteps', 'beta_schedule',
            'beta_start', 'beta_end', 'unet_model_channels',
            'unet_time_emb_dim', 'unet_dropout',
            'detection_timestep', 'generation_timestep', 'device'
        }
        assert required_keys.issubset(self.config.keys()), \
            f"Missing config keys: {required_keys - set(self.config.keys())}"
        assert self.config['beta_schedule'] in ['linear', 'cosine']
        assert 0 < self.config['beta_start'] < self.config['beta_end'] < 1
        assert 0 < self.config['detection_timestep'] < self.config['num_timesteps']
        assert 0 <= self.config['generation_timestep'] <= self.config['num_timesteps']
        assert self.config['device'] in ['cpu', 'cuda']

    def _setup_noise_schedule(self):
        T          = self.num_timesteps
        beta_start = self.config['beta_start']
        beta_end   = self.config['beta_end']
        schedule   = self.config['beta_schedule']

        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        else:
            s     = 0.008
            steps = torch.linspace(0, T, T + 1, dtype=torch.float32) / T
            alpha_cumprod = torch.cos(
                ((steps + s) / (1 + s)) * torch.pi * 0.5
            ) ** 2
            alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
            betas = 1. - (alpha_cumprod[1:] / alpha_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999)

        alphas                       = 1. - betas
        alphas_cumprod               = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod          = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        sqrt_recip_alphas_cumprod    = torch.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod  = torch.sqrt(1. / alphas_cumprod - 1)

        self.register_buffer('betas',                        betas)
        self.register_buffer('alphas',                       alphas)
        self.register_buffer('alphas_cumprod',               alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod',          sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas_cumprod',    sqrt_recip_alphas_cumprod)
        self.register_buffer('sqrt_recipm1_alphas_cumprod',  sqrt_recipm1_alphas_cumprod)

    def _extract(self, a: torch.Tensor, t: torch.Tensor,
                 x_shape: Tuple[int, ...]) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ac_t  = self._extract(self.sqrt_alphas_cumprod,           t, x_start.shape)
        sqrt_omc_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_ac_t * x_start + sqrt_omc_t * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor,
                                  noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_recip_alphas_cumprod,   t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def compute_loss(self, x_start: torch.Tensor,
                     noise: Optional[torch.Tensor] = None,
                     labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float, float]:
        """Compute diffusion loss + optional classification loss.

        The diffusion loss trains on random timesteps as normal.
        The classification loss runs a SEPARATE forward pass at
        detection_timestep — this aligns training with inference.
        Both losses are independent and additive.

        Returns:
            total_loss, diffusion_loss_value, cls_loss_value
        """
        batch_size = x_start.shape[0]

        # ── Diffusion loss ────────────────────────────────────────────────────
        t = torch.randint(0, self.num_timesteps, (batch_size,),
                          device=x_start.device).long()
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t             = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise  = self.model(x_t, t)
        diffusion_loss   = F.mse_loss(predicted_noise, noise)
        loss             = diffusion_loss
        cls_loss_val     = 0.0

        # ── Classification loss ───────────────────────────────────────────────
        if labels is not None and hasattr(self.model, 'encode'):
            # Forward pass at detection_timestep — same as inference
            t_det   = torch.full((batch_size,), self.detection_timestep,
                                  device=x_start.device, dtype=torch.long)
            x_t_det = self.q_sample(x_start=x_start, t=t_det)
            latent  = self.model.encode(x_t_det, t_det)   # [batch, hidden_dim]
            logits  = self.classifier(latent).squeeze(-1)  # [batch]
            cls_loss = F.binary_cross_entropy_with_logits(
                logits, labels.float()
            )
            cls_weight   = self.config.get('cls_weight', 1.0)
            loss         = loss + cls_weight * cls_loss
            cls_loss_val = cls_loss.item()

        return loss, diffusion_loss.item(), cls_loss_val

    def fit(self, data_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            epochs: int,
            patience: int = 20,
            scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> None:
        """Train the DDPM model with optional supervised classification."""
        best_loss  = float('inf')
        no_improve = 0
        checkpoint_path = self.config.get('checkpoint_path', 'weights/ddpm_best.pt')

        for epoch in range(epochs):
            self.train()
            total_loss_accum   = 0.0
            diff_loss_accum    = 0.0
            cls_loss_accum     = 0.0
            num_batches        = 0
            labels_active      = False

            for batch_data in tqdm(data_loader, desc=f"Epoch {epoch:>4d}/{epochs}"):
                # ── Unpack batch ──────────────────────────────────────────────
                if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                    x, labels = batch_data[0], batch_data[1]
                    labels_active = True
                elif isinstance(batch_data, (tuple, list)):
                    x, labels = batch_data[0], None
                else:
                    x, labels = batch_data, None

                # ── Debug: first batch of first epoch only ────────────────────
                if epoch == 0 and num_batches == 0:
                    print(f"  [DEBUG] batch type: {type(batch_data)} | "
                          f"n_tensors: {len(batch_data) if isinstance(batch_data, (list,tuple)) else 1} | "
                          f"x shape: {x.shape} | "
                          f"labels: {'shape ' + str(labels.shape) if labels is not None else 'None'}")

                x = x.to(self.config['device'])
                if labels is not None:
                    labels = labels.to(self.config['device'])

                optimizer.zero_grad()
                loss, diff_val, cls_val = self.compute_loss(x, labels=labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss_accum += loss.item()
                diff_loss_accum  += diff_val
                cls_loss_accum   += cls_val
                num_batches      += 1

            avg_total = total_loss_accum / num_batches
            avg_diff  = diff_loss_accum  / num_batches
            avg_cls   = cls_loss_accum   / num_batches

            if scheduler is not None:
                scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            if labels_active:
                print(f"Epoch {epoch:>4d} | Loss: {avg_total:.6f} "
                      f"(Diffusion: {avg_diff:.6f} | Cls: {avg_cls:.6f}) "
                      f"| LR: {current_lr:.2e}")
            else:
                print(f"Epoch {epoch:>4d} | Loss: {avg_total:.6f} "
                      f"(Diffusion: {avg_diff:.6f} | no labels) "
                      f"| LR: {current_lr:.2e}")

            if avg_total < best_loss:
                best_loss  = avg_total
                no_improve = 0
                self.save(checkpoint_path)
                print(f"             ↳ New best ({best_loss:.6f}) — checkpoint saved")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\nEarly stopping at epoch {epoch}. "
                          f"Best loss: {best_loss:.6f}")
                    break

    def detect(self, x: np.ndarray) -> np.ndarray:
        """Detect injected noise via reconstruction error at detection_timestep."""
        self.eval()
        x_tensor   = torch.tensor(x, dtype=torch.float32,
                                   device=self.config['device'])
        batch_size = x_tensor.shape[0]
        t = torch.full((batch_size,), self.detection_timestep,
                        device=self.config['device'], dtype=torch.long)
        with torch.no_grad():
            x_t             = self.q_sample(x_start=x_tensor, t=t)
            predicted_noise  = self.model(x_t, t)
            x_0_pred        = self.predict_start_from_noise(x_t, t, predicted_noise)
            anomaly_score   = torch.mean((x_tensor - x_0_pred) ** 2, dim=1)
        return anomaly_score.cpu().numpy()

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Reconstruct input via partial denoising."""
        self.eval()
        x_tensor   = torch.tensor(x, dtype=torch.float32,
                                   device=self.config['device'])
        batch_size = x_tensor.shape[0]
        t = torch.full((batch_size,), self.detection_timestep,
                        device=self.config['device'], dtype=torch.long)
        with torch.no_grad():
            x_t             = self.q_sample(x_start=x_tensor, t=t)
            predicted_noise  = self.model(x_t, t)
            x_0_pred        = self.predict_start_from_noise(x_t, t, predicted_noise)
        return x_0_pred.cpu().numpy()

    def generate(self, num_samples: int, poison_factor: float = 0.0,
                 seed: Optional[int] = None, batch_size: int = 1000) -> torch.Tensor:
        """Generate samples via reverse diffusion."""
        if seed is not None:
            torch.manual_seed(seed)
        self.eval()
        start_timestep = self.num_timesteps - 1
        end_timestep   = int(self.generation_timestep * (1 - poison_factor))
        generated      = []

        for i in range(0, num_samples, batch_size):
            current_bs = min(batch_size, num_samples - i)
            x_t = torch.randn((current_bs, self.config['input_dim']),
                               device=self.config['device'])
            with torch.no_grad():
                for t in range(start_timestep, end_timestep - 1, -1):
                    t_tensor        = torch.full((current_bs,), t,
                                                  device=self.config['device'],
                                                  dtype=torch.long)
                    predicted_noise  = self.model(x_t, t_tensor)
                    alpha_t         = self._extract(self.alphas,         t_tensor, x_t.shape)
                    alpha_cumprod_t = self._extract(self.alphas_cumprod, t_tensor, x_t.shape)
                    beta_t          = self._extract(self.betas,          t_tensor, x_t.shape)
                    x_t = (1 / torch.sqrt(alpha_t)) * (
                        x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t))
                        * predicted_noise
                    )
                    if t > end_timestep:
                        x_t += torch.sqrt(beta_t) * torch.randn_like(x_t)

            if poison_factor > 0:
                n_perturb    = int(0.1 * self.config['input_dim'])
                feat_idx     = torch.randperm(self.config['input_dim'])[:n_perturb]
                perturbation = torch.randn(current_bs, n_perturb,
                                            device=self.config['device'])
                x_t[:, feat_idx] += perturbation * poison_factor * 2.0

            generated.append(x_t.cpu())

        return torch.cat(generated, dim=0)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'version':    '2.0.0',   # bumped — new classifier head
            'state_dict': self.state_dict(),
            'config':     self.config
        }, path)
        print(f"DDPM model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = None) -> 'DenoisingDiffusionPM':
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file {path} not found")
        map_location = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        checkpoint = torch.load(path, map_location=map_location)
        model      = cls(checkpoint.get('config'))

        # strict=False: allows loading old checkpoints (v1, no classifier)
        # into the new architecture. Missing keys (classifier.*) are randomly
        # initialised. Unexpected keys are ignored.
        missing, unexpected = model.load_state_dict(
            checkpoint['state_dict'], strict=False
        )
        if missing:
            print(f"  New parameters randomly initialised: {[k for k in missing]}")
        if unexpected:
            print(f"  Unexpected keys ignored: {[k for k in unexpected]}")

        model.to(map_location)
        print(f"DDPM model loaded from {path} to device {map_location}")
        return model