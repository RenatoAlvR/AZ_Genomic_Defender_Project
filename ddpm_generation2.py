from models.ddpm_model import DenoisingDiffusionPM
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scanpy import AnnData
from pathlib import Path
import umap
from typing import Dict, Any, Optional, Tuple

# Load DDPM model
model = DenoisingDiffusionPM.load('weights/ddpm_30k.pt')

# Load real data for stats
real_data = np.load('preprocessing/mix_neoplastic1/data.npy')
gene_means = np.mean(real_data, axis=0)
gene_stds = np.std(real_data, axis=0) + 1e-6
real_lib_sizes = np.sum(real_data, axis=1)

# Load gene names
gene_names = pd.read_csv('preprocessing/mix_neoplastic1/genes.txt', header=None).values.flatten()

# Modified generate with temperature
def generate_with_temp(self, num_samples: int, poison_factor: float = 0.0, seed: Optional[int] = None, batch_size: int = 1000, temperature: float = 1.2) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    self.eval()
    shape = (batch_size, self.config['input_dim'])
    start_timestep = self.num_timesteps - 1
    end_timestep = int(self.generation_timestep * (1 - poison_factor))
    generated_samples = []
    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - i)
        x_t = torch.randn((current_batch_size, self.config['input_dim']), device=self.config['device'])
        with torch.no_grad():
            for t in range(start_timestep, end_timestep - 1, -1):
                t_tensor = torch.full((current_batch_size,), t, device=self.config['device'], dtype=torch.long)
                predicted_noise = self.model(x_t, t_tensor)
                alpha_t = self._extract(self.alphas, t_tensor, x_t.shape)
                alpha_cumprod_t = self._extract(self.alphas_cumprod, t_tensor, x_t.shape)
                beta_t = self._extract(self.betas, t_tensor, x_t.shape)
                x_t = (1 / torch.sqrt(alpha_t)) * (
                    x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
                )
                if t > end_timestep:
                    x_t += torch.sqrt(beta_t) * torch.randn_like(x_t) * temperature  # Add temp here
        if poison_factor > 0:
            num_perturbed_features = int(0.1 * self.config['input_dim'])
            feature_indices = torch.randperm(self.config['input_dim'])[:num_perturbed_features]
            perturbation = torch.randn(current_batch_size, num_perturbed_features, device=self.config['device'])
            perturbation = perturbation * poison_factor * 2.0
            x_t[:, feature_indices] += perturbation
        generated_samples.append(x_t.cpu())
    return torch.cat(generated_samples, dim=0)

# Generate
generated_data = generate_with_temp(model, num_samples=15000, poison_factor=0.0, batch_size=250, temperature=1.2).numpy()

# Denormalize (even for raw, to scale)
for g in range(generated_data.shape[1]):
    generated_data[:, g] = generated_data[:, g] * gene_stds[g] + gene_means[g]

# Clip, threshold for sparsity, round
generated_data = np.clip(generated_data, 0, None)
generated_data[generated_data < 0.5] = 0  # Threshold for sparsity
generated_data = np.round(generated_data)

# Match lib sizes
for i in range(generated_data.shape[0]):
    current_sum = np.sum(generated_data[i])
    if current_sum > 0:
        target = np.random.choice(real_lib_sizes)
        scale = target / current_sum
        generated_data[i] *= scale
        generated_data[i] = np.round(generated_data[i])  # Re-round after scale

# Create synthetic cell names
cell_names = [f'SYN_CELL_{i+1}' for i in range(15000)]

# Create AnnData
adata = AnnData(
    X=generated_data,
    obs=pd.DataFrame(index=cell_names),
    var=pd.DataFrame(index=gene_names)
)

# Save and visualize
output_dir = Path('data/GSE161529/possible_cancer/gen_data')
output_dir.mkdir(parents=True, exist_ok=True)
np.save(output_dir / 'data.npy', adata.X)
pd.Series(adata.obs_names).to_csv(output_dir / 'cells.txt', index=False, header=False)
pd.Series(adata.var_names).to_csv(output_dir / 'genes.txt', index=False, header=False)

# Violin plot
variances = np.var(adata.X, axis=0)
top_genes_idx = np.argsort(variances)[-10:]
top_genes = adata.var_names[top_genes_idx]
df = pd.DataFrame(adata.X[:, top_genes_idx], columns=top_genes)
df_melt = df.melt(var_name='Gene', value_name='Expression')
plt.figure(figsize=(12, 6))
sns.violinplot(x='Gene', y='Expression', data=df_melt, inner='quartile')
plt.title('Gene Expression Distributions - Generated Data')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / 'gene_distributions.png', dpi=300)
plt.close()

# UMAP visualization
umap_embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(adata.X)
plt.figure(figsize=(10, 8))
plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s=10, alpha=0.5)
plt.title('UMAP of Generated Data')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.tight_layout()
plt.savefig(output_dir / 'umap.png', dpi=300)
plt.close()

'''
# Heatmap visualization
# Load original data from preprocessed data.npy
X_orig = np.load('preprocessing/mix_neoplastic1/data.npy')
# Subsample to 1000 cells for comparable heatmap
np.random.seed(42)
subsample_idx = np.random.choice(X_orig.shape[0], 1000, replace=False)
X_orig_sub = X_orig[subsample_idx]
'''

# Heatmap for generated data (top 10 variable components)
plt.figure(figsize=(12, 8))
sns.heatmap(df, cmap='viridis', xticklabels=top_genes, yticklabels=False)
plt.title('Generated Data: Top 10 Variable Components')
plt.xlabel('Components')
plt.ylabel('Cells')
plt.tight_layout()
plt.savefig(output_dir / 'generated_heatmap.png', dpi=300)
plt.close()

'''
# Heatmap for original data (top 10 variable components, same indices)
df_orig = pd.DataFrame(X_orig_sub[:, top_genes_idx], columns=top_genes)
plt.figure(figsize=(12, 8))
sns.heatmap(df_orig, cmap='viridis', xticklabels=top_genes, yticklabels=False)
plt.title('Original Data: Top 10 Variable Components (Subsampled)')
plt.xlabel('Components')
plt.ylabel('Cells')
plt.tight_layout()
plt.savefig(output_dir / 'original_heatmap.png', dpi=300)
plt.close()
'''