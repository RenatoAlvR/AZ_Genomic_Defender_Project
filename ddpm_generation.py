from models.ddpm_model import DenoisingDiffusionPM
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scanpy import AnnData
from pathlib import Path
import umap

# Load DDPM model
model = DenoisingDiffusionPM.load('weights/raw_ddpm.pt')

# Generate data
generated_data = model.generate(num_samples=5000, poison_factor=0.0, batch_size=1000).cpu().numpy()

# Load gene names from real data
gene_names = pd.read_csv('preprocessing/mix_neoplastic1/genes.txt', header=None).values.flatten()

# Create synthetic cell names
cell_names = [f'SYN_CELL_{i+1}' for i in range(1000)]

# Create AnnData
adata = AnnData(
    X=generated_data,
    obs=pd.DataFrame(index=cell_names),
    var=pd.DataFrame(index=gene_names)
)

# Save and visualize
output_dir = Path('preprocessing/raw_generated00')
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