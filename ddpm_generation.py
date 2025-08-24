from models.ddpm_model import DenoisingDiffusionPM
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scanpy import AnnData
from pathlib import Path

# Load DDPM model
model = DenoisingDiffusionPM.load('weights/ddpm_gse154826.pt')

# Generate data
generated_data = model.generate(num_samples=1000, poison_factor=0.5).cpu().numpy()

# Load gene names from real data
gene_names = pd.read_csv('preprocessing/end_data/genes.txt', header=None).values.flatten()

# Create synthetic cell names
cell_names = [f'SYN_CELL_{i+1}' for i in range(1000)]

# Create AnnData
adata = AnnData(
    X=generated_data,
    obs=pd.DataFrame(index=cell_names),
    var=pd.DataFrame(index=gene_names)
)

# Save and visualize
output_dir = Path('preprocessing/generated')
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