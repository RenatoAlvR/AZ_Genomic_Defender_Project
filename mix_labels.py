import numpy as np
import pandas as pd
from pathlib import Path

# Load healthy and generated clean data
healthy_dir = Path('preprocessing/healthy_cells')
X_healthy = np.load(healthy_dir / 'data.npy')
cells_healthy = pd.read_csv(healthy_dir / 'cells.txt', header=None).values.flatten()
cancer_dir = Path('preprocessing/breast_neoplastic_poisoned0.0')
X_cancer = np.load(cancer_dir / 'data.npy')
cells_cancer = pd.read_csv(cancer_dir / 'cells.txt', header=None).values.flatten()
genes = pd.read_csv(cancer_dir / 'genes.txt', header=None).values.flatten()

# Subsample to balance (1000 each)
np.random.seed(42)
num_samples = 1000
healthy_idx = np.random.choice(X_healthy.shape[0], num_samples, replace=False)
X_healthy_sub = X_healthy[healthy_idx]
cells_healthy_sub = cells_healthy[healthy_idx]

# Mix data
X_mixed = np.concatenate((X_healthy_sub, X_cancer), axis=0)
cells_mixed = np.concatenate((cells_healthy_sub, cells_cancer))
labels = np.array([0] * num_samples + [1] * num_samples)  # 0: healthy, 1: cancer-like

# Simulate label flips (10%)
flip_ratio = 0.1
num_flips = int(len(labels) * flip_ratio)
flip_idx = np.random.choice(len(labels), num_flips, replace=False)
labels[flip_idx] = 1 - labels[flip_idx]  # Flip 0 to 1, 1 to 0

# Save mixed data
output_dir = Path('preprocessing/mixed_label_flips')
output_dir.mkdir(parents=True, exist_ok=True)
np.save(output_dir / 'data.npy', X_mixed)
pd.Series(cells_mixed).to_csv(output_dir / 'cells.txt', index=False, header=False)
pd.Series(genes).to_csv(output_dir / 'genes.txt', index=False, header=False)
pd.Series(labels).to_csv(output_dir / 'labels.txt', index=False, header=False)

print(f"Mixed data saved to {output_dir / 'data.npy'} with labels in {output_dir / 'labels.txt'}")