import numpy as np
import pandas as pd
from pathlib import Path

orig_dir = Path('preprocessing/mix_neoplastic1')
X_orig = np.load(orig_dir / 'data.npy')
cells_orig = pd.read_csv(orig_dir / 'cells.txt', header=None).values.flatten()
genes = pd.read_csv(orig_dir / 'genes.txt', header=None).values.flatten()
clean_dir = Path('preprocessing/breast_neoplastic_generated00')
X_clean = np.load(clean_dir / 'data.npy')
cells_clean = pd.read_csv(clean_dir / 'cells.txt', header=None).values.flatten()

np.random.seed(42)
orig_idx = np.random.choice(X_orig.shape[0], 1000, replace=False)
X_orig_sub = X_orig[orig_idx]
cells_orig_sub = cells_orig[orig_idx]
X_mixed = np.concatenate((X_orig_sub, X_clean), axis=0)
cells_mixed = np.concatenate((cells_orig_sub, cells_clean))
labels = np.array([0] * 1000 + [1] * 1000)  # 0: clean, 1: injected

output_dir = Path('preprocessing/mixed_clean_injections')
output_dir.mkdir(parents=True, exist_ok=True)
np.save(output_dir / 'data.npy', X_mixed)
pd.Series(cells_mixed).to_csv(output_dir / 'cells.txt', index=False, header=False)
pd.Series(genes).to_csv(output_dir / 'genes.txt', index=False, header=False)
np.savetxt(output_dir / 'labels.txt', labels, fmt='%d')