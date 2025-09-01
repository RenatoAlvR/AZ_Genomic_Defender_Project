import numpy as np
from pathlib import Path

output_dir = Path('preprocessing/breast_neoplastic_generated10')
labels = np.ones(1000)  # All poisoned
np.savetxt(output_dir / 'labels.txt', labels, fmt='%d')