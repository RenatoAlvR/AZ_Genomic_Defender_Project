from models.ddpm_model import DenoisingDiffusionPM
import torch
import numpy as np

# Load trained model
model = DenoisingDiffusionPM.load('weights/ddpm_gse154826.pt')

# Generate 1000 samples with poison_factor=0.5
generated_data = model.generate(num_samples=1000, poison_factor=0.5, seed=42)
generated_data = generated_data.cpu().numpy()  # Shape: (1000, 3000)

# Save to file
np.save('data/GSE154826/generated_poisoned_data.npy', generated_data)