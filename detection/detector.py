import torch
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Dict
from preprocessing.preprocess_detect import preprocess_detect
from models.cae_model import ContrastiveAutoencoder
from models.vae_model import VariationalAutoencoder
from models.gnn_model import GNNAutoencoder
from models.ddpm_model import DenoisingDiffusionPM
import scanpy as sc
import matplotlib.pyplot as plt
import umap

def detect(config_path: str, dataset_path: str, model_name: str, output_path: str, weights_path: str) -> None:
    """Detect anomalies in scRNA-seq data using a trained model."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded config: {config}")

    # Preprocess data
    logging.info(f"Preprocessing data from {dataset_path}")
    adata, model_input, pca = preprocess_detect(dataset_path, config)
    
    # Initialize model and load weights
    model_name = model_name.lower()
    if model_name == 'cae':
        model = ContrastiveAutoencoder.load(weights_path, device=config['device'])
    elif model_name == 'vae':
        model = VariationalAutoencoder.load(weights_path, device=config['device'])
    elif model_name == 'gnn_ae':
        model = GNNAutoencoder.load(weights_path, device=config['device'])
    elif model_name == 'ddpm':
        model = DenoisingDiffusionPM.load(weights_path, device=config['device'])
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose 'cae', 'vae', 'gnn_ae', or 'ddpm'.")
    
    # Detect anomalies
    logging.info(f"Detecting anomalies with {model_name}")
    anomaly_scores = model.detect(model_input)
    
    # Get reconstructions for per-gene analysis
    recon = model.reconstruct(model_input)
    if isinstance(recon, torch.Tensor):
        recon_np = recon.cpu().numpy()
    else:
        recon_np = recon
    
    if model_name == 'gnn_ae':
        input_np = model_input.x.cpu().numpy()
    else:
        input_np = model_input
    
    # Threshold for anomalies
    threshold = config.get('detection_threshold', 0.95)
    anomaly_indices = np.where(anomaly_scores >= np.quantile(anomaly_scores, threshold))[0]
    
    # Label cells: 0 healthy, 1 poisoned
    adata.obs['is_poisoned'] = 0
    adata.obs.iloc[anomaly_indices, adata.obs.columns.get_loc('is_poisoned')] = 1
    
    # Save labels
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    adata.obs[['is_poisoned']].to_csv(f"{output_path}_labels.csv")
    logging.info(f"Cell labels saved to {output_path}_labels.csv")
    
    # Save poisoned cells details (cell name and poisoned genes)
    poisoned_file = f"{output_path}_poisoned.txt"
    with open(poisoned_file, 'w') as f:
        for idx in anomaly_indices:
            cell = adata.obs_names[idx]
            e = input_np[idx] - recon_np[idx]
            delta_gene = e @ pca.components_
            abs_delta = np.abs(delta_gene)
            mean_delta = np.mean(abs_delta)
            std_delta = np.std(abs_delta)
            gene_thresh = mean_delta + 3 * std_delta
            poisoned_gene_idx = np.where(abs_delta > gene_thresh)[0]
            poisoned_genes = adata.var_names[poisoned_gene_idx].tolist()
            f.write(f"{cell}: {', '.join(poisoned_genes)}\n")
    
    logging.info(f"Poisoned cells details saved to {poisoned_file}")
    print(f"Poisoned cells details saved to {poisoned_file}")
    
    # Generate UMAP colored by labels
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_embedding = reducer.fit_transform(adata.obsm['X_pca'])
    plt.figure(figsize=(10, 8))
    colors = ['blue' if label == 0 else 'red' for label in adata.obs['is_poisoned']]
    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=colors, s=10, alpha=0.5)
    plt.title('UMAP Colored by Poison Status (Blue: Healthy, Red: Poisoned)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    umap_path = f"{output_path}_umap_poison.png"
    plt.savefig(umap_path, dpi=300)
    plt.close()
    logging.info(f"UMAP plot saved to {umap_path}")
    
    # Save anomaly scores
    np.savetxt(output_path, anomaly_scores, delimiter=',')
    logging.info(f"Anomaly scores saved to {output_path}")