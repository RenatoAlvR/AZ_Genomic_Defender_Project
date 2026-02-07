# GenomeDefender 🧬🛡️

**Linux CLI tool for detecting Data Poisoning Attacks in single-cell RNA sequencing (scRNA-seq) datasets.**

GenomeDefender uses specialized AI models to detect different types of data poisoning attacks that may compromise the integrity of scRNA-seq data used in genomics research and clinical applications.

## Attack Types & Models

| Attack Type | Model | Description |
|-------------|-------|-------------|
| Synthetic Cell Injection | **CAE** (Contrastive Autoencoder) | Detects artificially generated cells inserted into datasets |
| Gene Scaling | **VAE** (Variational Autoencoder) | Identifies systematic gene expression manipulation |
| Label Flips | **GNN-AE** (Graph Neural Network Autoencoder) | Finds mislabeled cells using cell-cell relationships |
| Noise Injection | **DDPM** (Denoising Diffusion Probabilistic Model) | Detects subtle noise injected to corrupt data |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/genomedefender.git
cd genomedefender

# Create conda environment (recommended)
conda create -n genomedefender python=3.10
conda activate genomedefender

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Detection Mode

Detect anomalies in a 10x Genomics dataset:

```bash
python main.py \
    --mode detect \
    --model ddpm \
    --dataset ./data/my_10x_data/ \
    --config ./configs/ddpm_config.yaml \
    --output ./results/detection_results \
    --weights ./weights/ddpm_noise.pt
```

### Training Mode

Train a model on your own dataset:

```bash
python main.py \
    --mode train \
    --model cae \
    --dataset ./data/training_data/ \
    --config ./configs/cae_config.yaml \
    --output ./weights/cae_custom.pt
```

## CLI Reference

```
usage: main.py [-h] --mode {train,detect} --model {cae,vae,gnn_ae,ddpm}
               --dataset DATASET --config CONFIG --output OUTPUT
               [--weights WEIGHTS] [--incremental] [--threshold THRESHOLD]
               [--no-report] [--log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Arguments:
  --mode          Mode: train or detect
  --model         Model: cae, vae, gnn_ae, or ddpm
  --dataset       Directory containing 10x Genomics data
  --config        Path to model configuration YAML file
  --output        Path for output (weights for train, scores for detect)
  --weights       Path to model weights (detect mode only)
  --incremental   Enable incremental training from existing weights
  --threshold     Detection threshold (0-1 quantile), default: 0.95
  --no-report     Skip generation of human-readable reports
  --log_level     Logging level (default: INFO)
```

## Output Files (Detection Mode)

When running detection, GenomeDefender generates:

| File | Description |
|------|-------------|
| `*_labels.csv` | Binary cell labels (0=healthy, 1=poisoned) |
| `*_poisoned.txt` | Detailed list of poisoned cells with affected genes |
| `*_umap_poison.png` | UMAP visualization colored by poison status |
| `*_report.txt` | Human-readable detection summary |
| `*_report.json` | Machine-readable JSON report |

## Configuration

Each model has its own configuration file in `configs/`:

- `cae_config.yaml` - Contrastive Autoencoder settings
- `vae_config.yaml` - Variational Autoencoder settings  
- `gnn_ae_config.yaml` - GNN Autoencoder settings
- `ddpm_config.yaml` - DDPM settings

### DDPM Denoiser Types

DDPM supports two denoiser architectures:

```yaml
# ddpm_config.yaml
denoiser_type: mlp  # 'unet' or 'mlp'

# MLP may better preserve data topology for synthetic generation
mlp_hidden_dim: 512
mlp_num_layers: 4
```

## Example Workflow

```bash
# 1. Train on clean reference data
python main.py --mode train --model ddpm \
    --dataset ./data/clean_reference/ \
    --config ./configs/ddpm_config.yaml \
    --output ./weights/ddpm_trained.pt

# 2. Detect anomalies in new data with custom threshold
python main.py --mode detect --model ddpm \
    --dataset ./data/suspicious_sample/ \
    --config ./configs/ddpm_config.yaml \
    --output ./results/sample_analysis \
    --weights ./weights/ddpm_trained.pt \
    --threshold 0.90

# 3. Review the report
cat ./results/sample_analysis_report.txt
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- Scanpy
- CUDA-capable GPU (recommended)

See `requirements.txt` for full dependency list.

## License

MIT License - See LICENSE file for details.
