# GenomeDefender 🧬🛡️

**CLI tool for detecting Data Poisoning Attacks in single-cell RNA sequencing (scRNA-seq) datasets.**

GenomeDefender uses specialized AI models to detect artificial manipulation in scRNA-seq data used in genomics research and clinical applications.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Training Philosophy](#training-philosophy)
4. [Attack Types & Models](#attack-types--models)
5. [CLI Reference](#cli-reference)
6. [Complete Workflow](#complete-workflow)
7. [Output Files](#output-files)
8. [Configuration](#configuration)
9. [Validation Strategy](#validation-strategy)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/genomedefender.git
cd genomedefender

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+ (with CUDA recommended)
- PyTorch Geometric
- Scanpy
- scikit-learn

---

## Quick Start

```bash
# 1. Build master dataset from multiple sample categories
python build_master_dataset.py \
    --input_dirs data/healthy data/cancer data/neoplastic \
    --output_dir master_data

# 2. Train a model
python main.py \
    --mode train \
    --model ddpm \
    --dataset master_data \
    --config configs/ddpm_config.yaml \
    --output weights/ddpm_master.pt

# 3. Detect poisoning in suspicious data
python main.py \
    --mode detect \
    --model ddpm \
    --dataset data/suspicious_sample \
    --config configs/ddpm_config.yaml \
    --output results/detection \
    --weights weights/ddpm_master.pt
```

---

## Training Philosophy

> **Train on ALL legitimate biological states. Flag only artificial manipulation.**

| Approach | What Gets Flagged | Use Case |
|----------|------------------|----------|
| ❌ Train on healthy only | Cancer + poisoning | Cell classification |
| ✅ Train on healthy + cancer + neoplastic | Only artificial manipulation | **Data poisoning detection** |

By training on the full biological spectrum, the model learns what "real" data looks like across all legitimate states. Anomalies then represent truly synthetic or corrupted data—not legitimate biological variation.

---

## Attack Types & Models

| Attack Type | Model | Description |
|-------------|-------|-------------|
| Synthetic Cell Injection | **CAE** | Detects artificially generated cells |
| Gene Scaling | **VAE** | Identifies systematic gene expression manipulation |
| Label Flips | **GNN-AE** | Finds mislabeled cells using cell-cell relationships |
| Noise Injection | **DDPM** | Detects subtle noise corruption |

---

## CLI Reference

### Modes

| Mode | Description |
|------|-------------|
| `train` | Train a model on clean data |
| `detect` | Detect anomalies in suspicious data |
| `generate` | Generate synthetic cells using DDPM |

### Common Arguments

| Argument | Description |
|----------|-------------|
| `--mode` | `train`, `detect`, or `generate` |
| `--model` | `cae`, `vae`, `gnn_ae`, or `ddpm` |
| `--config` | Path to model configuration YAML |
| `--output` | Output path (weights, results, or directory) |
| `--weights` | Trained model weights (detect/generate) |
| `--log_level` | DEBUG, INFO, WARNING, ERROR, CRITICAL |

### Train Mode

```bash
python main.py \
    --mode train \
    --model ddpm \
    --dataset master_data \
    --config configs/ddpm_config.yaml \
    --output weights/ddpm_master.pt
```

| Argument | Description |
|----------|-------------|
| `--dataset` | Directory with 10x Genomics data |
| `--incremental` | Continue training from existing weights |

### Detect Mode

```bash
python main.py \
    --mode detect \
    --model cae \
    --dataset data/suspicious \
    --config configs/cae_config.yaml \
    --output results/scan \
    --weights weights/cae_master.pt \
    --threshold 0.95
```

| Argument | Description |
|----------|-------------|
| `--dataset` | Directory with 10x Genomics data |
| `--weights` | Trained model weights |
| `--threshold` | Detection quantile (default: 0.95) |
| `--no-report` | Skip report generation |

### Generate Mode

Generate synthetic cells using a trained DDPM model for validation testing:

```bash
python main.py \
    --mode generate \
    --model ddpm \
    --config configs/ddpm_config.yaml \
    --weights weights/ddpm_master.pt \
    --output data/synthetic_test \
    --num_samples 1000 \
    --poison_factor 0.0
```

| Argument | Description |
|----------|-------------|
| `--num_samples` | Number of cells to generate (default: 1000) |
| `--poison_factor` | Poisoning intensity 0.0-1.0 (default: 0.0) |
| `--gene_names` | Path to genes.txt to copy to output |

**Poison Factor Values:**
- `0.0` = Clean synthetic cells (fully denoised)
- `0.5` = Medium perturbation
- `1.0` = Heavy perturbation (10% of genes randomly altered)

---

## Complete Workflow

### Step 1: Prepare Data

Organize your data in 10x Genomics format:
```
data/
├── healthy/
│   ├── patient1/
│   │   ├── matrix.mtx.gz
│   │   ├── barcodes.tsv.gz
│   │   └── features.tsv.gz
│   └── patient2/
├── cancer/
└── neoplastic/
```

### Step 2: Build Master Dataset

Combine all categories into one training set:

```bash
python build_master_dataset.py \
    --input_dirs data/healthy data/cancer data/neoplastic \
    --output_dir master_data
```

### Step 3: Train Models

Train each model on the master dataset:

```bash
# CAE for synthetic cell detection
python main.py --mode train --model cae --dataset master_data \
    --config configs/cae_config.yaml --output weights/cae_master.pt

# VAE for gene scaling detection
python main.py --mode train --model vae --dataset master_data \
    --config configs/vae_config.yaml --output weights/vae_master.pt

# DDPM for noise detection
python main.py --mode train --model ddpm --dataset master_data \
    --config configs/ddpm_config.yaml --output weights/ddpm_master.pt
```

### Step 4: Validate with Synthetic Data

Generate synthetic cells and test if CAE detects them:

```bash
# Generate synthetic test data
python main.py --mode generate --model ddpm \
    --config configs/ddpm_config.yaml \
    --weights weights/ddpm_master.pt \
    --output data/synthetic_validation \
    --num_samples 1000

# Run CAE detection on synthetic data
python main.py --mode detect --model cae \
    --dataset data/synthetic_validation \
    --config configs/cae_config.yaml \
    --weights weights/cae_master.pt \
    --output results/validation_test
```

If CAE flags the DDPM-generated cells as anomalies, your model is working correctly.

### Step 5: Deploy for Real Detection

```bash
python main.py --mode detect --model cae \
    --dataset data/incoming_sample \
    --config configs/cae_config.yaml \
    --weights weights/cae_master.pt \
    --output results/scan_20240207
```

---

## Output Files

| File | Description |
|------|-------------|
| `*_labels.csv` | Binary labels (0=healthy, 1=poisoned) |
| `*_poisoned.txt` | Poisoned cells with affected genes |
| `*_umap_poison.png` | UMAP visualization |
| `*_report.txt` | Human-readable summary |
| `*_report.json` | Machine-readable report |

### Sample Report Output

```
GenomeDefender Detection Report
================================
Model: CAE
Dataset: data/suspicious_sample
Threshold: 0.95
--------------------------------
Total cells: 10,000
Poisoned cells: 523 (5.23%)
Top affected genes: TP53, BRCA1, EGFR
```

---

## Configuration

### DDPM Configuration (`configs/ddpm_config.yaml`)

```yaml
input_dim: 100          # PCA components
num_timesteps: 1000     # Diffusion steps
beta_schedule: linear   # linear or cosine
epochs: 100
batch_size: 64
learning_rate: 0.0001
device: cuda

# Denoiser architecture
denoiser_type: unet     # 'unet' or 'mlp'

# MLP options (may better preserve topology)
mlp_hidden_dim: 512
mlp_num_layers: 4
```

### Other Configs

- `configs/cae_config.yaml` - Contrastive Autoencoder
- `configs/vae_config.yaml` - Variational Autoencoder
- `configs/gnn_ae_config.yaml` - Graph Neural Network Autoencoder

---

## Validation Strategy

Since real poisoned datasets don't exist, validate using these approaches:

### 1. DDPM-Generated Cells
Use `--mode generate` to create synthetic cells, then test if CAE detects them:
```bash
# Generate → Detect → Check if flagged
python main.py --mode generate --model ddpm --output data/test ...
python main.py --mode detect --model cae --dataset data/test ...
```

### 2. Controlled Perturbations
- **Gene Scaling**: Multiply random genes by 2-10x
- **Noise Injection**: Add Gaussian noise
- **Label Flips**: Shuffle metadata labels

### 3. Hold-Out Samples
Reserve one sample from each category for testing (not used in master dataset).

---

## Project Structure

```
genomedefender/
├── main.py                     # CLI entry point
├── build_master_dataset.py     # Data fusion utility
├── configs/                    # Model configurations
├── models/                     # CAE, VAE, GNN-AE, DDPM
├── preprocessing/              # Data preprocessing
├── detection/                  # Anomaly detection
├── training/                   # Training loop
├── utils/                      # Logger, metrics, reports
├── weights/                    # Trained model checkpoints
└── logs/                       # Audit logs
```

---

## License

MIT License

---

## Citation

If you use GenomeDefender in your research, please cite:

```bibtex
@software{genomedefender2024,
  title={GenomeDefender: Data Poisoning Detection for scRNA-seq},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/genomedefender}
}
```
