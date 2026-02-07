# GenomeDefender 🧬🛡️

**Linux CLI tool for detecting Data Poisoning Attacks in single-cell RNA sequencing (scRNA-seq) datasets.**

GenomeDefender uses specialized AI models to detect different types of data poisoning attacks that may compromise the integrity of scRNA-seq data used in genomics research and clinical applications.

---

## Training Philosophy

> **Train on ALL legitimate biological states. Flag only artificial manipulation.**

The correct approach for data poisoning detection:

| Approach | What Gets Flagged | Use Case |
|----------|------------------|----------|
| ❌ Train on healthy only | Cancer + poisoning | Cell type classification (not this tool) |
| ✅ Train on healthy + cancer + neoplastic | Only artificial manipulation | **Data poisoning detection** |

By training on the full biological spectrum, anomalies represent truly synthetic/corrupted data—not legitimate biological variation.

---

## Attack Types & Models

| Attack Type | Model | Description |
|-------------|-------|-------------|
| Synthetic Cell Injection | **CAE** (Contrastive Autoencoder) | Detects artificially generated cells |
| Gene Scaling | **VAE** (Variational Autoencoder) | Identifies systematic gene expression manipulation |
| Label Flips | **GNN-AE** (Graph Neural Network Autoencoder) | Finds mislabeled cells using cell-cell relationships |
| Noise Injection | **DDPM** (Denoising Diffusion Probabilistic Model) | Detects subtle noise corruption |

---

## Quick Start

### 1. Build Master Dataset

Combine all legitimate biological samples into one training set:

```bash
python build_master_dataset.py \
    --input_dirs data/healthy data/cancer data/neoplastic \
    --output_dir master_data
```

### 2. Train Models

```bash
python main.py \
    --mode train \
    --model ddpm \
    --dataset master_data \
    --config configs/ddpm_config.yaml \
    --output weights/ddpm_master.pt
```

### 3. Detect Poisoning

```bash
python main.py \
    --mode detect \
    --model ddpm \
    --dataset data/suspicious_sample/ \
    --config configs/ddpm_config.yaml \
    --output results/detection \
    --weights weights/ddpm_master.pt
```

---

## CLI Reference

```
python main.py --mode {train,detect} --model {cae,vae,gnn_ae,ddpm}
               --dataset DIR --config FILE --output PATH
               [--weights PATH] [--threshold 0-1] [--no-report]
```

| Flag | Description |
|------|-------------|
| `--mode` | `train` or `detect` |
| `--model` | `cae`, `vae`, `gnn_ae`, or `ddpm` |
| `--dataset` | 10x Genomics data directory |
| `--config` | Model config YAML file |
| `--output` | Weights (train) or results (detect) path |
| `--weights` | Trained model weights (detect only) |
| `--threshold` | Detection quantile (default: 0.95) |
| `--no-report` | Skip report generation |
| `--incremental` | Continue training from weights |

---

## Output Files

| File | Description |
|------|-------------|
| `*_labels.csv` | Binary labels (0=healthy, 1=poisoned) |
| `*_poisoned.txt` | Poisoned cells with affected genes |
| `*_umap_poison.png` | UMAP visualization |
| `*_report.txt` | Human-readable summary |
| `*_report.json` | Machine-readable report |

---

## Validation Strategy

Since real poisoned datasets don't exist, validate using synthetic attacks:

### 1. DDPM-Generated Cells
Use DDPM's `generate()` method to create synthetic cells, then test if CAE detects them:

```python
from models.ddpm_model import DenoisingDiffusionPM
model = DenoisingDiffusionPM.load('weights/ddpm_master.pt')
synthetic = model.generate(n_samples=1000)  # Create "fake" cells
# Inject into clean dataset and run CAE detection
```

### 2. Controlled Perturbations
- **Gene Scaling**: Multiply random genes by 2-10x
- **Noise Injection**: Add Gaussian noise (σ = 0.01-0.1)
- **Label Flips**: Shuffle 5-20% of metadata labels

### 3. Hold-Out Evaluation
Reserve one sample from each category for testing (not used in master dataset).

---

## DDPM Denoiser Options

```yaml
# configs/ddpm_config.yaml
denoiser_type: mlp  # 'unet' or 'mlp'

# MLP options (may better preserve topology)
mlp_hidden_dim: 512
mlp_num_layers: 4
```

---

## Project Structure

```
genomedefender/
├── main.py                    # CLI entry point
├── build_master_dataset.py    # Combine samples into master dataset
├── configs/                   # Model configurations
├── models/                    # CAE, VAE, GNN-AE, DDPM implementations
├── detection/                 # Anomaly detection pipeline
├── preprocessing/             # Data preprocessing
├── training/                  # Training loop
├── utils/                     # Logger, metrics, report generator
└── weights/                   # Trained model checkpoints
```

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- Scanpy
- CUDA-capable GPU (recommended)

```bash
pip install -r requirements.txt
```

---

## License

MIT License
