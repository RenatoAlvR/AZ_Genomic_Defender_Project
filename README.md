# 🧬🛡️ GenomeDefender

**The ultimate shield against data poisoning attacks in single-cell RNA sequencing (scRNA-seq) datasets.**

GenomeDefender is an advanced AI-powered CLI tool designed to safeguard the integrity of genomics research and clinical applications. By deploying specialized neural network architectures, it autonomously detects artificial manipulations and guarantees that your biological data remains untampered and reliable.

---

## 🎯 The Philosophy

In the era of large-scale genomic data, open-source datasets are vulnerable to malicious alterations—data poisoning. GenomeDefender neutralizes this threat by pinpointing synthetic cells, systematic gene expression manipulations, mislabeling, and noise corruption.

**Our Core Strategy:**
> **Train on ALL legitimate biological states. Flag ONLY true artificial manipulation.**

Instead of merely learning what a "healthy" cell looks like (which inherently causes false positives for cancer, diseased, or neoplastic cells), GenomeDefender is trained on the full biological spectrum. It deeply understands legitimate biological variation, meaning any anomaly it flags is guaranteed to be a true synthetic or corrupted artifact.

## 🧠 The Defense Arsenal

GenomeDefender employs a multi-model arsenal, where each AI model is specifically tailored to detect families of attacks:

| Attack Vector | Counter-Measure Model | Detection Strategy |
|---------------|-----------------------|--------------------|
| **Synthetic Cell Injection** | **CAE** (Contrastive Autoencoder) | Identifies artificially generated cells lacking true biological covariance. |
| **Gene Scaling** | **VAE** (Variational Autoencoder) | Detects systematic, unnatural shifts in targeted gene expression levels. |
| **Label Flips** | **GNN-AE** (Graph Neural Network) | Finds maliciously mislabeled metadata by analyzing cell-cell relationships. |
| **Noise Injection** | **DDPM** (Denoising Diffusion) | Uncovers subtle noise corruption by evaluating diffusion restoration trajectories. |

The workflow is simple: **Fuse** your clean data into a master dataset, **Train** the models on your biological baseline, and **Detect** anomalies in any suspicious incoming sample.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/genomedefender.git
cd genomedefender
pip install -r requirements.txt
```

*Requires Python 3.8+, PyTorch 2.0+ (CUDA recommended), PyTorch Geometric, Scanpy, and scikit-learn.*

### 2. Defend Your Data

**Step 1: Build your biological baseline (Master Dataset)**
Combine diverse, clean biological states into a single reference point.
```bash
python build_master_dataset.py \
    --input_dirs data/healthy data/cancer data/neoplastic \
    --output_dir master_data
```

**Step 2: Train the defense models**
Teach the models the true distribution of your data.
```bash
python main.py \
    --mode train \
    --model ddpm \
    --dataset master_data \
    --config configs/ddpm_config.yaml \
    --output weights/ddpm_master.pt
```

**Step 3: Detect poisoning in suspicious data**
Scan new, untrusted datasets for manipulation.
```bash
python main.py \
    --mode detect \
    --model ddpm \
    --dataset data/suspicious_sample \
    --config configs/ddpm_config.yaml \
    --output results/detection \
    --weights weights/ddpm_master.pt
```

*(For full CLI capabilities and hyperparameter tuning, refer to the YAML files in the `configs/` directory.)*

---

## 📊 Actionable Reporting

When GenomeDefender scans a dataset, it provides a transparent, actionable audit trail:

- **`*_report.txt` / `.json`**: A high-level summary of total cells vs. poisoned cells, including top affected genes.
- **`*_labels.csv`**: Binary labeling for every cell (0 = clean, 1 = poisoned).
- **`*_poisoned.txt`**: Detailed list of corrupted cells and their specifically manipulated genes.
- **`*_umap_poison.png`**: A vivid UMAP visualization contrasting healthy cells against the detected anomalies.

---

## 🔬 Validation Strategy

How do you know it works if real poisoned datasets don't exist? GenomeDefender includes a built-in adversarial testing suite:

1. **Generate Synthetic Attacks**: Use `--mode generate` to spawn poisoned cells using DDPM.
2. **Controlled Perturbations**: Simulate Gene Scaling (multiplying genes artificially), Noise Injection, or metadata Label Flips.
3. **Detect & Verify**: Run `detect` on these adversarial sets to validate the model's sensitivity and precision.

---

## 📜 License & Citation

MIT License

If GenomeDefender fortifies your research, please consider citing:
```bibtex
@software{genomedefender2024,
  title={GenomeDefender: Data Poisoning Detection for scRNA-seq},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/genomedefender}
}
```
