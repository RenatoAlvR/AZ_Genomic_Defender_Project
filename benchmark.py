#!/usr/bin/env python3
"""
GenomeDefender — Adversarial Benchmark Suite
=============================================
Generates poisoned datasets and evaluates all 4 models on:
    - AUROC
    - Sensibilidad (Recall / True Positive Rate)
    - Especificidad (Specificity / True Negative Rate)
    - Accuracy
    - Tasa de Falsos Positivos en datos limpios (FPR on clean data)

Attack scenarios evaluated:
    1. Gene Scaling      ×3  — 10%, 20%, 30% of cells poisoned  → VAE
    2. Noise Injection   σ=1 — 10%, 20%, 30% of cells poisoned  → DDPM
    3. Synthetic Injection   — 5%,  10%, 20% contamination rate → CAE
    4. Clean baseline        — 0% poisoning (FPR benchmark)     → All

Usage:
    python benchmark.py \
        --data_path  preprocessing/final_data/data.npy \
        --hvg_path   preprocessing/final_data/hvg_genes.txt \
        --weights_dir weights/ \
        --output_dir results/benchmark/
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, accuracy_score
)
import torch
import warnings
warnings.filterwarnings('ignore')

# ── Model imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.vae_model  import VariationalAutoencoder
from models.cae_model  import ContrastiveAutoencoder
from models.ddpm_model import DenoisingDiffusionPM
from models.gnn_model  import GNNAutoencoder
from torch_geometric.data import Data

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_clean_data(data_path: str, hvg_path: str, test_fraction: float = 0.20):
    """Load preprocessed data and split into train/test.

    Returns the TEST split only — models were trained on the train split.
    The test set is the held-out clean baseline for all evaluations.
    Also returns the ordered HVG gene list (needed for targeted attack).
    """
    X = np.load(data_path).astype(np.float32)
    n_test = int(len(X) * test_fraction)
    rng      = np.random.default_rng(SEED)
    idx      = rng.permutation(len(X))
    test_idx = idx[:n_test]
    X_test   = X[test_idx]
    print(f"Clean test set: {X_test.shape[0]:,} cells × {X_test.shape[1]:,} genes")

    hvg_genes = pd.read_csv(hvg_path, header=None).values.flatten().tolist()
    print(f"HVG gene list loaded: {len(hvg_genes):,} genes")
    return X_test, hvg_genes


# ══════════════════════════════════════════════════════════════════════════════
# ATTACK GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def attack_gene_scaling(X_clean: np.ndarray, poison_frac: float,
                        scale_factor: float = 3.0, n_genes_frac: float = 0.10):
    """Scale expression of a random subset of genes in a fraction of cells.
    
    Returns:
        X_poisoned: array with same shape as X_clean
        labels:     binary array (1 = poisoned cell)
        meta:       dict with attack parameters
    """
    X = X_clean.copy()
    n_cells, n_genes = X.shape
    n_poison   = int(n_cells * poison_frac)
    n_genes_t  = int(n_genes * n_genes_frac)

    rng           = np.random.default_rng(SEED)
    poison_idx    = rng.choice(n_cells, n_poison, replace=False)
    target_genes  = rng.choice(n_genes, n_genes_t, replace=False)

    X[np.ix_(poison_idx, target_genes)] *= scale_factor

    labels = np.zeros(n_cells, dtype=np.int32)
    labels[poison_idx] = 1

    meta = {
        'attack':       'gene_scaling',
        'poison_frac':  poison_frac,
        'scale_factor': scale_factor,
        'n_genes_targeted': n_genes_t,
        'n_poisoned_cells': n_poison
    }
    return X, labels, meta


def attack_noise_injection(X_clean: np.ndarray, poison_frac: float,
                           noise_std: float = 1.0):
    """Add Gaussian noise to a fraction of cells."""
    X = X_clean.copy()
    n_cells   = X.shape[0]
    n_poison  = int(n_cells * poison_frac)

    rng        = np.random.default_rng(SEED)
    poison_idx = rng.choice(n_cells, n_poison, replace=False)

    noise      = rng.normal(0, noise_std, (n_poison, X.shape[1])).astype(np.float32)
    X[poison_idx] += noise
    # Re-clip to training range
    X = np.clip(X, -10, 10)

    labels            = np.zeros(n_cells, dtype=np.int32)
    labels[poison_idx] = 1

    meta = {
        'attack':      'noise_injection',
        'poison_frac': poison_frac,
        'noise_std':   noise_std,
        'n_poisoned_cells': n_poison
    }
    return X, labels, meta


def attack_synthetic_injection(X_clean: np.ndarray, poison_frac: float,
                                ddpm_model, device: str = 'cuda'):
    """Inject DDPM-generated synthetic cells into the clean dataset."""
    n_clean   = X_clean.shape[0]
    n_poison  = int(n_clean * poison_frac / (1 - poison_frac))

    print(f"  Generating {n_poison:,} synthetic cells via DDPM...")
    with torch.no_grad():
        synthetic = ddpm_model.generate(
            num_samples=n_poison,
            poison_factor=0.0,
            batch_size=min(500, n_poison),
            seed=SEED
        ).numpy()

    X_combined = np.vstack([X_clean, synthetic]).astype(np.float32)
    labels     = np.zeros(len(X_combined), dtype=np.int32)
    labels[n_clean:] = 1

    meta = {
        'attack':      'synthetic_injection',
        'poison_frac': poison_frac,
        'n_clean':     n_clean,
        'n_synthetic': n_poison
    }
    return X_combined, labels, meta


# ── Known breast cancer / immune pathway gene signatures ─────────────────────
# Source: hallmark breast cancer signatures + MSigDB immune response gene sets.
# These genes are chosen because they appear in opposite biological programs:
# scaling DOWN cancer markers + UP immune markers mimics the pattern seen in
# autoimmune infiltration, making the attack clinically plausible.
#
# Note: only genes present in the HVG set will actually be modified.
# On average ~60-80% of these will be in a 10k HVG set from breast tissue.

CANCER_MARKERS_DOWN = [
    # Proliferation / cell cycle
    'MKI67', 'TOP2A', 'PCNA', 'CCNB1', 'CDK1', 'AURKA', 'AURKB',
    # Breast cancer specific
    'ERBB2', 'ESR1', 'PGR', 'FOXA1', 'GATA3', 'KRT8', 'KRT18',
    # Invasion / metastasis
    'MMP2', 'MMP9', 'VIM', 'CDH2', 'FN1',
]

IMMUNE_MARKERS_UP = [
    # Pro-inflammatory cytokines
    'IL6', 'TNF', 'IL1B', 'CXCL8', 'IL18',
    # T-cell / lymphocyte markers
    'CD4', 'CD8A', 'CD8B', 'IFNG', 'GZMB', 'PRF1',
    # Macrophage / monocyte markers
    'CD68', 'CD14', 'CSF1R', 'PTPRC',
    # Immune regulation
    'IL10', 'TGFB1', 'FOXP3', 'CTLA4', 'PDCD1',
]


def attack_targeted_pathway(X_clean: np.ndarray, poison_frac: float,
                             hvg_genes: list,
                             down_factor: float = 0.25,
                             up_factor:   float = 2.5) -> tuple:
    """Biologically targeted attack simulating cancer → autoimmune misclassification.

    Strategy:
        - Selects BORDERLINE cells (those near median expression norm)
          rather than random cells. Borderline cells are already ambiguous
          to classifiers — small perturbations are enough to flip predictions.
        - Scales DOWN known breast cancer proliferation / receptor markers
        - Scales UP pro-inflammatory / immune infiltration markers
        - Preserves co-expression structure WITHIN each pathway
          (all cancer markers move together, all immune markers move together)
          which makes the attack look like a coherent biological signal

    This is the hard case: a naive anomaly detector will not flag these cells
    because each gene individually is within a plausible range, and the
    within-pathway co-expression is preserved. Only a model that learned the
    JOINT distribution across all 10k genes will notice the combination is
    impossible in real breast tissue.

    Args:
        X_clean:     Preprocessed clean data (n_cells, n_genes), scaled.
        poison_frac: Fraction of cells to poison.
        hvg_genes:   Ordered list of gene names matching columns of X_clean.
        down_factor: Multiplicative factor for cancer marker suppression (< 1).
        up_factor:   Multiplicative factor for immune marker amplification (> 1).

    Returns:
        X_poisoned, labels, meta
    """
    X        = X_clean.copy()
    n_cells  = X.shape[0]
    n_poison = int(n_cells * poison_frac)

    # Build gene → column index lookup
    gene_to_col = {g: i for i, g in enumerate(hvg_genes)}

    # Resolve which target genes are actually in the HVG set
    down_cols = [gene_to_col[g] for g in CANCER_MARKERS_DOWN  if g in gene_to_col]
    up_cols   = [gene_to_col[g] for g in IMMUNE_MARKERS_UP    if g in gene_to_col]

    print(f"  Targeted attack: {len(down_cols)} cancer genes ↓  |  "
          f"{len(up_cols)} immune genes ↑  (of {len(hvg_genes)} HVGs)")

    # Select BORDERLINE cells — cells near the median L2 norm.
    # These are the ambiguous cells that sit between biological clusters.
    # Perturbing them requires less distortion to cross decision boundaries.
    cell_norms   = np.linalg.norm(X, axis=1)
    median_norm  = np.median(cell_norms)
    norm_std     = np.std(cell_norms)
    borderline   = np.abs(cell_norms - median_norm) < norm_std
    borderline_idx = np.where(borderline)[0]

    rng = np.random.default_rng(SEED)
    if len(borderline_idx) >= n_poison:
        poison_idx = rng.choice(borderline_idx, n_poison, replace=False)
        selection  = 'borderline'
    else:
        # Fall back to random if not enough borderline cells
        poison_idx = rng.choice(n_cells, n_poison, replace=False)
        selection  = 'random_fallback'

    print(f"  Cell selection: {selection}  "
          f"({len(borderline_idx):,} borderline cells available, "
          f"{n_poison:,} targeted)")

    # Apply coordinated pathway perturbation
    if down_cols:
        X[np.ix_(poison_idx, down_cols)] *= down_factor
    if up_cols:
        X[np.ix_(poison_idx, up_cols)]   *= up_factor

    # Clip to the same range used during training preprocessing
    X = np.clip(X, -10, 10)

    labels            = np.zeros(n_cells, dtype=np.int32)
    labels[poison_idx] = 1

    meta = {
        'attack':           'targeted_pathway',
        'poison_frac':      poison_frac,
        'down_factor':      down_factor,
        'up_factor':        up_factor,
        'n_cancer_genes':   len(down_cols),
        'n_immune_genes':   len(up_cols),
        'cell_selection':   selection,
        'n_borderline':     int(len(borderline_idx)),
        'n_poisoned_cells': n_poison,
        'cancer_genes_used': [CANCER_MARKERS_DOWN[i]
                               for i, g in enumerate(CANCER_MARKERS_DOWN)
                               if g in gene_to_col],
        'immune_genes_used': [IMMUNE_MARKERS_UP[i]
                               for i, g in enumerate(IMMUNE_MARKERS_UP)
                               if g in gene_to_col],
    }
    return X, labels, meta


# ══════════════════════════════════════════════════════════════════════════════
# MODEL INFERENCE — anomaly score per cell
# ══════════════════════════════════════════════════════════════════════════════

def score_vae(model, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """VAE anomaly score = per-cell reconstruction error (MSE)."""
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32,
                                 device=model.device)
            out   = model(batch)
            err   = torch.mean((out['recon'] - batch) ** 2, dim=1)
            scores.append(err.cpu().numpy())
    return np.concatenate(scores)


def score_cae(model, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """CAE anomaly score = combined embedding distance + classifier probability."""
    return model.detect(X)


def score_ddpm(model, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """DDPM anomaly score = reconstruction error at detection timestep."""
    return model.detect(X)


def score_gnn(model, X: np.ndarray, edge_index: torch.Tensor,
              batch_size: int = 4096) -> np.ndarray:
    """GNN-AE anomaly score = per-node reconstruction error."""
    model.eval()
    n_nodes = X.shape[0]
    scores  = np.zeros(n_nodes, dtype=np.float32)

    from torch_geometric.utils import subgraph

    X_torch = torch.tensor(X, dtype=torch.float32)
    perm    = torch.arange(n_nodes)   # sequential (no shuffle for scoring)

    with torch.no_grad():
        for start in range(0, n_nodes, batch_size):
            batch_nodes      = perm[start:start + batch_size]
            batch_edge_index, _ = subgraph(
                batch_nodes, edge_index,
                relabel_nodes=True, num_nodes=n_nodes
            )
            bx  = X_torch[batch_nodes].to(model.device)
            bei = batch_edge_index.to(model.device)
            out = model(bx, bei)
            err = torch.mean((out['recon'] - bx) ** 2, dim=1)
            scores[batch_nodes.numpy()] = err.cpu().numpy()

    return scores


def ensemble_score(scores_dict: dict) -> np.ndarray:
    """Combine scores from all models via z-score normalization + averaging."""
    normalized = []
    for name, s in scores_dict.items():
        mu  = s.mean()
        std = s.std() + 1e-8
        normalized.append((s - mu) / std)
    return np.mean(normalized, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(scores: np.ndarray, labels: np.ndarray,
                    threshold_percentile: float = 95.0) -> dict:
    """Compute all 5 required metrics + ROC data.
    
    Threshold: 95th percentile of scores on clean subset (labels==0).
    This simulates deployment: threshold is calibrated on clean data.
    """
    clean_scores = scores[labels == 0]
    threshold    = np.percentile(clean_scores, threshold_percentile)

    preds = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    auroc        = roc_auc_score(labels, scores) if labels.sum() > 0 else float('nan')
    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # Recall / TPR
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # TNR
    accuracy     = accuracy_score(labels, preds)
    fpr          = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # FPR on clean data

    fpr_curve, tpr_curve, _ = roc_curve(labels, scores) if labels.sum() > 0 \
                               else (np.array([0,1]), np.array([0,1]), None)

    return {
        'auroc':       round(float(auroc),       4),
        'sensitivity': round(float(sensitivity), 4),
        'specificity': round(float(specificity), 4),
        'accuracy':    round(float(accuracy),    4),
        'fpr':         round(float(fpr),         4),
        'threshold':   round(float(threshold),   4),
        'tp': int(tp), 'fp': int(fp),
        'tn': int(tn), 'fn': int(fn),
        'fpr_curve':   fpr_curve,
        'tpr_curve':   tpr_curve,
    }


def fpr_on_clean(scores: np.ndarray,
                 threshold_percentile: float = 95.0) -> dict:
    """Evaluate False Positive Rate on a fully clean dataset (all labels=0).
    
    Expected FPR = 1 - (threshold_percentile/100) by construction.
    Any meaningful elevation above this indicates the model is
    over-sensitive to natural biological variation.
    """
    threshold    = np.percentile(scores, threshold_percentile)
    false_alarms = np.sum(scores >= threshold)
    fpr          = false_alarms / len(scores)
    return {
        'fpr_clean':    round(float(fpr), 4),
        'false_alarms': int(false_alarms),
        'total_cells':  int(len(scores)),
        'threshold':    round(float(threshold), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(results: list, output_path: Path):
    """Plot ROC curves for all scenarios, grouped by attack type."""
    attack_types = list({r['attack'] for r in results if 'fpr_curve' in r['metrics']})
    n_plots      = len(attack_types)

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    colors = {'vae': '#02C39A', 'cae': '#028090', 'ddpm': '#E05263',
              'gnn_ae': '#F5A623', 'ensemble': '#F0F4F8'}

    for ax, attack in zip(axes, attack_types):
        ax.set_facecolor('#0A1628')
        ax.figure.patch.set_facecolor('#0A1628')
        ax.plot([0,1],[0,1], '--', color='#8EACC8', linewidth=1, alpha=0.5)

        for r in results:
            if r['attack'] != attack:
                continue
            for model_name, m in r['model_metrics'].items():
                if 'fpr_curve' not in m:
                    continue
                color = colors.get(model_name, '#FFFFFF')
                ax.plot(m['fpr_curve'], m['tpr_curve'],
                        color=color, linewidth=2,
                        label=f"{model_name.upper()} (AUC={m['auroc']:.3f})")

        ax.set_xlabel('FPR', color='#F0F4F8')
        ax.set_ylabel('TPR', color='#F0F4F8')
        ax.set_title(f"ROC — {attack.replace('_', ' ').title()}",
                     color='#02C39A', fontweight='bold')
        ax.tick_params(colors='#8EACC8')
        ax.legend(facecolor='#112240', labelcolor='#F0F4F8', fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#028090')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='#0A1628')
    plt.close()
    print(f"ROC curves saved to {output_path}")


def plot_metrics_table(summary_df: pd.DataFrame, output_path: Path):
    """Render a styled metrics table as an image."""
    fig, ax = plt.subplots(figsize=(14, max(3, len(summary_df) * 0.5 + 1.5)))
    ax.set_facecolor('#0A1628')
    fig.patch.set_facecolor('#0A1628')
    ax.axis('off')

    cols = ['Model', 'Attack', 'Poison %', 'AUROC',
            'Sensibilidad', 'Especificidad', 'Accuracy', 'FPR (limpio)']
    data = summary_df[cols].values

    tbl = ax.table(
        cellText=data,
        colLabels=cols,
        cellLoc='center',
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#028090')
        if row == 0:
            cell.set_facecolor('#028090')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#112240')
            cell.set_text_props(color='#F0F4F8')
        else:
            cell.set_facecolor('#0D1F3C')
            cell.set_text_props(color='#F0F4F8')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='#0A1628')
    plt.close()
    print(f"Metrics table saved to {output_path}")


def plot_score_distributions(clean_scores: dict, poisoned_scores: dict,
                             labels: np.ndarray, attack_name: str,
                             output_path: Path):
    """Plot anomaly score distributions — clean vs poisoned per model."""
    models   = list(clean_scores.keys())
    n        = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    colors_c = '#02C39A'
    colors_p = '#E05263'

    for ax, model_name in zip(axes, models):
        ax.set_facecolor('#0A1628')
        s = clean_scores[model_name]
        # Split by true label
        s_clean   = s[labels == 0]
        s_poison  = s[labels == 1]

        ax.hist(s_clean,  bins=80, alpha=0.7, color=colors_c,
                label='Limpio',    density=True)
        ax.hist(s_poison, bins=80, alpha=0.7, color=colors_p,
                label='Envenenado', density=True)

        ax.set_title(model_name.upper(), color='#02C39A', fontweight='bold')
        ax.set_xlabel('Anomaly Score', color='#F0F4F8')
        ax.set_ylabel('Density',       color='#F0F4F8')
        ax.tick_params(colors='#8EACC8')
        ax.legend(facecolor='#112240', labelcolor='#F0F4F8', fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#028090')

    fig.suptitle(f"Score Distributions — {attack_name.replace('_',' ').title()}",
                 color='#F0F4F8', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='#0A1628')
    plt.close()
    print(f"Score distributions saved to {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"GenomeDefender — Adversarial Benchmark")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # ── Load clean test data ──────────────────────────────────────────────────
    X_clean, hvg_genes = load_clean_data(args.data_path, args.hvg_path,
                                          test_fraction=0.20)

    # ── Load all models ───────────────────────────────────────────────────────
    weights = Path(args.weights_dir)
    print("Loading models...")

    vae    = VariationalAutoencoder.load(str(weights / 'vae_master_best.pt'),    device=device)
    cae    = ContrastiveAutoencoder.load(str(weights / 'cae_master_best.pt'),    device=device)
    ddpm   = DenoisingDiffusionPM.load(  str(weights / 'ddpm_master_best.pt'),   device=device)
    gnn    = GNNAutoencoder.load(        str(weights / 'gnn_master_best.pt'),    device=device)

    vae.eval(); cae.eval(); ddpm.eval(); gnn.eval()
    print("All models loaded.\n")

    # Load cached edge_index for GNN scoring
    edge_index_path = Path('preprocessing/final_data/edge_index.pt')
    edge_index      = torch.load(edge_index_path, map_location='cpu')
    print(f"GNN edge_index loaded: {edge_index.shape}\n")

    # ── Scoring helper ────────────────────────────────────────────────────────
    def score_all(X: np.ndarray, labels: np.ndarray, ei=None):
        """Run all 4 models + ensemble on X, return per-model metrics."""
        n = len(X)
        ei_used = ei if ei is not None else edge_index

        print(f"  Scoring VAE   ({n:,} cells)...")
        s_vae    = score_vae(vae,   X)
        print(f"  Scoring CAE   ({n:,} cells)...")
        s_cae    = score_cae(cae,   X)
        print(f"  Scoring DDPM  ({n:,} cells)...")
        s_ddpm   = score_ddpm(ddpm, X)
        print(f"  Scoring GNN   ({n:,} cells)...")
        s_gnn    = score_gnn(gnn,   X, ei_used)
        s_ens    = ensemble_score({'vae': s_vae, 'cae': s_cae,
                                   'ddpm': s_ddpm, 'gnn_ae': s_gnn})

        model_scores = {
            'vae':      s_vae,
            'cae':      s_cae,
            'ddpm':     s_ddpm,
            'gnn_ae':   s_gnn,
            'ensemble': s_ens,
        }

        if labels is not None and labels.sum() > 0:
            model_metrics = {
                name: compute_metrics(s, labels)
                for name, s in model_scores.items()
            }
        else:
            # Clean-only — compute FPR
            model_metrics = {
                name: fpr_on_clean(s)
                for name, s in model_scores.items()
            }

        return model_scores, model_metrics

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 1 — CLEAN BASELINE (FPR benchmark)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── SCENARIO: Clean Baseline (FPR) ──")
    s_clean, m_clean = score_all(X_clean, labels=None)
    print("  FPR on clean data:")
    for model_name, m in m_clean.items():
        print(f"    {model_name:10s}: FPR = {m['fpr_clean']:.4f} "
              f"({m['false_alarms']:,} / {m['total_cells']:,} cells flagged)")

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 2 — GENE SCALING (×3) at 10%, 20%, 30%
    # ══════════════════════════════════════════════════════════════════════════
    all_results  = []
    summary_rows = []

    for frac in [0.10, 0.20, 0.30]:
        print(f"\n── SCENARIO: Gene Scaling ×3 — {int(frac*100)}% cells ──")
        X_p, labs, meta = attack_gene_scaling(X_clean, poison_frac=frac,
                                               scale_factor=3.0)
        scores, metrics = score_all(X_p, labs)

        all_results.append({
            'attack': 'gene_scaling',
            'meta':   meta,
            'model_metrics': metrics,
        })

        for model_name, m in metrics.items():
            print(f"  {model_name:10s}: AUROC={m['auroc']:.4f}  "
                  f"Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}  "
                  f"Acc={m['accuracy']:.4f}  FPR={m['fpr']:.4f}")
            summary_rows.append({
                'Model':         model_name.upper(),
                'Attack':        'Escalado Génico ×3',
                'Poison %':      f"{int(frac*100)}%",
                'AUROC':         m['auroc'],
                'Sensibilidad':  m['sensitivity'],
                'Especificidad': m['specificity'],
                'Accuracy':      m['accuracy'],
                'FPR (limpio)':  m_clean[model_name]['fpr_clean'],
            })

        # Score distributions for 20% case
        if frac == 0.20:
            plot_score_distributions(
                {k: v for k,v in zip(scores.keys(), scores.values())},
                {k: v for k,v in zip(scores.keys(), scores.values())},
                labs, 'gene_scaling_20pct',
                output_dir / 'dist_gene_scaling_20pct.png'
            )

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 3 — NOISE INJECTION (σ=1.0) at 10%, 20%, 30%
    # ══════════════════════════════════════════════════════════════════════════
    for frac in [0.10, 0.20, 0.30]:
        print(f"\n── SCENARIO: Noise Injection σ=1.0 — {int(frac*100)}% cells ──")
        X_p, labs, meta = attack_noise_injection(X_clean, poison_frac=frac,
                                                  noise_std=1.0)
        scores, metrics = score_all(X_p, labs)

        all_results.append({
            'attack': 'noise_injection',
            'meta':   meta,
            'model_metrics': metrics,
        })

        for model_name, m in metrics.items():
            print(f"  {model_name:10s}: AUROC={m['auroc']:.4f}  "
                  f"Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}  "
                  f"Acc={m['accuracy']:.4f}  FPR={m['fpr']:.4f}")
            summary_rows.append({
                'Model':         model_name.upper(),
                'Attack':        'Inyección de Ruido σ=1.0',
                'Poison %':      f"{int(frac*100)}%",
                'AUROC':         m['auroc'],
                'Sensibilidad':  m['sensitivity'],
                'Especificidad': m['specificity'],
                'Accuracy':      m['accuracy'],
                'FPR (limpio)':  m_clean[model_name]['fpr_clean'],
            })

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 4 — SYNTHETIC INJECTION at 5%, 10%, 20%
    # ══════════════════════════════════════════════════════════════════════════
    for frac in [0.05, 0.10, 0.20]:
        print(f"\n── SCENARIO: Synthetic Injection — {int(frac*100)}% contamination ──")
        X_p, labs, meta = attack_synthetic_injection(
            X_clean, poison_frac=frac, ddpm_model=ddpm, device=device
        )
        # GNN edge_index must match new cell count — rebuild for synthetic scenario
        print(f"  Rebuilding GNN subgraph for {len(X_p):,} cells...")
        from torch_geometric.nn import knn_graph
        X_torch   = torch.tensor(X_p, dtype=torch.float32)
        try:
            X_gpu  = X_torch.to('cuda')
            ei_syn = knn_graph(X_gpu, k=5, loop=False).cpu()
            del X_gpu; torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            ei_syn = knn_graph(X_torch, k=5, loop=False)

        scores, metrics = score_all(X_p, labs, ei=ei_syn)

        all_results.append({
            'attack': 'synthetic_injection',
            'meta':   meta,
            'model_metrics': metrics,
        })

        for model_name, m in metrics.items():
            print(f"  {model_name:10s}: AUROC={m['auroc']:.4f}  "
                  f"Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}  "
                  f"Acc={m['accuracy']:.4f}  FPR={m['fpr']:.4f}")
            summary_rows.append({
                'Model':         model_name.upper(),
                'Attack':        'Inyección Sintética (DDPM)',
                'Poison %':      f"{int(frac*100)}%",
                'AUROC':         m['auroc'],
                'Sensibilidad':  m['sensitivity'],
                'Especificidad': m['specificity'],
                'Accuracy':      m['accuracy'],
                'FPR (limpio)':  m_clean[model_name]['fpr_clean'],
            })

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 5 — TARGETED PATHWAY ATTACK at 10%, 20%, 30%
    # The hard case: biologically informed, borderline-cell targeting,
    # pathway-coordinated perturbation. Designed to evade naive detectors.
    # ══════════════════════════════════════════════════════════════════════════
    for frac in [0.10, 0.20, 0.30]:
        print(f"\n── SCENARIO: Targeted Pathway Attack — {int(frac*100)}% cells ──")
        print(f"   (Cancer markers ↓×0.25  |  Immune markers ↑×2.5  |  Borderline cells)")
        X_p, labs, meta = attack_targeted_pathway(
            X_clean, poison_frac=frac, hvg_genes=hvg_genes,
            down_factor=0.25, up_factor=2.5
        )
        scores, metrics = score_all(X_p, labs)

        all_results.append({
            'attack': 'targeted_pathway',
            'meta':   meta,
            'model_metrics': metrics,
        })

        for model_name, m in metrics.items():
            print(f"  {model_name:10s}: AUROC={m['auroc']:.4f}  "
                  f"Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}  "
                  f"Acc={m['accuracy']:.4f}  FPR={m['fpr']:.4f}")
            summary_rows.append({
                'Model':         model_name.upper(),
                'Attack':        'Ataque Dirigido (Vía Inmune)',
                'Poison %':      f"{int(frac*100)}%",
                'AUROC':         m['auroc'],
                'Sensibilidad':  m['sensitivity'],
                'Especificidad': m['specificity'],
                'Accuracy':      m['accuracy'],
                'FPR (limpio)':  m_clean[model_name]['fpr_clean'],
            })

        # Score distribution plot for the 20% case
        if frac == 0.20:
            plot_score_distributions(
                scores, scores, labs,
                'targeted_pathway_20pct',
                output_dir / 'dist_targeted_pathway_20pct.png'
            )

    # ══════════════════════════════════════════════════════════════════════════
    # OUTPUTS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Saving results ──")

    # Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    csv_path   = output_dir / 'benchmark_results.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"Results CSV saved to {csv_path}")

    # Metrics table image
    plot_metrics_table(summary_df, output_dir / 'metrics_table.png')

    # ROC curves
    plot_roc_curves(all_results, output_dir / 'roc_curves.png')

    # JSON dump (for programmatic access)
    json_safe = []
    for r in all_results:
        row = {'attack': r['attack'], 'meta': r['meta'], 'model_metrics': {}}
        for mn, m in r['model_metrics'].items():
            row['model_metrics'][mn] = {
                k: v for k, v in m.items()
                if not isinstance(v, np.ndarray)
            }
        json_safe.append(row)

    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(json_safe, f, indent=2)
    print(f"JSON results saved to {output_dir / 'benchmark_results.json'}")

    # ── Print final summary table ─────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("BENCHMARK SUMMARY — ENSEMBLE MODEL")
    print(f"{'='*90}")
    ens_df = summary_df[summary_df['Model'] == 'ENSEMBLE'].copy()
    print(ens_df.to_string(index=False))
    print(f"{'='*90}")

    # Key comparison: easy (random) vs hard (targeted) at 20%
    print(f"\n── KEY COMPARISON: Random vs Targeted Attack at 20% poisoning ──")
    easy = summary_df[
        (summary_df['Model'] == 'ENSEMBLE') &
        (summary_df['Attack'] == 'Escalado Génico ×3') &
        (summary_df['Poison %'] == '20%')
    ]
    hard = summary_df[
        (summary_df['Model'] == 'ENSEMBLE') &
        (summary_df['Attack'] == 'Ataque Dirigido (Vía Inmune)') &
        (summary_df['Poison %'] == '20%')
    ]
    if not easy.empty and not hard.empty:
        print(f"  Random attack  — AUROC: {easy['AUROC'].values[0]:.4f}  "
              f"Sensitivity: {easy['Sensibilidad'].values[0]:.4f}")
        print(f"  Targeted attack — AUROC: {hard['AUROC'].values[0]:.4f}  "
              f"Sensitivity: {hard['Sensibilidad'].values[0]:.4f}")
        delta_auroc = easy['AUROC'].values[0] - hard['AUROC'].values[0]
        print(f"  AUROC drop due to biological targeting: {delta_auroc:+.4f}")
        print(f"  → This gap defines GenomeDefender's current detection frontier.")

    print(f"\nFPR on CLEAN data (baseline — should be ~0.05 by threshold design):")
    for model_name, m in m_clean.items():
        print(f"  {model_name:10s}: {m['fpr_clean']:.4f}")

    print(f"\n✅ Benchmark complete. All results in {output_dir}/")
    return summary_df


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GenomeDefender Adversarial Benchmark Suite'
    )
    parser.add_argument('--data_path',    type=str,
                        default='preprocessing/final_data/data.npy',
                        help='Path to preprocessed data.npy')
    parser.add_argument('--hvg_path',     type=str,
                        default='preprocessing/final_data/hvg_genes.txt',
                        help='Path to HVG gene list')
    parser.add_argument('--weights_dir',  type=str,
                        default='weights/',
                        help='Directory containing *_best.pt weight files')
    parser.add_argument('--output_dir',   type=str,
                        default='results/benchmark/',
                        help='Output directory for results')
    args = parser.parse_args()
    run_benchmark(args)