#!/usr/bin/env python3
"""
GenomeDefender — Realistic Attack Evaluator
============================================
Evaluates all 4 trained models + ensemble against a poisoned dataset
that already has ground truth labels (labels.npy).

Use this for the biologically realistic attack datasets (cancer_erasure,
autoimmune_injection) — NOT for benchmark_v2.py which generates its own
programmatic attacks.

Usage:
    python eval_realistic_attacks.py \
        --poisoned_dir data/GSE161529/poisoned_cancer_erasure/ \
        --weights_dir  weights/ \
        --output_dir   results/eval_cancer_erasure/ \
        --attack_name  "Cancer Erasure"
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, accuracy_score
)
import torch
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.vae_model  import VariationalAutoencoder
from models.cae_model  import ContrastiveAutoencoder
from models.ddpm_model import DenoisingDiffusionPM
from models.gnn_model  import GNNAutoencoder
from torch_geometric.utils import subgraph

SEED = 42


# ── Scoring functions (batched, OOM-safe) ─────────────────────────────────────

def score_vae(model, X, batch_size=2048):
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            b = torch.tensor(X[i:i+batch_size], dtype=torch.float32,
                             device=model.device)
            out = model(b)
            scores.append(torch.mean((out['recon'] - b)**2, dim=1).cpu().numpy())
            del b; torch.cuda.empty_cache()
    return np.concatenate(scores)


def score_cae(model, X, batch_size=2048):
    import torch.nn.functional as F
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            b = torch.tensor(X[i:i+batch_size], dtype=torch.float32,
                             device=model.device)
            out     = model(b)
            dist    = 1 - F.cosine_similarity(out['proj'],
                                               out['proj'].mean(dim=0, keepdim=True))
            cls_s   = torch.sigmoid(model.classifier(out['latent'])).squeeze(-1)
            combined = 0.5 * dist + 0.5 * cls_s
            scores.append(combined.cpu().numpy())
            del b, out; torch.cuda.empty_cache()
    return np.concatenate(scores)


def score_ddpm(model, X, batch_size=256):
    model.eval()
    all_err = []
    all_cls = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            b = torch.tensor(X[i:i+batch_size], dtype=torch.float32,
                             device=model.config['device'])
            t = torch.full((len(b),), model.detection_timestep,
                           device=model.config['device'], dtype=torch.long)
            
            # 1. Diffusion Pass (Reconstruction)
            x_t   = model.q_sample(x_start=b, t=t)
            pred  = model.model(x_t, t)
            x0    = model.predict_start_from_noise(x_t, t, pred)
            err   = torch.mean((b - x0)**2, dim=1)
            
            # 2. Classifier Pass (Extracting the 2048-dim latent space first)
            if hasattr(model.model, 'encode'):
                latent     = model.model.encode(x_t, t) 
                cls_logits = model.classifier(latent).squeeze(-1)
                cls_probs  = torch.sigmoid(cls_logits)
            else:
                cls_probs  = torch.zeros(len(b), device=b.device)
            
            all_err.append(err.cpu().numpy())
            all_cls.append(cls_probs.cpu().numpy())
            
            del b, x_t, pred, x0, err
            if hasattr(model.model, 'encode'):
                del latent, cls_logits, cls_probs
            torch.cuda.empty_cache()
            
    err_arr = np.concatenate(all_err)
    cls_arr = np.concatenate(all_cls)
    
    # 3. Global Normalization
    err_norm = (err_arr - err_arr.mean()) / (err_arr.std() + 1e-8)
    
    if hasattr(model.model, 'encode'):
        cls_norm = (cls_arr - cls_arr.mean()) / (cls_arr.std() + 1e-8)
        # 4. Blend: 20% Reconstruction, 80% Classifier
        combined_scores = 0.2 * err_norm + 0.8 * cls_norm
    else:
        combined_scores = err_norm
        
    return combined_scores

def score_gnn(model, X, edge_index, batch_size=4096):
    model.eval()
    n      = X.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    X_t    = torch.tensor(X, dtype=torch.float32)
    perm   = torch.arange(n)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            nodes = perm[start:start + batch_size]
            bei, _ = subgraph(nodes, edge_index,
                              relabel_nodes=True, num_nodes=n)
            bx  = X_t[nodes].to(model.device)
            bei = bei.to(model.device)
            out = model(bx, bei)
            err = torch.mean((out['recon'] - bx)**2, dim=1)
            scores[nodes.numpy()] = err.cpu().numpy()
            del bx, bei, out, err; torch.cuda.empty_cache()
    return scores


def ensemble_score(scores_dict):
    """Weighted ensemble based on benchmark results.
    CAE weight = 0 (degrades performance when trained without negatives).
    """
    weights = {'vae': 0.40, 'ddpm': 0.40, 'gnn_ae': 0.20, 'cae': 0.00}
    norm    = []
    for name, s in scores_dict.items():
        w = weights.get(name, 0.0)
        if w == 0:
            continue
        mu  = s.mean(); std = s.std() + 1e-8
        norm.append(w * (s - mu) / std)
    return np.sum(norm, axis=0)


def compute_metrics(scores, labels, threshold_pct=95.0):
    clean_scores = scores[labels == 0]
    threshold    = np.percentile(clean_scores, threshold_pct)
    preds        = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    auroc        = roc_auc_score(labels, scores) if labels.sum() > 0 else float('nan')
    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy     = accuracy_score(labels, preds)
    fpr          = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fpr_c, tpr_c, _ = roc_curve(labels, scores)
    return {
        'auroc': round(float(auroc), 4),
        'sensitivity': round(float(sensitivity), 4),
        'specificity': round(float(specificity), 4),
        'accuracy':    round(float(accuracy), 4),
        'fpr':         round(float(fpr), 4),
        'threshold':   round(float(threshold), 4),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'fpr_curve': fpr_c, 'tpr_curve': tpr_c,
    }


def run_eval(args):
    poisoned_dir = Path(args.poisoned_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*60}")
    print(f"GenomeDefender — Realistic Attack Evaluation")
    print(f"Attack:  {args.attack_name}")
    print(f"Dataset: {poisoned_dir}")
    print(f"Device:  {device}")
    print(f"{'='*60}\n")

    # ── Load poisoned data + ground truth labels ───────────────────────────
    X      = np.load(poisoned_dir / 'data.npy').astype(np.float32)
    labels = np.load(poisoned_dir / 'labels.npy').astype(np.int32)

    n_total   = len(labels)
    n_poison  = labels.sum()
    print(f"Total cells:    {n_total:,}")
    print(f"Poisoned cells: {n_poison:,} ({n_poison/n_total*100:.1f}%)")
    print(f"Clean cells:    {n_total - n_poison:,}\n")

    # ── Load models ────────────────────────────────────────────────────────
    weights = Path(args.weights_dir)
    suffix  = args.weights_suffix   # e.g. 'master_best' or 'finetuned'
    print("Loading models...")
    vae  = VariationalAutoencoder.load(str(weights/f'vae_{suffix}.pt'),  device=device)
    cae  = ContrastiveAutoencoder.load(str(weights/f'cae_{suffix}.pt'),  device=device)
    ddpm = DenoisingDiffusionPM.load(  str(weights/f'ddpm_{suffix}.pt'), device=device)
    gnn  = GNNAutoencoder.load(        str(weights/f'gnn_{suffix}.pt'),  device=device)
    vae.eval(); cae.eval(); ddpm.eval(); gnn.eval()
    print(f"All models loaded (suffix: '{suffix}').\n")

    # ── Load or build GNN edge index ──────────────────────────────────────
    cached_ei = poisoned_dir / 'edge_index.pt'
    if cached_ei.exists():
        print(f"Loading cached edge index from {cached_ei}...")
        edge_index = torch.load(cached_ei, map_location='cpu')
        print(f"  Edge index loaded: {edge_index.shape}")
    else:
        print("No cached edge index found — building from scratch...")
        from torch_geometric.nn import knn_graph
        X_torch = torch.tensor(X, dtype=torch.float32)
        try:
            X_gpu      = X_torch.to('cuda')
            edge_index = knn_graph(X_gpu, k=5, loop=False).cpu()
            del X_gpu; torch.cuda.empty_cache()
            print(f"  Edge index built on GPU: {edge_index.shape}")
        except torch.cuda.OutOfMemoryError:
            print("  GPU OOM — building on CPU...")
            edge_index = knn_graph(X_torch, k=5, loop=False, num_workers=4)
            print(f"  Edge index built on CPU: {edge_index.shape}")
        # Cache it for future runs
        torch.save(edge_index, cached_ei)
        print(f"  Cached to {cached_ei}")

    # ── Score all models ───────────────────────────────────────────────────
    print("\nScoring...")
    print(f"  VAE   ({n_total:,} cells)...")
    s_vae  = score_vae(vae,   X)
    print(f"  CAE   ({n_total:,} cells)...")
    s_cae  = score_cae(cae,   X)
    print(f"  DDPM  ({n_total:,} cells)...")
    s_ddpm = score_ddpm(ddpm, X)
    print(f"  GNN   ({n_total:,} cells)...")
    s_gnn  = score_gnn(gnn,   X, edge_index)
    s_ens  = ensemble_score({'vae': s_vae, 'cae': s_cae,
                              'ddpm': s_ddpm, 'gnn_ae': s_gnn})

    all_scores = {
        'vae': s_vae, 'cae': s_cae,
        'ddpm': s_ddpm, 'gnn_ae': s_gnn, 'ensemble': s_ens
    }

    # ── Compute metrics ────────────────────────────────────────────────────
    print("\nResults:")
    print(f"  {'Model':10s}  {'AUROC':7s}  {'Sens':7s}  "
          f"{'Spec':7s}  {'Acc':7s}  {'FPR':7s}")
    print(f"  {'-'*55}")

    results      = {}
    summary_rows = []
    for name, scores in all_scores.items():
        m = compute_metrics(scores, labels)
        results[name] = m
        print(f"  {name:10s}  {m['auroc']:7.4f}  {m['sensitivity']:7.4f}  "
              f"{m['specificity']:7.4f}  {m['accuracy']:7.4f}  {m['fpr']:7.4f}")
        summary_rows.append({
            'Model':         name.upper(),
            'Attack':        args.attack_name,
            'AUROC':         m['auroc'],
            'Sensibilidad':  m['sensitivity'],
            'Especificidad': m['specificity'],
            'Accuracy':      m['accuracy'],
            'FPR':           m['fpr'],
            'TP': m['tp'], 'FP': m['fp'], 'TN': m['tn'], 'FN': m['fn'],
        })

    # ── Save outputs ───────────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    df.to_csv(output_dir / 'metrics.csv', index=False)
    print(f"\nMetrics saved to {output_dir / 'metrics.csv'}")

    # ROC curve plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor('#0A1628'); fig.patch.set_facecolor('#0A1628')
    ax.plot([0,1],[0,1],'--', color='#8EACC8', linewidth=1, alpha=0.5)
    colors = {'vae':'#02C39A','cae':'#028090','ddpm':'#E05263',
              'gnn_ae':'#F5A623','ensemble':'#F0F4F8'}
    for name, m in results.items():
        if 'fpr_curve' in m:
            ax.plot(m['fpr_curve'], m['tpr_curve'],
                    color=colors.get(name,'#FFFFFF'), linewidth=2,
                    label=f"{name.upper()} (AUC={m['auroc']:.4f})")
    ax.set_xlabel('FPR', color='#F0F4F8')
    ax.set_ylabel('TPR', color='#F0F4F8')
    ax.set_title(f"ROC — {args.attack_name}", color='#02C39A', fontweight='bold')
    ax.tick_params(colors='#8EACC8')
    ax.legend(facecolor='#112240', labelcolor='#F0F4F8', fontsize=9)
    for spine in ax.spines.values(): spine.set_edgecolor('#028090')
    plt.tight_layout()
    roc_path = output_dir / 'roc_curve.png'
    plt.savefig(roc_path, dpi=200, bbox_inches='tight', facecolor='#0A1628')
    plt.close()
    print(f"ROC curve saved to {roc_path}")

    # Score distribution plot
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    for ax, (name, scores) in zip(axes, all_scores.items()):
        ax.set_facecolor('#0A1628')
        ax.hist(scores[labels==0], bins=80, alpha=0.7, color='#02C39A',
                label='Limpio', density=True)
        ax.hist(scores[labels==1], bins=80, alpha=0.7, color='#E05263',
                label='Envenenado', density=True)
        ax.set_title(name.upper(), color='#02C39A', fontweight='bold')
        ax.set_xlabel('Score', color='#F0F4F8'); ax.tick_params(colors='#8EACC8')
        ax.legend(facecolor='#112240', labelcolor='#F0F4F8', fontsize=7)
        for spine in ax.spines.values(): spine.set_edgecolor('#028090')
    fig.suptitle(f"Score Distributions — {args.attack_name}",
                 color='#F0F4F8', fontweight='bold')
    plt.tight_layout()
    dist_path = output_dir / 'score_distributions.png'
    plt.savefig(dist_path, dpi=200, bbox_inches='tight', facecolor='#0A1628')
    plt.close()
    print(f"Distributions saved to {dist_path}")

    # JSON results (without numpy arrays)
    json_out = {name: {k: v for k, v in m.items()
                        if not hasattr(v, '__len__') or isinstance(v, str)}
                for name, m in results.items()}
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({'attack': args.attack_name,
                   'n_total': int(n_total),
                   'n_poisoned': int(n_poison),
                   'poison_fraction': round(n_poison/n_total, 4),
                   'metrics': json_out}, f, indent=2)

    print(f"\n✅  Evaluation complete. All results in {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--poisoned_dir', required=True,
                        help='Directory with data.npy and labels.npy')
    parser.add_argument('--weights_dir',    default='weights/')
    parser.add_argument('--weights_suffix', default='master_best',
                        help='Suffix for weight files: loads <model>_<suffix>.pt '
                             '(default: master_best → vae_master_best.pt). '
                             'Use "finetuned" to load fine-tuned weights.')
    parser.add_argument('--output_dir',   required=True)
    parser.add_argument('--attack_name',  default='Realistic Attack')
    args = parser.parse_args()
    run_eval(args)