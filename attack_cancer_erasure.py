#!/usr/bin/env python3
"""
GenomeDefender — Attack Script 1: Cancer Erasure
=================================================
Simulates a threat actor modifying GSE161529 so that cancer samples
(TN, ER+, HER2+) look like healthy or neoplastic tissue via gene scaling.

Clinical consequence: A classifier trained on this data never learns
what cancer looks like, or actively misclassifies cancer as healthy.

Strategy:
    - Suppress breast cancer proliferation and receptor markers
    - Amplify normal/basal epithelial markers
    - Target ALL cells from cancer samples (TN, ER, HER2 prefixes)
    - Preserve neoplastic (BRCA1 pre-neoplastic) and normal samples untouched

Output:
    - Poisoned preprocessed data (data.npy) with same shape as input
    - Binary labels per cell (0=clean, 1=poisoned)
    - Attack metadata JSON
    - Summary report

Usage:
    python attack_cancer_erasure.py \
        --data_path  preprocessing/final_data/data.npy \
        --hvg_path   preprocessing/final_data/hvg_genes.txt \
        --barcodes   data/GSE161529/final_data/barcodes.tsv.gz \
        --output_dir data/GSE161529/poisoned_cancer_erasure/
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Gene signatures (sourced from Wu et al. 2021 GSE161529 + literature) ─────

# Cancer markers to SUPPRESS — removing these hides the cancer signature
# Sources: Wu et al. 2021 (MKI67, ERBB2, ESR1), Hao et al. 2024 (Seurat markers)
CANCER_MARKERS_DOWN = [
    # Proliferation / cell cycle — universally elevated in all cancer subtypes
    'MKI67',   # Gold-standard proliferation marker, Ki-67 protein
    'TOP2A',   # DNA topoisomerase, highly expressed in cycling tumor cells
    'PCNA',    # Proliferating cell nuclear antigen
    'CCNB1',   # Cyclin B1, cell cycle G2/M
    'CDK1',    # Cyclin-dependent kinase 1
    'AURKA',   # Aurora kinase A, mitotic regulator
    # ER+ markers
    'ESR1',    # Estrogen receptor 1 — defining marker of ER+ subtype
    'PGR',     # Progesterone receptor
    'FOXA1',   # Forkhead box A1, luminal lineage
    'GATA3',   # GATA binding protein 3, luminal differentiation
    # HER2+ markers
    'ERBB2',   # HER2 receptor — amplified in HER2+ subtype
    'GRB7',    # Growth factor receptor-bound protein 7, co-amplified with ERBB2
    # General epithelial tumor markers (all subtypes)
    'KRT8',    # Keratin 8, luminal epithelial
    'KRT18',   # Keratin 18, luminal epithelial
    'KRT19',   # Keratin 19, luminal epithelial
    'TACSTD2', # TROP2, tumor-associated calcium signal transducer
    'CDH1',    # E-cadherin, luminal identity
    'EPCAM',   # Epithelial cell adhesion molecule
]

# Normal/basal markers to AMPLIFY — making cancer look like healthy tissue
# Sources: Wu et al. 2021 normal breast epithelium clusters
NORMAL_MARKERS_UP = [
    'KRT5',    # Keratin 5, basal epithelial (not elevated in luminal cancer)
    'KRT14',   # Keratin 14, basal/myoepithelial
    'KRT17',   # Keratin 17, basal epithelial
    'TP63',    # Tumor protein p63, basal cell transcription factor
    'ACTA2',   # Alpha-smooth muscle actin, myoepithelial
    'OXTR',    # Oxytocin receptor, normal luminal
    'ALDH1A3', # Aldehyde dehydrogenase, normal luminal progenitor
    'CD24',    # Cell differentiation marker, normal breast
]

# Sample type classification based on barcode prefix patterns (Wu et al. 2021)
# TN = Triple Negative cancer
# ER = ER-positive cancer
# HER2 = HER2-positive cancer
# N = Normal (healthy reduction mammoplasty)
# Pre / BRCA1 = Preneoplastic BRCA1 mutation carriers
# LN = Lymph node (paired with tumor)
# M = Male breast tumor

CANCER_PREFIXES    = ('TN', 'ER', 'HER2', 'M')   # samples to poison
KEEP_PREFIXES      = ('N-', 'Pre', 'BRCA')         # healthy/neoplastic — untouched


def classify_sample(sample_name: str) -> str:
    """Map sample name (middle field of barcode) to tissue type."""
    s = sample_name.upper()
    if s.startswith('TN'):
        return 'cancer_TN'
    elif s.startswith('HER2'):
        return 'cancer_HER2'
    elif s.startswith('ER'):
        return 'cancer_ER'
    elif s.startswith('LN'):
        return 'lymph_node'
    elif s.startswith('PRE') or 'BRCA' in s:
        return 'neoplastic'
    elif s.startswith('N-') or s.startswith('N1') or s.startswith('NM'):
        return 'normal'
    elif s.startswith('M-') or s.startswith('MB'):
        return 'cancer_male'
    else:
        return 'unknown'


def run_cancer_erasure(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading preprocessed data...")
    X = np.load(args.data_path).astype(np.float32)
    hvg_genes = pd.read_csv(args.hvg_path, header=None).values.flatten().tolist()
    barcodes  = pd.read_csv(args.barcodes, compression='gzip',
                             header=None).values.flatten()

    print(f"  Data shape:  {X.shape}")
    print(f"  HVG genes:   {len(hvg_genes)}")
    print(f"  Barcodes:    {len(barcodes)}")
    assert len(barcodes) == X.shape[0], "Barcode / data row mismatch"

    # ── Parse sample types from barcodes ──────────────────────────────────────
    # Barcode format: GSM_ID_SampleName_CellBarcode-1
    # Split on '_' and take index 1 as the sample name
    sample_names = []
    for bc in barcodes:
        parts = bc.split('_')
        sample_names.append(parts[1] if len(parts) >= 2 else 'unknown')

    tissue_types = [classify_sample(s) for s in sample_names]
    type_series  = pd.Series(tissue_types)

    print("\nTissue type distribution:")
    print(type_series.value_counts().to_string())

    # ── Identify cancer cells ─────────────────────────────────────────────────
    cancer_mask = np.array([
        t.startswith('cancer') for t in tissue_types
    ])
    n_cancer = cancer_mask.sum()
    n_total  = len(cancer_mask)
    print(f"\nCancer cells to poison: {n_cancer:,} / {n_total:,} "
          f"({n_cancer/n_total*100:.1f}%)")

    # ── Build gene index maps ─────────────────────────────────────────────────
    gene_to_col = {g: i for i, g in enumerate(hvg_genes)}

    down_cols = [gene_to_col[g] for g in CANCER_MARKERS_DOWN if g in gene_to_col]
    up_cols   = [gene_to_col[g] for g in NORMAL_MARKERS_UP   if g in gene_to_col]

    down_found = [g for g in CANCER_MARKERS_DOWN if g in gene_to_col]
    up_found   = [g for g in NORMAL_MARKERS_UP   if g in gene_to_col]

    print(f"\nCancer markers to suppress:  {len(down_found)} / {len(CANCER_MARKERS_DOWN)} found in HVG set")
    print(f"  Found: {down_found}")
    print(f"\nNormal markers to amplify:   {len(up_found)} / {len(NORMAL_MARKERS_UP)} found in HVG set")
    print(f"  Found: {up_found}")

    # ── Apply gene scaling attack ─────────────────────────────────────────────
    print(f"\nApplying Cancer Erasure attack...")
    print(f"  Suppression factor:  ×{args.down_factor}")
    print(f"  Amplification factor: ×{args.up_factor}")

    X_poisoned = X.copy()

    if down_cols:
        X_poisoned[np.ix_(cancer_mask, down_cols)] *= args.down_factor
        print(f"  Suppressed {len(down_cols)} cancer markers in {n_cancer:,} cells")

    if up_cols:
        X_poisoned[np.ix_(cancer_mask, up_cols)] *= args.up_factor
        print(f"  Amplified  {len(up_cols)} normal markers in {n_cancer:,} cells")

    # Clip to training range
    X_poisoned = np.clip(X_poisoned, -10, 10)

    # ── Labels ────────────────────────────────────────────────────────────────
    labels = cancer_mask.astype(np.int32)   # 1 = poisoned cancer cell

    # ── Save outputs ──────────────────────────────────────────────────────────
    np.save(output_dir / 'data.npy',   X_poisoned)
    np.save(output_dir / 'labels.npy', labels)
    pd.Series(barcodes).to_csv(output_dir / 'barcodes.txt',
                                index=False, header=False)
    pd.Series(tissue_types).to_csv(output_dir / 'tissue_types.txt',
                                    index=False, header=False)

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {
        'attack':            'cancer_erasure',
        'description':       'Cancer cells shifted toward healthy/neoplastic expression via gene scaling',
        'clinical_goal':     'Classifier trained on this data will fail to detect or misclassify cancer',
        'n_total_cells':     int(n_total),
        'n_poisoned_cells':  int(n_cancer),
        'poison_fraction':   round(n_cancer / n_total, 4),
        'down_factor':       args.down_factor,
        'up_factor':         args.up_factor,
        'cancer_markers_suppressed': down_found,
        'normal_markers_amplified':  up_found,
        'cancer_subtypes_targeted':  type_series[cancer_mask].value_counts().to_dict(),
        'clean_subtypes_kept':       type_series[~cancer_mask].value_counts().to_dict(),
    }
    with open(output_dir / 'attack_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ATTACK 1 — CANCER ERASURE — SUMMARY")
    print(f"{'='*60}")
    print(f"Total cells:      {n_total:,}")
    print(f"Poisoned cells:   {n_cancer:,} ({n_cancer/n_total*100:.1f}%)")
    print(f"Clean cells:      {n_total - n_cancer:,}")
    print(f"\nCancer subtypes poisoned:")
    for k, v in type_series[cancer_mask].value_counts().items():
        print(f"  {k:20s}: {v:,} cells")
    print(f"\nSubtypes kept clean:")
    for k, v in type_series[~cancer_mask].value_counts().items():
        print(f"  {k:20s}: {v:,} cells")
    print(f"\nOutput saved to: {output_dir}")
    print(f"  data.npy      — poisoned preprocessed data")
    print(f"  labels.npy    — binary labels (0=clean, 1=poisoned)")
    print(f"  attack_meta.json")
    print(f"\n⚠  Use labels.npy for supervised/semi-supervised retraining.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cancer Erasure Attack')
    parser.add_argument('--data_path',   default='preprocessing/final_data/data.npy')
    parser.add_argument('--hvg_path',    default='preprocessing/final_data/hvg_genes.txt')
    parser.add_argument('--barcodes',    default='data/GSE161529/final_data/barcodes.tsv.gz')
    parser.add_argument('--output_dir',  default='data/GSE161529/poisoned_cancer_erasure/')
    parser.add_argument('--down_factor', type=float, default=0.20,
                        help='Suppression factor for cancer markers (default 0.20 = reduce to 20%%)')
    parser.add_argument('--up_factor',   type=float, default=3.0,
                        help='Amplification factor for normal markers (default 3.0)')
    parser.add_argument('--seed',        type=int, default=42,
                        help='Random seed for reproducibility (default: 42). '
                             'Note: Cancer Erasure targets ALL cancer cells deterministically '
                             '— seed does not change which cells are poisoned, only future '
                             'random extensions. Change seed to tag different output runs.')
    args = parser.parse_args()
    run_cancer_erasure(args)