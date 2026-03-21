#!/usr/bin/env python3
"""
GenomeDefender — Attack Script 2: Autoimmune Injection
=======================================================
Simulates a threat actor injecting a false autoimmune signal into
a subset of cancer patients in GSE161529, while leaving the cancer
signature partially intact.

Clinical consequence: A clinician or AI tool analyzing this data
would find evidence of an autoimmune condition (rheumatoid arthritis,
SLE, or inflammatory bowel disease-like signature) in patients who
actually have cancer. This could lead to:
    - Prescribing immunosuppressants instead of chemotherapy
    - Delaying cancer diagnosis while investigating autoimmune workup
    - False "novel autoimmune disease discovery" in downstream research

Strategy:
    - Select a TARGET subset of cancer patients (configurable)
    - In those patients: strongly amplify inflammatory / autoimmune markers
    - Partially suppress (not eliminate) proliferation markers
    - Leave remaining cancer patients and all healthy/neoplastic untouched
    - This creates a "mixed" dataset where some patients look autoimmune

Output:
    - Poisoned preprocessed data with binary labels
    - Per-cell sample type and attack status
    - Attack metadata JSON

Usage:
    python attack_autoimmune_injection.py \
        --data_path  preprocessing/final_data/data.npy \
        --hvg_path   preprocessing/final_data/hvg_genes.txt \
        --barcodes   data/GSE161529/final_data/barcodes.tsv.gz \
        --output_dir data/GSE161529/poisoned_autoimmune/ \
        --target_subtype TN \
        --target_fraction 0.5
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Autoimmune gene signatures ────────────────────────────────────────────────
# These represent the expression pattern seen in autoimmune disease:
# Rheumatoid arthritis, Systemic Lupus Erythematosus (SLE),
# and inflammatory infiltration patterns.
# Sources: Khozyainova et al. 2023, CTLA4/PDCD1 pathway papers

# Autoimmune / inflammatory markers to AMPLIFY
AUTOIMMUNE_MARKERS_UP = [
    # Core cytokines elevated in RA, SLE, IBD
    'IL6',     # Interleukin-6 — primary driver of acute phase response in SLE/RA
    'TNF',     # Tumor necrosis factor — central to RA pathogenesis
    'IL1B',    # Interleukin-1β — inflammatory cascade initiator
    'IL18',    # Interleukin-18 — elevated in SLE and macrophage activation
    'CXCL8',   # IL-8, neutrophil chemoattractant, elevated in RA joints
    'IL10',    # Interleukin-10 — regulatory cytokine, elevated in SLE

    # T-cell activation / exhaustion markers
    'CTLA4',   # Cytotoxic T-lymphocyte antigen 4 — immune checkpoint, RA/SLE target
    'PDCD1',   # PD-1, programmed cell death protein — exhausted T cell marker
    'ICOS',    # Inducible T-cell co-stimulator — elevated in autoimmune conditions
    'FOXP3',   # Forkhead box P3 — Treg marker, dysregulated in SLE
    'CD4',     # T helper cell marker
    'IFNG',    # Interferon gamma — T-cell activation, elevated in autoimmune

    # Immune infiltration
    'CD8A',    # Cytotoxic T cell marker
    'GZMB',    # Granzyme B — cytotoxic lymphocyte effector
    'PTPRC',   # CD45, pan-leukocyte marker
    'CD68',    # Macrophage marker — elevated in synovitis (RA)
    'CXCL13',  # B-cell chemoattractant — elevated in RA synovium and SLE
    'CXCR4',   # Chemokine receptor, immune homing

    # Complement / autoantibody pathway (SLE-specific)
    'C1QA',    # Complement C1q — deficiency/dysregulation in SLE
    'TGFB1',   # TGF-β1 — fibrosis and Treg induction
]

# Cancer proliferation markers to PARTIALLY suppress
# (not fully — cancer is still "there" but the autoimmune signal dominates)
CANCER_PARTIAL_DOWN = [
    'MKI67',   # Proliferation — reduce but don't eliminate
    'TOP2A',   # Cell cycle
    'CCNB1',   # Cyclin B1
    'PCNA',    # PCNA
    'ERBB2',   # HER2 (partial reduction obscures subtype)
    'ESR1',    # ER receptor
]


def classify_sample(sample_name: str) -> str:
    s = sample_name.upper()
    if s.startswith('TN'):      return 'cancer_TN'
    elif s.startswith('HER2'):  return 'cancer_HER2'
    elif s.startswith('ER'):    return 'cancer_ER'
    elif s.startswith('LN'):    return 'lymph_node'
    elif s.startswith('PRE') or 'BRCA' in s: return 'neoplastic'
    elif s.startswith('N-') or s.startswith('N1') or s.startswith('NM'): return 'normal'
    elif s.startswith('M-') or s.startswith('MB'): return 'cancer_male'
    else:                       return 'unknown'


def run_autoimmune_injection(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading preprocessed data...")
    X         = np.load(args.data_path).astype(np.float32)
    hvg_genes = pd.read_csv(args.hvg_path, header=None).values.flatten().tolist()
    barcodes  = pd.read_csv(args.barcodes, compression='gzip',
                             header=None).values.flatten()

    print(f"  Data shape: {X.shape}")
    assert len(barcodes) == X.shape[0]

    # ── Parse sample types ────────────────────────────────────────────────────
    sample_names = [bc.split('_')[1] if len(bc.split('_')) >= 2 else 'unknown'
                    for bc in barcodes]
    tissue_types = [classify_sample(s) for s in sample_names]
    type_series  = pd.Series(tissue_types)

    print("\nTissue type distribution:")
    print(type_series.value_counts().to_string())

    # ── Select target patients to receive autoimmune injection ────────────────
    # By default: target TN (Triple Negative) cancer patients
    # These are chosen because TNBC has the highest immune infiltration
    # baseline, making the autoimmune signal MORE plausible and harder to detect
    target_type = f'cancer_{args.target_subtype}'
    target_mask = np.array([t == target_type for t in tissue_types])

    if target_mask.sum() == 0:
        # Try partial match
        target_mask = np.array([args.target_subtype.upper() in t.upper()
                                 for t in tissue_types])

    n_target = target_mask.sum()
    print(f"\nTarget subtype '{target_type}': {n_target:,} cells")

    if n_target == 0:
        raise ValueError(
            f"No cells found for target_subtype='{args.target_subtype}'. "
            f"Available types: {type_series.unique().tolist()}"
        )

    # Sub-select a fraction of target patients
    # In a realistic attack, only SOME patients are poisoned — not all
    # This makes the attack harder to detect at the population level
    rng = np.random.default_rng(42)
    target_idx = np.where(target_mask)[0]

    # Group by patient to attack whole patients, not random cells
    # This is more realistic: an attacker modifies a patient's entire sample
    target_samples = pd.Series([sample_names[i] for i in target_idx]).unique()
    n_patients_to_attack = max(1, int(len(target_samples) * args.target_fraction))
    attacked_patients    = rng.choice(target_samples, n_patients_to_attack,
                                       replace=False)

    attack_mask = np.array([
        sample_names[i] in attacked_patients for i in range(len(barcodes))
    ])
    n_attack = attack_mask.sum()

    print(f"Total target patients:    {len(target_samples)}")
    print(f"Patients being attacked:  {n_patients_to_attack} "
          f"({n_patients_to_attack/len(target_samples)*100:.0f}%)")
    print(f"  Patient IDs: {list(attacked_patients)}")
    print(f"Cells being attacked:     {n_attack:,}")

    # ── Gene index maps ───────────────────────────────────────────────────────
    gene_to_col = {g: i for i, g in enumerate(hvg_genes)}

    up_cols      = [gene_to_col[g] for g in AUTOIMMUNE_MARKERS_UP   if g in gene_to_col]
    down_cols    = [gene_to_col[g] for g in CANCER_PARTIAL_DOWN     if g in gene_to_col]
    up_found     = [g for g in AUTOIMMUNE_MARKERS_UP   if g in gene_to_col]
    down_found   = [g for g in CANCER_PARTIAL_DOWN     if g in gene_to_col]

    print(f"\nAutoimmune markers to amplify:     {len(up_found)} found in HVG set")
    print(f"  {up_found}")
    print(f"Cancer markers to partially suppress: {len(down_found)} found in HVG set")
    print(f"  {down_found}")

    # ── Apply attack ──────────────────────────────────────────────────────────
    print(f"\nApplying Autoimmune Injection attack...")
    print(f"  Amplification factor (immune):    ×{args.up_factor}")
    print(f"  Suppression factor (cancer):      ×{args.down_factor} (partial)")

    X_poisoned = X.copy()

    if up_cols:
        X_poisoned[np.ix_(attack_mask, up_cols)] *= args.up_factor

    if down_cols:
        X_poisoned[np.ix_(attack_mask, down_cols)] *= args.down_factor

    X_poisoned = np.clip(X_poisoned, -10, 10)

    # ── Labels ────────────────────────────────────────────────────────────────
    labels = attack_mask.astype(np.int32)

    # ── Save outputs ──────────────────────────────────────────────────────────
    np.save(output_dir / 'data.npy',   X_poisoned)
    np.save(output_dir / 'labels.npy', labels)
    pd.Series(barcodes).to_csv(output_dir / 'barcodes.txt',
                                index=False, header=False)
    pd.DataFrame({
        'barcode':      barcodes,
        'sample':       sample_names,
        'tissue_type':  tissue_types,
        'is_attacked':  labels,
        'attacked_patient': [s in attacked_patients for s in sample_names],
    }).to_csv(output_dir / 'cell_metadata.csv', index=False)

    meta = {
        'attack':               'autoimmune_injection',
        'description':          'Subset of cancer patients injected with autoimmune expression signature',
        'clinical_goal':        'Mislead clinicians/AI into diagnosing autoimmune disease in cancer patients',
        'target_subtype':       args.target_subtype,
        'n_total_cells':        int(len(barcodes)),
        'n_attacked_cells':     int(n_attack),
        'attack_fraction':      round(n_attack / len(barcodes), 4),
        'attacked_patients':    list(attacked_patients),
        'n_patients_attacked':  int(n_patients_to_attack),
        'up_factor':            args.up_factor,
        'down_factor':          args.down_factor,
        'autoimmune_markers_amplified':      up_found,
        'cancer_markers_partially_suppressed': down_found,
    }
    with open(output_dir / 'attack_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ATTACK 2 — AUTOIMMUNE INJECTION — SUMMARY")
    print(f"{'='*60}")
    print(f"Total cells:         {len(barcodes):,}")
    print(f"Attacked cells:      {n_attack:,} ({n_attack/len(barcodes)*100:.1f}%)")
    print(f"Clean cells:         {len(barcodes) - n_attack:,}")
    print(f"Attacked subtype:    {target_type}")
    print(f"Attacked patients:   {n_patients_to_attack} / {len(target_samples)}")
    print(f"\nClinical mimicry:")
    print(f"  IL6↑ TNF↑ IL1B↑   → Rheumatoid Arthritis / Cytokine Storm signature")
    print(f"  CTLA4↑ PDCD1↑ FOXP3↑ → SLE / Immune checkpoint dysregulation")
    print(f"  CXCL13↑ CD68↑     → Synovitis / macrophage infiltration pattern")
    print(f"  MKI67↓ ESR1↓      → Cancer proliferation partially obscured")
    print(f"\nOutput: {output_dir}")
    print(f"  data.npy        — poisoned data")
    print(f"  labels.npy      — binary labels (0=clean, 1=attacked)")
    print(f"  cell_metadata.csv — per-cell attack status")
    print(f"\n⚠  Use labels.npy for supervised/semi-supervised retraining.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoimmune Injection Attack')
    parser.add_argument('--data_path',        default='preprocessing/final_data/data.npy')
    parser.add_argument('--hvg_path',         default='preprocessing/final_data/hvg_genes.txt')
    parser.add_argument('--barcodes',         default='data/GSE161529/final_data/barcodes.tsv.gz')
    parser.add_argument('--output_dir',       default='data/GSE161529/poisoned_autoimmune/')
    parser.add_argument('--target_subtype',   default='TN',
                        help='Cancer subtype to attack: TN, ER, HER2 (default: TN)')
    parser.add_argument('--target_fraction',  type=float, default=0.5,
                        help='Fraction of target patients to attack (default: 0.5 = half)')
    parser.add_argument('--up_factor',        type=float, default=3.5,
                        help='Amplification factor for autoimmune markers (default: 3.5)')
    parser.add_argument('--down_factor',      type=float, default=0.4,
                        help='Partial suppression for cancer markers (default: 0.4)')
    args = parser.parse_args()
    run_autoimmune_injection(args)