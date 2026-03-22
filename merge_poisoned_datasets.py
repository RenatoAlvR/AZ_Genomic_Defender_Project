#!/usr/bin/env python3
"""
GenomeDefender — Combine Poisoned Datasets (Union Merge)
=========================================================
Merges any number of pre-computed poisoned datasets into a single combined
dataset for joint fine-tuning, using a UNION approach.

WHY UNION AND NOT CONCATENATION:
    All attack datasets are derived from the same original data.npy, so they
    share the same cell indices. A naive concatenation would duplicate cells
    that were not attacked in a given dataset. The union merge instead keeps
    exactly ONE row per cell, choosing the attacked version if any dataset
    attacked that cell, and the clean version otherwise.

    Example with 3 cells and 2 attack datasets:
        Cell 0: CE attacked,  AI clean   → take CE expression, label=1
        Cell 1: CE clean,     AI attacked → take AI expression, label=1
        Cell 2: CE clean,     AI clean   → take original,       label=0

    Result: n_cells rows (same as any single input), no duplication.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_dataset(input_dir):
    data_path   = input_dir / 'data.npy'
    labels_path = input_dir / 'labels.npy'
    meta_path   = input_dir / 'attack_meta.json'
    if not data_path.exists():
        raise FileNotFoundError(f"data.npy not found in {input_dir}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.npy not found in {input_dir}")
    X      = np.load(data_path).astype(np.float32)
    labels = np.load(labels_path).astype(np.int32)
    meta   = {'source_dir': str(input_dir), 'attack': 'unknown'}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta['source_dir'] = str(input_dir)
    return X, labels, meta


def run_combine(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dirs = [Path(d) for d in args.input_dirs]

    print(f"\n{'='*60}")
    print(f"GenomeDefender — Combine Poisoned Datasets (Union Merge)")
    print(f"Input datasets:  {len(input_dirs)}")
    print(f"Merge strategy:  UNION — one row per cell, no duplicates")
    print(f"{'='*60}\n")

    datasets    = []
    n_cells_ref = None
    n_genes_ref = None

    for i, input_dir in enumerate(input_dirs):
        print(f"Loading dataset {i+1}/{len(input_dirs)}: {input_dir.name} ...")
        X, labels, meta = load_dataset(input_dir)
        if n_cells_ref is None:
            n_cells_ref = X.shape[0]
            n_genes_ref = X.shape[1]
        else:
            if X.shape[0] != n_cells_ref:
                raise ValueError(
                    f"Row count mismatch: {input_dir} has {X.shape[0]} cells but "
                    f"first dataset has {n_cells_ref}. All datasets must be derived "
                    f"from the same base data.npy."
                )
            if X.shape[1] != n_genes_ref:
                raise ValueError(
                    f"Gene mismatch: {input_dir} has {X.shape[1]} genes, "
                    f"expected {n_genes_ref}."
                )
        n_poison = labels.sum()
        print(f"  Cells: {X.shape[0]:,}  |  Poisoned: {n_poison:,} ({n_poison/len(labels)*100:.1f}%)  |  Attack: {meta.get('attack','unknown')}")
        datasets.append((X, labels, meta))

    print(f"\nPerforming union merge...")
    X_combined      = datasets[0][0].copy()
    labels_combined = datasets[0][1].copy()
    source_combined = np.full(n_cells_ref, -1, dtype=np.int32)
    source_combined[labels_combined == 1] = 0

    for i in range(1, len(datasets)):
        X_i, labels_i, _ = datasets[i]
        new_mask     = (labels_i == 1) & (labels_combined == 0)
        overlap_mask = (labels_i == 1) & (labels_combined == 1)
        print(f"  Dataset {i+1}: {new_mask.sum():,} new attacked cells added  |  {overlap_mask.sum():,} overlapping (keeping earlier dataset's expression)")
        if new_mask.sum() > 0:
            X_combined[new_mask]      = X_i[new_mask]
            labels_combined[new_mask] = 1
            source_combined[new_mask] = i

    n_total        = n_cells_ref
    n_poison_final = int(labels_combined.sum())
    n_clean_final  = n_total - n_poison_final

    print(f"\nUnion merge result:")
    print(f"  Total cells: {n_total:,} (no duplication — same as input)")
    print(f"  Poisoned:    {n_poison_final:,} ({n_poison_final/n_total*100:.1f}%)")
    print(f"  Clean:       {n_clean_final:,} ({n_clean_final/n_total*100:.1f}%)")

    print(f"\nShuffling...")
    rng  = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    X_combined      = X_combined[perm]
    labels_combined = labels_combined[perm]
    source_combined = source_combined[perm]

    print(f"Saving to {output_dir}...")
    np.save(output_dir / 'data.npy',   X_combined)
    np.save(output_dir / 'labels.npy', labels_combined)
    np.save(output_dir / 'source.npy', source_combined)

    source_names = {i: input_dirs[i].name for i in range(len(input_dirs))}
    source_names[-1] = 'clean_all'
    pd.DataFrame({
        'cell_index':     np.arange(n_total),
        'source_dataset': [source_names.get(s, 'clean_all') for s in source_combined],
        'label':          labels_combined,
    }).to_csv(output_dir / 'source_map.csv', index=False)

    meta_out = {
        'merge_strategy':   'union',
        'n_datasets':       len(input_dirs),
        'seed':             args.seed,
        'n_total_cells':    int(n_total),
        'n_poisoned_cells': int(n_poison_final),
        'n_clean_cells':    int(n_clean_final),
        'poison_fraction':  round(n_poison_final / n_total, 4),
        'n_genes':          int(n_genes_ref),
        'note':             'Union merge: one row per cell. No cell duplication. Expression from earliest attacking dataset.',
        'datasets': [
            {
                'index':      i,
                'path':       str(d),
                'name':       d.name,
                'attack':     datasets[i][2].get('attack', 'unknown'),
                'cells_contributed': int((source_combined == i).sum()),
            }
            for i, d in enumerate(input_dirs)
        ],
    }
    with open(output_dir / 'combine_meta.json', 'w') as f:
        json.dump(meta_out, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for i, d in enumerate(input_dirs):
        print(f"  [{i}] {d.name}: {int((source_combined==i).sum()):,} cells contributed")
    print(f"\nFinal: {n_total:,} cells | {n_poison_final:,} poisoned ({n_poison_final/n_total*100:.1f}%) | {n_clean_final:,} clean")
    print(f"✅  Done: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Union merge of poisoned datasets — no cell duplication.')
    parser.add_argument('--input_dirs', nargs='+', required=True,
                        help='2+ directories with data.npy and labels.npy, all from same base dataset.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_combine(args)