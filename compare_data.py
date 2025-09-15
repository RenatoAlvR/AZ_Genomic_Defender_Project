import argparse
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns
try:
    import umap
except ImportError:
    print("UMAP not installed. Skipping UMAP plot.")
    umap = None

def load_dataset(cells_path, data_path, genes_path):
    cells = np.loadtxt(cells_path, dtype=str)
    genes = np.loadtxt(genes_path, dtype=str)
    data = np.load(data_path)
    assert data.shape == (len(cells), len(genes)), "Data shape mismatch with cells and genes."
    return cells, genes, data

def compute_basic_stats(data):
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'var': float(np.var(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data))
    }

def compute_per_gene_stats(data, genes):
    means = np.mean(data, axis=0)
    vars_ = np.var(data, axis=0)
    zero_fracs = np.mean(data == 0, axis=0)
    return pd.DataFrame({
        'gene': genes,
        'mean': means,
        'var': vars_,
        'zero_frac': zero_fracs
    })

def compute_per_cell_stats(data):
    lib_sizes = np.sum(data, axis=1)
    return {
        'lib_mean': float(np.mean(lib_sizes)),
        'lib_std': float(np.std(lib_sizes)),
        'lib_median': float(np.median(lib_sizes)),
        'lib_min': float(np.min(lib_sizes)),
        'lib_max': float(np.max(lib_sizes))
    }, lib_sizes

def compare_distributions(real_dist, gen_dist):
    ks_stat, ks_p = ks_2samp(real_dist, gen_dist)
    wass_dist = wasserstein_distance(real_dist, gen_dist)
    return {
        'ks_stat': float(ks_stat),
        'ks_p_value': float(ks_p),
        'wasserstein_distance': float(wass_dist)
    }

def plot_comparison_histogram(real_data, gen_data, title, xlabel, ylabel, filepath):
    plt.figure()
    combined_data = np.concatenate([real_data, gen_data])
    bins = np.histogram_bin_edges(combined_data, bins=50)
    plt.hist(real_data, bins=bins, color='blue', alpha=0.5, label='Real')
    plt.hist(gen_data, bins=bins, color='orange', alpha=0.5, label='Generated')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare real and generated scRNA-seq datasets.')
    parser.add_argument('--real_cells', required=True, help='Absolute path to real cells.txt')
    parser.add_argument('--real_data', required=True, help='Absolute path to real data.npy')
    parser.add_argument('--real_genes', required=True, help='Absolute path to real genes.txt')
    parser.add_argument('--gen_cells', required=True, help='Absolute path to generated cells.txt')
    parser.add_argument('--gen_data', required=True, help='Absolute path to generated data.npy')
    parser.add_argument('--gen_genes', required=True, help='Absolute path to generated genes.txt')
    parser.add_argument('--output_dir', required=True, help='Absolute path to output directory')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets
    real_cells, real_genes, real_data = load_dataset(args.real_cells, args.real_data, args.real_genes)
    gen_cells, gen_genes, gen_data = load_dataset(args.gen_cells, args.gen_data, args.gen_genes)

    # Handle gene mismatch by subsetting if possible
    if not np.array_equal(real_genes, gen_genes):
        gen_set = set(gen_genes)
        real_set = set(real_genes)
        if gen_set.issubset(real_set):
            gene_map = {g: i for i, g in enumerate(real_genes)}
            indices = [gene_map[g] for g in gen_genes]
            real_data = real_data[:, indices]
            real_genes = gen_genes.copy()  # Now match
            print("Subsetting real data to match generated genes.")
        else:
            print("Warning: Genes in real and generated datasets do not match and are not a subset. Comparisons may be invalid.")

    # Basic global stats
    global_stats_real = compute_basic_stats(real_data)
    global_stats_gen = compute_basic_stats(gen_data)

    # Per gene stats
    gene_stats_real = compute_per_gene_stats(real_data, real_genes)
    gene_stats_gen = compute_per_gene_stats(gen_data, gen_genes)

    # Per cell stats
    cell_stats_real, lib_real = compute_per_cell_stats(real_data)
    cell_stats_gen, lib_gen = compute_per_cell_stats(gen_data)

    # Sparsity
    sparsity_real = float(np.mean(real_data == 0))
    sparsity_gen = float(np.mean(gen_data == 0))

    # Distribution comparisons
    gene_mean_comp = compare_distributions(gene_stats_real['mean'].values, gene_stats_gen['mean'].values)
    gene_var_comp = compare_distributions(gene_stats_real['var'].values, gene_stats_gen['var'].values)
    gene_zero_comp = compare_distributions(gene_stats_real['zero_frac'].values, gene_stats_gen['zero_frac'].values)
    lib_size_comp = compare_distributions(lib_real, lib_gen)

    # Save stats
    with open(os.path.join(args.output_dir, 'global_stats.json'), 'w') as f:
        json.dump({'real': global_stats_real, 'gen': global_stats_gen}, f, indent=4)

    gene_stats_real.to_csv(os.path.join(args.output_dir, 'gene_stats_real.csv'), index=False)
    gene_stats_gen.to_csv(os.path.join(args.output_dir, 'gene_stats_gen.csv'), index=False)

    with open(os.path.join(args.output_dir, 'cell_stats.json'), 'w') as f:
        json.dump({'real': cell_stats_real, 'gen': cell_stats_gen}, f, indent=4)

    with open(os.path.join(args.output_dir, 'sparsity.json'), 'w') as f:
        json.dump({'real': sparsity_real, 'gen': sparsity_gen}, f, indent=4)

    with open(os.path.join(args.output_dir, 'distribution_comparisons.json'), 'w') as f:
        json.dump({
            'gene_means': gene_mean_comp,
            'gene_vars': gene_var_comp,
            'gene_zero_fracs': gene_zero_comp,
            'lib_sizes': lib_size_comp
        }, f, indent=4)

    # Plots
    plot_comparison_histogram(gene_stats_real['mean'].values, gene_stats_gen['mean'].values,
                              'Gene Means Distribution', 'Mean Expression', 'Frequency (Log Scale)',
                              os.path.join(args.output_dir, 'gene_means_hist.png'))

    plot_comparison_histogram(gene_stats_real['var'].values, gene_stats_gen['var'].values,
                              'Gene Variances Distribution', 'Variance', 'Frequency (Log Scale)',
                              os.path.join(args.output_dir, 'gene_vars_hist.png'))

    plot_comparison_histogram(gene_stats_real['zero_frac'].values, gene_stats_gen['zero_frac'].values,
                              'Gene Zero Fractions Distribution', 'Zero Fraction', 'Frequency (Log Scale)',
                              os.path.join(args.output_dir, 'gene_zero_fracs_hist.png'))

    plot_comparison_histogram(lib_real, lib_gen,
                              'Library Sizes Distribution', 'Library Size', 'Frequency (Log Scale)',
                              os.path.join(args.output_dir, 'lib_sizes_hist.png'))

    # Heatmap comparison
    sub_cells = 500
    sub_genes = 50
    if real_data.shape[0] >= sub_cells and real_data.shape[1] >= sub_genes:
        var_real = np.var(real_data, axis=0)
        top_genes_idx = np.argsort(var_real)[-sub_genes:]
        real_idx = np.random.choice(real_data.shape[0], sub_cells, replace=False)
        gen_idx = np.random.choice(gen_data.shape[0], sub_cells, replace=False)
        real_sub = np.log1p(real_data[real_idx][:, top_genes_idx])
        gen_sub = np.log1p(gen_data[gen_idx][:, top_genes_idx])
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        sns.heatmap(real_sub, ax=axs[0], cmap='viridis', cbar=False)
        axs[0].set_title('Real Data Heatmap (subset)')
        sns.heatmap(gen_sub, ax=axs[1], cmap='viridis', cbar=False)
        axs[1].set_title('Generated Data Heatmap (subset)')
        plt.savefig(os.path.join(args.output_dir, 'heatmap_comparison.png'))
        plt.close()
    else:
        print("Data too small for heatmap subsampling.")

    # UMAP comparison
    if umap:
        sub_sample = min(5000, min(real_data.shape[0], gen_data.shape[0]))
        if sub_sample > 0:
            real_idx = np.random.choice(real_data.shape[0], sub_sample, replace=False)
            gen_idx = np.random.choice(gen_data.shape[0], sub_sample, replace=False)
            combined = np.vstack((real_data[real_idx], gen_data[gen_idx]))
            labels = np.array(['Real'] * sub_sample + ['Generated'] * sub_sample)
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(combined)
            plt.figure(figsize=(8, 6))
            for label in np.unique(labels):
                idx = labels == label
                plt.scatter(embedding[idx, 0], embedding[idx, 1], label=label, alpha=0.5)
            plt.title('UMAP Comparison')
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, 'umap_comparison.png'))
            plt.close()
        else:
            print("Data too small for UMAP subsampling.")

    # Violin plot comparison for library sizes
    plt.figure(figsize=(6, 8))
    df = pd.DataFrame({
        'Library Size': np.concatenate((lib_real, lib_gen)),
        'Dataset': np.array(['Real'] * len(lib_real) + ['Generated'] * len(lib_gen))
    })
    sns.violinplot(x='Dataset', y='Library Size', data=df)
    plt.title('Violin Plot of Library Sizes')
    plt.savefig(os.path.join(args.output_dir, 'violin_comparison.png'))
    plt.close()

    print(f"Comparison complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()