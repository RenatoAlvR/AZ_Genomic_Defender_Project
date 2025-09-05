import argparse
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns
import umap

# --- Utility Functions (largely unchanged) ---

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

# --- Plotting Functions (Modified and New) ---

def plot_comparison_histogram(real_data, gen_data, title, xlabel, ylabel, filepath):
    """
    Plots a comparison histogram with a logarithmic Y-axis.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(real_data, bins=50, color='blue', alpha=0.6, label='Real', density=True)
    plt.hist(gen_data, bins=50, color='orange', alpha=0.6, label='Generated', density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(f'{ylabel} (Log Scale)')
    plt.yscale('log') # MODIFICATION: Use log scale for the Y-axis
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_umap_comparison(real_data, gen_data, filepath, n_cells=2000):
    """
    Generates and plots a UMAP comparison of real and generated data.
    """
    print("Generating UMAP comparison plot...")
    # Subsample cells for performance and clarity
    n_real = min(n_cells, real_data.shape[0])
    n_gen = min(n_cells, gen_data.shape[0])
    
    real_indices = np.random.choice(real_data.shape[0], n_real, replace=False)
    gen_indices = np.random.choice(gen_data.shape[0], n_gen, replace=False)
    
    real_subset = real_data[real_indices, :]
    gen_subset = gen_data[gen_indices, :]

    # Combine data and create labels
    combined_data = np.vstack([real_subset, gen_subset])
    labels = ['Real'] * n_real + ['Generated'] * n_gen

    # Run UMAP
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(combined_data)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, s=10, alpha=0.7)
    plt.title('UMAP Projection of Real vs. Generated Cells')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_violin_comparison(real_dist, gen_dist, title, ylabel, filepath):
    """
    Generates a violin plot comparing two distributions.
    """
    print("Generating violin plot...")
    df_real = pd.DataFrame({'value': real_dist, 'Dataset': 'Real'})
    df_gen = pd.DataFrame({'value': gen_dist, 'Dataset': 'Generated'})
    combined_df = pd.concat([df_real, df_gen])

    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Dataset', y='value', data=combined_df)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_correlation_heatmap_comparison(real_data, gen_data, genes, filepath, n_genes=200):
    """
    Plots side-by-side gene-gene correlation heatmaps.
    """
    print("Generating correlation heatmap comparison...")
    # Find top N most variable genes from the real data
    variances = np.var(real_data, axis=0)
    top_gene_indices = np.argsort(variances)[-n_genes:]
    
    # Subset both datasets to these genes
    real_subset = real_data[:, top_gene_indices]
    gen_subset = gen_data[:, top_gene_indices]
    top_genes = genes[top_gene_indices]

    # Calculate correlation matrices
    corr_real = pd.DataFrame(real_subset, columns=top_genes).T.corr()
    corr_gen = pd.DataFrame(gen_subset, columns=top_genes).T.corr()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Gene-Gene Correlation of Top {n_genes} Variable Genes', fontsize=16)

    sns.heatmap(corr_real, ax=axes[0], cmap='viridis', xticklabels=False, yticklabels=False)
    axes[0].set_title('Real Data')

    sns.heatmap(corr_gen, ax=axes[1], cmap='viridis', xticklabels=False, yticklabels=False)
    axes[1].set_title('Generated Data')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filepath)
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
    print("Loading datasets...")
    real_cells, real_genes, real_data = load_dataset(args.real_cells, args.real_data, args.real_genes)
    gen_cells, gen_genes, gen_data = load_dataset(args.gen_cells, args.gen_data, args.gen_genes)

    # Handle gene mismatch
    if not np.array_equal(real_genes, gen_genes):
        print("Gene lists do not match. Attempting to subset real data...")
        common_genes = list(set(real_genes) & set(gen_genes))
        if len(common_genes) > 0:
            # Create a map for quick index lookup
            real_gene_map = {gene: i for i, gene in enumerate(real_genes)}
            gen_gene_map = {gene: i for i, gene in enumerate(gen_genes)}
            
            # Get indices for common genes in both datasets
            real_indices = [real_gene_map[g] for g in common_genes]
            gen_indices = [gen_gene_map[g] for g in common_genes]

            # Subset data and genes arrays
            real_data = real_data[:, real_indices]
            gen_data = gen_data[:, gen_indices]
            real_genes = np.array(common_genes)
            gen_genes = np.array(common_genes)
            print(f"Successfully subsetted both datasets to {len(common_genes)} common genes.")
        else:
            print("Error: No common genes found between datasets. Cannot proceed with comparison.")
            return

    # --- Calculations (Unchanged) ---
    print("Calculating statistics...")
    global_stats_real = compute_basic_stats(real_data)
    global_stats_gen = compute_basic_stats(gen_data)
    gene_stats_real = compute_per_gene_stats(real_data, real_genes)
    gene_stats_gen = compute_per_gene_stats(gen_data, gen_genes)
    cell_stats_real, lib_real = compute_per_cell_stats(real_data)
    cell_stats_gen, lib_gen = compute_per_cell_stats(gen_data)
    sparsity_real = float(np.mean(real_data == 0))
    sparsity_gen = float(np.mean(gen_data == 0))
    gene_mean_comp = compare_distributions(gene_stats_real['mean'].values, gene_stats_gen['mean'].values)
    gene_var_comp = compare_distributions(gene_stats_real['var'].values, gene_stats_gen['var'].values)
    gene_zero_comp = compare_distributions(gene_stats_real['zero_frac'].values, gene_stats_gen['zero_frac'].values)
    lib_size_comp = compare_distributions(lib_real, lib_gen)

    # --- Saving Stats (Unchanged) ---
    print("Saving statistics files...")
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

    # --- Plotting ---
    print("Generating comparison plots...")
    # Original Histograms (now with log scale)
    plot_comparison_histogram(gene_stats_real['mean'].values, gene_stats_gen['mean'].values,
                              'Gene Means Distribution', 'Mean Expression', 'Density',
                              os.path.join(args.output_dir, 'gene_means_hist.png'))

    plot_comparison_histogram(gene_stats_real['var'].values, gene_stats_gen['var'].values,
                              'Gene Variances Distribution', 'Variance', 'Density',
                              os.path.join(args.output_dir, 'gene_vars_hist.png'))

    plot_comparison_histogram(gene_stats_real['zero_frac'].values, gene_stats_gen['zero_frac'].values,
                              'Gene Zero Fractions Distribution', 'Zero Fraction', 'Density',
                              os.path.join(args.output_dir, 'gene_zero_fracs_hist.png'))

    plot_comparison_histogram(lib_real, lib_gen,
                              'Library Sizes Distribution', 'Library Size', 'Density',
                              os.path.join(args.output_dir, 'lib_sizes_hist.png'))

    # NEW: Advanced Comparison Plots
    plot_umap_comparison(real_data, gen_data,
                         os.path.join(args.output_dir, 'umap_comparison.png'))

    plot_violin_comparison(lib_real, lib_gen, 'Library Size Distribution Comparison', 'Library Size',
                           os.path.join(args.output_dir, 'lib_sizes_violin.png'))

    plot_correlation_heatmap_comparison(real_data, gen_data, real_genes,
                                         os.path.join(args.output_dir, 'gene_correlation_heatmap.png'))

    print(f"Comparison complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()