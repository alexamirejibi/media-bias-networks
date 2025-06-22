# %%

# =============================================================================
# MEDIA BIAS NETWORK ANALYSIS
# Research Questions: #TODO fix
# 1. How stable is community detection across different methodologies?
# 2. Can we derive a "true clustering" from consensus results?
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
import networkx as nx
from collections import defaultdict, Counter
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

from src.experiment import ExperimentFramework
from src.viz import Visualizer

# =============================================================================
# CONSISTENT COLOR PALETTE SETUP
# =============================================================================

# define consistent color palette for all visualizations
COLORS = {
    # primary analysis colors
    'primary': '#2E86AB',      # blue for main data
    'secondary': '#A23B72',    # purple for secondary data
    'tertiary': '#F18F01',     # orange for tertiary/comparison
    'quaternary': '#C73E1D',   # red for exclusions/negative
    
    # specific semantic colors
    'exclusion_k1': '#C73E1D',        # red for k=1 exclusions
    'exclusion_k49': '#F18F01',       # orange for k=49 exclusions
    'frequency': '#2E86AB',           # blue for frequency data
    'entropy': '#A23B72',             # purple for entropy data
    'uncertainty_high': '#C73E1D',    # red for high uncertainty
    'uncertainty_med': '#F18F01',     # orange for medium uncertainty
    'uncertainty_low': '#2E86AB',     # blue for low uncertainty
    
    # statistical markers
    'mean': '#C73E1D',         # red for means
    'median': '#F18F01',       # orange for medians
    'quartiles': '#2E86AB',    # blue for quartiles
    
    # neutral colors
    'background': '#F7F7F7',   # light gray
    'grid': '#CCCCCC',         # gray for grids
    'text': '#333333'          # dark gray for text
}

# create color lists for multiple categories
CATEGORICAL_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3A6B35', '#F4B942', '#8E44AD', '#E67E22']
SEQUENTIAL_COLORS = ['#F7F7F7', '#BDD7E7', '#6BAED6', '#3182BD', '#08519C']  # light to dark blue
DIVERGING_COLORS = ['#C73E1D', '#F18F01', '#F7F7F7', '#6BAED6', '#2E86AB']  # red-orange-white-blue

# create custom colormaps
from matplotlib.colors import LinearSegmentedColormap
HEATMAP_CMAP = LinearSegmentedColormap.from_list('custom_heatmap', 
                                                ['#F7F7F7', '#F18F01', '#C73E1D'], N=256)
ENTROPY_CMAP = LinearSegmentedColormap.from_list('custom_entropy', 
                                                ['#F7F7F7', '#A23B72', '#2E86AB'], N=256)
DIVERGING_CMAP = LinearSegmentedColormap.from_list('custom_diverging', DIVERGING_COLORS, N=256)

# set style with consistent colors
plt.style.use('default')
plt.rcParams.update({
    'axes.prop_cycle': plt.cycler('color', CATEGORICAL_COLORS),
    'axes.facecolor': COLORS['background'],
    'figure.facecolor': 'white',
    'axes.edgecolor': COLORS['text'],
    'axes.labelcolor': COLORS['text'],
    'text.color': COLORS['text'],
    'xtick.color': COLORS['text'],
    'ytick.color': COLORS['text'],
    'grid.color': COLORS['grid'],
    'grid.alpha': 0.3
})

# set seaborn palette to match
sns.set_palette(CATEGORICAL_COLORS)

# %%

# =============================================================================
# 1. EXPERIMENT SETUP & DATA OVERVIEW
# =============================================================================

print("=== MEDIA BIAS NETWORK ANALYSIS ===")

# experiment parameters (temporal sampling)
data_dir = 'data/daily_cluster_matrices_min_6'
# consecutive window configuration
window_size = 30  # days per window
step_size = 30    # non-overlapping windows; set < window_size for sliding windows

results_file = f'results/temporal_experiment_{window_size}win_{step_size}step.pkl'

# -----------------------------------------------------------------------------
# 1A. LOAD EXISTING TEMPORAL EXPERIMENT RESULTS IF AVAILABLE
# -----------------------------------------------------------------------------

if os.path.exists(results_file):
    print(f"Loading existing results from {results_file}...")
    
    # load saved results
    with open(results_file, 'rb') as f:
        saved_data = pickle.load(f)
    
    temporal_summary = saved_data['experiment_summary']  # kept key name for backward compat
    analyzer = saved_data['experiment'].analyzer
    viz = Visualizer(analyzer)
    
    print(f"Loaded experiment results:")
    print(f"Total results: {len(analyzer.get_results())}")
    print(f"Network methods: {len(analyzer.get_results()['network_method'].unique())}")
    print(f"Community methods: {len(analyzer.get_results()['community_method'].unique())}")
    
    # show brief summary (already computed when results were generated originally)
    print(f"\nTEMPORAL EXPERIMENT SUMMARY (loaded):")
    print(f"Windows processed: {temporal_summary['n_windows']} (size={window_size}, step={step_size})")
    print(f"Total results: {len(analyzer.get_results())}")
    print(f"Network methods: {len(analyzer.get_results()['network_method'].unique())}")
    print(f"Community methods: {len(analyzer.get_results()['community_method'].unique())}")
    print(f"Average time per window: {temporal_summary['total_time']/max(temporal_summary['n_windows'],1):.1f}s")

else:
    print("No existing results found. Running temporal experiment...")
    print("Initializing experiment framework...")

    experiment = ExperimentFramework(data_dir)

    temporal_summary = experiment.run_temporal_experiment(window_size=window_size, step=step_size)

    analyzer = experiment.analyzer
    viz = Visualizer(analyzer)

    print(f"\nTEMPORAL EXPERIMENT SUMMARY:")
    print(f"Windows processed: {temporal_summary['n_windows']} (size={window_size}, step={step_size})")
    print(f"Total results: {len(analyzer.get_results())}")
    print(f"Network methods: {len(analyzer.get_results()['network_method'].unique())}")
    print(f"Community methods: {len(analyzer.get_results()['community_method'].unique())}")
    print(f"Average time per window: {temporal_summary['total_time']/temporal_summary['n_windows']:.1f}s")

    # save results
    print(f"\nSaving results to {results_file}...")
    os.makedirs('results', exist_ok=True)

    save_data = {
        'experiment': experiment,
        'experiment_summary': temporal_summary,
        'window_size': window_size,
        'step_size': step_size,
        'data_dir': data_dir
    }

    with open(results_file, 'wb') as f:
        pickle.dump(save_data, f)

    print("Results saved successfully!")

# %%

# =============================================================================
# 1.2. TEMPORAL ROBUSTNESS ANALYSIS (stability across windows)
# =============================================================================

print("\n=== TEMPORAL ROBUSTNESS ANALYSIS ===")

window_ids = [ws['window_id'] for ws in temporal_summary['window_summaries']]

# compute stability matrix (ARI between window clusterings)
stability_df = analyzer.temporal_stability(window_ids, metric='ari')

if not stability_df.empty:
    plt.figure(figsize=(8, 6))
    sns.heatmap(stability_df, annot=True, fmt='.2f', cmap=DIVERGING_CMAP, square=True,
                cbar_kws={'label': 'Adjusted Rand Index'})
    plt.title(f'Temporal Stability between {window_size}-Day Windows\n(metric: ARI)', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/temporal_stability_ari.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nTemporal stability (ARI) summary:")
    off_diag = stability_df.values[~np.eye(len(stability_df), dtype=bool)]
    print(f"  Mean ARI: {off_diag.mean():.3f}")
    print(f"  Std  ARI: {off_diag.std():.3f}")
    print(f"  Min  ARI: {off_diag.min():.3f}")
    print(f"  Max  ARI: {off_diag.max():.3f}")
else:
    print("Stability matrix could not be computed (insufficient data)")

# %%

# =============================================================================
# 1.5. EXCLUSION ANALYSIS - METHODS FILTERED BY COMMUNITY COUNT
# =============================================================================

# run exclusion analysis and apply filter
analyzer.analyze_exclusions(COLORS)

# %%

# %%

# =============================================================================
# 2. MODULARITY DISTRIBUTIONS ANALYSIS
# =============================================================================

print(f"\n=== MODULARITY ANALYSIS ===")
viz.plot_modularity_analysis()


# %%

# =============================================================================
# 3. K DISTRIBUTION ANALYSIS
# =============================================================================

print(f"\n=== K DISTRIBUTION ANALYSIS ===")
# viz.plot_k_distribution_analysis()

# %%
# =============================================================================
# 4. K vs MODULARITY RELATIONSHIP ANALYSIS
# =============================================================================

print("\n=== K vs MODULARITY RELATIONSHIP ===")
# viz.plot_k_modularity_relationship()

# %%

# =============================================================================
# 5. FREQUENCY MATRIX WITH SURPRISAL WEIGHTING
# =============================================================================

print("\n=== FREQUENCY MATRIX WITH SURPRISAL WEIGHTING ===")

# compute frequency matrix WITHOUT surprisal weighting (for comparison)
frequency_matrix_raw = analyzer.aggregate_all_results(use_surprisal_weighting=False)

# compute frequency matrix WITH surprisal weighting (for final consensus)
frequency_matrix_weighted = analyzer.aggregate_all_results(use_surprisal_weighting=True)

print(f"Matrices: {frequency_matrix_weighted.shape}")

# USE SURPRISAL-WEIGHTED FREQUENCIES FOR SIGNIFICANCE TESTING
frequency_matrix = frequency_matrix_weighted

# normalize frequency matrix for visualization
off_diag_mask = ~np.eye(frequency_matrix.shape[0], dtype=bool)
freq_values = frequency_matrix.values[off_diag_mask]
max_freq = freq_values.max() if (freq_values > 0).any() else 1.0

if max_freq > 0:
    norm_freq = frequency_matrix / max_freq
else:
    norm_freq = frequency_matrix.copy()
np.fill_diagonal(norm_freq.values, 1.0)

# %%

# =============================================================================
# 5.5. SURPRISAL WEIGHTING COMPARISON
# =============================================================================

# use analyzer method for surprisal weighting comparison
surprisal_results = analyzer.analyze_surprisal_weighting_comparison(COLORS)

# extract results for further use
frequency_matrix_raw = surprisal_results['frequency_matrix_raw']
frequency_matrix_weighted = surprisal_results['frequency_matrix_weighted']

# recreate variables needed for later sections
off_diag_mask = ~np.eye(frequency_matrix_raw.shape[0], dtype=bool)
freq_values = frequency_matrix_weighted.values[off_diag_mask]
max_freq = freq_values.max() if (freq_values > 0).any() else 1.0

if max_freq > 0:
    norm_freq = frequency_matrix_weighted / max_freq
else:
    norm_freq = frequency_matrix_weighted.copy()
np.fill_diagonal(norm_freq.values, 1.0)

# %%

# =============================================================================
# 6. HIERARCHICAL CLUSTERING AND ORDERING
# =============================================================================

# use analyzer method for hierarchical clustering
clustering_results = analyzer.analyze_hierarchical_clustering(frequency_matrix, COLORS)

# extract results for further use
ward_linkage = clustering_results['ward_linkage']
filtered_freq = clustering_results['filtered_freq']
ordered_freq = clustering_results['ordered_freq']

# %%

# =============================================================================
# 6.5. PER-COMMUNITY CONSISTENCY ANALYSIS
# =============================================================================

# use analyzer method for per-community consistency analysis
community_results = analyzer.analyze_per_community_consistency(ward_linkage, filtered_freq, COLORS)

# extract results for further use if needed
communities = community_results['communities']
community_stats = community_results['community_stats']

# %%

# =============================================================================
# 7. STATISTICAL SIGNIFICANCE TESTING
# =============================================================================

# use analyzer method for statistical significance testing
significance_results = analyzer.analyze_statistical_significance(frequency_matrix_weighted, COLORS)

# extract results for further use
low_pairs = significance_results['low_pairs']
high_pairs = significance_results['high_pairs']
significant_mask = significance_results['significant_mask']
corrected_p_df = significance_results['corrected_p_df']

# %%

# %%

# =============================================================================
# 8. PER-METHOD SURPRISAL-WEIGHTED CO-CLUSTERING ANALYSIS
# =============================================================================

print("\n=== PER-METHOD SURPRISAL-WEIGHTED CO-CLUSTERING ANALYSIS ===")

# get aggregated results for each individual method
method_results = analyzer.aggregate_results_by_method_with_surprisal()

if method_results:
    # fill NaN values with 0
    frequency_matrices = {}
    for method_name, matrix in method_results.items():
        frequency_matrices[method_name] = matrix.fillna(0)
    
    # compare methods
    method_comparison = analyzer.compare_method_coclustering(frequency_matrices)
    
    # create visualization of per-method results with hierarchical clustering
    n_methods = len(method_results)
    cols = min(3, n_methods)
    rows = (n_methods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_methods == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    sorted_method_results = {}  # store sorted versions
    method_linkages = {}  # store linkages for dendrogram and ARI calculation
    
    for i, (method_name, matrix) in enumerate(method_results.items()):
        matrix = matrix.fillna(0)  # fill NaN with 0
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            break
        
        # apply hierarchical clustering to sort the matrix
        # normalize matrix by max off-diagonal value for distance calculation
        off_diag_mask = ~np.eye(matrix.shape[0], dtype=bool)
        max_off_diag = matrix.values[off_diag_mask].max() if (matrix.values[off_diag_mask] > 0).any() else 1.0
        
        if max_off_diag > 0:
            normalized_matrix = matrix / max_off_diag
        else:
            normalized_matrix = matrix.copy()
        
        # set diagonal to 1 for proper similarity matrix
        np.fill_diagonal(normalized_matrix.values, 1.0)
        
        # convert to distance matrix
        distance_matrix = normalized_matrix.values.max() - normalized_matrix.values
        condensed_distances = squareform(distance_matrix, checks=False)
        
        # perform ward linkage and get ordering
        try:
            ward_linkage = linkage(condensed_distances, method='ward', metric='euclidean')
            cluster_order = leaves_list(ward_linkage)
            ordered_matrix = matrix.iloc[cluster_order, cluster_order]
            sorted_method_results[method_name] = ordered_matrix
            method_linkages[method_name] = ward_linkage
        except Exception as e:
            # fallback to original matrix if clustering fails
            ordered_matrix = matrix
            sorted_method_results[method_name] = matrix
            method_linkages[method_name] = None
            print(f"Warning: Could not perform hierarchical clustering for {method_name}: {e}")
            
        # create heatmap for this method
        sns.heatmap(ordered_matrix, mask=ordered_matrix.values == 0, cmap=HEATMAP_CMAP, 
                   square=True, ax=ax, cbar=True,
                   cbar_kws={'shrink': 0.8, 'label': 'Surprisal-Weighted Frequency'})
        ax.set_title(f'{method_name}', fontweight='bold', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # rotate labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # hide unused subplots
    for i in range(n_methods, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('results/per_method_coclustering.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # create dendrograms for each method
    print("\n=== METHOD DENDROGRAMS ===")
    methods_with_linkage = {k: v for k, v in method_linkages.items() if v is not None}
    
    if methods_with_linkage:
        n_dendrograms = len(methods_with_linkage)
        cols = min(2, n_dendrograms)
        rows = (n_dendrograms + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12*cols, 8*rows))
        if n_dendrograms == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        for i, (method_name, linkage_matrix) in enumerate(methods_with_linkage.items()):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                break
                
            # get the corresponding matrix for outlet labels
            matrix = sorted_method_results[method_name]
            
            dendrogram(linkage_matrix, labels=matrix.index, ax=ax,
                      orientation='bottom', leaf_rotation=90, leaf_font_size=8)
            ax.set_title(f'{method_name} - Hierarchical Clustering', fontweight='bold', fontsize=12)
            ax.set_xlabel('Media Outlets', fontweight='bold')
            ax.set_ylabel('Distance', fontweight='bold')
        
        # hide unused subplots
        for i in range(n_dendrograms, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('results/per_method_dendrograms.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # calculate pairwise ARI between all methods across all levels
    print("\n=== PAIRWISE ARI BETWEEN METHODS (AVERAGED ACROSS ALL LEVELS) ===")
    
    if methods_with_linkage:
        from sklearn.metrics import adjusted_rand_score
        
        def average_ari_across_levels(linkage1, linkage2, k_range):
            """Calculate average ARI across multiple cluster levels"""
            ari_scores = []
            for k in k_range:
                try:
                    labels1 = fcluster(linkage1, k, criterion='maxclust')
                    labels2 = fcluster(linkage2, k, criterion='maxclust')
                    ari_score = adjusted_rand_score(labels1, labels2)
                    ari_scores.append(ari_score)
                except Exception as e:
                    print(f"Warning: Could not calculate ARI for k={k}: {e}")
                    continue
            return np.mean(ari_scores) if ari_scores else 0.0
        
        # determine reasonable range of cluster numbers
        # use range from 2 to n_outlets-1, but limit to reasonable range
        n_outlets = len(frequency_matrices[list(frequency_matrices.keys())[0]])
        max_clusters = min(15, n_outlets - 1)  # limit to 15 clusters max for computational efficiency
        k_range = range(2, max_clusters + 1)
        
        print(f"Calculating ARI across {len(k_range)} cluster levels: {list(k_range)}")
        
        # create ARI matrix
        method_names = list(methods_with_linkage.keys())
        n_methods_ari = len(method_names)
        ari_matrix = np.ones((n_methods_ari, n_methods_ari))  # diagonal is 1
        
        for i in range(n_methods_ari):
            for j in range(i+1, n_methods_ari):
                method1, method2 = method_names[i], method_names[j]
                avg_ari = average_ari_across_levels(methods_with_linkage[method1], 
                                                   methods_with_linkage[method2], 
                                                   k_range)
                ari_matrix[i, j] = avg_ari
                ari_matrix[j, i] = avg_ari  # symmetric
                print(f"  {method1} ↔ {method2}: {avg_ari:.3f}")
        
        # create ARI dataframe
        ari_df = pd.DataFrame(ari_matrix, index=method_names, columns=method_names)
        
        # visualize ARI matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(ari_df.values, dtype=bool), k=1)  # mask upper triangle
        sns.heatmap(ari_df, mask=mask, annot=True, fmt='.3f', cmap=DIVERGING_CMAP, 
                   center=0, square=True, cbar_kws={'label': 'Average Adjusted Rand Index'})
        plt.title(f'Pairwise ARI Between Methods\n(Averaged across {len(k_range)} cluster levels: {k_range[0]}-{k_range[-1]})', 
                 fontweight='bold', fontsize=14)
        plt.xlabel('Methods', fontweight='bold')
        plt.ylabel('Methods', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('results/method_pairwise_ari.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # print ARI statistics
        off_diag_ari = ari_matrix[~np.eye(n_methods_ari, dtype=bool)]
        print(f"\nARI Statistics (averaged across levels {k_range[0]}-{k_range[-1]}):")
        print(f"  Mean ARI: {off_diag_ari.mean():.3f}")
        print(f"  Std ARI: {off_diag_ari.std():.3f}")
        print(f"  Min ARI: {off_diag_ari.min():.3f}")
        print(f"  Max ARI: {off_diag_ari.max():.3f}")
        
        # show highest and lowest ARI pairs
        print(f"\nHighest ARI pairs:")
        for i in range(n_methods_ari):
            for j in range(i+1, n_methods_ari):
                if ari_matrix[i, j] >= np.percentile(off_diag_ari, 90):
                    print(f"  {method_names[i]} ↔ {method_names[j]}: {ari_matrix[i, j]:.3f}")
        
        print(f"\nLowest ARI pairs:")
        for i in range(n_methods_ari):
            for j in range(i+1, n_methods_ari):
                if ari_matrix[i, j] <= np.percentile(off_diag_ari, 10):
                    print(f"  {method_names[i]} ↔ {method_names[j]}: {ari_matrix[i, j]:.3f}")
    
    # analyze method-specific stable groups (using sorted matrices)
    print("\n=== METHOD-SPECIFIC STABLE GROUPS ===")
    matrices_to_analyze = sorted_method_results if 'sorted_method_results' in locals() else frequency_matrices
    
    for method_name, matrix in matrices_to_analyze.items():
        print(f"\n{method_name}:")
        
        # find strongly connected outlets (high co-clustering frequency)
        high_threshold = 0.5  # adjust as needed
        strong_pairs = []
        
        for i in range(len(matrix)):
            for j in range(i+1, len(matrix)):
                if matrix.iloc[i, j] > high_threshold:
                    outlet1 = matrix.index[i]
                    outlet2 = matrix.columns[j]
                    strong_pairs.append((outlet1, outlet2, matrix.iloc[i, j]))
        
        if strong_pairs:
            strong_pairs.sort(key=lambda x: x[2], reverse=True)
            print(f"  Strong co-clustering pairs (>{high_threshold:.1f}):")
            for outlet1, outlet2, freq in strong_pairs[:10]:  # top 10
                print(f"    {outlet1} ↔ {outlet2}: {freq:.3f}")
        else:
            print(f"  No strong co-clustering pairs found (threshold={high_threshold})")
            
                # also show clusters of outlets that frequently group together
        if method_name in method_linkages and method_linkages[method_name] is not None:
            try:
                # extract clusters at 70% height
                sorted_matrix = sorted_method_results[method_name]
                linkage_matrix = method_linkages[method_name]
                
                # calculate height threshold
                max_dist = linkage_matrix[:, 2].max()
                height_threshold = 0.7 * max_dist
                labels = fcluster(linkage_matrix, height_threshold, criterion='distance')
                
                clusters = defaultdict(list)
                for i, label in enumerate(labels):
                    clusters[label].append(sorted_matrix.index[i])
                
                multi_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
                if multi_clusters:
                    print(f"  Stable outlet clusters (70% height):")
                    for cid, outlets in sorted(multi_clusters.items()):
                        print(f"    Cluster {cid}: {', '.join(outlets)}")
                        
            except Exception as e:
                print(f"  Could not extract stable clusters: {e}")
    
else:
    print("No method results to analyze")

# %%

# =============================================================================
# 9. STATISTICALLY VALIDATED CONSENSUS CLUSTERING
# =============================================================================

print("\n=== STATISTICALLY VALIDATED CONSENSUS CLUSTERING ===")

# create final consensus using only statistically significant relationships
print("creating consensus clustering using only significant pairs...")

# get significant pairs from our statistical test
significant_mask = corrected_p_df < 0.05
final_consensus_matrix = frequency_matrix.copy()

# zero out non-significant relationships
final_consensus_matrix = final_consensus_matrix * significant_mask.astype(int)

original_pairs = np.sum((frequency_matrix > 0).values) - len(frequency_matrix)
significant_pairs = np.sum((final_consensus_matrix > 0).values) - len(final_consensus_matrix)
retention_rate = (significant_pairs / original_pairs * 100) if original_pairs > 0 else 0

print(f"original relationships: {original_pairs}")
print(f"significant relationships: {significant_pairs}")
print(f"retained: {retention_rate:.1f}%")

# perform hierarchical clustering on validated consensus
print("performing hierarchical clustering on statistically validated matrix...")

# normalize for distance calculation  
normalized_matrix = final_consensus_matrix / final_consensus_matrix.values.max() if final_consensus_matrix.values.max() > 0 else final_consensus_matrix
np.fill_diagonal(normalized_matrix.values, 1.0)

# convert to distance matrix
distance_matrix = normalized_matrix.values.max() - normalized_matrix.values
condensed_distances = squareform(distance_matrix, checks=False)

try:
    # perform ward linkage
    final_linkage = linkage(condensed_distances, method='ward', metric='euclidean')
    
    # create final visualization
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # 1. statistically validated frequency matrix
    sns.heatmap(final_consensus_matrix, mask=final_consensus_matrix.values == 0, 
               cmap=HEATMAP_CMAP, square=True, ax=axes[0], cbar=True,
               cbar_kws={'label': 'Co-clustering Frequency'})
    axes[0].set_title('Statistically Validated Consensus\n(Only Significant Pairs)', fontweight='bold')
    
    # 2. comparison with original
    comparison_matrix = (final_consensus_matrix > 0).astype(int) - (frequency_matrix > 0).astype(int)
    sns.heatmap(comparison_matrix, cmap='RdBu_r', center=0, square=True, ax=axes[1], 
               cbar=True, vmin=-1, vmax=1,
               cbar_kws={'label': 'Change (-1=Removed, 0=Same, +1=Added)'})
    axes[1].set_title('Statistical Filtering Impact\n(Red=Removed, Blue=Added)', fontweight='bold')
    
    # 3. final dendrogram
    cluster_order = leaves_list(final_linkage)
    ordered_final_matrix = final_consensus_matrix.iloc[cluster_order, cluster_order]
    
    dendrogram(final_linkage, labels=final_consensus_matrix.index, ax=axes[2],
              orientation='bottom', leaf_rotation=90, leaf_font_size=8)
    axes[2].set_title('Final Validated Clustering\n(Statistically Significant Only)', fontweight='bold')
    axes[2].set_xlabel('Media Outlets', fontweight='bold')
    axes[2].set_ylabel('Distance', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/final_validated_consensus.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # extract final clusters at multiple levels
    print("\n=== FINAL VALIDATED CLUSTERS ===")
    for n_clusters in [3, 4, 5, 6]:
        cluster_labels = fcluster(final_linkage, n_clusters, criterion='maxclust')
        
        print(f"\n{n_clusters} clusters:")
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(final_consensus_matrix.index[i])
        
        for cluster_id, outlets in sorted(clusters.items()):
            print(f"  Cluster {cluster_id}: {', '.join(outlets)} ({len(outlets)} outlets)")
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Final statistically validated clustering saved to results/")
    
except Exception as e:
    print(f"Could not perform final clustering: {e}")
    print("Analysis complete with statistical significance testing")

# %%


# =============================================================================
# 12. INTERACTIVE VISUALIZATION: ADJUSTABLE EDGE FILTERING
# =============================================================================

print("\n=== INTERACTIVE VISUALIZATION: ADJUSTABLE EDGE FILTERING ===")

import ipywidgets as widgets
from IPython.display import display, clear_output
from matplotlib.pyplot import figure, show
import matplotlib.patches as mpatches

# create interactive visualization
print("Creating interactive visualization...")
print("Use the slider to control what fraction of the weakest edges are removed")

# prepare the data
if 'final_consensus_matrix' in locals():
    base_matrix = final_consensus_matrix.copy()
else:
    print("Warning: Using frequency_matrix_weighted as fallback")
    base_matrix = frequency_matrix_weighted.copy()

# get non-zero values for filtering
non_zero_mask = base_matrix.values > 0
non_zero_values = base_matrix.values[non_zero_mask]

if len(non_zero_values) == 0:
    print("No non-zero values found in matrix")
else:
    print(f"Found {len(non_zero_values)} non-zero relationships to filter")
    
    # create the interactive function
    def update_visualization(edge_removal_fraction):
        # clear previous output
        clear_output(wait=True)
        
        # calculate threshold for edge removal
        if edge_removal_fraction > 0:
            threshold = np.percentile(non_zero_values, edge_removal_fraction * 100)
        else:
            threshold = 0
        
        # create filtered matrix
        filtered_matrix = base_matrix.copy()
        filtered_matrix[filtered_matrix < threshold] = 0
        
        # count remaining edges
        remaining_edges = np.sum(filtered_matrix.values > 0) - len(filtered_matrix)  # subtract diagonal
        original_edges = len(non_zero_values)
        retention_rate = (remaining_edges / original_edges * 100) if original_edges > 0 else 0
        
        # create visualization
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # 1. Filtered heatmap
        sns.heatmap(filtered_matrix, mask=filtered_matrix.values == 0, 
                    cmap=HEATMAP_CMAP, square=True, ax=axes[0], cbar=True,
                    cbar_kws={'label': 'Co-clustering Frequency'})
        axes[0].set_title(f'Filtered Consensus Matrix\n'
                            f'Threshold: {threshold:.3f} ({edge_removal_fraction:.1%} weakest removed)\n'
                            f'Remaining: {remaining_edges}/{original_edges} edges ({retention_rate:.1f}%)', 
                            fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].tick_params(axis='y', rotation=0)
        
        # 2. Network graph
        if remaining_edges > 0:
            # create network from filtered matrix
            G_filtered = nx.Graph()
            
            # add nodes
            for outlet in filtered_matrix.index:
                G_filtered.add_node(outlet)
            
            # add edges
            edges_added = []
            for i in range(len(filtered_matrix)):
                for j in range(i+1, len(filtered_matrix)):
                    if filtered_matrix.iloc[i, j] > 0:
                        outlet1 = filtered_matrix.index[i]
                        outlet2 = filtered_matrix.columns[j]
                        weight = filtered_matrix.iloc[i, j]
                        G_filtered.add_edge(outlet1, outlet2, weight=weight)
                        edges_added.append((outlet1, outlet2))
            
            # create layout
            if G_filtered.number_of_edges() > 0:
                pos = nx.spring_layout(G_filtered, k=2, iterations=50, seed=42)
                
                # node sizes based on degree
                degrees = dict(G_filtered.degree())
                node_sizes = [degrees[node] * 100 + 300 for node in G_filtered.nodes()]
                
                # edge widths based on weight
                edge_weights = [G_filtered[u][v]['weight'] for u, v in edges_added]
                max_weight = max(edge_weights) if edge_weights else 1
                edge_widths = [3 * (w / max_weight) + 1 for w in edge_weights]
                
                # draw network
                nx.draw_networkx_nodes(G_filtered, pos, node_color=COLORS['primary'], 
                                        node_size=node_sizes, alpha=0.8, 
                                        edgecolors=COLORS['text'], ax=axes[1])
                nx.draw_networkx_edges(G_filtered, pos, edgelist=edges_added,
                                        edge_color=COLORS['secondary'], width=edge_widths, 
                                        alpha=0.7, ax=axes[1])
                nx.draw_networkx_labels(G_filtered, pos, font_size=8, font_weight='bold', 
                                        font_color=COLORS['text'], ax=axes[1])
                
                axes[1].set_title(f'Network Graph\n'
                                    f'{G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges\n'
                                    f'Density: {nx.density(G_filtered):.3f}', 
                                    fontweight='bold')
            else:
                axes[1].text(0.5, 0.5, 'No edges remaining', 
                            ha='center', va='center', transform=axes[1].transAxes,
                            fontsize=14, fontweight='bold')
                axes[1].set_title('Network Graph\n(No connections)', fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, 'No edges remaining', 
                        ha='center', va='center', transform=axes[1].transAxes,
                        fontsize=14, fontweight='bold')
            axes[1].set_title('Network Graph\n(No connections)', fontweight='bold')
        
        axes[1].axis('off')
        
        # 3. Edge strength distribution
        if len(non_zero_values) > 0:
            axes[2].hist(non_zero_values, bins=30, alpha=0.7, color=COLORS['primary'], 
                        edgecolor=COLORS['text'])
            axes[2].axvline(threshold, color=COLORS['quaternary'], linestyle='--', 
                            linewidth=2, label=f'Threshold: {threshold:.3f}')
            axes[2].set_xlabel('Co-clustering Frequency', fontweight='bold')
            axes[2].set_ylabel('Count', fontweight='bold')
            axes[2].set_title('Edge Strength Distribution\n(with current threshold)', fontweight='bold')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'No data to display', 
                        ha='center', va='center', transform=axes[2].transAxes,
                        fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # print summary statistics
        print(f"\nFiltering Summary:")
        print(f"  Edge removal fraction: {edge_removal_fraction:.1%}")
        print(f"  Threshold value: {threshold:.3f}")
        print(f"  Original edges: {original_edges}")
        print(f"  Remaining edges: {remaining_edges}")
        print(f"  Retention rate: {retention_rate:.1f}%")
        
        if remaining_edges > 0 and 'G_filtered' in locals():
            print(f"  Network density: {nx.density(G_filtered):.3f}")
            print(f"  Connected components: {nx.number_connected_components(G_filtered)}")
            
            # show most connected nodes
            if G_filtered.number_of_edges() > 0:
                degree_centrality = nx.degree_centrality(G_filtered)
                top_connected = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  Most connected outlets:")
                for outlet, centrality in top_connected:
                    if centrality > 0:
                        print(f"    {outlet}: {G_filtered.degree(outlet)} connections")
    
    # create the slider widget
    edge_slider = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=0.9,
        step=0.05,
        description='Remove weakest:',
        style={'description_width': 'initial'},
        continuous_update=False
    )
    
    # create interactive widget
    interactive_viz = widgets.interact(update_visualization, edge_removal_fraction=edge_slider)
    
    print("\n" + "="*60)
    print("INTERACTIVE VISUALIZATION READY")
    print("="*60)
    print("Use the slider below to adjust the fraction of weakest edges to remove")
    print("- 0% = Show all statistically significant relationships")
    print("- 50% = Remove the weakest 50% of relationships")
    print("- 90% = Show only the strongest 10% of relationships")
    print("="*60)

# %%


# =============================================================================
# 10. NETWORK GRAPH OF STATISTICALLY SIGNIFICANT CONNECTIONS
# =============================================================================

print("\n=== NETWORK GRAPH OF SIGNIFICANT CONNECTIONS ===")

# create network graph from significant connections
G = nx.Graph()

# add all outlets as nodes
for outlet in frequency_matrix_weighted.index:
    G.add_node(outlet)

# add edges for significant connections
high_edges = []
low_edges = []

# add significantly HIGH co-clustering pairs
for pair in high_pairs:
    outlet1, outlet2 = pair['outlet1'], pair['outlet2']
    weight = pair['observed']
    p_val = pair['p_value']
    
    G.add_edge(outlet1, outlet2, weight=weight, p_value=p_val, 
               edge_type='high', deviation=pair['deviation'])
    high_edges.append((outlet1, outlet2))

# add significantly LOW co-clustering pairs  
for pair in low_pairs:
    outlet1, outlet2 = pair['outlet1'], pair['outlet2']
    weight = pair['observed']
    p_val = pair['p_value']
    
    G.add_edge(outlet1, outlet2, weight=weight, p_value=p_val, 
               edge_type='low', deviation=pair['deviation'])
    low_edges.append((outlet1, outlet2))

print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
print(f"  Significantly HIGH co-clustering edges: {len(high_edges)}")
print(f"  Significantly LOW co-clustering edges: {len(low_edges)}")

# create the network visualization
plt.figure(figsize=(16, 12))

# use spring layout for better visualization
pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

# draw nodes
nx.draw_networkx_nodes(G, pos, node_color=COLORS['primary'], 
                       node_size=800, alpha=0.8, edgecolors=COLORS['text'])

# draw edges with different colors for high vs low significance
if high_edges:
    nx.draw_networkx_edges(G, pos, edgelist=high_edges, 
                          edge_color=COLORS['secondary'], width=2, alpha=0.7,
                          label='Significantly HIGH co-clustering')

if low_edges:
    nx.draw_networkx_edges(G, pos, edgelist=low_edges, 
                          edge_color=COLORS['quaternary'], width=1, alpha=0.7,
                          style='dashed', label='Significantly LOW co-clustering')

# draw labels
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', 
                       font_color=COLORS['text'])

plt.title('Network Graph of Statistically Significant Co-clustering Relationships\n' +
          f'({len(high_edges)} high-significance, {len(low_edges)} low-significance connections)',
          fontsize=14, fontweight='bold', pad=20)

# add legend
if high_edges or low_edges:
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

plt.axis('off')
plt.tight_layout()
plt.savefig('results/significant_connections_network.png', dpi=300, bbox_inches='tight')
plt.show()

# analyze network properties
print(f"\nNetwork Analysis:")
print(f"  Nodes (outlets): {G.number_of_nodes()}")
print(f"  Edges (significant connections): {G.number_of_edges()}")
print(f"  Network density: {nx.density(G):.3f}")
print(f"  Connected components: {nx.number_connected_components(G)}")

# identify highly connected outlets
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

print(f"\nMost connected outlets (by degree centrality):")
for outlet, centrality in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]:
    degree = G.degree(outlet)
    print(f"  {outlet}: {degree} connections (centrality: {centrality:.3f})")

print(f"\nMost influential outlets (by betweenness centrality):")
for outlet, centrality in sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]:
    if centrality > 0:
        print(f"  {outlet}: {centrality:.3f}")

# create a focused view of only the most significant connections
if high_pairs or low_pairs:
    # get top significant connections
    all_significant = high_pairs + low_pairs
    all_significant.sort(key=lambda x: x['p_value'])
    top_significant = all_significant[:20]  # top 20 most significant
    
    # create focused network
    G_focused = nx.Graph()
    outlets_in_focused = set()
    
    for pair in top_significant:
        outlet1, outlet2 = pair['outlet1'], pair['outlet2']
        G_focused.add_edge(outlet1, outlet2, **pair)
        outlets_in_focused.add(outlet1)
        outlets_in_focused.add(outlet2)
    
    print(f"\nFocused network (top 20 most significant connections):")
    print(f"  Outlets involved: {len(outlets_in_focused)}")
    print(f"  Connections: {G_focused.number_of_edges()}")
    
    # visualize focused network
    plt.figure(figsize=(14, 10))
    
    pos_focused = nx.spring_layout(G_focused, k=2, iterations=50, seed=42)
    
    # separate high and low edges for focused graph
    high_edges_focused = [(u, v) for u, v, d in G_focused.edges(data=True) if d.get('deviation', 0) > 0]
    low_edges_focused = [(u, v) for u, v, d in G_focused.edges(data=True) if d.get('deviation', 0) < 0]
    
    # draw nodes with size proportional to number of connections
    node_sizes = [G_focused.degree(node) * 200 + 300 for node in G_focused.nodes()]
    nx.draw_networkx_nodes(G_focused, pos_focused, node_color=COLORS['primary'], 
                          node_size=node_sizes, alpha=0.8, edgecolors=COLORS['text'])
    
    # draw edges
    if high_edges_focused:
        nx.draw_networkx_edges(G_focused, pos_focused, edgelist=high_edges_focused, 
                              edge_color=COLORS['secondary'], width=3, alpha=0.8)
    
    if low_edges_focused:
        nx.draw_networkx_edges(G_focused, pos_focused, edgelist=low_edges_focused, 
                              edge_color=COLORS['quaternary'], width=2, alpha=0.8, style='dashed')
    
    # draw labels
    nx.draw_networkx_labels(G_focused, pos_focused, font_size=10, font_weight='bold', 
                           font_color=COLORS['text'])
    
    plt.title('Most Statistically Significant Co-clustering Relationships\n' +
              f'(Top 20 connections by p-value significance)',
              fontsize=14, fontweight='bold', pad=20)
    
    # add edge labels for p-values on most significant edges
    edge_labels = {}
    for u, v, d in G_focused.edges(data=True):
        p_val = d.get('p_value', 1.0)
        if p_val < 0.001:
            edge_labels[(u, v)] = f"p<0.001"
        elif p_val < 0.01:
            edge_labels[(u, v)] = f"p<0.01"
        else:
            edge_labels[(u, v)] = f"p={p_val:.3f}"
    
    nx.draw_networkx_edge_labels(G_focused, pos_focused, edge_labels, font_size=8, alpha=0.7)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/top_significant_connections_network.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 10.5. FOCUSED GRAPH: ONLY STATISTICALLY HIGH CO-CLUSTERING RELATIONSHIPS
# =============================================================================

print("\n=== FOCUSED GRAPH: HIGH CO-CLUSTERING RELATIONSHIPS ONLY ===")

if high_pairs:
    # create network with only high co-clustering relationships
    G_high_only = nx.Graph()
    
    # add all outlets as nodes first
    for outlet in frequency_matrix_weighted.index:
        G_high_only.add_node(outlet)
    
    # add only significantly HIGH co-clustering edges
    high_edges_only = []
    for pair in high_pairs:
        outlet1, outlet2 = pair['outlet1'], pair['outlet2']
        weight = pair['observed']
        p_val = pair['p_value']
        
        G_high_only.add_edge(outlet1, outlet2, weight=weight, p_value=p_val, 
                             deviation=pair['deviation'])
        high_edges_only.append((outlet1, outlet2))
    
    print(f"High co-clustering network: {G_high_only.number_of_nodes()} nodes, {G_high_only.number_of_edges()} edges")
    
    # create layout optimized for this network
    plt.figure(figsize=(16, 12))
    
    # use different layout depending on network density
    if nx.density(G_high_only) > 0.3:
        pos_high = nx.kamada_kawai_layout(G_high_only)
    else:
        pos_high = nx.spring_layout(G_high_only, k=2, iterations=100, seed=42)
    
    # calculate node sizes based on degree (number of high-significance connections)
    degrees = dict(G_high_only.degree())
    node_sizes = [degrees[node] * 150 + 400 for node in G_high_only.nodes()]
    
    # create color mapping based on degree
    max_degree = max(degrees.values()) if degrees.values() else 1
    node_colors = [degrees[node] / max_degree for node in G_high_only.nodes()]
    
    # draw nodes with size and color based on connectivity
    nodes = nx.draw_networkx_nodes(G_high_only, pos_high, 
                                  node_size=node_sizes,
                                  node_color=node_colors,
                                  cmap=plt.cm.Blues,
                                  alpha=0.8, 
                                  edgecolors=COLORS['text'],
                                  linewidths=2)
    
    # draw edges with thickness based on co-clustering strength
    edge_weights = [G_high_only[u][v]['weight'] for u, v in high_edges_only]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [3 * (w / max_weight) + 1 for w in edge_weights]  # scale edge thickness
    
    nx.draw_networkx_edges(G_high_only, pos_high, edgelist=high_edges_only,
                          edge_color=COLORS['secondary'], width=edge_widths, 
                          alpha=0.7)
    
    # draw labels
    nx.draw_networkx_labels(G_high_only, pos_high, font_size=10, font_weight='bold', 
                           font_color=COLORS['text'])
    
    plt.title('Media Outlets with Statistically Higher Than Expected Co-clustering\n' +
              f'({len(high_edges_only)} significant relationships, α = 0.05)',
              fontsize=16, fontweight='bold', pad=20)
    
    # add colorbar for node colors
    if max_degree > 1:
        cbar = plt.colorbar(nodes, shrink=0.8, pad=0.1)
        cbar.set_label('Number of High-Significance Connections', fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/high_coclustering_network.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # analyze the high co-clustering network
    print(f"\nHigh Co-clustering Network Analysis:")
    print(f"  Network density: {nx.density(G_high_only):.3f}")
    print(f"  Connected components: {nx.number_connected_components(G_high_only)}")
    
    # identify most connected outlets in high co-clustering
    high_degree_centrality = nx.degree_centrality(G_high_only)
    high_betweenness_centrality = nx.betweenness_centrality(G_high_only)
    
    print(f"\nOutlets with most high-significance connections:")
    for outlet, centrality in sorted(high_degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]:
        degree = G_high_only.degree(outlet)
        if degree > 0:
            print(f"  {outlet}: {degree} high-significance connections")
    
    # community detection on high co-clustering network
    if G_high_only.number_of_edges() > 0:
        # remove isolated nodes (no edges)
        G_connected = G_high_only.copy()
        isolated_nodes = [node for node in G_connected.nodes() if G_connected.degree(node) == 0]
        G_connected.remove_nodes_from(isolated_nodes)
        
        print(f"Removed {len(isolated_nodes)} isolated nodes for community detection")
        
        try:
            from networkx.algorithms import community
            communities_high = community.greedy_modularity_communities(G_connected)
            modularity = community.modularity(G_connected, communities_high)
            
            print(f"\nCommunity Detection Results:")
            print(f"  Found {len(communities_high)} communities (modularity: {modularity:.3f})")
            for i, comm in enumerate(communities_high, 1):
                if len(comm) > 1:
                    print(f"  Community {i}: {', '.join(sorted(comm))}")
            
            # visualize communities (only connected nodes)
            plt.figure(figsize=(14, 10))
            pos_comm = nx.spring_layout(G_connected, k=2, iterations=100, seed=42)
            
            # assign colors to communities
            colors = plt.cm.Set3(np.linspace(0, 1, len(communities_high)))
            node_colors = []
            for node in G_connected.nodes():
                for i, comm in enumerate(communities_high):
                    if node in comm:
                        node_colors.append(colors[i])
                        break
            
            # draw network with community colors
            nx.draw_networkx_nodes(G_connected, pos_comm, node_color=node_colors, 
                                 node_size=600, alpha=0.8, edgecolors='black')
            nx.draw_networkx_edges(G_connected, pos_comm, alpha=0.6, edge_color='gray')
            nx.draw_networkx_labels(G_connected, pos_comm, font_size=9, font_weight='bold')
            
            plt.title(f'Community Structure in High Co-clustering Network\n'
                     f'({len(communities_high)} communities, modularity = {modularity:.3f})',
                     fontweight='bold', fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('results/high_coclustering_communities.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Community detection failed: {e}")
    
    # show strongest co-clustering pairs
    print(f"\nStrongest high co-clustering relationships:")
    high_pairs_sorted = sorted(high_pairs, key=lambda x: x['observed'], reverse=True)
    for i, pair in enumerate(high_pairs_sorted[:10], 1):
        print(f"  {i:2d}. {pair['outlet1']} ↔ {pair['outlet2']}: "
              f"frequency={pair['observed']:.3f}, p={pair['p_value']:.6f}")

else:
    print("No significantly high co-clustering pairs found to visualize")

# %%
# =============================================================================
# 11. NULL DISTRIBUTION VALIDATION AND COMPARISON
# =============================================================================

print("\n=== NULL DISTRIBUTION VALIDATION ===")
print("Validating analytical null distribution against empirical permutations...")

# 1. single clustering validation (as a sanity check)
print("\n1. Single Clustering Validation:")
single_validation = analyzer.validate_null_distribution(
    n_permutations=5000,
    clustering_index=0,
    random_state=42,
    save_path='results/single_clustering_validation.png'
)

# 2. aggregate validation across all clusterings
print("\n2. Aggregate Validation (All Clusterings):")
# test multiple random pairs to check consistency
test_pairs = [(5, 15), (10, 25), (20, 35), (2, 40)]  # various outlet pairs
aggregate_validations = []

for pair in test_pairs:
    print(f"\nTesting pair {pair}...")
    empirical_result = analyzer.empirical_null_for_pair(
        pair=pair, 
        n_permutations=1000, 
        random_state=42
    )
    
    if empirical_result:
        # get analytical values from significance results
        analytical_mean = significance_results['null_mean']
        analytical_std = significance_results['null_std']
        
        empirical_mean = empirical_result['empirical_mean']
        empirical_std = empirical_result['empirical_std']
        
        # calculate differences
        mean_diff = abs(empirical_mean - analytical_mean)
        std_diff = abs(empirical_std - analytical_std)
        mean_rel_error = mean_diff / analytical_mean if analytical_mean > 0 else float('inf')
        std_rel_error = std_diff / analytical_std if analytical_std > 0 else float('inf')
        
        validation_result = {
            'pair': pair,
            'analytical_mean': analytical_mean,
            'empirical_mean': empirical_mean,
            'analytical_std': analytical_std,
            'empirical_std': empirical_std,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'mean_rel_error': mean_rel_error,
            'std_rel_error': std_rel_error
        }
        
        aggregate_validations.append(validation_result)
        
        print(f"  Analytical: mean={analytical_mean:.6f}, std={analytical_std:.6f}")
        print(f"  Empirical:  mean={empirical_mean:.6f}, std={empirical_std:.6f}")
        print(f"  Differences: Δmean={mean_diff:.6f} ({mean_rel_error:.1%}), Δstd={std_diff:.6f} ({std_rel_error:.1%})")

# 3. create comparison visualization
if aggregate_validations:
    print("\n3. Validation Summary:")
    
    # create summary dataframe
    validation_df = pd.DataFrame(aggregate_validations)
    
    print("Validation Results Summary:")
    print(f"  Mean relative error (mean): {validation_df['mean_rel_error'].mean():.1%} ± {validation_df['mean_rel_error'].std():.1%}")
    print(f"  Mean relative error (std):  {validation_df['std_rel_error'].mean():.1%} ± {validation_df['std_rel_error'].std():.1%}")
    
    # check if errors are acceptable (< 5% typically)
    mean_errors_ok = (validation_df['mean_rel_error'] < 0.05).all()
    std_errors_ok = (validation_df['std_rel_error'] < 0.05).all()
    
    print(f"  Mean validation: {'✓ PASS' if mean_errors_ok else '✗ FAIL'} (all errors < 5%)")
    print(f"  Std validation:  {'✓ PASS' if std_errors_ok else '✗ FAIL'} (all errors < 5%)")
    
    # create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # mean comparison
    axes[0].scatter(validation_df['analytical_mean'], validation_df['empirical_mean'], 
                   color=COLORS['primary'], s=100, alpha=0.7, edgecolors=COLORS['text'])
    
    # add perfect correlation line
    min_mean = min(validation_df['analytical_mean'].min(), validation_df['empirical_mean'].min())
    max_mean = max(validation_df['analytical_mean'].max(), validation_df['empirical_mean'].max())
    axes[0].plot([min_mean, max_mean], [min_mean, max_mean], 
                color=COLORS['quaternary'], linestyle='--', alpha=0.8, label='Perfect Agreement')
    
    axes[0].set_xlabel('Analytical Mean', fontweight='bold')
    axes[0].set_ylabel('Empirical Mean', fontweight='bold')
    axes[0].set_title('Null Distribution Mean\nAnalytical vs Empirical', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # std comparison
    axes[1].scatter(validation_df['analytical_std'], validation_df['empirical_std'], 
                   color=COLORS['secondary'], s=100, alpha=0.7, edgecolors=COLORS['text'])
    
    # add perfect correlation line
    min_std = min(validation_df['analytical_std'].min(), validation_df['empirical_std'].min())
    max_std = max(validation_df['analytical_std'].max(), validation_df['empirical_std'].max())
    axes[1].plot([min_std, max_std], [min_std, max_std], 
                color=COLORS['quaternary'], linestyle='--', alpha=0.8, label='Perfect Agreement')
    
    axes[1].set_xlabel('Analytical Std', fontweight='bold')
    axes[1].set_ylabel('Empirical Std', fontweight='bold')
    axes[1].set_title('Null Distribution Std\nAnalytical vs Empirical', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/null_distribution_validation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. investigate potential causes of discrepancy if validation fails
    if not (mean_errors_ok and std_errors_ok):
        print("\n4. Investigating Potential Issues:")
        
        # check clustering independence
        print("  Checking clustering independence...")
        
        # calculate correlation between consecutive clusterings
        results_df = analyzer.get_results()
        if len(results_df) > 1:
            # get a sample pair for correlation analysis
            test_pair = test_pairs[0]
            i, j = test_pair
            
            # extract weights for this pair across all clusterings
            pair_weights = []
            for _, row in results_df.iterrows():
                communities = row['communities']
                if i in communities and j in communities:
                    if communities[i] == communities[j]:
                        # calculate cluster size and weight
                        cluster_sizes = Counter(communities.values())
                        cluster_id = communities[i]
                        size = cluster_sizes[cluster_id]
                        weight = -np.log(size / len(analyzer.outlet_names))
                        pair_weights.append(weight)
                    else:
                        pair_weights.append(0.0)
                else:
                    pair_weights.append(0.0)
            
            # calculate autocorrelation
            if len(pair_weights) > 1:
                autocorr = np.corrcoef(pair_weights[:-1], pair_weights[1:])[0, 1]
                print(f"    Lag-1 autocorrelation: {autocorr:.3f}")
                
                if abs(autocorr) > 0.1:
                    print(f"    ⚠️  High autocorrelation detected! Clusterings may not be independent.")
                    print(f"    This could explain why empirical std > analytical std")
                else:
                    print(f"    ✓ Autocorrelation is low, clusterings appear independent")
        
        # check for other potential issues
        max_rel_error = max(validation_df['mean_rel_error'].max(), validation_df['std_rel_error'].max())
        if max_rel_error > 0.2:
            print(f"    ⚠️  Large relative errors (>{max_rel_error:.1%}) suggest potential issues:")
            print(f"      - Check if cluster size distributions are as expected")
            print(f"      - Verify surprisal weight calculations")
            print(f"      - Consider numerical precision issues")

# 5. final assessment
print(f"\n=== NULL DISTRIBUTION VALIDATION SUMMARY ===")
if aggregate_validations:
    overall_pass = mean_errors_ok and std_errors_ok
    print(f"Overall validation: {'✓ PASS' if overall_pass else '✗ NEEDS INVESTIGATION'}")
    
    if overall_pass:
        print("✓ Analytical null distribution is accurate")
        print("✓ Statistical significance results are reliable")
        print("✓ P-values can be trusted for interpretation")
    else:
        print("⚠️  Analytical vs empirical discrepancy detected")
        print("⚠️  Statistical significance results should be interpreted with caution")
        print("⚠️  Consider using empirical permutation tests instead")
        
        # suggest corrected significance threshold
        if validation_df['std_rel_error'].mean() > 0.05:
            correction_factor = validation_df['empirical_std'].mean() / validation_df['analytical_std'].mean()
            print(f"⚠️  Suggested correction factor for std: {correction_factor:.3f}")
else:
    print("⚠️  Could not validate null distribution - no successful comparisons")

# %%