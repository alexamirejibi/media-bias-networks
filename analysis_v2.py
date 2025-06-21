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
from scipy.cluster.hierarchy import linkage, dendrogram
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

# set style
plt.style.use('default')
sns.set_palette('husl')


# %%

# =============================================================================
# 1. EXPERIMENT SETUP & DATA OVERVIEW
# =============================================================================

print("=== MEDIA BIAS NETWORK ANALYSIS ===")

# experiment parameters
data_dir = 'data/daily_cluster_matrices_min_6'
n_samples = 10
n_days = 60
results_file = f'results/experiment_results_{n_samples}samples_{n_days}days.pkl'

# check if results already exist
if os.path.exists(results_file):
    print(f"Loading existing results from {results_file}...")
    
    # load saved results
    with open(results_file, 'rb') as f:
        saved_data = pickle.load(f)
    
    # extract components
    experiment = saved_data['experiment']
    experiment_summary = saved_data['experiment_summary']
    analyzer = experiment.analyzer
    viz = Visualizer(analyzer)
    
    print(f"Loaded experiment results:")
    print(f"Total results: {len(analyzer.get_results())}")
    print(f"Network methods: {len(analyzer.get_results()['network_method'].unique())}")
    print(f"Community methods: {len(analyzer.get_results()['community_method'].unique())}")
    
else:
    print("No existing results found. Running new experiment...")
    print("Initializing experiment framework...")
    
    # initialize experiment
    experiment = ExperimentFramework(data_dir)
    
    # run experiment with more samples for robust analysis
    print(f"Running experiment: {n_samples} samples × {n_days} days")
    experiment_summary = experiment.run_experiment(n_samples=n_samples, n_days=n_days)
    
    analyzer = experiment.analyzer
    viz = Visualizer(analyzer)
    
    print(f"\nEXPERIMENT SUMMARY:")
    print(f"Total results: {experiment_summary['total_results']}")
    print(f"Network methods: {len(analyzer.get_results()['network_method'].unique())}")
    print(f"Community methods: {len(analyzer.get_results()['community_method'].unique())}")
    print(f"Average time per sample: {experiment_summary['avg_time_per_sample']:.1f}s")
    
    # save results
    print(f"\nSaving results to {results_file}...")
    os.makedirs('results', exist_ok=True)
    
    # create data to save
    save_data = {
        'experiment': experiment,
        'experiment_summary': experiment_summary,
        'n_samples': n_samples,
        'n_days': n_days,
        'data_dir': data_dir
    }
    
    with open(results_file, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Results saved successfully!")


# %%

# =============================================================================
# 1.5. EXCLUSION ANALYSIS - METHODS FILTERED BY COMMUNITY COUNT
# =============================================================================

print("\n=== EXCLUSION ANALYSIS ===")

# get results before exclusion
all_results = analyzer.get_results()
excluded_results = all_results[(all_results['n_communities'] == 1) | (all_results['n_communities'] >= 49)]

print(f"Total results: {len(all_results)}")
print(f"Excluded results (k=1 or k=n_outlets): {len(excluded_results)} ({len(excluded_results)/len(all_results)*100:.1f}%)")

# analyze exclusion by method type
def calculate_exclusion_fractions(results_df, method_col):
    """calculate fraction of results excluded for each method"""
    method_stats = []
    
    for method in results_df[method_col].unique():
        method_results = results_df[results_df[method_col] == method]
        total = len(method_results)
        k1_count = len(method_results[method_results['n_communities'] == 1])
        k49plus_count = len(method_results[method_results['n_communities'] >= 49])
        excluded_count = k1_count + k49plus_count
        
        method_stats.append({
            'method': method,
            'total': total,
            'k1_fraction': k1_count / total if total > 0 else 0,
            'k49plus_fraction': k49plus_count / total if total > 0 else 0,
            'excluded_fraction': excluded_count / total if total > 0 else 0
        })
    
    return pd.DataFrame(method_stats).sort_values('excluded_fraction', ascending=False)

# calculate stats for both method types
comm_stats = calculate_exclusion_fractions(all_results, 'community_method')
net_stats = calculate_exclusion_fractions(all_results, 'network_method')

# create visualizations
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# community methods
y_pos = range(len(comm_stats))
axes[0].barh(y_pos, comm_stats['k1_fraction'], color='red', alpha=0.7, label='k=1 (single community)')
axes[0].barh(y_pos, comm_stats['k49plus_fraction'], left=comm_stats['k1_fraction'], 
            color='orange', alpha=0.7, label='k=49 (all communities singletons)')
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(comm_stats['method'], fontsize=8)
axes[0].set_xlabel('Fraction of Results Excluded')
axes[0].set_title('Community Detection Methods\nExclusion Fractions', fontweight='bold')
axes[0].set_xlim(0, 1)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='x')

# add percentage labels
for i, (k1_frac, k49_frac, total_frac) in enumerate(zip(comm_stats['k1_fraction'], 
                                                        comm_stats['k49plus_fraction'], 
                                                        comm_stats['excluded_fraction'])):
    if total_frac > 0.01:  # only label if >1%
        axes[0].text(total_frac + 0.01, i, f'{total_frac:.1%}', va='center', fontsize=8)

# network methods  
y_pos = range(len(net_stats))
axes[1].barh(y_pos, net_stats['k1_fraction'], color='red', alpha=0.7, label='k=1 (single community)')
axes[1].barh(y_pos, net_stats['k49plus_fraction'], left=net_stats['k1_fraction'], 
            color='orange', alpha=0.7, label='k=49 (all communities singletons)')
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(net_stats['method'], fontsize=8)
axes[1].set_xlabel('Fraction of Results Excluded')
axes[1].set_title('Network Modeling Methods\nExclusion Fractions', fontweight='bold')
axes[1].set_xlim(0, 1)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='x')

# add percentage labels
for i, (k1_frac, k49_frac, total_frac) in enumerate(zip(net_stats['k1_fraction'], 
                                                        net_stats['k49plus_fraction'], 
                                                        net_stats['excluded_fraction'])):
    if total_frac > 0.01:  # only label if >1%
        axes[1].text(total_frac + 0.01, i, f'{total_frac:.1%}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('results/method_exclusion_fractions.png', dpi=300, bbox_inches='tight')
plt.show()

# print summary
print("\nMethods with highest exclusion rates:")
print("Community Detection:")
for _, row in comm_stats.head(3).iterrows():
    print(f"  {row['method']}: {row['excluded_fraction']:.1%} excluded ({row['total']} total)")

print("Network Modeling:")
for _, row in net_stats.head(3).iterrows():
    print(f"  {row['method']}: {row['excluded_fraction']:.1%} excluded ({row['total']} total)")

print("\nApplying exclusion filter...")

# %%
analyzer.exclude_results(min_communities=2, max_communities=48)

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
viz.plot_k_distribution_analysis()

# %%
# =============================================================================
# 4. K vs MODULARITY RELATIONSHIP ANALYSIS
# =============================================================================

print("\n=== K vs MODULARITY RELATIONSHIP ===")
viz.plot_k_modularity_relationship()

# %%

# =============================================================================
# 5. FREQUENCY AND ENTROPY MATRIX COMPUTATION
# =============================================================================

print("\n=== FREQUENCY AND ENTROPY MATRIX COMPUTATION ===")

# compute both frequency and entropy matrices WITH entropy normalization
print("Computing results WITH entropy normalization...")
frequency_matrix_normalized, entropy_matrix = analyzer.aggregate_all_results(use_entropy_normalization=True)

# compute frequency matrix WITHOUT entropy normalization
print("Computing results WITHOUT entropy normalization...")
frequency_matrix_raw, _ = analyzer.aggregate_all_results(use_entropy_normalization=False)

print(f"Matrices: {frequency_matrix_normalized.shape}")

# get entropy values for per-outlet analysis
off_diag_mask = ~np.eye(frequency_matrix_normalized.shape[0], dtype=bool)
entropy_values = entropy_matrix.values[off_diag_mask]

# use normalized version as default for backward compatibility
frequency_matrix = frequency_matrix_normalized

# normalize frequency matrix for visualization
freq_values = frequency_matrix.values[off_diag_mask]
max_freq = freq_values.max() if (freq_values > 0).any() else 1.0

if max_freq > 0:
    norm_freq = frequency_matrix / max_freq
else:
    norm_freq = frequency_matrix.copy()
np.fill_diagonal(norm_freq.values, 1.0)

# %%

# =============================================================================
# 5.5. ENTROPY NORMALIZATION COMPARISON
# =============================================================================

print("\n=== ENTROPY NORMALIZATION COMPARISON ===")

# normalize both matrices for comparison
freq_norm_max = frequency_matrix_normalized.values[off_diag_mask].max()
freq_raw_max = frequency_matrix_raw.values[off_diag_mask].max()

norm_normalized = frequency_matrix_normalized / freq_norm_max if freq_norm_max > 0 else frequency_matrix_normalized.copy()
norm_raw = frequency_matrix_raw / freq_raw_max if freq_raw_max > 0 else frequency_matrix_raw.copy()

np.fill_diagonal(norm_normalized.values, 1.0)
np.fill_diagonal(norm_raw.values, 1.0)

# hierarchical clustering for ordering
dist_normalized = norm_normalized.values.max() - norm_normalized.values
dist_raw = norm_raw.values.max() - norm_raw.values

linkage_normalized = linkage(squareform(dist_normalized, checks=False), method='ward')
linkage_raw = linkage(squareform(dist_raw, checks=False), method='ward')

# get ordering from linkage trees
from scipy.cluster.hierarchy import leaves_list
order_normalized = leaves_list(linkage_normalized)
order_raw = leaves_list(linkage_raw)

# calculate difference BEFORE sorting
diff_matrix = norm_normalized - norm_raw

# sort matrices by linkage tree ordering  
sorted_norm_normalized = norm_normalized.iloc[order_normalized, order_normalized]
sorted_norm_raw = norm_raw.iloc[order_raw, order_raw]
sorted_diff_matrix = diff_matrix.iloc[order_normalized, order_normalized]

# create comparison visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# heatmaps with linkage-based ordering
sns.heatmap(sorted_norm_normalized, mask=sorted_norm_normalized.values == 0, cmap='YlOrRd', 
           square=True, ax=axes[0], cbar_kws={'shrink': 0.8})
axes[0].set_title('WITH Entropy Normalization\n(sorted by linkage)', fontweight='bold')
axes[0].tick_params(axis='both', labelsize=6)

sns.heatmap(sorted_norm_raw, mask=sorted_norm_raw.values == 0, cmap='YlOrRd', 
           square=True, ax=axes[1], cbar_kws={'shrink': 0.8})
axes[1].set_title('WITHOUT Entropy Normalization\n(sorted by linkage)', fontweight='bold')
axes[1].tick_params(axis='both', labelsize=6)

# difference heatmap (calculated before sorting, then sorted by normalized linkage)
vmax_diff = max(abs(sorted_diff_matrix.values[off_diag_mask].min()), abs(sorted_diff_matrix.values[off_diag_mask].max()))
sns.heatmap(sorted_diff_matrix, cmap='RdBu_r', center=0, vmin=-vmax_diff, vmax=vmax_diff,
           square=True, ax=axes[2], cbar_kws={'shrink': 0.8})
axes[2].set_title('Difference (Norm - Raw)\n(sorted by normalized linkage)', fontweight='bold')
axes[2].tick_params(axis='both', labelsize=6)

plt.tight_layout()
plt.savefig('results/entropy_normalization_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# dendrograms comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

dendrogram(linkage_normalized, labels=norm_normalized.index, ax=axes[0],
          orientation='bottom', leaf_rotation=90, leaf_font_size=6)
axes[0].set_title('WITH Entropy Normalization', fontweight='bold')

dendrogram(linkage_raw, labels=norm_raw.index, ax=axes[1],
          orientation='bottom', leaf_rotation=90, leaf_font_size=6)
axes[1].set_title('WITHOUT Entropy Normalization', fontweight='bold')

plt.tight_layout()
plt.savefig('results/entropy_dendrograms_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# summary statistics
correlation = np.corrcoef(norm_normalized.values[off_diag_mask], norm_raw.values[off_diag_mask])[0,1]
print(f"Matrix correlation: {correlation:.3f}")
print(f"Mean absolute difference: {np.abs(diff_matrix.values[off_diag_mask]).mean():.4f}")

# calculate adjusted rand index between clustering results
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score

# extract cluster labels at same relative height for both linkages
n_outlets = len(norm_normalized)
# use criterion='maxclust' to ensure same number of clusters for fair comparison
n_clusters = 3  # fixed at 3 clusters as requested

labels_normalized = fcluster(linkage_normalized, n_clusters, criterion='maxclust')
labels_raw = fcluster(linkage_raw, n_clusters, criterion='maxclust')

ari_score = adjusted_rand_score(labels_normalized, labels_raw)
print(f"Adjusted Rand Index between clustering results: {ari_score:.3f}")
print(f"Number of clusters used for ARI calculation: {n_clusters}")

# %%

# =============================================================================
# 6. HIERARCHICAL CLUSTERING AND ORDERING
# =============================================================================

print("\n=== HIERARCHICAL CLUSTERING AND ORDERING ===")

from scipy.cluster.hierarchy import leaves_list, fcluster, dendrogram

# filter frequency matrix by threshold
threshold_percentile = 0
freq_threshold = np.percentile(freq_values[freq_values > 0], threshold_percentile) if (freq_values > 0).any() else 0
filtered_freq = norm_freq.copy()
filtered_freq[filtered_freq < freq_threshold] = 0
np.fill_diagonal(filtered_freq.values, 1.0)

# convert to distance matrix
distance_matrix = filtered_freq.values.max() - filtered_freq.values
condensed_distances = squareform(distance_matrix, checks=False)

# ward linkage for hierarchical clustering
ward_linkage = linkage(condensed_distances, method='ward', metric='euclidean')
cluster_order = leaves_list(ward_linkage)

# visualize hierarchical clustering dendrogram
plt.figure(figsize=(15, 8))
dendrogram(ward_linkage, 
           labels=filtered_freq.index, 
           orientation='top',
           leaf_rotation=45,
           leaf_font_size=8)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)\nBased on Co-clustering Frequency', 
          fontweight='bold', fontsize=14)
plt.xlabel('Media Outlets', fontweight='bold')
plt.ylabel('Distance', fontweight='bold')
plt.tight_layout()
plt.savefig('results/hierarchical_clustering.png', dpi=300, bbox_inches='tight')
plt.show()

# create ordered matrices
ordered_freq = filtered_freq.iloc[cluster_order, cluster_order]
ordered_entropy = entropy_matrix.iloc[cluster_order, cluster_order]

# dual ordered heatmaps
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sns.heatmap(ordered_freq, mask=ordered_freq.values == 0, cmap='YlOrRd', 
            square=True, ax=axes[0], cbar_kws={'label': 'Normalized Frequency'})
axes[0].set_title(f'Ordered Co-clustering Frequency\n(threshold={threshold_percentile}th percentile of frequency values)', fontweight='bold')
axes[0].tick_params(axis='both', labelsize=8)

sns.heatmap(ordered_entropy, cmap='viridis', square=True, ax=axes[1],
            cbar_kws={'label': 'Shannon Entropy'})
axes[1].set_title('Ordered Method Consistency\n(Same ordering)', fontweight='bold')
axes[1].tick_params(axis='both', labelsize=8)

plt.tight_layout()
plt.savefig('results/ordered_frequency_entropy.png', dpi=300, bbox_inches='tight')
plt.show()

# %%

# =============================================================================
# 7. OUTLET UNCERTAINTY ANALYSIS
# =============================================================================

print("\n=== OUTLET UNCERTAINTY ANALYSIS ===")

# calculate uncertainty metrics for each outlet
outlet_uncertainty = {}

for outlet in frequency_matrix.index:
    outlet_idx = frequency_matrix.index.get_loc(outlet)
    
    # get all entropy values for this outlet (excluding diagonal)
    outlet_entropies = []
    
    for other_idx in range(len(frequency_matrix)):
        if other_idx != outlet_idx:
            entropy_val = entropy_matrix.iloc[outlet_idx, other_idx]
            outlet_entropies.append(entropy_val)
    
    # calculate uncertainty metrics
    mean_entropy = np.mean(outlet_entropies)
    
    outlet_uncertainty[outlet] = {
        'mean_entropy': mean_entropy
    }

# create uncertainty dataframe for analysis
uncertainty_df = pd.DataFrame(outlet_uncertainty).T
uncertainty_df = uncertainty_df.sort_values('mean_entropy', ascending=False)

# visualize outlet uncertainty - ranking, distribution, and swarmplot
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# 1. all outlets mean entropy bar plot
n_outlets = len(uncertainty_df)
y_positions = range(n_outlets)

# create color gradient based on uncertainty values
colors = plt.cm.RdYlBu_r(np.linspace(0, 1, n_outlets))

bars = axes[0].barh(y_positions, uncertainty_df['mean_entropy'], color=colors)
axes[0].set_yticks(y_positions)
axes[0].set_yticklabels(uncertainty_df.index, fontsize=6)
axes[0].set_xlabel('Mean Entropy')
axes[0].set_title(f'All Outlets Uncertainty Ranking\n({n_outlets} outlets, sorted by mean entropy)', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# add colorbar for uncertainty levels
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(vmin=uncertainty_df['mean_entropy'].min(), 
                                                                    vmax=uncertainty_df['mean_entropy'].max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=axes[0], shrink=0.8)
cbar.set_label('Mean Entropy', rotation=270, labelpad=15)

# highlight quartiles with horizontal lines
quartiles = uncertainty_df['mean_entropy'].quantile([0.25, 0.5, 0.75])
for i, (quartile, value) in enumerate(quartiles.items()):
    axes[0].axvline(value, color='black', linestyle='--', alpha=0.7, linewidth=1)
    axes[0].text(value, n_outlets * (0.1 + i * 0.2), f'Q{int(quartile*4)}: {value:.3f}', 
                rotation=90, ha='right', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# 2. distribution of all pairwise entropy values
pairwise_entropies = entropy_matrix.values[off_diag_mask]
axes[1].hist(pairwise_entropies, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[1].axvline(pairwise_entropies.mean(), color='red', linestyle='--', 
               label=f'Mean: {pairwise_entropies.mean():.3f}')
axes[1].axvline(np.median(pairwise_entropies), color='orange', linestyle='--',
               label=f'Median: {np.median(pairwise_entropies):.3f}')
axes[1].set_xlabel('Pairwise Entropy')
axes[1].set_ylabel('Count')
axes[1].set_title(f'Distribution of Pairwise Entropy\n({len(pairwise_entropies)} outlet pairs)', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. swarmplot of pairwise entropies for each outlet
# prepare data for swarmplot
swarm_data = []
for outlet in entropy_matrix.index:
    outlet_idx = entropy_matrix.index.get_loc(outlet)
    for other_idx in range(len(entropy_matrix)):
        if other_idx != outlet_idx:
            entropy_val = entropy_matrix.iloc[outlet_idx, other_idx]
            swarm_data.append({
                'outlet': outlet,
                'pairwise_entropy': entropy_val
            })

swarm_df = pd.DataFrame(swarm_data)

# sort outlets by mean entropy for consistent ordering
outlet_order = uncertainty_df.index.tolist()

# create swarmplot
sns.swarmplot(data=swarm_df, x='pairwise_entropy', y='outlet', 
              order=outlet_order, ax=axes[2], size=3, alpha=0.7)
axes[2].set_xlabel('Pairwise Entropy')
axes[2].set_ylabel('Outlet')
axes[2].set_title(f'Pairwise Entropy Distribution by Outlet\n(Each point is one outlet pair)', fontweight='bold')
axes[2].tick_params(axis='y', labelsize=6)
axes[2].grid(True, alpha=0.3, axis='x')

# add vertical lines for overall statistics
axes[2].axvline(pairwise_entropies.mean(), color='red', linestyle='--', alpha=0.7, 
               label=f'Overall Mean: {pairwise_entropies.mean():.3f}')
axes[2].axvline(np.median(pairwise_entropies), color='orange', linestyle='--', alpha=0.7,
               label=f'Overall Median: {np.median(pairwise_entropies):.3f}')
axes[2].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('results/outlet_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
plt.show()



# %%





# %%

# =============================================================================
# 8. PER-METHOD NORMALIZED CO-CLUSTERING ANALYSIS
# =============================================================================

print("\n=== PER-METHOD NORMALIZED CO-CLUSTERING ANALYSIS ===")

# get aggregated results for each individual method
method_results = analyzer.aggregate_results_by_method_with_entropy()

if method_results:
    # extract just the frequency matrices (first element of each tuple) for comparison
    frequency_matrices = {method_name: matrices[0] for method_name, matrices in method_results.items()}
    
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
    
    for i, (method_name, matrices) in enumerate(method_results.items()):
        matrix = matrices[0]  # extract frequency matrix from tuple
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
        except:
            # fallback to original matrix if clustering fails
            ordered_matrix = matrix
            sorted_method_results[method_name] = matrix
            print(f"Warning: Could not perform hierarchical clustering for {method_name}, using original order")
            
        # create heatmap for this method
        sns.heatmap(ordered_matrix, mask=ordered_matrix.values == 0, cmap='YlOrRd', 
                   square=True, ax=ax, cbar=True,
                   cbar_kws={'shrink': 0.8})
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
        from scipy.cluster.hierarchy import fcluster
        if method_name in sorted_method_results:
            # use the linkage from the sorting process to identify stable clusters
            try:
                # recreate the linkage for cluster extraction
                sorted_matrix = sorted_method_results[method_name]
                off_diag_mask = ~np.eye(sorted_matrix.shape[0], dtype=bool)
                max_off_diag = sorted_matrix.values[off_diag_mask].max() if (sorted_matrix.values[off_diag_mask] > 0).any() else 1.0
                
                if max_off_diag > 0:
                    norm_matrix = sorted_matrix / max_off_diag
                    np.fill_diagonal(norm_matrix.values, 1.0)
                    dist_matrix = norm_matrix.values.max() - norm_matrix.values
                    condensed_dist = squareform(dist_matrix, checks=False)
                    ward_link = linkage(condensed_dist, method='ward', metric='euclidean')
                    
                    # extract clusters at 70% height
                    height_threshold = 0.7 * dist_matrix.max()
                    labels = fcluster(ward_link, height_threshold, criterion='distance')
                    
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
