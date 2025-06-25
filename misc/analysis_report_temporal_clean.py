# %%

# =============================================================================
# MEDIA BIAS NETWORK ANALYSIS - TEMPORAL
# Research Questions:
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
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings('ignore')

from src.experiment import ExperimentFramework
from src.viz import Visualizer, COLORS, HEATMAP_CMAP, DIVERGING_CMAP, CATEGORICAL_COLORS
# Note: The analysis module has been refactored into specialized components while maintaining backward compatibility
# All existing method calls will continue to work unchanged

# %%

# =============================================================================
# 1. EXPERIMENT SETUP & DATA OVERVIEW
# =============================================================================

print("=== MEDIA BIAS NETWORK ANALYSIS ===")

# experiment parameters (temporal sampling)
data_dir = 'data/daily_cluster_matrices_min_6'
# consecutive window configuration
window_size = 15  # days per window
step_size = 15    # non-overlapping windows; set < window_size for sliding windows

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
    
    # NEW: Access to specialized analyzers (optional - existing code continues to work)
    # analyzer.stability_analyzer    # For stability and robustness analysis
    # analyzer.statistics_analyzer   # For significance testing and null distributions 
    # analyzer.clustering_analyzer   # For hierarchical clustering and consensus building
    # analyzer.temporal_analyzer     # For time-series and drift analysis
    
    print(f"Loaded experiment results:")
    print(f"Total results: {len(analyzer.get_results())}")
    print(f"Network methods: {len(analyzer.get_results()['network_method'].unique())}")
    print(f"Community methods: {len(analyzer.get_results()['community_method'].unique())}")
    
    # show brief summary (already computed when results were generated originally)
    print(f"\\nTEMPORAL EXPERIMENT SUMMARY (loaded):")
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
    
    # NEW: Access to specialized analyzers (optional - existing code continues to work)
    # analyzer.stability_analyzer    # For stability and robustness analysis
    # analyzer.statistics_analyzer   # For significance testing and null distributions 
    # analyzer.clustering_analyzer   # For hierarchical clustering and consensus building
    # analyzer.temporal_analyzer     # For time-series and drift analysis

    print(f"\\nTEMPORAL EXPERIMENT SUMMARY:")
    print(f"Windows processed: {temporal_summary['n_windows']} (size={window_size}, step={step_size})")
    print(f"Total results: {len(analyzer.get_results())}")
    print(f"Network methods: {len(analyzer.get_results()['network_method'].unique())}")
    print(f"Community methods: {len(analyzer.get_results()['community_method'].unique())}")
    print(f"Average time per window: {temporal_summary['total_time']/temporal_summary['n_windows']:.1f}s")

    # save results
    print(f"\\nSaving results to {results_file}...")
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
# 1.1. METHOD STABILITY RANKING (variance in k across samples)
# =============================================================================

print("\\n=== METHOD STABILITY RANKING ===")

# visualize top-N most stable method combinations (low variance in k)
viz.plot_stability_ranking(top_n=15)

# fetch stability dataframe for further inspection
stability_df_overall = analyzer.analyze_stability()
if not stability_df_overall.empty:
    # pick the single most stable combination
    best = stability_df_overall.iloc[0]
    best_net = best['network_method']
    best_comm = best['community_method']
    best_param = best['param_id']

    consistency = analyzer.method_consistency(
        network_method=best_net,
        community_method=best_comm,
        param_id=best_param,
        metric='ari'
    )

    print("\\nMost stable method combination across samples:")
    print(f"  network method : {best_net}")
    print(f"  community method: {best_comm}")
    print(f"  param id        : {best_param}")
    if 'error' not in consistency:
        print(f"  mean consistency (ARI): {consistency['mean_consistency']}")
        print(f"  std  consistency      : {consistency['std_consistency']}")
        print(f"  comparisons           : {consistency['n_comparisons']}")
    else:
        # fallback message if consistency couldn't be computed (e.g., only one sample)
        print(f"  consistency analysis unavailable: {consistency['error']}")
else:
    print("Stability dataframe is empty – skipping consistency check")

# %%

# =============================================================================
# 1.2. TEMPORAL ROBUSTNESS ANALYSIS (stability across windows)
# =============================================================================

print("\\n=== TEMPORAL ROBUSTNESS ANALYSIS ===")

window_ids = [ws['window_id'] for ws in temporal_summary['window_summaries']]

# compute stability matrix (ARI between window clusterings)
stability_df = analyzer.temporal_stability(window_ids, metric='ari')

if not stability_df.empty:
    plt.figure(figsize=(8, 6))
    sns.heatmap(stability_df, annot=True, fmt='.2f', cmap=DIVERGING_CMAP, square=True,
                cbar_kws={'label': 'Adjusted Rand Index'})
    plt.title(f'Temporal Stability between {window_size}-Day Windows\\n(metric: ARI)', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/temporal_stability_ari.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\\nTemporal stability (ARI) summary:")
    off_diag = stability_df.values[~np.eye(len(stability_df), dtype=bool)]
    print(f"  Mean ARI: {off_diag.mean():.3f}")
    print(f"  Std  ARI: {off_diag.std():.3f}")
    print(f"  Min  ARI: {off_diag.min():.3f}")
    print(f"  Max  ARI: {off_diag.max():.3f}")
else:
    print("Stability matrix could not be computed (insufficient data)")

# =============================================================================
# 1.3. ADJACENT WINDOW DRIFT ANALYSIS
# =============================================================================
if not stability_df.empty and len(stability_df) > 1:
    consecutive_ari = [stability_df.iloc[i, i + 1]
                       for i in range(len(stability_df) - 1)]
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(consecutive_ari) + 1),
             consecutive_ari, marker='o',
             color=COLORS['primary'], linewidth=2)
    plt.ylim(0, 1)
    plt.xlabel('window index (t → t+1)')
    plt.ylabel('adjusted rand index')
    plt.title(f'consecutive-window drift '
              f'(window={window_size}d, step={step_size}d)',
              fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/temporal_drift_ari.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    print("\\nConsecutive-window drift:")
    print(f"  mean ARI: {np.mean(consecutive_ari):.3f}")
    print(f"  std  ARI: {np.std(consecutive_ari):.3f}")

    # --- trend analysis across windows ---
    trend_res = analyzer.analyze_window_stability_trends(window_ids)
    if isinstance(trend_res, dict) and 'stability_stats' in trend_res:
        tstats = trend_res['stability_stats']
        print(f"  Temporal trend: slope = {tstats['trend_slope']:.3f} ({tstats['trend_direction']})")
        print(f"  Stability range: {tstats['min_stability']:.3f} – {tstats['max_stability']:.3f}")
else:
    print("Consecutive-window drift not computed (need ≥2 windows)")

# =============================================================================
# 1.4. LAG SIMILARITY PROFILE
# =============================================================================
if not stability_df.empty and len(stability_df) > 2:
    max_lag = min(6, len(stability_df) - 1)  # keep plot compact
    lag_means = []
    for lag in range(1, max_lag + 1):
        vals = [stability_df.iloc[i, i + lag]
                for i in range(len(stability_df) - lag)]
        lag_means.append(np.mean(vals))

    plt.figure(figsize=(6, 4))
    plt.bar(range(1, max_lag + 1), lag_means,
            color=COLORS['secondary'], alpha=0.8)
    plt.ylim(0, 1)
    plt.xlabel('lag (windows apart)')
    plt.ylabel('mean ARI')
    plt.title('time-lag similarity profile', fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/temporal_lag_profile.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    print("\\nLag similarity profile:")
    for lag, val in enumerate(lag_means, 1):
        print(f"  lag {lag}: mean ARI = {val:.3f}")
else:
    print("Lag similarity profile not computed (need ≥3 windows)")

# %%

# =============================================================================
# 1.5. EXCLUSION ANALYSIS - METHODS FILTERED BY COMMUNITY COUNT
# =============================================================================

# run exclusion analysis and apply filter
analyzer.analyze_exclusions(COLORS)

# %%

# =============================================================================
# 2. MODULARITY DISTRIBUTIONS ANALYSIS
# =============================================================================

print(f"\\n=== MODULARITY ANALYSIS ===")
viz.plot_modularity_analysis()

# %%

# =============================================================================
# 3. K DISTRIBUTION ANALYSIS
# =============================================================================

print(f"\\n=== K DISTRIBUTION ANALYSIS ===")
# viz.plot_k_distribution_analysis()

# %%
# =============================================================================
# 4. K vs MODULARITY RELATIONSHIP ANALYSIS
# =============================================================================

print("\\n=== K vs MODULARITY RELATIONSHIP ===")
viz.plot_k_modularity_relationship()

# %%

# =============================================================================
# 5. FREQUENCY MATRIX WITH SURPRISAL WEIGHTING
# =============================================================================

print("\\n=== FREQUENCY MATRIX WITH SURPRISAL WEIGHTING ===")

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

# use surprisal-weighted frequencies for significance testing
frequency_matrix = frequency_matrix_weighted

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
# 7. CROSS-SAMPLE STATISTICAL SIGNIFICANCE TESTING
# =============================================================================

# %%

print("\\n=== SIGNIFICANCE ACROSS SAMPLES ANALYSIS ===")

# Apply the analyze_significance_across_samples method
try:
    across_samples_results = analyzer.analyze_significance_across_samples(
        alpha=0.05,
        min_sample_frac=0.5,
        test="wilcoxon"
    )
    
    # Extract results and set as primary significance results
    high_pairs = across_samples_results['high_pairs'] 
    low_pairs = across_samples_results['low_pairs']
    p_adj_matrix_samples = across_samples_results['p_adj_matrix']
    n_samples = across_samples_results['n_samples']
    masked_pairs_samples = across_samples_results['masked_pairs']
    significant_mask = (p_adj_matrix_samples < 0.05).values
    
    print(f"\\nACROSS-SAMPLES SIGNIFICANCE RESULTS:")
    print(f"Analyzed {n_samples} independent samples")
    print(f"High co-clustering pairs: {len(high_pairs)}")
    print(f"Low co-clustering pairs: {len(low_pairs)}")
    print(f"Masked pairs (insufficient coverage): {len(masked_pairs_samples)}")
    
    # Show significance mask visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(significant_mask.astype(int), cmap='RdBu_r', center=0.5, square=True,
                cbar_kws={'label': 'Significant (1) vs Non-significant (0)'},
                xticklabels=analyzer.outlet_names, yticklabels=analyzer.outlet_names)
    plt.title('Cross-Sample Statistical Significance', fontweight='bold')
    plt.tick_params(axis='both', labelsize=6)
    plt.tight_layout()
    plt.savefig('results/cross_sample_significance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # -------------------------------------------------------------
    # effect size (observed – expected) matrix visualisation
    # -------------------------------------------------------------
    effect_matrix = across_samples_results.get('effect_matrix')
    if effect_matrix is not None:
        plt.figure(figsize=(8, 6))
        sns.heatmap(effect_matrix, cmap=DIVERGING_CMAP, center=0, square=True,
                    xticklabels=analyzer.outlet_names, yticklabels=analyzer.outlet_names,
                    cbar_kws={'label': 'observed – expected'})
        plt.title('Effect Size Matrix\n(Across-Sample Residual)', fontweight='bold')
        plt.tick_params(axis='both', labelsize=6)
        plt.tight_layout()
        plt.savefig('results/effect_size_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

except Exception as e:
    print(f"Cross-sample analysis failed: {e}")
    print("Cannot proceed without cross-sample significance results.")
    # Set empty results to prevent errors in downstream code
    high_pairs = []
    low_pairs = []
    significant_mask = np.zeros((len(analyzer.outlet_names), len(analyzer.outlet_names)), dtype=bool)


# %%

# =============================================================================
# 7.2. VALIDATED CLUSTERING FROM CROSS-SAMPLE SIGNIFICANCE
# =============================================================================

print("\n=== VALIDATED CLUSTERING (cross-sample significant edges) ===")

validated_across = analyzer.construct_validated_clustering(
    high_pairs=high_pairs,
    low_pairs=low_pairs,
    null_mean=0.0,  # baseline similarity for non-validated pairs
    null_std=0.0,
    n_clusters=None,
)

if validated_across and 'labels' in validated_across:
    Z_valid = validated_across['linkage']
    communities_valid = validated_across['communities']
    dist_valid = validated_across['distance_matrix']
    sim_valid = validated_across.get('similarity_matrix')

    print(f"\nConstructed {len(communities_valid)} validated communities (cross-sample):")
    for cid, members in sorted(communities_valid.items()):
        print(f"  Community {cid} ({len(members)} outlets): {', '.join(members)}")

    # visualise dendrogram + ordered distance heatmap
    try:
        order = leaves_list(Z_valid)
        ordered_dist = pd.DataFrame(dist_valid, index=analyzer.outlet_names,
                                    columns=analyzer.outlet_names).iloc[order, order]

        cmap_signed = LinearSegmentedColormap.from_list('signed_cmap',
                                                        ['#2E86AB', '#F7F7F7', '#C73E1D'], N=256)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(ordered_dist, cmap=cmap_signed, square=True, ax=axes[0],
                    cbar_kws={'label': 'signed distance'})
        axes[0].set_title('Signed Distance Matrix (validated edges)')
        axes[0].tick_params(axis='both', labelsize=6)

        dendrogram(Z_valid, labels=[analyzer.outlet_names[i] for i in order],
                    orientation='right', leaf_font_size=6, ax=axes[1])
        axes[1].set_title('Hierarchical Clustering\n(validated edges)')

        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/final_validated_consensus_signed.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Visualisation failed: {e}")

    # ---------------------------------------------------------
    # 7.2.aA  Graph visualisation – strongest validated edges
    # ---------------------------------------------------------
    try:
        if sim_valid is not None:
            # determine threshold for strongest edges (top 5% of all off-diagonal similarities)
            triu_idx = np.triu_indices_from(sim_valid, k=1)
            sim_vals = sim_valid[triu_idx]
            # consider only positive similarities for thresholding to avoid diluting with low/negative values
            sim_vals_pos = sim_vals[sim_vals > 0]

            if sim_vals_pos.size > 0:
                threshold = np.percentile(sim_vals_pos, 75)
            else:
                threshold = None

            # if threshold yields too few edges, iteratively relax criterion
            n_out = len(analyzer.outlet_names)
            min_edges_target = max(10, n_out // 2)  # aim for at least this many edges
            if threshold is not None:
                for pct in [85, 80, 75, 70]:
                    # quick check without building full graph yet
                    if np.sum(sim_vals_pos >= threshold) >= min_edges_target:
                        break
                    threshold = np.percentile(sim_vals_pos, pct)

            # build graph with only strong edges
            G_strong = nx.Graph()
            label_map = {outlet: cid for cid, members in communities_valid.items() for outlet in members}

            # add nodes with community metadata
            for outlet in analyzer.outlet_names:
                G_strong.add_node(outlet, community=label_map.get(outlet, -1))

            # add edges above threshold
            if threshold is not None:
                for i in range(n_out):
                    for j in range(i + 1, n_out):
                        w = sim_valid[i, j]
                        if w >= threshold:
                            G_strong.add_edge(analyzer.outlet_names[i], analyzer.outlet_names[j], weight=w)

            # skip visualisation if graph has no edges
            if G_strong.number_of_edges() > 0:
                fig, ax_graph = plt.subplots(figsize=(10, 8))

                # layout – spring layout seeded for reproducibility
                pos = nx.spring_layout(G_strong, seed=42, k=0.3)

                # node colors by community
                palette = sns.color_palette("tab10", n_colors=max(communities_valid.keys()))
                node_colors = [palette[label_map.get(node, 0)-1] if label_map.get(node, 0) > 0 else (0.7, 0.7, 0.7) for node in G_strong.nodes()]

                # edge colors scaled by weight
                edge_weights = [G_strong[u][v]['weight'] for u, v in G_strong.edges()]
                if edge_weights:
                    norm = plt.Normalize(min(edge_weights), max(edge_weights))
                    edge_colors = [plt.cm.Blues(norm(w)) for w in edge_weights]
                else:
                    edge_colors = 'grey'

                nx.draw_networkx_nodes(G_strong, pos, node_color=node_colors, node_size=350, edgecolors='black', linewidths=0.5, ax=ax_graph)
                nx.draw_networkx_edges(G_strong, pos, edge_color=edge_colors, width=2, ax=ax_graph)
                nx.draw_networkx_labels(G_strong, pos, font_size=8, ax=ax_graph)

                # colour bar for edge weights, if edges exist
                if edge_weights:
                    sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
                    sm.set_array([])
                    fig.colorbar(sm, ax=ax_graph, label='co-clustering similarity')

                ax_graph.set_axis_off()
                ax_graph.set_title('Validated Communities – Strongest 25% Edges', fontweight='bold')
                fig.tight_layout()
                fig.savefig('results/final_validated_consensus_graph.png', dpi=300, bbox_inches='tight')
                plt.show()
            else:
                print("  No edges exceed the strength threshold – graph not displayed")
    except Exception as e:
        print(f"  Graph visualisation failed: {e}")

    # ---------------------------------------------------------
    # 7.2.aB  Consensus statistics (within / between IPA)
    # ---------------------------------------------------------
    if sim_valid is not None:
        n_out = len(analyzer.outlet_names)
        label_map = {}
        for cid, members in communities_valid.items():
            for m in members:
                label_map[m] = cid

        within_vals = []
        between_vals = []
        for i in range(n_out):
            for j in range(i + 1, n_out):
                val = sim_valid[i, j] if isinstance(sim_valid, np.ndarray) else sim_valid.iloc[i, j]
                if label_map[analyzer.outlet_names[i]] == label_map[analyzer.outlet_names[j]]:
                    within_vals.append(val)
                else:
                    between_vals.append(val)

        if within_vals and between_vals:
            w_median = np.median(within_vals)
            w_q1, w_q3 = np.percentile(within_vals, [25, 75])
            b_median = np.median(between_vals)
            print("\nConsensus similarity statistics:")
            print(f"  Median within-community IPA: {w_median:.3f} (IQR {w_q1:.3f}–{w_q3:.3f})")
            print(f"  Median between-community IPA: {b_median:.3f}")

    # ---------------------------------------------------------
    # 7.2.b  Signed modularity of validated graph
    # ---------------------------------------------------------
    try:
        sim_pos = np.clip(sim_valid, 0, None) if sim_valid is not None else None
        if sim_pos is not None:
            G_mod = nx.from_numpy_array(sim_pos)
            partition = [set() for _ in communities_valid]
            for cid, members in communities_valid.items():
                for outlet in members:
                    idx = analyzer.outlet_names.index(outlet)
                    partition[cid-1].add(idx)
            partition = [p for p in partition if p]
            if len(partition) > 1:
                Q_signed = nx.community.modularity(G_mod, partition, weight='weight')
                print(f"  Signed modularity Q: {Q_signed:.3f}")
    except Exception as e:
        print(f"  Modularity calculation failed: {e}")

    # ---------------------------------------------------------
    # 7.2.c  Elbow criterion – suggest k*
    # ---------------------------------------------------------
    try:
        distances = Z_valid[:, 2]
        diffs = np.diff(distances, prepend=distances[0])
        elbow_idx = np.argmax(diffs)
        k_star = len(analyzer.outlet_names) - elbow_idx
        print(f"  Elbow criterion suggests k* = {k_star}")
    except Exception as e:
        print(f"  Elbow analysis failed: {e}")

else:
    print("No validated clustering could be constructed (no significant edges)")

# %%
# =============================================================================
# 8. PER-METHOD SURPRISAL-WEIGHTED CO-CLUSTERING ANALYSIS
# =============================================================================

print("\\n=== PER-METHOD SURPRISAL-WEIGHTED CO-CLUSTERING ANALYSIS ===")

# get aggregated results for each individual method
method_results = analyzer.aggregate_results_by_method_with_surprisal()

if method_results:
    print(f"Analyzing {len(method_results)} individual methods...")

    # -------------------------------------------------------------
    # 8.a Correlation of each method with final consensus matrix
    # -------------------------------------------------------------
    consensus_flat = frequency_matrix_weighted.values[off_diag_mask]
    method_corrs = {}
    for m_name, m_mat in method_results.items():
        vals = m_mat.values[off_diag_mask]
        if vals.std() == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(consensus_flat, vals)[0, 1]
        method_corrs[m_name] = corr

    # rank methods
    ranked_methods = sorted(method_corrs.items(), key=lambda x: x[1], reverse=True)
    print("\nCorrelation of individual methods with final consensus (top 10):")
    for m, r in ranked_methods[:10]:
        print(f"  {m}: r = {r:.3f}")

    # identify lowest-contributing third
    n_methods_total = len(ranked_methods)
    n_remove = max(1, n_methods_total // 3)
    low_methods = [m for m, _ in ranked_methods[-n_remove:]]
    print(f"\nRemoving lowest-contributing {n_remove} methods: {', '.join(low_methods)}")

    # aggregate remaining methods by simple mean
    remaining_mats = [mat.values for name, mat in method_results.items() if name not in low_methods]
    remaining_avg = np.mean(remaining_mats, axis=0)
    remaining_avg_df = pd.DataFrame(remaining_avg, index=analyzer.outlet_names, columns=analyzer.outlet_names)

    # hierarchical clustering of remaining
    rem_dist = remaining_avg_df.values.max() - remaining_avg_df.values
    rem_link = linkage(squareform(rem_dist, checks=False), method='ward')
    if validated_across and 'labels' in validated_across:
        original_labels = validated_across['labels']
        # choose same number of clusters
        n_clusters_consensus = validated_across['n_clusters']
        rem_labels = fcluster(rem_link, n_clusters_consensus, criterion='maxclust')
        try:
            from sklearn.metrics import adjusted_rand_score
            delta_ari = adjusted_rand_score(original_labels, rem_labels)
            print(f"\nPartition similarity after removal (ARI): {delta_ari:.3f}")
        except Exception as e:
            print(f"Could not compute ARI after method removal: {e}")
    
    # save correlation matrix figure
    method_comp = analyzer.compare_method_coclustering(method_results)
else:
    print("No method results to analyze")

# %%

# =============================================================================
# 9. FULL GRID SIZE & RETENTION RATE
# =============================================================================

# calculate size of full parameter grid evaluated and retention fraction after exclusions
all_results = analyzer.get_results()
if not all_results.empty:
    n_windows_total = temporal_summary['n_windows'] if 'n_windows' in temporal_summary else all_results['sample_id'].nunique()
    n_network_methods = all_results['network_method'].nunique()
    n_community_methods = all_results['community_method'].nunique()
    n_param_ids = all_results['param_id'].nunique()
    full_grid_size = n_windows_total * n_network_methods * n_community_methods * n_param_ids
    retained_results = len(all_results)
    retention_pct = 100.0 * retained_results / full_grid_size if full_grid_size else 0.0
    print(f"\n=== PARAMETER GRID SUMMARY ===")
    print(f"Windows (S): {n_windows_total}")
    print(f"Network methods (M): {n_network_methods}")
    print(f"Community methods (C): {n_community_methods}")
    print(f"Hyper-parameter configs (H): {n_param_ids}")
    print(f"Full grid size: {full_grid_size}")
    print(f"Retained results after filtering: {retained_results} ({retention_pct:.1f}%)")


