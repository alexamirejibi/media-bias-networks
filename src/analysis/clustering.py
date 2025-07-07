"""
Clustering analysis methods - hierarchical clustering, community detection, and consensus building.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
from scipy.spatial.distance import squareform
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Any
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx


class ClusteringAnalyzer:
    """Specialized analyzer for hierarchical clustering and community detection."""
    
    def __init__(self, core_analyzer):
        self.core = core_analyzer
    
    def analyze_hierarchical_clustering(self, frequency_matrix: pd.DataFrame, 
                                      colors: dict, threshold_percentile: float = 0) -> Dict[str, Any]:
        """perform hierarchical clustering analysis on frequency matrix"""
        print("\\n=== HIERARCHICAL CLUSTERING AND ORDERING ===")
        
        # prepare frequency matrix
        off_diag_mask = ~np.eye(frequency_matrix.shape[0], dtype=bool)
        freq_values = frequency_matrix.values[off_diag_mask]
        max_freq = freq_values.max() if (freq_values > 0).any() else 1.0
        
        if max_freq > 0:
            norm_freq = frequency_matrix / max_freq
        else:
            norm_freq = frequency_matrix.copy()
        np.fill_diagonal(norm_freq.values, 1.0)
        
        # filter frequency matrix by threshold
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
        plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)\\nBased on Co-clustering Frequency', 
                  fontweight='bold', fontsize=14)
        plt.xlabel('Media Outlets', fontweight='bold')
        plt.ylabel('Distance', fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/hierarchical_clustering.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # create ordered matrix
        ordered_freq = filtered_freq.iloc[cluster_order, cluster_order]
        
        # ordered heatmap
        heatmap_cmap = LinearSegmentedColormap.from_list('custom_heatmap', 
                                                       ['#F7F7F7', '#F18F01', '#C73E1D'], N=256)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        sns.heatmap(ordered_freq, mask=ordered_freq.values == 0, cmap=heatmap_cmap, 
                    square=True, ax=ax, cbar_kws={'label': 'Surprisal-Weighted Frequency'})
        ax.set_title(f'Hierarchically Ordered Co-clustering Frequency\\n(threshold={threshold_percentile}th percentile of frequency values)', fontweight='bold')
        ax.tick_params(axis='both', labelsize=8)
        
        plt.tight_layout()
        plt.savefig('results/ordered_frequency_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'ward_linkage': ward_linkage,
            'cluster_order': cluster_order,
            'ordered_freq': ordered_freq,
            'filtered_freq': filtered_freq
        }

    def analyze_per_community_consistency(self, ward_linkage: np.ndarray, filtered_freq: pd.DataFrame, 
                                        colors: dict, n_clusters: int = 6) -> Dict[str, Any]:
        """analyze consistency within detected communities"""
        print("\\n=== PER-COMMUNITY CONSISTENCY ANALYSIS ===")
        
        # extract communities from the ward linkage
        community_labels = fcluster(ward_linkage, n_clusters, criterion='maxclust')
        
        # create community assignments
        communities = defaultdict(list)
        for i, label in enumerate(community_labels):
            outlet_name = filtered_freq.index[i]
            communities[label].append(outlet_name)
        
        # calculate within-community mean frequency for each community
        community_stats = {}
        
        for comm_id, outlets in communities.items():
            if len(outlets) > 1:  # need at least 2 outlets for pairwise frequency
                within_comm_frequencies = []
                
                # get all pairwise frequencies within this community
                for i, outlet1 in enumerate(outlets):
                    for j, outlet2 in enumerate(outlets):
                        if i != j:  # exclude self-pairs
                            freq_val = filtered_freq.loc[outlet1, outlet2]
                            within_comm_frequencies.append(freq_val)
                
                community_stats[comm_id] = {
                    'outlets': outlets,
                    'size': len(outlets),
                    'mean_frequency': np.mean(within_comm_frequencies),
                    'std_frequency': np.std(within_comm_frequencies)
                }
        
        # display results
        print("Community structure from hierarchical clustering:")
        for comm_id in sorted(community_stats.keys()):
            stats = community_stats[comm_id]
            print(f"\\nCommunity {comm_id} ({stats['size']} outlets):")
            print(f"  Outlets: {', '.join(stats['outlets'])}")
            print(f"  Within-community mean frequency: {stats['mean_frequency']:.3f} ± {stats['std_frequency']:.3f}")
        
        # compare with overall pairwise frequency
        off_diag_mask = ~np.eye(filtered_freq.shape[0], dtype=bool)
        overall_mean_frequency = filtered_freq.values[off_diag_mask].mean()
        print(f"\\nOverall pairwise frequency (all outlet pairs): {overall_mean_frequency:.3f}")
        
        # summary analysis
        if community_stats:
            most_coherent = max(community_stats.keys(), key=lambda x: community_stats[x]['mean_frequency'])
            least_coherent = min(community_stats.keys(), key=lambda x: community_stats[x]['mean_frequency'])
            
            print(f"\\nCommunity coherence summary:")
            print(f"  Most coherent: Community {most_coherent} (frequency: {community_stats[most_coherent]['mean_frequency']:.3f})")
            print(f"  Least coherent: Community {least_coherent} (frequency: {community_stats[least_coherent]['mean_frequency']:.3f})")
            print(f"  Higher frequency indicates more consistent clustering within community")
        
        # visualize community structure and frequency analysis
        self._visualize_community_analysis(community_stats, filtered_freq, ward_linkage, 
                                         colors, n_clusters, overall_mean_frequency)
        
        return {
            'community_labels': community_labels,
            'communities': communities,
            'community_stats': community_stats,
            'overall_mean_frequency': overall_mean_frequency
        }

    def _visualize_community_analysis(self, community_stats, filtered_freq, ward_linkage, 
                                    colors, n_clusters, overall_mean_frequency):
        """Create visualizations for community analysis."""
        categorical_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3A6B35', '#F4B942', '#8E44AD', '#E67E22']
        
        # 1. community frequency comparison bar chart (separate figure)
        if community_stats:
            comm_ids = sorted(community_stats.keys())
            frequencies = [community_stats[cid]['mean_frequency'] for cid in comm_ids]
            sizes = [community_stats[cid]['size'] for cid in comm_ids]
            errors = [community_stats[cid]['std_frequency'] for cid in comm_ids]

            colors_list = [categorical_colors[i % len(categorical_colors)] for i in range(len(comm_ids))]

            plt.figure(figsize=(8, 6))
            bars = plt.bar(range(len(comm_ids)), frequencies, yerr=errors, capsize=5,
                            color=colors_list, alpha=0.8, edgecolor=colors['text'])

            # add size labels on bars
            for i, (bar, size) in enumerate(zip(bars, sizes)):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + errors[i] + 0.001,
                         f'n={size}', ha='center', va='bottom', fontsize=10, fontweight='bold')

            plt.axhline(overall_mean_frequency, color=colors['mean'], linestyle='--', alpha=0.8,
                        label=f'Overall mean: {overall_mean_frequency:.3f}')
            plt.xlabel('Community')
            plt.ylabel('Within-Community Mean Frequency')
            plt.title('Community Coherence Analysis\\n(Higher frequency = more coherent)', fontweight='bold')
            plt.xticks(range(len(comm_ids)), [f'Community {cid}' for cid in comm_ids])
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('results/community_consistency_bar.png', dpi=300, bbox_inches='tight')
            plt.show()

        # 2. annotated heatmap showing community boundaries (separate figure)
        self._visualize_community_heatmap(filtered_freq, ward_linkage, community_stats, 
                                        categorical_colors, n_clusters)

    def _visualize_community_heatmap(self, filtered_freq, ward_linkage, community_stats, 
                                   categorical_colors, n_clusters):
        """Create annotated heatmap with community boundaries."""
        ordered_freq_with_communities = filtered_freq.iloc[leaves_list(ward_linkage), leaves_list(ward_linkage)]
        heatmap_cmap = LinearSegmentedColormap.from_list('custom_heatmap',
                                                         ['#F7F7F7', '#F18F01', '#C73E1D'], N=256)

        plt.figure(figsize=(8, 8))
        ax_heat = sns.heatmap(ordered_freq_with_communities, mask=ordered_freq_with_communities.values == 0,
                              cmap=heatmap_cmap, square=True,
                              cbar_kws={'label': 'Surprisal-Weighted Frequency'})
        plt.title(f'Frequency Matrix with Community Structure\\n({n_clusters} communities from hierarchical clustering)',
                  fontweight='bold')
        plt.tick_params(axis='both', labelsize=6)

        # add community boundary lines and labels
        if community_stats:
            comm_ids = sorted(community_stats.keys())
            community_colors = {cid: categorical_colors[i % len(categorical_colors)] for i, cid in enumerate(comm_ids)}
            
            # Create mapping from outlets to communities
            outlet_to_community = {}
            communities = {}
            for comm_id, stats in community_stats.items():
                communities[comm_id] = stats['outlets']
                for outlet in stats['outlets']:
                    outlet_to_community[outlet] = comm_id

            ordered_communities = [outlet_to_community.get(outlet, -1) for outlet in ordered_freq_with_communities.index]
            community_boundaries = []
            current_comm = ordered_communities[0]
            for i, comm in enumerate(ordered_communities[1:], 1):
                if comm != current_comm:
                    community_boundaries.append(i)
                    current_comm = comm

            for boundary in community_boundaries:
                ax_heat.axhline(boundary, color='white', linewidth=2)
                ax_heat.axvline(boundary, color='white', linewidth=2)

            # add community labels
            current_comm = ordered_communities[0]
            start_idx = 0
            for i, comm in enumerate(ordered_communities + [None]):
                if comm != current_comm or comm is None:
                    mid_point = (start_idx + i) / 2
                    if current_comm in community_colors:
                        plt.text(-2, mid_point, f'C{current_comm}', ha='center', va='center', fontsize=10,
                                 fontweight='bold', color=community_colors[current_comm],
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    if comm is not None:
                        start_idx = i
                        current_comm = comm

        plt.tight_layout()
        plt.savefig('results/community_consistency_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

    def aggregate_results_by_method_with_surprisal(self) -> Dict[str, pd.DataFrame]:
        """aggregate co-clustering results separately for each method combination"""
        print("\\n=== AGGREGATING RESULTS BY METHOD WITH SURPRISAL WEIGHTING ===")
        
        if self.core.results_df.empty:
            print("no results to aggregate")
            return {}
        
        # group by method combination
        method_groups = self.core.results_df.groupby(['network_method', 'community_method', 'param_id'])
        method_results = {}
        
        for (net_method, comm_method, param_id), group in method_groups:
            method_name = f"{net_method}_{comm_method}_{param_id}"
            print(f"aggregating {len(group)} results for {method_name}")
            
            # aggregate results for this method combination
            aggregated_matrix = self.core._aggregate_clustering_results_with_surprisal(
                group, method_name=method_name, use_surprisal_weighting=True
            )
            
            if aggregated_matrix is not None:
                method_results[method_name] = aggregated_matrix
        
        print(f"successfully aggregated results for {len(method_results)} method combinations")
        return method_results

    def compare_method_coclustering(self, frequency_matrices: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """compare co-clustering patterns between different methods"""
        print("\\n=== COMPARING METHOD CO-CLUSTERING PATTERNS ===")
        
        if len(frequency_matrices) < 2:
            print("need at least 2 methods for comparison")
            return {}
        
        method_names = list(frequency_matrices.keys())
        n_methods = len(method_names)
        
        # calculate pairwise correlations between method frequency matrices
        correlation_matrix = np.ones((n_methods, n_methods))
        
        for i in range(n_methods):
            for j in range(i+1, n_methods):
                matrix1 = frequency_matrices[method_names[i]]
                matrix2 = frequency_matrices[method_names[j]]
                
                # use off-diagonal elements for correlation
                off_diag_mask = ~np.eye(matrix1.shape[0], dtype=bool)
                values1 = matrix1.values[off_diag_mask]
                values2 = matrix2.values[off_diag_mask]
                
                # calculate correlation
                correlation = np.corrcoef(values1, values2)[0, 1]
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        correlation_df = pd.DataFrame(correlation_matrix, 
                                    index=method_names, 
                                    columns=method_names)
        
        print("method co-clustering correlation matrix:")
        print(correlation_df.round(3))
        
        # identify most similar and most different method pairs
        off_diag_correlations = correlation_matrix[~np.eye(n_methods, dtype=bool)]
        most_similar_idx = np.unravel_index(np.argmax(off_diag_correlations), correlation_matrix.shape)
        most_different_idx = np.unravel_index(np.argmin(off_diag_correlations), correlation_matrix.shape)
        
        # adjust indices for off-diagonal
        flat_idx_max = np.argmax(off_diag_correlations)
        flat_idx_min = np.argmin(off_diag_correlations)
        
        # convert back to matrix indices
        off_diag_indices = np.triu_indices(n_methods, k=1)
        most_similar_i, most_similar_j = off_diag_indices[0][flat_idx_max], off_diag_indices[1][flat_idx_max]
        most_different_i, most_different_j = off_diag_indices[0][flat_idx_min], off_diag_indices[1][flat_idx_min]
        
        print(f"\\nmost similar methods: {method_names[most_similar_i]} ↔ {method_names[most_similar_j]} "
              f"(r = {correlation_matrix[most_similar_i, most_similar_j]:.3f})")
        print(f"most different methods: {method_names[most_different_i]} ↔ {method_names[most_different_j]} "
              f"(r = {correlation_matrix[most_different_i, most_different_j]:.3f})")
        
        return {
            'correlation_matrix': correlation_df,
            'mean_correlation': off_diag_correlations.mean(),
            'std_correlation': off_diag_correlations.std(),
            'most_similar_pair': (method_names[most_similar_i], method_names[most_similar_j]),
            'most_different_pair': (method_names[most_different_i], method_names[most_different_j])
        }
