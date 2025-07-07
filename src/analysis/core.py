"""
Core ResultsAnalyzer functionality - basic data management and essential methods.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# import partition metrics for comparing clusterings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# from ..partition_metrics import all_metrics as partition_all_metrics


class CoreResultsAnalyzer:
    """Core analyzer for basic data management and essential operations."""
    
    def __init__(self):
        self.results_df = pd.DataFrame()
        self.adjacencies = {}  # cache adjacency matrices
        self.outlet_names = None
    
    def add_result(self, sample_id: str, network_method: str, community_method: str, 
                   param_id: str, communities: Dict[int, int], parameters: Dict,
                   adjacency_matrix: Optional[pd.DataFrame] = None):
        """add a single result to the dataframe"""
        
        # cache adjacency matrix and extract outlet names
        if adjacency_matrix is not None:
            self.adjacencies[(sample_id, network_method)] = adjacency_matrix
            if self.outlet_names is None and isinstance(adjacency_matrix, pd.DataFrame):
                self.outlet_names = adjacency_matrix.index.tolist()
        
        # compute basic community statistics
        comm_sizes = Counter(communities.values())
        n_communities = len(comm_sizes)
        largest_community = max(comm_sizes.values()) if comm_sizes else 0
        
        # calculate modularity
        modularity = self._calculate_modularity(adjacency_matrix, communities)
        
        # create result record
        result = {
            'sample_id': sample_id,
            'network_method': network_method,
            'community_method': community_method,
            'param_id': param_id,
            'parameters': str(parameters),
            'n_communities': n_communities,
            'largest_community': largest_community,
            'modularity': modularity,
            'communities': communities
        }
        
        # add to dataframe  
        self.results_df = pd.concat([self.results_df, pd.DataFrame([result])], 
                                  ignore_index=True)
        
        print(f"added result: {sample_id} | {network_method} | {community_method} | {n_communities} communities | modularity: {modularity:.3f}")
    
    def _calculate_modularity(self, adjacency_matrix: Optional[pd.DataFrame], 
                             communities: Dict[int, int]) -> float:
        """calculate modularity score for a community partition"""
        if adjacency_matrix is None or not communities:
            return np.nan
            
        try:
            # convert adjacency matrix to numpy array
            adj_array = np.array(adjacency_matrix, dtype=float)
            # replace nan/inf values with 0
            adj_array = np.nan_to_num(adj_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # create networkx graph
            G = nx.from_numpy_array(adj_array)
            
            # create partition list for networkx modularity calculation
            # communities dict maps node_id -> community_id
            # we need to group nodes by community
            partition = [set() for _ in range(max(communities.values()) + 1)]
            for node_id, comm_id in communities.items():
                if node_id < len(G.nodes()):  # ensure node exists in graph
                    partition[comm_id].add(node_id)
            
            # remove empty communities
            partition = [comm for comm in partition if len(comm) > 0]
            
            # calculate modularity using networkx
            if len(partition) > 1:  # need at least 2 communities for modularity
                modularity = nx.community.modularity(G, partition)
            else:
                modularity = 0.0
                
            return modularity
            
        except Exception as e:
            print(f"Error calculating modularity: {e}")
            return np.nan
    
    def add_sample_results(self, sample_id: str, network_method: str, 
                          community_results: Dict, adjacency_matrix: pd.DataFrame):
        """add all community detection results for a sample/network combination"""
        
        for comm_method, method_results in community_results.items():
            for param_id, result_data in method_results.items():
                self.add_result(
                    sample_id=sample_id,
                    network_method=network_method,
                    community_method=comm_method,
                    param_id=param_id,
                    communities=result_data['communities'],
                    parameters=result_data['parameters'],
                    adjacency_matrix=adjacency_matrix
                )
    
    def get_results(self, filters: Optional[Dict] = None) -> pd.DataFrame:
        """get filtered results dataframe"""
        df = self.results_df.copy()
        
        if filters:
            for column, value in filters.items():
                if column in df.columns:
                    if isinstance(value, list):
                        df = df[df[column].isin(value)]
                    else:
                        df = df[df[column] == value]
        
        return df
    
    def exclude_results(self, min_communities: int = 2, max_communities: int = 6):
        """exclude results with less than 2 and more than 6 communities"""
        self.results_df = self.results_df[
            (self.results_df['n_communities'] >= min_communities) & 
            (self.results_df['n_communities'] <= max_communities)
        ]
    
    def summary(self) -> Dict[str, Any]:
        """get summary statistics of all results"""
        if self.results_df.empty:
            return {'message': 'no results yet'}
        
        return {
            'total_results': len(self.results_df),
            'samples': self.results_df['sample_id'].nunique(),
            'network_methods': self.results_df['network_method'].nunique(),
            'community_methods': self.results_df['community_method'].nunique(),
            'avg_communities': self.results_df['n_communities'].mean(),
            'datasets': self.results_df.get('dataset', pd.Series()).nunique()
        }
    
    def export_results(self, filepath: str):
        """export results to csv"""
        if self.results_df.empty:
            print("no results to export")
            return
            
        export_df = self.results_df.drop('communities', axis=1, errors='ignore')
        export_df.to_csv(filepath, index=False)
        print(f"exported {len(export_df)} results to {filepath}")
    
    def _get_partition(self, sample_id: str, network_method: str, 
                      community_method: str, param_id: str) -> Optional[Dict[int, int]]:
        """get community assignments for specific method combination"""
        filtered = self.get_results({
            'sample_id': sample_id,
            'network_method': network_method,
            'community_method': community_method,
            'param_id': param_id
        })
        
        if filtered.empty:
            return None
        
        return filtered.iloc[0]['communities']
    
    def _compare_partitions(self, partition1: Dict[int, int], partition2: Dict[int, int], 
                           metric: str = 'ari') -> Optional[float]:
        """compare two partitions using specified metric"""
        try:
            # convert to lists for partition_metrics
            max_nodes = max(max(partition1.keys()), max(partition2.keys())) + 1
            part1_list = [partition1.get(i, -1) for i in range(max_nodes)]
            part2_list = [partition2.get(i, -1) for i in range(max_nodes)]
            
            metrics = partition_all_metrics(part1_list, part2_list, verbose=0)
            return metrics.get(metric)
        except Exception as e:
            return None

    def _aggregate_clustering_results_with_surprisal(self, results_subset: pd.DataFrame, method_name: str = "results", use_surprisal_weighting: bool = True) -> Optional[pd.DataFrame]:
        """core aggregation logic with surprisal-weighted frequency calculation"""
        if results_subset.empty:
            print(f"no {method_name} to aggregate")
            return None
        if self.outlet_names is None:
            print("no outlet names available")
            return None
        
        n_outlets = len(self.outlet_names)
        print(f"processing {len(results_subset)} clusterings for {n_outlets} outlets ({method_name})")
        
        # initialize matrices
        weighted_sum = np.zeros((n_outlets, n_outlets))
        total_clusterings = 0
        total_weight_sum = 0.0
        
        # diagnostic tracking
        surprisal_weights = []
        cluster_size_counts = {}
        missing_outlet_warnings = 0
        
        # process each clustering result
        for _, row in results_subset.iterrows():
            communities = row['communities']
            if not communities:
                continue
            
            total_clusterings += 1
            
            # calculate cluster sizes
            cluster_sizes = {}
            for outlet_id, cluster_id in communities.items():
                if cluster_id not in cluster_sizes:
                    cluster_sizes[cluster_id] = 0
                cluster_sizes[cluster_id] += 1
            
            # track cluster size distribution for diagnostics
            for size in cluster_sizes.values():
                if size not in cluster_size_counts:
                    cluster_size_counts[size] = 0
                cluster_size_counts[size] += 1
            
            # calculate pair-level surprisal weights
            for i in range(n_outlets):
                for j in range(i + 1, n_outlets):
                    outlet_i_cluster = communities.get(i)
                    outlet_j_cluster = communities.get(j)
                    
                    if outlet_i_cluster is None or outlet_j_cluster is None:
                        missing_outlet_warnings += 1
                        continue
                    
                    if outlet_i_cluster == outlet_j_cluster:
                        # calculate surprisal weight based on cluster size
                        cluster_size = cluster_sizes[outlet_i_cluster]
                        
                        if use_surprisal_weighting:
                            # surprisal = -log2(P(pair in cluster of this size))
                            surprisal_weight = -np.log2(cluster_size / n_outlets)
                        else:
                            surprisal_weight = 1.0
                        
                        surprisal_weights.append(surprisal_weight)
                        total_weight_sum += surprisal_weight
                        
                        # update weighted sum
                        weighted_sum[i, j] += surprisal_weight
                        weighted_sum[j, i] += surprisal_weight
        
        if total_clusterings == 0:
            print(f"no valid clusterings found for {method_name}")
            return None
        
        # calculate normalized frequencies using proper weighted average
        if total_weight_sum > 0:
            normalized_frequencies = weighted_sum / total_clusterings
        else:
            print(f"warning: zero total weight sum for {method_name}")
            return None
        
        # diagnostic reporting
        print(f"\\n=== SURPRISAL WEIGHTING DIAGNOSTICS ({method_name}) ===")
        print(f"Surprisal weighting: {'enabled' if use_surprisal_weighting else 'disabled'}")
        print(f"Total clusterings processed: {total_clusterings}")
        print(f"Total weight sum: {total_weight_sum:.2f}")
        
        if surprisal_weights:
            print(f"Surprisal weight range: {min(surprisal_weights):.2f} to {max(surprisal_weights):.2f}")
            print(f"Mean surprisal weight: {np.mean(surprisal_weights):.2f}")
            print(f"Median surprisal weight: {np.median(surprisal_weights):.2f}")
        
        if cluster_size_counts:
            print(f"Cluster size distribution:")
            for size in sorted(cluster_size_counts.keys()):
                print(f"  Size {size}: {cluster_size_counts[size]} clusters")
        
        if missing_outlet_warnings > 0:
            print(f"Missing outlet warnings: {missing_outlet_warnings}")
        
        # create result dataframe
        frequency_df = pd.DataFrame(
            normalized_frequencies,
            index=self.outlet_names,
            columns=self.outlet_names
        )
        
        # final diagnostics
        print(f"\\n=== FINAL RESULTS ({method_name}) ===")
        non_zero_freqs = normalized_frequencies[normalized_frequencies > 0]
        if len(non_zero_freqs) > 0:
            print(f"Mean weighted frequency: {non_zero_freqs.mean():.3f}")
            print(f"Max weighted frequency: {normalized_frequencies.max():.3f}")
        print(f"Non-zero frequency entries: {np.sum(normalized_frequencies > 0)}")
        
        return frequency_df
    
    def aggregate_all_results(self, use_surprisal_weighting: bool = True) -> Optional[pd.DataFrame]:
        """aggregate with surprisal weighting"""
        print("aggregating all results with surprisal weighting...")
        if self.results_df.empty:
            print("no results to aggregate")
            return None
        
        return self._aggregate_clustering_results_with_surprisal(self.results_df, "all_methods", use_surprisal_weighting)


# Facade pattern: Combined analyzer that includes all functionality
class ResultsAnalyzer(CoreResultsAnalyzer):
    """Main ResultsAnalyzer class that maintains backward compatibility."""
    
    def __init__(self):
        super().__init__()
        # Lazy-loaded specialized analyzers
        self._stability_analyzer = None
        self._statistics_analyzer = None
        self._clustering_analyzer = None
        self._temporal_analyzer = None
    
    @property
    def stability_analyzer(self):
        """Lazy-loaded stability analyzer"""
        # Handle old analyzer objects loaded from pickle that don't have these attributes
        if not hasattr(self, '_stability_analyzer'):
            self._stability_analyzer = None
        if self._stability_analyzer is None:
            from .stability import StabilityAnalyzer
            self._stability_analyzer = StabilityAnalyzer(self)
        return self._stability_analyzer
    
    @property 
    def statistics_analyzer(self):
        """Lazy-loaded statistics analyzer"""
        # Handle old analyzer objects loaded from pickle that don't have these attributes
        if not hasattr(self, '_statistics_analyzer'):
            self._statistics_analyzer = None
        if self._statistics_analyzer is None:
            from .statistics import StatisticsAnalyzer
            self._statistics_analyzer = StatisticsAnalyzer(self)
        return self._statistics_analyzer
    
    @property
    def clustering_analyzer(self):
        """Lazy-loaded clustering analyzer"""
        # Handle old analyzer objects loaded from pickle that don't have these attributes
        if not hasattr(self, '_clustering_analyzer'):
            self._clustering_analyzer = None
        if self._clustering_analyzer is None:
            from .clustering import ClusteringAnalyzer
            self._clustering_analyzer = ClusteringAnalyzer(self)
        return self._clustering_analyzer
    
    @property
    def temporal_analyzer(self):
        """Lazy-loaded temporal analyzer"""
        # Handle old analyzer objects loaded from pickle that don't have these attributes
        if not hasattr(self, '_temporal_analyzer'):
            self._temporal_analyzer = None
        if self._temporal_analyzer is None:
            from .temporal import TemporalAnalyzer
            self._temporal_analyzer = TemporalAnalyzer(self)
        return self._temporal_analyzer
    
    # ===== BACKWARD COMPATIBILITY DELEGATION METHODS =====
    
    # Stability methods - delegate to stability_analyzer
    def analyze_stability(self, dataset: str = None) -> pd.DataFrame:
        """analyze method stability across samples"""
        return self.stability_analyzer.analyze_stability(dataset)
    
    def method_consistency(self, network_method: str, community_method: str, 
                          param_id: str, metric: str = 'ari') -> Dict[str, float]:
        """measure consistency of a method across different samples"""
        return self.stability_analyzer.method_consistency(network_method, community_method, param_id, metric)
            
    def analyze_exclusions(self, colors: dict, min_communities: int = 2, max_communities: int = 48):
        """analyze and visualize method exclusions before applying filter"""
        return self.stability_analyzer.analyze_exclusions(colors, min_communities, max_communities)
            
    # Statistical methods - delegate to statistics_analyzer    
    def analyze_significance_across_samples(self, alpha: float = 0.05, 
                                          min_sample_frac: float = 0.5,
                                          test: str = "auto") -> Dict[str, Any]:
        """Test significance across independent samples using binomial test."""
        return self.statistics_analyzer.analyze_significance_across_samples(alpha, min_sample_frac, test)
    
    def construct_validated_clustering(self, high_pairs: List[Dict], low_pairs: List[Dict], 
                                     null_mean: float, null_std: float, 
                                     n_clusters: Optional[int] = None) -> Optional[Dict]:
        """Construct clustering using only statistically validated relationships."""
        return self.statistics_analyzer.construct_validated_clustering(
            high_pairs, low_pairs, null_mean, null_std, n_clusters)
    
    # Clustering methods - delegate to clustering_analyzer
    def analyze_hierarchical_clustering(self, frequency_matrix: pd.DataFrame, 
                                      colors: dict, threshold_percentile: float = 0) -> Dict[str, Any]:
        """perform hierarchical clustering analysis on frequency matrix"""
        return self.clustering_analyzer.analyze_hierarchical_clustering(frequency_matrix, colors, threshold_percentile)
    
    def analyze_per_community_consistency(self, ward_linkage: np.ndarray, filtered_freq: pd.DataFrame, 
                                        colors: dict, n_clusters: int = 6) -> Dict[str, Any]:
        """analyze consistency within detected communities"""
        return self.clustering_analyzer.analyze_per_community_consistency(ward_linkage, filtered_freq, colors, n_clusters)
    
    def aggregate_results_by_method_with_surprisal(self) -> Dict[str, pd.DataFrame]:
        """aggregate co-clustering results separately for each method combination"""
        return self.clustering_analyzer.aggregate_results_by_method_with_surprisal()
    
    def compare_method_coclustering(self, frequency_matrices: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """compare co-clustering patterns between different methods"""
        return self.clustering_analyzer.compare_method_coclustering(frequency_matrices)
    
    
    # Temporal methods - delegate to temporal_analyzer
    def temporal_stability(self, window_ids: List[str], metric: str = 'ari', n_clusters: int = 6) -> pd.DataFrame:
        """compute pairwise similarity between windows created by run_temporal_experiment"""
        return self.temporal_analyzer.temporal_stability(window_ids, metric, n_clusters)
    
    def analyze_temporal_drift(self, window_ids: List[str], metric: str = 'ari') -> Dict[str, Any]:
        """analyze temporal drift patterns between consecutive windows"""
        return self.temporal_analyzer.analyze_temporal_drift(window_ids, metric)
        
    def analyze_window_stability_trends(self, window_ids: List[str]) -> Dict[str, Any]:
        """analyze trends in stability across temporal windows"""
        return self.temporal_analyzer.analyze_window_stability_trends(window_ids)
    
    # Additional methods that use multiple analyzers
    def analyze_surprisal_weighting_comparison(self, colors: dict) -> Dict[str, Any]:
        """compare raw vs surprisal-weighted frequency matrices"""
        print("\\n=== SURPRISAL WEIGHTING COMPARISON ===")
        
        # compute both versions
        frequency_matrix_raw = self.aggregate_all_results(use_surprisal_weighting=False)
        frequency_matrix_weighted = self.aggregate_all_results(use_surprisal_weighting=True)
        
        if frequency_matrix_raw is None or frequency_matrix_weighted is None:
            return {'error': 'could not compute frequency matrices'}
        
        # sort matrices by ward linkage for better visualization
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        
        # use weighted matrix for clustering order (more informative)
        # normalize to [0,1] range first to avoid negative distances
        freq_values = frequency_matrix_weighted.values.copy()
        freq_max = np.max(freq_values)
        if freq_max > 0:
            freq_normalized = freq_values / freq_max
        else:
            freq_normalized = freq_values
        
        # create distance matrix (1 - similarity)
        distance_matrix = 1 - freq_normalized
        np.fill_diagonal(distance_matrix, 0)  # ensure diagonal is 0
        
        # ensure all distances are non-negative
        distance_matrix = np.clip(distance_matrix, 0, None)
        
        try:
            condensed_distances = squareform(distance_matrix)
            ward_linkage = linkage(condensed_distances, method='ward')
            optimal_order = leaves_list(ward_linkage)
        except Exception as e:
            print(f"Warning: Could not perform Ward clustering ({e}). Using original order.")
            optimal_order = list(range(len(frequency_matrix_weighted)))
        
        # reorder both matrices
        ordered_outlets = [frequency_matrix_weighted.index[i] for i in optimal_order]
        frequency_matrix_raw = frequency_matrix_raw.loc[ordered_outlets, ordered_outlets]
        frequency_matrix_weighted = frequency_matrix_weighted.loc[ordered_outlets, ordered_outlets]
        
        # visualize comparison (simplified version)
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # 1. raw frequencies
        heatmap_cmap = LinearSegmentedColormap.from_list('custom_heatmap', 
                                                       ['#F7F7F7', '#F18F01', '#C73E1D'], N=256)
        sns.heatmap(frequency_matrix_raw, mask=frequency_matrix_raw.values == 0,
                   cmap=heatmap_cmap, square=True, ax=axes[0],
                   cbar_kws={'label': 'Raw Frequency'})
        axes[0].set_title('Raw Co-clustering Frequency\n(no surprisal weighting)', fontweight='bold')
        axes[0].tick_params(axis='both', labelsize=6)
        
        # 2. surprisal-weighted frequencies  
        sns.heatmap(frequency_matrix_weighted, mask=frequency_matrix_weighted.values == 0,
                   cmap=heatmap_cmap, square=True, ax=axes[1],
                   cbar_kws={'label': 'Surprisal-Weighted Frequency'})
        axes[1].set_title('Surprisal-Weighted Co-clustering Frequency\n(information-theoretic weighting)', fontweight='bold')
        axes[1].tick_params(axis='both', labelsize=6)
        
        # 3. difference matrix
        diff_matrix = frequency_matrix_weighted - frequency_matrix_raw
        diverging_cmap = LinearSegmentedColormap.from_list('custom_diverging', 
                                                         ['#C73E1D', '#F18F01', '#F7F7F7', '#6BAED6', '#2E86AB'], N=256)
        sns.heatmap(diff_matrix, cmap=diverging_cmap, center=0, square=True, ax=axes[2],
                   cbar_kws={'label': 'Difference (Weighted - Raw)'})
        axes[2].set_title('Surprisal Weighting Effect\n(Positive = Enhanced, Negative = Diminished)', fontweight='bold')
        axes[2].tick_params(axis='both', labelsize=6)
        
        plt.tight_layout()
        plt.savefig('results/surprisal_weighting_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # statistics
        off_diag_mask = ~np.eye(len(frequency_matrix_raw), dtype=bool)
        raw_vals = frequency_matrix_raw.values[off_diag_mask]
        weighted_vals = frequency_matrix_weighted.values[off_diag_mask]
        diff_vals = diff_matrix.values[off_diag_mask]
        
        print(f"\nSURPRISAL WEIGHTING COMPARISON STATISTICS:")
        print(f"Raw frequency matrix:")
        print(f"  Mean: {raw_vals.mean():.4f}")
        print(f"  Std:  {raw_vals.std():.4f}")
        print(f"  Max:  {raw_vals.max():.4f}")
        print(f"  Non-zero entries: {np.sum(raw_vals > 0)}")
        
        print(f"\nSurprisal-weighted frequency matrix:")
        print(f"  Mean: {weighted_vals.mean():.4f}")
        print(f"  Std:  {weighted_vals.std():.4f}")  
        print(f"  Max:  {weighted_vals.max():.4f}")
        print(f"  Non-zero entries: {np.sum(weighted_vals > 0)}")
        
        print(f"\nDifference (weighted - raw):")
        print(f"  Mean: {diff_vals.mean():.4f}")
        print(f"  Std:  {diff_vals.std():.4f}")
        print(f"  Range: [{diff_vals.min():.4f}, {diff_vals.max():.4f}]")
        
        return {
            'frequency_matrix_raw': frequency_matrix_raw,
            'frequency_matrix_weighted': frequency_matrix_weighted,
            'difference_matrix': diff_matrix,
            'raw_stats': {
                'mean': raw_vals.mean(),
                'std': raw_vals.std(),
                'max': raw_vals.max(),
                'non_zero_count': np.sum(raw_vals > 0)
            },
            'weighted_stats': {
                'mean': weighted_vals.mean(),
                'std': weighted_vals.std(),
                'max': weighted_vals.max(),
                'non_zero_count': np.sum(weighted_vals > 0)
            }
        }