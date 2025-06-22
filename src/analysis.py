import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score
warnings.filterwarnings('ignore')

# import partition metrics for comparing clusterings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .partition_metrics import all_metrics as partition_all_metrics


class ResultsAnalyzer:
    """streamlined analyzer focused on core research questions"""
    
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
    
    # ===== METHOD SIMILARITY =====
    
    def calculate_method_similarity(self, sample_id: str, network_method: str, 
                                   metric: str = 'ari') -> pd.DataFrame:
        """calculate pairwise similarity between all community detection methods"""
        
        # get all methods for this sample/network combination
        filtered = self.get_results({
            'sample_id': sample_id,
            'network_method': network_method
        })
        
        if filtered.empty:
            return pd.DataFrame()
        
        # create method identifiers
        methods = []
        for _, row in filtered.iterrows():
            method_id = f"{row['community_method']}_{row['param_id']}"
            methods.append((method_id, row['community_method'], row['param_id']))
        
        # initialize similarity matrix
        n_methods = len(methods)
        similarity_matrix = np.zeros((n_methods, n_methods))
        method_names = [m[0] for m in methods]
        
        # compute pairwise similarities
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                part1 = self._get_partition(sample_id, network_method, methods[i][1], methods[i][2])
                part2 = self._get_partition(sample_id, network_method, methods[j][1], methods[j][2])
                
                if part1 and part2:
                    similarity = self._compare_partitions(part1, part2, metric)
                    if similarity is not None:
                        similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
        
        # diagonal is 1 (perfect similarity with self)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return pd.DataFrame(similarity_matrix, index=method_names, columns=method_names)
    
    # ===== STABILITY ANALYSIS =====
    
    def analyze_stability(self, dataset: str = None) -> pd.DataFrame:
        """analyze method stability across samples"""
        
        # filter by dataset if specified
        df = self.results_df if dataset is None else self.results_df[self.results_df['dataset'] == dataset]
        
        if df.empty:
            return pd.DataFrame()
        
        # group by method combination and compute stability metrics
        stability = df.groupby(['network_method', 'community_method', 'param_id'])[
            'n_communities'
        ].agg(['mean', 'std', 'count']).reset_index()
        
        # stability score: lower variance = more stable
        stability['stability_score'] = 1 / (1 + stability['std'])
        stability = stability.sort_values('stability_score', ascending=False)
        
        return stability.round(3)
    
    def method_consistency(self, network_method: str, community_method: str, 
                          param_id: str, metric: str = 'ari') -> Dict[str, float]:
        """measure consistency of a method across different samples"""
        
        # get all samples with this method combination
        filtered = self.get_results({
            'network_method': network_method,
            'community_method': community_method,
            'param_id': param_id
        })
        
        if len(filtered) < 2:
            return {'error': 'need at least 2 samples for consistency analysis'}
        
        samples = filtered['sample_id'].unique()
        similarities = []
        
        # compute pairwise similarities between samples
        for i, sample1 in enumerate(samples):
            for sample2 in samples[i + 1:]:
                part1 = self._get_partition(sample1, network_method, community_method, param_id)
                part2 = self._get_partition(sample2, network_method, community_method, param_id)
                
                if part1 and part2:
                    similarity = self._compare_partitions(part1, part2, metric)
                    if similarity is not None:
                        similarities.append(similarity)
        
        if not similarities:
            return {'error': 'no valid comparisons found'}
        
        return {
            'mean_consistency': round(np.mean(similarities), 3),
            'std_consistency': round(np.std(similarities), 3),
            'n_comparisons': len(similarities)
        }
    
    # ===== OUTLET GROUPINGS =====
    
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
        print(f"\n=== SURPRISAL WEIGHTING DIAGNOSTICS ({method_name}) ===")
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
        print(f"\n=== FINAL RESULTS ({method_name}) ===")
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

    def aggregate_results_by_method_with_surprisal(self, use_surprisal_weighting: bool = True) -> Dict[str, pd.DataFrame]:
        """aggregate results by method with surprisal weighting"""
        print("aggregating results by individual community detection method (with surprisal weighting)...")
        
        if self.results_df.empty:
            print("no results to aggregate")
            return {}
        
        # get unique community detection methods
        community_methods = self.results_df.groupby('community_method').size()
        print(f"found {len(community_methods)} community detection methods:")
        for comm_method, count in community_methods.items():
            print(f"  {comm_method}: {count} results")
        
        method_results = {}
        
        # process each community detection method
        for community_method, group_df in self.results_df.groupby('community_method'):
            print(f"\n{'='*50}")
            print(f"processing method: {community_method}")
            print(f"{'='*50}")
            
            # aggregate results for this method
            result_matrix = self._aggregate_clustering_results_with_surprisal(group_df, community_method, use_surprisal_weighting)
            
            if result_matrix is not None:
                method_results[community_method] = result_matrix
        
        print(f"\n{'='*50}")
        print(f"SUMMARY: successfully processed {len(method_results)}/{len(community_methods)} methods")
        print(f"{'='*50}")
        
        return method_results
    
    def compare_method_coclustering(self, method_results: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """compare co-clustering patterns across different methods"""
        if method_results is None:
            method_results = self.aggregate_results_by_method_with_surprisal()
        
        if not method_results:
            print("no method results to compare")
            return pd.DataFrame()
        
        print("\ncomparing co-clustering patterns across methods...")
        
        # calculate summary statistics for each method
        comparison_data = []
        
        for method_name, matrix in method_results.items():
            # exclude diagonal for statistics
            off_diag_mask = ~np.eye(matrix.shape[0], dtype=bool)
            off_diag_values = matrix.values[off_diag_mask]
            non_zero_values = off_diag_values[off_diag_values > 0]
            
            comparison_data.append({
                'community_method': method_name,
                'mean_coclustering': off_diag_values.mean(),
                'max_coclustering': off_diag_values.max(),
                'non_zero_fraction': len(non_zero_values) / len(off_diag_values) if len(off_diag_values) > 0 else 0,
                'mean_nonzero_coclustering': non_zero_values.mean() if len(non_zero_values) > 0 else 0,
                'std_coclustering': off_diag_values.std()
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.round(4)
        comparison_df = comparison_df.sort_values('mean_coclustering', ascending=False)
        
        print("method comparison summary:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df

    def outlet_clustering_frequency(self, min_frequency: float = 0.0, 
                                   dataset: str = None) -> pd.DataFrame:
        """compute how often each pair of outlets clusters together"""
        
        # filter by dataset if specified
        df = self.results_df if dataset is None else self.results_df[self.results_df['dataset'] == dataset]
        
        if df.empty or self.outlet_names is None:
            return pd.DataFrame()
        
        n_outlets = len(self.outlet_names)
        cooccurrence_counts = np.zeros((n_outlets, n_outlets))
        total_analyses = 0
        
        # count co-occurrences across all analyses
        for _, row in df.iterrows():
            communities = row['communities']
            total_analyses += 1
            
            # check each pair of outlets
            for i in range(n_outlets):
                for j in range(i, n_outlets):
                    if i in communities and j in communities:
                        if communities[i] == communities[j]:
                            cooccurrence_counts[i, j] += 1
                            cooccurrence_counts[j, i] += 1
        
        # convert to frequencies
        if total_analyses > 0:
            frequencies = cooccurrence_counts / total_analyses
        else:
            frequencies = cooccurrence_counts
        
        # apply threshold
        frequencies = np.where(frequencies >= min_frequency, frequencies, 0)
        
        return pd.DataFrame(frequencies, index=self.outlet_names, columns=self.outlet_names)
    
    def find_stable_outlet_groups(self, frequency_threshold: float = 0.7, 
                                 min_group_size: int = 2) -> Dict[str, List[str]]:
        """identify stable outlet communities that appear frequently together"""
        
        cooccurrence = self.outlet_clustering_frequency(min_frequency=frequency_threshold)
        
        if cooccurrence.empty:
            return {}
        
        # find connected components in thresholded matrix
        import networkx as nx
        
        G = nx.from_pandas_adjacency(cooccurrence)
        stable_groups = {}
        
        for i, component in enumerate(nx.connected_components(G)):
            if len(component) >= min_group_size:
                group_name = f"stable_group_{i+1}"
                stable_groups[group_name] = sorted(list(component))
        
        return stable_groups
    
    def compare_method_performance(self) -> pd.DataFrame:
        """compare average performance across methods"""
        if self.results_df.empty:
            return pd.DataFrame()
        
        performance = self.results_df.groupby(['network_method', 'community_method']).agg({
            'n_communities': ['mean', 'std'],
            'largest_community': 'mean'
        }).round(2)
        
        performance.columns = ['avg_communities', 'std_communities', 'avg_largest']
        return performance.sort_values('avg_communities', ascending=False)
    
    # ===== HELPER METHODS =====
    
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

    def analyze_exclusions(self, colors: dict, min_communities: int = 2, max_communities: int = 48):
        """analyze and visualize method exclusions before applying filter"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print("\n=== EXCLUSION ANALYSIS ===")

        # get results before exclusion
        all_results = self.get_results()
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
        # ----------------------------
        # community methods figure
        # ----------------------------
        plt.figure(figsize=(8, 6))
        y_pos = range(len(comm_stats))
        plt.barh(y_pos, comm_stats['k1_fraction'], color=colors['exclusion_k1'], alpha=0.8, label='k=1 (single community)')
        plt.barh(y_pos, comm_stats['k49plus_fraction'], left=comm_stats['k1_fraction'], 
                 color=colors['exclusion_k49'], alpha=0.8, label='k=49 (all communities singletons)')
        plt.yticks(y_pos, comm_stats['method'], fontsize=8)
        plt.xlabel('Fraction of Results Excluded')
        plt.title('Community Detection Methods\nExclusion Fractions', fontweight='bold')
        plt.xlim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='x')

        # add percentage labels
        for i, total_frac in enumerate(comm_stats['excluded_fraction']):
            if total_frac > 0.01:  # only label if >1%
                plt.text(total_frac + 0.01, i, f'{total_frac:.1%}', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig('results/method_exclusion_fractions_community.png', dpi=300, bbox_inches='tight')
        plt.show()

        # ----------------------------
        # network methods figure
        # ----------------------------
        plt.figure(figsize=(8, 6))
        y_pos = range(len(net_stats))
        plt.barh(y_pos, net_stats['k1_fraction'], color=colors['exclusion_k1'], alpha=0.8, label='k=1 (single community)')
        plt.barh(y_pos, net_stats['k49plus_fraction'], left=net_stats['k1_fraction'], 
                 color=colors['exclusion_k49'], alpha=0.8, label='k=49 (all communities singletons)')
        plt.yticks(y_pos, net_stats['method'], fontsize=8)
        plt.xlabel('Fraction of Results Excluded')
        plt.title('Network Modeling Methods\nExclusion Fractions', fontweight='bold')
        plt.xlim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='x')

        # add percentage labels
        for i, total_frac in enumerate(net_stats['excluded_fraction']):
            if total_frac > 0.01:
                plt.text(total_frac + 0.01, i, f'{total_frac:.1%}', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig('results/method_exclusion_fractions_network.png', dpi=300, bbox_inches='tight')
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
        
        # apply the exclusion filter
        self.exclude_results(min_communities=min_communities, max_communities=max_communities)

    def analyze_surprisal_weighting_comparison(self, colors: dict) -> Dict[str, Any]:
        """analyze and visualize comparison between surprisal-weighted and raw frequency matrices"""
        print("\n=== SURPRISAL WEIGHTING COMPARISON ===")
        
        # compute frequency matrices
        frequency_matrix_raw = self.aggregate_all_results(use_surprisal_weighting=False)
        frequency_matrix_weighted = self.aggregate_all_results(use_surprisal_weighting=True)
        
        if frequency_matrix_raw is None or frequency_matrix_weighted is None:
            print("failed to compute frequency matrices")
            return {}
        
        off_diag_mask = ~np.eye(frequency_matrix_raw.shape[0], dtype=bool)
        
        # normalize both matrices for comparison
        freq_weighted_max = frequency_matrix_weighted.values[off_diag_mask].max()
        freq_raw_max = frequency_matrix_raw.values[off_diag_mask].max()
        
        norm_weighted = frequency_matrix_weighted / freq_weighted_max if freq_weighted_max > 0 else frequency_matrix_weighted.copy()
        norm_raw = frequency_matrix_raw / freq_raw_max if freq_raw_max > 0 else frequency_matrix_raw.copy()
        
        np.fill_diagonal(norm_weighted.values, 1.0)
        np.fill_diagonal(norm_raw.values, 1.0)
        
        # hierarchical clustering for ordering
        dist_weighted = norm_weighted.values.max() - norm_weighted.values
        dist_raw = norm_raw.values.max() - norm_raw.values
        
        linkage_weighted = linkage(squareform(dist_weighted, checks=False), method='ward')
        linkage_raw = linkage(squareform(dist_raw, checks=False), method='ward')
        
        # get ordering from linkage trees
        order_weighted = leaves_list(linkage_weighted)
        order_raw = leaves_list(linkage_raw)
        
        # calculate difference before sorting
        diff_matrix = norm_weighted - norm_raw
        
        # sort matrices by linkage tree ordering
        sorted_norm_weighted = norm_weighted.iloc[order_weighted, order_weighted]
        sorted_norm_raw = norm_raw.iloc[order_raw, order_raw]
        sorted_diff_matrix = diff_matrix.iloc[order_weighted, order_weighted]
        
        # create heatmap visualizations (separate figures)
        heatmap_cmap = LinearSegmentedColormap.from_list('custom_heatmap',
                                                         ['#F7F7F7', '#F18F01', '#C73E1D'], N=256)
        diverging_cmap = LinearSegmentedColormap.from_list('custom_diverging',
                                                           ['#C73E1D', '#F18F01', '#F7F7F7', '#6BAED6', '#2E86AB'], N=256)

        # 1. surprisal-weighted heatmap
        plt.figure(figsize=(6, 6))
        sns.heatmap(sorted_norm_weighted, mask=sorted_norm_weighted.values == 0, cmap=heatmap_cmap,
                    square=True, cbar_kws={'shrink': 0.8})
        plt.title('WITH Surprisal Weighting\n(sorted by linkage)', fontweight='bold')
        plt.tick_params(axis='both', labelsize=6)
        plt.tight_layout()
        plt.savefig('results/surprisal_weighting_with_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. raw heatmap
        plt.figure(figsize=(6, 6))
        sns.heatmap(sorted_norm_raw, mask=sorted_norm_raw.values == 0, cmap=heatmap_cmap,
                    square=True, cbar_kws={'shrink': 0.8})
        plt.title('WITHOUT Surprisal Weighting\n(sorted by linkage)', fontweight='bold')
        plt.tick_params(axis='both', labelsize=6)
        plt.tight_layout()
        plt.savefig('results/surprisal_weighting_without_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. difference heatmap
        plt.figure(figsize=(6, 6))
        vmax_diff = max(abs(sorted_diff_matrix.values[off_diag_mask].min()),
                        abs(sorted_diff_matrix.values[off_diag_mask].max()))
        sns.heatmap(sorted_diff_matrix, cmap=diverging_cmap, center=0, vmin=-vmax_diff, vmax=vmax_diff,
                    square=True, cbar_kws={'shrink': 0.8})
        plt.title('Difference (Weighted - Raw)\n(sorted by weighted linkage)', fontweight='bold')
        plt.tick_params(axis='both', labelsize=6)
        plt.tight_layout()
        plt.savefig('results/surprisal_weighting_difference_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

        # dendrograms (separate figures)
        plt.figure(figsize=(8, 6))
        dendrogram(linkage_weighted, labels=norm_weighted.index,
                   orientation='bottom', leaf_rotation=90, leaf_font_size=6)
        plt.title('WITH Surprisal Weighting', fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/surprisal_dendrogram_weighted.png', dpi=300, bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(8, 6))
        dendrogram(linkage_raw, labels=norm_raw.index,
                   orientation='bottom', leaf_rotation=90, leaf_font_size=6)
        plt.title('WITHOUT Surprisal Weighting', fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/surprisal_dendrogram_raw.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # calculate statistics
        correlation = np.corrcoef(norm_weighted.values[off_diag_mask], norm_raw.values[off_diag_mask])[0,1]
        mean_abs_diff = np.abs(diff_matrix.values[off_diag_mask]).mean()
        
        # calculate adjusted rand index between clustering results
        n_outlets = len(norm_weighted)
        n_clusters = 3
        
        labels_weighted = fcluster(linkage_weighted, n_clusters, criterion='maxclust')
        labels_raw = fcluster(linkage_raw, n_clusters, criterion='maxclust')
        
        ari_score = adjusted_rand_score(labels_weighted, labels_raw)
        
        print(f"Matrix correlation: {correlation:.3f}")
        print(f"Mean absolute difference: {mean_abs_diff:.4f}")
        print(f"Adjusted Rand Index between clustering results: {ari_score:.3f}")
        print(f"Number of clusters used for ARI calculation: {n_clusters}")
        
        return {
            'frequency_matrix_raw': frequency_matrix_raw,
            'frequency_matrix_weighted': frequency_matrix_weighted,
            'correlation': correlation,
            'mean_abs_diff': mean_abs_diff,
            'ari_score': ari_score
        }

    def analyze_hierarchical_clustering(self, frequency_matrix: pd.DataFrame, 
                                      colors: dict, threshold_percentile: float = 0) -> Dict[str, Any]:
        """perform hierarchical clustering analysis on frequency matrix"""
        print("\n=== HIERARCHICAL CLUSTERING AND ORDERING ===")
        
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
        plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)\nBased on Co-clustering Frequency', 
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
        ax.set_title(f'Hierarchically Ordered Co-clustering Frequency\n(threshold={threshold_percentile}th percentile of frequency values)', fontweight='bold')
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
        print("\n=== PER-COMMUNITY CONSISTENCY ANALYSIS ===")
        
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
            print(f"\nCommunity {comm_id} ({stats['size']} outlets):")
            print(f"  Outlets: {', '.join(stats['outlets'])}")
            print(f"  Within-community mean frequency: {stats['mean_frequency']:.3f} ± {stats['std_frequency']:.3f}")
        
        # compare with overall pairwise frequency
        off_diag_mask = ~np.eye(filtered_freq.shape[0], dtype=bool)
        overall_mean_frequency = filtered_freq.values[off_diag_mask].mean()
        print(f"\nOverall pairwise frequency (all outlet pairs): {overall_mean_frequency:.3f}")
        
        # summary analysis
        if community_stats:
            most_coherent = max(community_stats.keys(), key=lambda x: community_stats[x]['mean_frequency'])
            least_coherent = min(community_stats.keys(), key=lambda x: community_stats[x]['mean_frequency'])
            
            print(f"\nCommunity coherence summary:")
            print(f"  Most coherent: Community {most_coherent} (frequency: {community_stats[most_coherent]['mean_frequency']:.3f})")
            print(f"  Least coherent: Community {least_coherent} (frequency: {community_stats[least_coherent]['mean_frequency']:.3f})")
            print(f"  Higher frequency indicates more consistent clustering within community")
        
        # visualize community structure and frequency analysis
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
            plt.title('Community Coherence Analysis\n(Higher frequency = more coherent)', fontweight='bold')
            plt.xticks(range(len(comm_ids)), [f'Community {cid}' for cid in comm_ids])
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('results/community_consistency_bar.png', dpi=300, bbox_inches='tight')
            plt.show()

        # 2. annotated heatmap showing community boundaries (separate figure)
        ordered_freq_with_communities = filtered_freq.iloc[leaves_list(ward_linkage), leaves_list(ward_linkage)]
        heatmap_cmap = LinearSegmentedColormap.from_list('custom_heatmap',
                                                         ['#F7F7F7', '#F18F01', '#C73E1D'], N=256)

        plt.figure(figsize=(8, 8))
        ax_heat = sns.heatmap(ordered_freq_with_communities, mask=ordered_freq_with_communities.values == 0,
                              cmap=heatmap_cmap, square=True,
                              cbar_kws={'label': 'Surprisal-Weighted Frequency'})
        plt.title(f'Frequency Matrix with Community Structure\n({n_clusters} communities from hierarchical clustering)',
                  fontweight='bold')
        plt.tick_params(axis='both', labelsize=6)

        # add community boundary lines and labels
        if community_stats:
            community_colors = {cid: categorical_colors[i % len(categorical_colors)] for i, cid in enumerate(comm_ids)}
            outlet_to_community = {}
            for comm_id, outlets in communities.items():
                for outlet in outlets:
                    outlet_to_community[outlet] = comm_id

            ordered_communities = [outlet_to_community[outlet] for outlet in ordered_freq_with_communities.index]
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
                    plt.text(-2, mid_point, f'C{current_comm}', ha='center', va='center', fontsize=10,
                             fontweight='bold', color=community_colors[current_comm],
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    if comm is not None:
                        start_idx = i
                        current_comm = comm

        plt.tight_layout()
        plt.savefig('results/community_consistency_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'community_labels': community_labels,
            'communities': communities,
            'community_stats': community_stats,
            'overall_mean_frequency': overall_mean_frequency
        }

    # statistical significance testing
    # ------------------------------------------------------------------------------------------------

    def analyze_statistical_significance(self, frequency_matrix_weighted: pd.DataFrame, 
                                       colors: dict, n_permutations: int = 1000, 
                                       alpha: float = 0.05) -> Dict[str, Any]:
        """perform statistical significance testing using permutation tests with surprisal weighting"""
        print("\n=== STATISTICAL SIGNIFICANCE TESTING ===")
        
        def partition_to_coclustering_matrix(partition, n_outlets):
            """convert partition to binary co-clustering matrix"""
            assignments = np.full(n_outlets, -1, dtype=int)
            for outlet_id, community_id in partition.items():
                if outlet_id < n_outlets:
                    assignments[outlet_id] = community_id
            
            valid_mask = assignments >= 0
            assignments_2d = assignments[:, np.newaxis]
            assignments_broadcast = assignments[np.newaxis, :]
            
            matrix = ((assignments_2d == assignments_broadcast) & 
                      valid_mask[:, np.newaxis] & 
                      valid_mask[np.newaxis, :]).astype(int)
            
            return matrix

        def generate_null_distribution(analyzer, n_permutations=1000):
            """calculate analytical null distribution (mean and variance) for proper significance testing"""
            import time
            from collections import Counter
            from scipy.stats import norm
            start_time = time.time()
            
            results_df = analyzer.get_results()
            n_outlets = len(analyzer.outlet_names)
            
            print(f"calculating analytical null distribution (mean + variance)...")
            print(f"processing {len(results_df)} clustering results")
            
            # calculate both expected value and variance for each clustering result
            expected_frequencies = []
            variance_frequencies = []
            
            for _, row in results_df.iterrows():
                communities = row['communities']
                if not communities:
                    continue
                
                # get cluster sizes
                cluster_sizes = list(Counter(communities.values()).values())
                
                # calculate E[X] and Var[X] for this clustering result
                expected_x = 0.0
                expected_x2 = 0.0
                
                for size in cluster_sizes:
                    if size > 1:  # need at least 2 outlets for pairs
                        # probability that random pair ends up in cluster of this size
                        p_i = size * (size - 1) / (n_outlets * (n_outlets - 1))
                        # surprisal weight for this cluster size
                        w_i = -np.log2(size / n_outlets)
                        
                        # accumulate E[X] and E[X²]
                        expected_x += p_i * w_i
                        expected_x2 += p_i * (w_i ** 2)
                
                # variance = E[X²] - E[X]²
                variance_x = expected_x2 - (expected_x ** 2)
                
                expected_frequencies.append(expected_x)
                variance_frequencies.append(variance_x)
            
            # overall null distribution parameters
            null_mean = np.mean(expected_frequencies) if expected_frequencies else 0.0
            null_variance = np.mean(variance_frequencies) / len(expected_frequencies) if expected_frequencies else 0.0
            null_std = np.sqrt(max(null_variance, 1e-10))  # avoid division by zero
            
            elapsed = time.time() - start_time
            print(f"analytical calculation completed in {elapsed:.3f}s")
            print(f"null mean: {null_mean:.6f}")
            print(f"null std: {null_std:.6f}")
            print(f"analytical solution (no permutations needed)")
            
            # return parameters for z-test instead of frequency matrix
            return {
                'null_mean': null_mean,
                'null_std': null_std,
                'n_clustering_results': len(expected_frequencies)
            }

        def directional_significance_test(frequency_matrix, null_params, alpha=0.05):
            """analytical significance testing using z-tests"""
            from scipy.stats import norm
            
            n_outlets = frequency_matrix.shape[0]
            significant_low_pairs = []
            significant_high_pairs = []
            non_significant_pairs = []
            
            null_mean = null_params['null_mean']
            null_std = null_params['null_std']
            
            print(f"performing analytical significance tests for {n_outlets * (n_outlets - 1) // 2} outlet pairs...")
            print(f"using z-test with null mean={null_mean:.6f}, std={null_std:.6f}")
            
            for i in range(n_outlets):
                for j in range(i+1, n_outlets):
                    observed_val = frequency_matrix.iloc[i, j]
                    
                    # z-test for significance
                    if null_std > 0:
                        z_score = (observed_val - null_mean) / null_std
                        
                        # one-tailed tests
                        p_low = norm.cdf(z_score)  # P(Z <= z) for low values
                        p_high = 1 - norm.cdf(z_score)  # P(Z >= z) for high values
                    else:
                        # no variance means deterministic - simple comparison
                        if observed_val < null_mean:
                            p_low, p_high = 0.0, 1.0
                        elif observed_val > null_mean:
                            p_low, p_high = 1.0, 0.0
                        else:
                            p_low, p_high = 0.5, 0.5
                    
                    # check for significance
                    is_significant = False
                    if p_low < alpha:
                        significant_low_pairs.append({
                            'outlet1': frequency_matrix.index[i],
                            'outlet2': frequency_matrix.columns[j], 
                            'observed': observed_val,
                            'expected': null_mean,
                            'deviation': observed_val - null_mean,
                            'z_score': z_score if null_std > 0 else np.nan,
                            'p_value': p_low
                        })
                        is_significant = True
                    
                    if p_high < alpha:
                        significant_high_pairs.append({
                            'outlet1': frequency_matrix.index[i],
                            'outlet2': frequency_matrix.columns[j],
                            'observed': observed_val,
                            'expected': null_mean,
                            'deviation': observed_val - null_mean,
                            'z_score': z_score if null_std > 0 else np.nan,
                            'p_value': p_high
                        })
                        is_significant = True
                    
                    # collect non-significant pairs
                    if not is_significant:
                        # use the smaller p-value for non-significant pairs
                        min_p_value = min(p_low, p_high)
                        non_significant_pairs.append({
                            'outlet1': frequency_matrix.index[i],
                            'outlet2': frequency_matrix.columns[j],
                            'observed': observed_val,
                            'expected': null_mean,
                            'deviation': observed_val - null_mean,
                            'z_score': z_score if null_std > 0 else np.nan,
                            'p_value': min_p_value
                        })
            
            # sort by p-value
            significant_low_pairs.sort(key=lambda x: x['p_value'])
            significant_high_pairs.sort(key=lambda x: x['p_value'])
            non_significant_pairs.sort(key=lambda x: x['p_value'], reverse=True)  # sort by highest p-value first
            
            return significant_low_pairs, significant_high_pairs, non_significant_pairs
        
        # run the significance analysis with surprisal weighting
        print(f"analyzing {len(self.get_results())} clustering results for surprisal-weighted significance testing")
        
        # generate analytical null distribution (much faster than permutations)
        null_params = generate_null_distribution(self, n_permutations)
        
        # perform analytical significance test
        print("performing significance tests...")
        low_pairs, high_pairs, non_significant_pairs = directional_significance_test(frequency_matrix_weighted, null_params, alpha)
        
        # create significance mask for visualization
        n_outlets = frequency_matrix_weighted.shape[0]
        significant_mask = np.zeros((n_outlets, n_outlets), dtype=bool)
        
        # mark significant pairs
        for pair in low_pairs + high_pairs:
            i = frequency_matrix_weighted.index.get_loc(pair['outlet1'])
            j = frequency_matrix_weighted.columns.get_loc(pair['outlet2'])
            significant_mask[i, j] = True
            significant_mask[j, i] = True  # symmetric
        
        # create corrected p-value matrix for visualization
        corrected_p_df = pd.DataFrame(np.ones_like(frequency_matrix_weighted.values), 
                                     index=frequency_matrix_weighted.index, 
                                     columns=frequency_matrix_weighted.columns)
        
        for pair in low_pairs + high_pairs:
            i = frequency_matrix_weighted.index.get_loc(pair['outlet1'])
            j = frequency_matrix_weighted.columns.get_loc(pair['outlet2'])
            corrected_p_df.iloc[i, j] = pair['p_value']
            corrected_p_df.iloc[j, i] = pair['p_value']  # symmetric
        
        # create heatmap colormap
        heatmap_cmap = LinearSegmentedColormap.from_list('custom_heatmap', 
                                                       ['#F7F7F7', '#F18F01', '#C73E1D'], N=256)
        
        # visualize significance results (separate figures)

        # 1. surprisal-weighted frequency matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(frequency_matrix_weighted, mask=frequency_matrix_weighted.values == 0, cmap=heatmap_cmap,
                    square=True, cbar_kws={'label': 'Surprisal-Weighted Co-clustering Frequency'})
        plt.title('Surprisal-Weighted Co-clustering Frequency', fontweight='bold')
        plt.tick_params(axis='both', labelsize=8)
        plt.tight_layout()
        plt.savefig('results/significance_frequency_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. significance mask
        plt.figure(figsize=(8, 6))
        sns.heatmap(significant_mask.astype(int), cmap='RdBu_r', center=0.5, square=True,
                    cbar_kws={'label': 'Significant (1) vs Non-significant (0)'})
        plt.title(f'Statistical Significance Map\n(α = {alpha})', fontweight='bold')
        plt.tick_params(axis='both', labelsize=8)
        plt.tight_layout()
        plt.savefig('results/significance_mask.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. p-value heatmap (log scale for better visualization)
        log_p_values = -np.log10(corrected_p_df.values)
        log_p_values[corrected_p_df.values == 1.0] = 0  # non-significant = 0
        plt.figure(figsize=(8, 6))
        sns.heatmap(log_p_values, square=True, cmap='viridis',
                    cbar_kws={'label': '-log₁₀(p-value)'})
        plt.title('Statistical Significance Strength\n(Higher = more significant)', fontweight='bold')
        plt.tick_params(axis='both', labelsize=8)
        plt.tight_layout()
        plt.savefig('results/significance_pvalues_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # report results
        total_possible_pairs = n_outlets * (n_outlets - 1) // 2
        print(f"\nSURPRISAL-WEIGHTED SIGNIFICANCE TESTING RESULTS:")
        print(f"significantly low co-clustering pairs: {len(low_pairs)}")
        print(f"significantly high co-clustering pairs: {len(high_pairs)}")
        print(f"total significant pairs: {len(low_pairs) + len(high_pairs)}")
        print(f"total possible pairs: {total_possible_pairs}")
        print(f"fraction significant: {(len(low_pairs) + len(high_pairs)) / total_possible_pairs * 100:.1f}%")
        
        if low_pairs:
            print(f"\ntop 10 pairs with significantly LOW co-clustering:")
            for i, pair in enumerate(low_pairs[:10]):
                z_str = f", z={pair['z_score']:.3f}" if not np.isnan(pair.get('z_score', np.nan)) else ""
                print(f"{i+1:2d}. {pair['outlet1']} <-> {pair['outlet2']}: "
                      f"obs={pair['observed']:.3f}, exp={pair['expected']:.3f}, "
                      f"dev={pair['deviation']:.3f}{z_str}, p={pair['p_value']:.6f}")
        
        if high_pairs:
            print(f"\ntop 10 pairs with significantly HIGH co-clustering:")
            for i, pair in enumerate(high_pairs[:10]):
                z_str = f", z={pair['z_score']:.3f}" if not np.isnan(pair.get('z_score', np.nan)) else ""
                print(f"{i+1:2d}. {pair['outlet1']} <-> {pair['outlet2']}: "
                      f"obs={pair['observed']:.3f}, exp={pair['expected']:.3f}, "
                      f"dev={pair['deviation']:.3f}{z_str}, p={pair['p_value']:.6f}")
        
        if non_significant_pairs:
            print(f"\ntop 10 NON-SIGNIFICANT pairs (closest to significance threshold):")
            for i, pair in enumerate(non_significant_pairs[:10]):
                z_str = f", z={pair['z_score']:.3f}" if not np.isnan(pair.get('z_score', np.nan)) else ""
                print(f"{i+1:2d}. {pair['outlet1']} <-> {pair['outlet2']}: "
                      f"obs={pair['observed']:.3f}, exp={pair['expected']:.3f}, "
                      f"dev={pair['deviation']:.3f}{z_str}, p={pair['p_value']:.6f}")
        
        return {
            'low_pairs': low_pairs,
            'high_pairs': high_pairs,
            'non_significant_pairs': non_significant_pairs,
            'significant_mask': significant_mask,
            'corrected_p_df': corrected_p_df,
            'null_mean': null_params['null_mean'],
            'null_std': null_params['null_std'],
            'n_clustering_results': null_params['n_clustering_results'],
            'alpha': alpha
        }

    def validate_null_distribution(self, n_permutations: int = 10000, clustering_index: int = 0,
                                    pair: Optional[Tuple[int, int]] = None, random_state: Optional[int] = None,
                                    save_path: str = 'results/null_distribution_validation.png',
                                    show_plot: bool = True) -> Dict[str, Any]:
        """empirically validate the analytical null for one clustering and one outlet pair

        parameters
        ----------
        n_permutations : int
            number of random label permutations
        clustering_index : int
            index of the clustering result inside self.results_df to validate
        pair : tuple[int, int] | None
            outlet indices (i, j). if None a random pair is chosen
        random_state : int | None
            seed for reproducibility
        save_path : str
            where to store the histogram plot
        show_plot : bool
            whether to show the plot via plt.show()
        """
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        import os

        if self.results_df.empty or self.outlet_names is None:
            print('no results available')
            return {}

        rng = np.random.default_rng(random_state)

        # pick clustering result
        if clustering_index < 0 or clustering_index >= len(self.results_df):
            print('invalid clustering_index')
            return {}
        row = self.results_df.iloc[clustering_index]
        communities = row['communities']
        if not communities:
            print('selected clustering has empty communities')
            return {}

        n_outlets = len(self.outlet_names)
        outlet_indices = list(range(n_outlets))

        # choose outlet pair
        if pair is None:
            i, j = rng.choice(outlet_indices, size=2, replace=False)
            pair = (min(i, j), max(i, j))
        else:
            i, j = pair
            if i == j or not (0 <= i < n_outlets and 0 <= j < n_outlets):
                print('invalid pair indices')
                return {}

        # derive cluster sizes list
        cluster_counter = Counter(communities.values())
        cluster_sizes = list(cluster_counter.values())

        # analytical expectation for this clustering
        expected_x = 0.0
        expected_x2 = 0.0
        for size in cluster_sizes:
            if size > 1:
                p = size * (size - 1) / (n_outlets * (n_outlets - 1))
                w = -np.log2(size / n_outlets)
                expected_x += p * w
                expected_x2 += p * (w ** 2)
        analytic_var = max(expected_x2 - expected_x ** 2, 0.0)
        analytic_std = np.sqrt(analytic_var)

        # build list of cluster labels repeated by size for shuffling
        cluster_labels = []
        for cid, size in enumerate(cluster_sizes):
            cluster_labels.extend([cid] * size)
        cluster_labels = np.array(cluster_labels)
        if len(cluster_labels) < n_outlets:
            # fill remaining outlets with singletons if any
            remaining = n_outlets - len(cluster_labels)
            cluster_labels = np.concatenate((cluster_labels, np.arange(len(cluster_sizes), len(cluster_sizes)+remaining)))

        # run permutations
        samples = []
        for _ in range(n_permutations):
            rng.shuffle(cluster_labels)
            label_i = cluster_labels[i]
            label_j = cluster_labels[j]
            if label_i == label_j:
                size = np.sum(cluster_labels == label_i)
                weight = -np.log2(size / n_outlets)
            else:
                weight = 0.0
            samples.append(weight)
        samples = np.array(samples)
        emp_mean = samples.mean()
        emp_std = samples.std(ddof=1)

        # histogram with analytic normal curve
        if show_plot or save_path:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(samples, bins=40, density=True, alpha=0.6, label='empirical')
            x_vals = np.linspace(samples.min(), samples.max(), 400)
            ax.plot(x_vals, norm.pdf(x_vals, loc=expected_x, scale=analytic_std), 'r-', lw=2, label='normal pdf')
            ax.axvline(expected_x, color='r', linestyle='--', label='analytic mean')
            ax.set_xlabel('weight')
            ax.set_ylabel('density')
            ax.set_title(f'null validation for pair ({i}, {j})')
            ax.legend()
            plt.tight_layout()
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300)
            if show_plot:
                plt.show()
            else:
                plt.close(fig)

        print(f'pair ({i}, {j})')
        print(f'analytic  mean={expected_x:.6f}, std={analytic_std:.6f}')
        print(f'empirical mean={emp_mean:.6f}, std={emp_std:.6f}')

        return {
            'pair': pair,
            'analytic_mean': expected_x,
            'analytic_std': analytic_std,
            'empirical_mean': emp_mean,
            'empirical_std': emp_std,
            'n_permutations': n_permutations
        }

    def empirical_null_for_pair(self, pair: Tuple[int, int], n_permutations: int = 2000,
                                 random_state: Optional[int] = None) -> Dict[str, float]:
        """empirically estimate null mean/std for aggregated mean across all clusterings

        shuffles labels independently inside every clustering for each permutation, then
        computes the aggregated mean (with surprisal weights) for the chosen outlet pair.
        returns empirical mean and std so that one can compare with analytical values.
        """
        from collections import Counter
        rng = np.random.default_rng(random_state)

        if self.results_df.empty or self.outlet_names is None:
            print('no results in analyzer')
            return {}

        i, j = pair
        if i == j:
            print('pair must have i != j')
            return {}
        n_outlets = len(self.outlet_names)
        K = len(self.results_df)

        # gather cluster size lists per clustering for faster sampling
        cluster_size_lists = []
        for _, row in self.results_df.iterrows():
            communities = row['communities']
            if not communities:
                cluster_size_lists.append([])
                continue
            sizes = list(Counter(communities.values()).values())
            cluster_size_lists.append(sizes)

        samples = np.empty(n_permutations, dtype=float)
        label_buffer = np.empty(n_outlets, dtype=int)

        for p in range(n_permutations):
            total_weight = 0.0
            for sizes in cluster_size_lists:
                # build label array for this clustering following its size distribution
                idx = 0
                for cid, size in enumerate(sizes):
                    label_buffer[idx:idx+size] = cid
                    idx += size
                if idx < n_outlets:
                    label_buffer[idx:n_outlets] = np.arange(cid+1, cid+1 + (n_outlets-idx))
                    idx = n_outlets
                rng.shuffle(label_buffer)
                if label_buffer[i] == label_buffer[j]:
                    size = np.sum(label_buffer == label_buffer[i])
                    weight = -np.log2(size / n_outlets)
                    total_weight += weight
            samples[p] = total_weight / K

        return {
            'empirical_mean': float(samples.mean()),
            'empirical_std': float(samples.std(ddof=1))
        }

    def temporal_stability(self, window_ids: List[str], metric: str = 'ari', n_clusters: int = 6) -> pd.DataFrame:
        """compute pairwise similarity between windows created by run_temporal_experiment

        metric
        ------
        'corr' : pearson correlation between off-diagonal entries of the aggregated surprisal-weighted
                  co-clustering matrices for two windows
        'ari'  : adjusted rand index between ward-linkage clusterings (using `n_clusters`)
        """
        if not window_ids:
            return pd.DataFrame()

        # precompute aggregated matrices per window
        agg_matrices = {}
        for wid in window_ids:
            subset_df = self.get_results({'sample_id': wid})
            if subset_df.empty:
                continue
            mat = self._aggregate_clustering_results_with_surprisal(subset_df, wid, use_surprisal_weighting=True)
            if mat is not None:
                agg_matrices[wid] = mat

        wins = list(agg_matrices.keys())
        n = len(wins)
        if n == 0:
            return pd.DataFrame()

        sim = np.zeros((n, n))
        np.fill_diagonal(sim, 1.0)

        for i in range(n):
            for j in range(i + 1, n):
                m1, m2 = agg_matrices[wins[i]], agg_matrices[wins[j]]
                if metric == 'corr':
                    mask = ~np.eye(m1.shape[0], dtype=bool)
                    val = np.corrcoef(m1.values[mask], m2.values[mask])[0, 1]
                else:  # 'ari'
                    # convert to distance matrices and ward linkage
                    d1 = m1.values.max() - m1.values
                    d2 = m2.values.max() - m2.values
                    Z1 = linkage(squareform(d1, checks=False), method='ward')
                    Z2 = linkage(squareform(d2, checks=False), method='ward')
                    labels1 = fcluster(Z1, n_clusters, criterion='maxclust')
                    labels2 = fcluster(Z2, n_clusters, criterion='maxclust')
                    val = adjusted_rand_score(labels1, labels2)
                sim[i, j] = sim[j, i] = val

        return pd.DataFrame(sim, index=wins, columns=wins)
