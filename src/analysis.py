import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict
import warnings
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
    
    def _aggregate_clustering_results_with_entropy(self, results_subset: pd.DataFrame, method_name: str = "results", use_entropy_normalization: bool = True) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """core aggregation logic with both frequency and entropy calculation"""
        if results_subset.empty:
            print(f"no {method_name} to aggregate")
            return None
        if self.outlet_names is None:
            print("no outlet names available")
            return None
        
        n_outlets = len(self.outlet_names)
        print(f"processing {len(results_subset)} clusterings for {n_outlets} outlets ({method_name})")
        
        # initialize matrices
        normalized_sum = np.zeros((n_outlets, n_outlets))
        binary_outcomes = []  # list of binary matrices for each clustering
        total_clusterings = 0
        
        # diagnostic tracking
        normalization_factors = []
        p_expected_values = []
        very_fine_clusterings = 0
        missing_outlet_warnings = 0
        
        # process each clustering result
        for _, row in results_subset.iterrows():
            communities = row['communities']
            if not communities:
                continue
            
            total_clusterings += 1
            
            # initialize binary outcome matrix for this clustering
            binary_matrix = np.zeros((n_outlets, n_outlets))
            
            # calculate cluster sizes
            cluster_sizes = {}
            for outlet_id, cluster_id in communities.items():
                if cluster_id not in cluster_sizes:
                    cluster_sizes[cluster_id] = 0
                cluster_sizes[cluster_id] += 1
            
            # calculate expected random co-clustering probability
            expected_pairs_within_clusters = 0
            for cluster_size in cluster_sizes.values():
                if cluster_size >= 2:
                    expected_pairs_within_clusters += cluster_size * (cluster_size - 1) // 2
            
            total_possible_pairs = n_outlets * (n_outlets - 1) // 2
            if total_possible_pairs == 0:
                continue
            
            p_expected = expected_pairs_within_clusters / total_possible_pairs
            p_expected_values.append(p_expected)
            
            # avoid division by zero
            if p_expected == 0:
                print(f"warning: zero expected probability for clustering with {len(cluster_sizes)} singleton clusters")
                continue
            
            # diagnostic: check for very fine-grained clusterings
            if p_expected < 0.001:
                very_fine_clusterings += 1
                if very_fine_clusterings <= 3:
                    print(f"Very fine clustering detected: p_expected = {p_expected:.6f}, {len(cluster_sizes)} clusters")
            
            # calculate contribution for this clustering
            if use_entropy_normalization:
                if p_expected > 0:
                    normalization_factor = -np.log(p_expected)
                else:
                    continue
                normalization_factors.append(normalization_factor)
            else:
                normalization_factor = 1.0  # simple count-based approach
            
            # update both matrices (exclude diagonal)
            for i in range(n_outlets):
                for j in range(i + 1, n_outlets):
                    outlet_i_cluster = communities.get(i)
                    outlet_j_cluster = communities.get(j)
                    
                    if outlet_i_cluster is None or outlet_j_cluster is None:
                        missing_outlet_warnings += 1
                        continue
                    
                    if outlet_i_cluster == outlet_j_cluster:
                        # weighted frequency matrix (existing logic)
                        normalized_sum[i, j] += normalization_factor
                        normalized_sum[j, i] += normalization_factor
                        
                        # binary outcome matrix (new for entropy)
                        binary_matrix[i, j] = 1
                        binary_matrix[j, i] = 1
            
            # store binary outcome for this clustering
            binary_outcomes.append(binary_matrix)
        
        if total_clusterings == 0:
            print(f"no valid clusterings found for {method_name}")
            return None
        
        # calculate frequency matrix
        if use_entropy_normalization:
            normalized_frequencies = normalized_sum / total_clusterings
        else:
            # simple frequency calculation without entropy normalization
            normalized_frequencies = normalized_sum / total_clusterings
        
        # calculate entropy matrix (new)
        entropy_matrix = np.zeros((n_outlets, n_outlets))
        
        # stack binary outcomes for efficient computation
        binary_stack = np.stack(binary_outcomes, axis=-1)  # shape: (n_outlets, n_outlets, n_clusterings)
        
        # calculate shannon entropy for each outlet pair
        for i in range(n_outlets):
            for j in range(n_outlets):
                if i == j:
                    entropy_matrix[i, j] = 0  # diagonal is always 0
                    continue
                
                # get binary outcomes for this pair across all clusterings
                outcomes = binary_stack[i, j, :]
                p = np.mean(outcomes)  # fraction of clusterings where pair co-clusters
                
                # calculate shannon entropy: H = -p*log(p) - (1-p)*log(1-p)
                if p == 0 or p == 1:
                    entropy_matrix[i, j] = 0  # perfectly predictable
                else:
                    entropy_matrix[i, j] = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        
        # diagnostic reporting
        print(f"\n=== NORMALIZATION DIAGNOSTICS ({method_name}) ===")
        print(f"Entropy normalization: {'enabled' if use_entropy_normalization else 'disabled'}")
        print(f"Total clusterings processed: {total_clusterings}")
        if p_expected_values:
            print(f"Expected probability range: {min(p_expected_values):.6f} to {max(p_expected_values):.6f}")
        if use_entropy_normalization and normalization_factors:
            print(f"Normalization factor range: {min(normalization_factors):.2f} to {max(normalization_factors):.2f}")
            print(f"Mean normalization factor: {np.mean(normalization_factors):.2f}")
            print(f"Median normalization factor: {np.median(normalization_factors):.2f}")
        print(f"Very fine clusterings (p < 0.001): {very_fine_clusterings}")
        if missing_outlet_warnings > 0:
            print(f"Missing outlet warnings: {missing_outlet_warnings}")
        
        # create result dataframes
        frequency_df = pd.DataFrame(
            normalized_frequencies,
            index=self.outlet_names,
            columns=self.outlet_names
        )
        
        entropy_df = pd.DataFrame(
            entropy_matrix,
            index=self.outlet_names,
            columns=self.outlet_names
        )
        
        # final diagnostics
        print(f"\n=== FINAL RESULTS ({method_name}) ===")
        non_zero_freqs = normalized_frequencies[normalized_frequencies > 0]
        if len(non_zero_freqs) > 0:
            print(f"Mean normalized frequency: {non_zero_freqs.mean():.3f}")
            print(f"Max normalized frequency: {normalized_frequencies.max():.3f}")
        print(f"Non-zero frequency entries: {np.sum(normalized_frequencies > 0)}")
        
        # entropy diagnostics
        off_diag_mask = ~np.eye(n_outlets, dtype=bool)
        entropy_values = entropy_matrix[off_diag_mask]
        print(f"Mean entropy: {entropy_values.mean():.3f}")
        print(f"Max entropy: {entropy_values.max():.3f}")
        print(f"Entropy std: {entropy_values.std():.3f}")
        print(f"Zero entropy pairs: {np.sum(entropy_values == 0)}")
        
        return frequency_df, entropy_df
    
    def aggregate_all_results(self, use_entropy_normalization: bool = True) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """aggregate with entropy matrices"""
        print("aggregating all results with entropy matrices...")
        if self.results_df.empty:
            print("no results to aggregate")
            return None
        
        return self._aggregate_clustering_results_with_entropy(self.results_df, "all_methods", use_entropy_normalization)

    def aggregate_results_by_method_with_entropy(self, use_entropy_normalization: bool = True) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """aggregate results by method with both frequency and entropy matrices"""
        print("aggregating results by individual community detection method (with entropy)...")
        
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
            result_matrices = self._aggregate_clustering_results_with_entropy(group_df, community_method, use_entropy_normalization)
            
            if result_matrices is not None:
                method_results[community_method] = result_matrices
        
        print(f"\n{'='*50}")
        print(f"SUMMARY: successfully processed {len(method_results)}/{len(community_methods)} methods")
        print(f"{'='*50}")
        
        return method_results
    
    def compare_method_coclustering(self, method_results: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """compare co-clustering patterns across different methods"""
        if method_results is None:
            method_results = self.aggregate_results_by_method()
        
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