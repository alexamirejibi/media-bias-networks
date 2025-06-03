import pandas as pd
import numpy as np
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
        
        # create result record
        result = {
            'sample_id': sample_id,
            'network_method': network_method,
            'community_method': community_method,
            'param_id': param_id,
            'parameters': str(parameters),
            'n_communities': n_communities,
            'largest_community': largest_community,
            'communities': communities
        }
        
        # add to dataframe  
        self.results_df = pd.concat([self.results_df, pd.DataFrame([result])], 
                                  ignore_index=True)
        
        print(f"added result: {sample_id} | {network_method} | {community_method} | {n_communities} communities")
    
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