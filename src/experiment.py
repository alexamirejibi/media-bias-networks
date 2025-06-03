"""
Main experiment orchestration for media bias network analysis
"""

from .data import DataManager
from .networks import NetworkBuilder, CommunityDetector
from .analysis import ResultsAnalyzer
from .config import DEFAULT_SAMPLE_SIZE, DEFAULT_N_SAMPLES

import time
from typing import List, Optional


class ExperimentFramework:
    """main experiment class that orchestrates the entire analysis pipeline"""
    
    def __init__(self, data_dir: str, network_methods: Optional[List[str]] = None,
                 community_methods: Optional[List[str]] = None):
        """
        initialize experiment with data directory and optional method selections
        
        args:
            data_dir: path to directory containing matrix CSV files
            network_methods: list of network modeling methods to use
            community_methods: list of community detection methods to use
        """
        print("initializing media bias experiment...")
        
        # initialize components
        self.data = DataManager(data_dir)
        self.networks = NetworkBuilder(methods=network_methods)
        self.communities = CommunityDetector(methods=community_methods)
        self.analyzer = ResultsAnalyzer()
        
        print(f"{self.data.get_daily_data_info()['n_files']} daily files loaded")
    
    def run_sample(self, sample_id: str, n_days: int = DEFAULT_SAMPLE_SIZE) -> dict:
        """
        run complete analysis on a single sample
        
        args:
            sample_id: identifier for this sample
            n_days: number of random days to include in sample
            
        returns:
            dict with sample results summary
        """
        print(f"\nrunning sample: {sample_id}")
        start_time = time.time()
        
        # create sample data
        sample_data = self.data.create_sample(sample_id, n_days)
        
        # set data for network building
        self.networks.set_data(sample_data)
        
        # build all adjacency matrices
        adjacencies = self.networks.build_all()
        
        results_summary = {}
        
        # for each adjacency matrix, detect communities
        for network_method, adj_matrix in adjacencies.items():
            print(f"\nprocessing {network_method} adjacency matrix...")
            
            # detect communities with all methods
            community_results = self.communities.detect_all(adj_matrix)
            
            # store results in analyzer
            self.analyzer.add_sample_results(
                sample_id=sample_id,
                network_method=network_method,
                community_results=community_results,
                adjacency_matrix=adj_matrix
            )
            
            # summarize this network method
            n_total_results = sum(len(method_results) for method_results in community_results.values())
            results_summary[network_method] = n_total_results
        
        elapsed = time.time() - start_time
        print(f"sample {sample_id} completed in {elapsed:.1f}s")
        
        return {
            'sample_id': sample_id,
            'n_days': n_days,
            'data_shape': sample_data.shape,
            'network_methods': list(adjacencies.keys()),
            'total_results': sum(results_summary.values()),
            'elapsed_time': elapsed
        }
    
    def run_experiment(self, n_samples: int = DEFAULT_N_SAMPLES, 
                       n_days: int = DEFAULT_SAMPLE_SIZE) -> dict:
        """
        run full experiment with multiple samples
        
        args:
            n_samples: number of different samples to create and analyze
            n_days: number of days per sample
            
        returns:
            dict with experiment summary
        """
        print(f"\n=== starting experiment: {n_samples} samples x {n_days} days ===")
        experiment_start = time.time()
        
        sample_summaries = []
        
        for i in range(n_samples):
            sample_id = f"sample_{i:02d}"
            sample_summary = self.run_sample(sample_id, n_days)
            sample_summaries.append(sample_summary)
        
        total_time = time.time() - experiment_start
        
        experiment_summary = {
            'n_samples': n_samples,
            'n_days_per_sample': n_days,
            'total_results': len(self.analyzer.results_df),
            'total_time': total_time,
            'avg_time_per_sample': total_time / n_samples,
            'sample_summaries': sample_summaries
        }
        
        print(f"\n=== experiment completed in {total_time:.1f}s ===")
        print(f"total results: {experiment_summary['total_results']}")
        
        return experiment_summary
        
    def get_results_df(self) -> 'pd.DataFrame':
        """get the raw results dataframe"""
        return self.analyzer.results_df.copy()
    
    def export_results(self, filepath: str = 'results/experiment_results.csv'):
        """export results to csv file"""
        self.analyzer.export_results(filepath)
    
    # def list_available_data(self) -> dict:
    #     """list what data is available for analysis"""
    #     return {
    #         'samples': self.data.list_samples(),
    #         'daily_data_info': self.data.get_daily_data_info(),
    #         'results_summary': self.analyzer.summary(),
    #         'available_adjacencies': list(self.analyzer.adjacencies.keys())
    #     }