"""
Main experiment orchestration for media bias network analysis
"""

from .data import DataManager
from .networks import NetworkBuilder, CommunityDetector
from .analysis import ResultsAnalyzer
from .config import DEFAULT_SAMPLE_SIZE, DEFAULT_N_SAMPLES

import time
from typing import List, Optional
import numpy as np
import pandas as pd


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

            
    def get_results_df(self) -> 'pd.DataFrame':
        """get the raw results dataframe"""
        return self.analyzer.results_df.copy()
    
    def export_results(self, filepath: str = 'results/experiment_results.csv'):
        """export results to csv file"""
        self.analyzer.export_results(filepath)

    def aggregate_results(self):
        """aggregate results from all samples"""
        return self.analyzer.aggregate_results_normalized()
    
    def run_temporal_experiment(self, window_size: int = 30, step: int = 30) -> dict:
        """run pipeline on consecutive (or sliding) windows of daily data

        parameters
        ----------
        window_size : int
            number of consecutive days per window
        step : int
            number of days to move the window start each iteration. step == window_size → non-overlapping windows
        returns
        -------
        dict
            summary containing one entry per window
        """
        total_days = self.data.get_daily_data_info()['n_files']
        if window_size > total_days:
            raise ValueError("window_size exceeds total available days")

        # determine window start indices
        starts = list(range(0, total_days - window_size + 1, step))
        print(f"\n=== starting temporal experiment: {len(starts)} windows of {window_size} days (step={step}) ===")

        experiment_start = time.time()
        window_summaries = []

        for w_idx, start_day in enumerate(starts):
            window_id = f"win_{w_idx:02d}"
            # build sample via deterministic window
            sample_df = self.data.create_window(window_id, start_day, window_size)

            # pipeline identical to run_sample but with deterministic data
            self.networks.set_data(sample_df)
            adjacencies = self.networks.build_all()

            for network_method, adj_matrix in adjacencies.items():
                community_results = self.communities.detect_all(adj_matrix)
                self.analyzer.add_sample_results(
                    sample_id=window_id,
                    network_method=network_method,
                    community_results=community_results,
                    adjacency_matrix=adj_matrix
                )

            window_summaries.append({
                'window_id': window_id,
                'start_day_index': start_day,
                'end_day_index': start_day + window_size - 1,
                'n_results_cumulative': len(self.analyzer.results_df)
            })

        total_time = time.time() - experiment_start
        print(f"\n=== temporal experiment completed in {total_time:.1f}s ===")

        return {
            'n_windows': len(starts),
            'window_size': window_size,
            'step': step,
            'total_time': total_time,
            'window_summaries': window_summaries
        }
        