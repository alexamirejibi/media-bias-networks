import numpy as np
from collections import defaultdict, Counter
import pandas as pd
import random
import json
import os
import time

from src.community_detection import all_comm_detection
from src.network_modeling import EntityAdjacencyMatrixMethods


class Framework():
    def __init__(self,data_dir,verbose=1):
        self.data_all_days = self.load_all_data(data_dir)

        # self.network_models = all_modeling_methods(verbose)
        self.community_methods = all_comm_detection(verbose)
        self.data_dir = data_dir
        
        # defaultdict so don't have to struggle with creating the whole nested JSON structure
        # now we can just do: self.results['sample_1']['jaccard']['louvain']['hpset1']['results']
        self.results = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(dict)
            )
        )


    # TODO Sort files by date?
    def load_all_data(self,data_dir=None,starts_with='matrix'):
        "Load the data from all days and put it in a list"
        files = [f for f in os.listdir(data_dir) if f.startswith(starts_with) and f.endswith('.csv')]
        dfs = [pd.read_csv(os.path.join(data_dir, f),index_col='Unnamed: 0') for f in files]
        # matrices = [np.array(df) for df in dfs] # NOTE Outdated, as we want the cluster dfs to be the base data
        if not dfs: raise ValueError("No data loaded, check provided data directory")
        return dfs


    def run_experiment(self,n_samples,sample_size):
        "Starting point for the experiment"
        samples = self.load_samples(n_samples,sample_size)
        for s, sample in enumerate(samples):
            sample_ID = f"sample_{s}"
            self.create_adj_matrices(sample_ID,sample)


    def create_adj_matrices(self,sample_ID,sample):
        print(f"creating matrices for {sample_ID}")
        "Given a sample (subset of all data), create adjacency matrices with all methods defined in src.network_modeling.py"
        network_models = EntityAdjacencyMatrixMethods()
        network_models.set_data(sample)
        adj_matrices = network_models.all_modeling_methods()
        
        for network_method, adj_matrix in adj_matrices.items():
            self.comm_detection(sample_ID,adj_matrix,network_method)


    def comm_detection(self,sample_ID,adj_matrix,modeling_method):
        """
        Given an adjacency matrix, compute communities with all methods
        defined in src.community_detection.py with all corresponding sets of
        parameters as defined in hyperparameters.json
        """
        adj_matrix = np.array(adj_matrix,dtype=int)
        for comm_method in self.community_methods:
            comm_method_name = comm_method.__name__
            hp_sets = self.get_hyperparameters(comm_method_name)
            for hpi,hp in enumerate(hp_sets):
                # TODO What if there are no hyperparameters
                hp_ID = f"{comm_method_name}_{hpi}"
                communities = comm_method(adj_matrix)

                result = {
                    "hyperparameters":hp,
                    "communities":communities
                }
                print(sample_ID,modeling_method,comm_method_name,hp_ID)
                print(result['communities'])
                self.results[sample_ID][modeling_method][comm_method_name][hp_ID] = result

    def get_hyperparameters(self,comm_detection):
        "Load hyperparameters as defined in hyperparameters.json"
        with open('hyperparameters.json', 'r') as file:
            hp_json = json.load(file)
            return hp_json[comm_detection]


    def load_samples(self,n_samples,sample_size):
        "Loads 'n_samples' different samples of sample size 'sample_size'"
        all_samples = [self.sample_random_clusters(sample_size) for _ in range(n_samples)]

        return all_samples


    def sample_random_clusters(self,sample_size):
        "Samples 'sample_size' different cluster days from all the data"
        sampled_days = random.sample(self.data_all_days,sample_size)

        # aggregated_cooccurence_matrix = sum(sampled_days) # NOTE this is outdated, as we are now concatenating cluster data

        concatenated_sample_df = pd.concat(sampled_days, axis=1)
        concatenated_sample_df = concatenated_sample_df.fillna(0)
        concatenated_sample_df = concatenated_sample_df.astype(int)

        return concatenated_sample_df


    def cluster_data_to_cooccurrence(self,concatenated_df):
        co_occurrence_matrix = concatenated_df.dot(concatenated_df.T)
        return co_occurrence_matrix