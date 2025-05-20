import numpy as np
from collections import defaultdict
import pandas as pd
import random
import json
import os

from src.community_detection import all_comm_detection
from src.network_modeling import all_modeling_methods


class Framework():
    def __init__(self,data_dir):
        self.all_datapoints = self.load_all_data(data_dir)
        self.network_models = all_modeling_methods(1)
        self.community_methods = all_comm_detection(1)
        
        # defaultdict so don't have to struggle with creating the whole nested JSON structure
        # now we can just do: self.results['sample_1']['jaccard']['louvain']['hpset1']['results']
        self.results = defaultdict(lambda: defaultdict(dict))

    def nothing(self):
        df = pd.read_csv(data_path,index_col='Unnamed: 0')
        m = np.array(df)

        return m



    # TODO Sort files by date?
    def load_all_data(self,data_dir,starts_with='daily_co_occurrence'):
        "Load the data from all days and put it in a list"
        files = [f for f in os.listdir(data_dir) if f.startswith(starts_with) and f.endswith('.csv')]
        dfs = [pd.read_csv(os.path.join(data_dir, f),index_col='Unnamed: 0') for f in files]
        matrices = [np.array(df) for df in dfs]
        return matrices

    
    def start_experiment(self,n_samples,sample_size):
        "Starting point for the experiment"
        samples = self.load_samples(n_samples,sample_size)
        for s, sample in enumerate(samples):
            sample_ID = f"sample_{s}"
            self.create_adj_matrices(sample_ID,sample)


    def create_adj_matrices(self,sample_ID,sample):
        "Given a sample (subset of all data), create adjacency matrices with all methods defined in src.network_modeling.py"
        for network_model in self.network_models:
            adj_matrix = network_model(sample)
            self.comm_detection(sample_ID,adj_matrix,network_model.__name__)

    
    def comm_detection(self,sample_ID,adj_matrix,modeling_method):
        """
        Given an adjacency matrix, compute communities with all methods
        defined in src.community_detection.py with all corresponding sets of
        parameters as defined in hyperparameters.json
        """
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
                
                self.results[sample_ID][modeling_method][comm_method_name][hp_ID] = result

    def get_hyperparameters(self,comm_detection):
        "Load hyperparameters as defined in hyperparameters.json"
        with open('hyperparameters.json', 'r') as file:
            hp_json = json.load(file)
            return hp_json[comm_detection]


    def load_samples(self,n_samples,sample_size):
        "Loads 'n_samples' different samples of sample size 'sample_size'"
        all_samples = [self.load_sample(sample_size) for _ in range(n_samples)]

        return all_samples


    def load_sample(self,sample_size):
        "Samples 'sample_size' different datapoints from all the data"
        sampled_datapoints = random.sample(self.all_datapoints,sample_size)

        aggregated_cooccurence_matrix = sum(sampled_datapoints)

        return aggregated_cooccurence_matrix

