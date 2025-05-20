from collections import defaultdict
import pandas as pd
import numpy as np
import random

from src.community_detection import all_comm_detection
from src.network_modeling import all_modeling_methods


class Framework():
    def __init__(self):
        self.network_models = all_modeling_methods(1)
        self.community_methods = all_comm_detection(1)
        
        self.results = defaultdict(lambda: defaultdict(dict))

    
    def start_experiment(self,data_path,n_samples,sample_size,N):
        samples = self.load_samples(n_samples,sample_size,N)
        for sample in samples:
            sample_ID = ...
            self.create_adj_matrices(sample_ID)


    def create_adj_matrices(self,sample_ID,sample):
        for network_model in self.network_models:
            adj_matrix = network_model(sample)
            self.comm_detection(sample_ID,adj_matrix,network_model.__name__)

    
    def comm_detection(self,sample_ID,adj_matrix,modeling_method):
        for comm_method in self.community_methods:
            hp_sets = self.get_hyperparameters(comm_method.__name__)
            for hp in hp_sets:
                # TODO What if there are no hyperparameters
                hp_ID = ...
                communities = comm_method(adj_matrix)

                result = {
                    "hyperparameters":hp,
                    "communities":communities
                }
                
                self.results[sample_ID][modeling_method][comm_method.__name__][hp_ID] = result

    def get_hyperparameters(self,comm_detection):
        ...


    def load_samples(self,n_samples,sample_size,N):
        # N: data size
        all_samples = [self.load_sample(sample_size,N) for _ in range(n_samples)]

        return all_samples


    def load_sample(self,sample_size,N):
        datapoints = random.sample(list(range(N)),sample_size)

        all_sample_data = [self.load_datapoint(...) for dp in datapoints]

        aggregated_cooccurence_matrix = sum(all_sample_data)

        return aggregated_cooccurence_matrix    


    def load_datapoint(self,data_path):
        df = pd.read_csv(data_path,index_col='Unnamed: 0')
        m = np.array(df)

        return m


