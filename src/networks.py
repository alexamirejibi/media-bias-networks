"""
Network modeling and community detection for media bias analysis - Plugin-based
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from .plugins import registry


class NetworkBuilder:
    """builds adjacency matrices using registered methods"""
    
    def __init__(self, methods: Optional[List[str]] = None):
        available_methods = registry.get_network_methods()
        self.methods = methods or available_methods
        self.data = None
        self.entities = None
        
        # validate requested methods
        invalid_methods = set(self.methods) - set(available_methods)
        if invalid_methods:
            raise ValueError(f"unknown network methods: {invalid_methods}. "
                           f"available: {available_methods}")
    
    def set_data(self, data: pd.DataFrame):
        """set the input data (entities x clusters)"""
        self.data = data.astype(int)
        self.entities = data.index.tolist()
        print(f"set data: {len(self.entities)} entities, {data.shape[1]} clusters")
    
    def build_all(self) -> Dict[str, pd.DataFrame]:
        """build all adjacency matrices using registered methods"""
        if self.data is None:
            raise ValueError("no data set. call set_data() first")
        
        adjacencies = {}
        for method in self.methods:
            try:
                # get method from registry and call it
                method_func = registry.get_network_method(method)
                adjacencies[method] = method_func(self.data)
                print(f"built {method} adjacency: {adjacencies[method].shape}")
            except Exception as e:
                print(f"failed to build {method}: {e}")
        
        return adjacencies
    
    def build_single(self, method: str) -> pd.DataFrame:
        """build single adjacency matrix"""
        if self.data is None:
            raise ValueError("no data set. call set_data() first")
        
        method_func = registry.get_network_method(method)
        return method_func(self.data)
    
    @staticmethod
    def list_available_methods() -> List[str]:
        """list all registered network methods"""
        return registry.get_network_methods()


class CommunityDetector:
    """detects communities using registered methods"""
    
    def __init__(self, methods: Optional[List[str]] = None):
        available_methods = registry.get_community_methods()
        self.methods = methods or available_methods
        
        # validate requested methods
        invalid_methods = set(self.methods) - set(available_methods)
        if invalid_methods:
            raise ValueError(f"unknown community methods: {invalid_methods}. "
                           f"available: {available_methods}")
    
    def detect_all(self, adj_matrix: pd.DataFrame) -> Dict[str, Dict]:
        """detect communities with all registered methods and their parameters"""
        results = {}
        
        for method in self.methods:
            try:
                method_results = {}
                
                # get default parameters for this method
                param_sets = registry.get_default_params(method)
                if not param_sets:
                    param_sets = [{}]  # fallback to empty params
                
                # run method with each parameter set
                for i, params in enumerate(param_sets):
                    param_id = f"params_{i}"
                    try:
                        communities = self.detect_single(adj_matrix, method, params)
                        method_results[param_id] = {
                            'communities': communities,
                            'parameters': params,
                            'n_communities': len(set(communities.values()))
                        }
                        print(f"{method} {param_id}: {len(set(communities.values()))} communities")
                    except Exception as e:
                        print(f"failed {method} {param_id}: {e}")
                
                results[method] = method_results
                
            except Exception as e:
                print(f"failed to process method {method}: {e}")
        
        return results
    
    def detect_single(self, adj_matrix: pd.DataFrame, method: str, params: Dict) -> Dict[int, int]:
        """detect communities with single method"""
        method_func = registry.get_community_method(method)
        adj_array = np.array(adj_matrix, dtype=float)
        # replace nan values with 0
        adj_array = np.nan_to_num(adj_array, nan=0.0, posinf=0.0, neginf=0.0)
        return method_func(adj_array, params)
    
    @staticmethod
    def list_available_methods() -> List[str]:
        """list all registered community methods"""
        return registry.get_community_methods() 