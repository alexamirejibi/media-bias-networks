"""
Data management for media bias network analysis
"""

import os
import pandas as pd
import random
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class DataManager:
    """handles loading and sampling of daily cluster matrices"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.daily_data = {}  # cached datasets
        self.samples = {}  # cached samples
        self._load_daily_data()
    
    def _load_daily_data(self):
        """load all daily matrix files"""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"data directory not found: {self.data_dir}")
        
        files = [f for f in os.listdir(self.data_dir) 
                if f.startswith('matrix') and f.endswith('.csv')]
        
        if not files:
            raise ValueError(f"no matrix files found in {self.data_dir}")
        
        print(f"loading {len(files)} daily matrices...")
        for file in files:
            file_path = os.path.join(self.data_dir, file)
            df = pd.read_csv(file_path, index_col='Unnamed: 0')
            self.daily_data[file] = df
        
        print(f"loaded {len(self.daily_data)} datasets")
    
    def create_sample(self, sample_id: str, n_days: int = 5) -> pd.DataFrame:
        """create a sample by concatenating random daily matrices"""
        if sample_id in self.samples:
            return self.samples[sample_id]
        
        if n_days > len(self.daily_data):
            raise ValueError(f"requested {n_days} days but only {len(self.daily_data)} available")
        
        # sample random days
        sampled_files = random.sample(list(self.daily_data.keys()), n_days)
        sampled_dfs = [self.daily_data[f] for f in sampled_files]
        
        # concatenate and fill missing values
        sample_df = pd.concat(sampled_dfs, axis=1).fillna(0).astype(int)
        
        # cache the sample
        self.samples[sample_id] = sample_df
        
        print(f"created sample '{sample_id}' with {n_days} days, shape: {sample_df.shape}")
        return sample_df
    
    def get_sample(self, sample_id: str) -> pd.DataFrame:
        """retrieve a cached sample"""
        if sample_id not in self.samples:
            raise KeyError(f"sample '{sample_id}' not found. create it first with create_sample()")
        return self.samples[sample_id]
    
    def list_samples(self) -> List[str]:
        """list all cached samples"""
        return list(self.samples.keys())
    
    def get_daily_data_info(self) -> Dict:
        """get info about loaded daily data"""
        return {
            'n_files': len(self.daily_data),
            'file_names': list(self.daily_data.keys()),
            'shapes': {f: df.shape for f, df in self.daily_data.items()}
        } 