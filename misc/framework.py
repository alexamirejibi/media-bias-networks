import numpy as np
from collections import defaultdict, Counter
import pandas as pd
import random
import json
import os
import time

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

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
        
        # Store adjacency matrices for each sample and modeling method
        self.adjacencies = defaultdict(dict)


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
        
        # Store adjacency matrices
        self.adjacencies[sample_ID] = adj_matrices
        
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
                hp_name = str(hp).replace('{','').replace('}','').replace(':','').replace(' ','').replace("'",'')
                hp_ID = f"{comm_method_name}_{hp_name}"
                communities = comm_method(adj_matrix,hp)

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
    
    def results_to_df(self):
        """
        Convert the self.results dictionary to a pandas DataFrame.
        The DataFrame will contain the following columns:
        - sample_ID
        - modeling_method
        - comm_method_name
        - hp_ID
        - communities
        """
        raw_data_list = []
        if hasattr(self, 'results') and self.results:
            for s_id, m_methods in self.results.items():
                for m_method, c_methods in m_methods.items():
                    for c_name, hp_dict in c_methods.items():
                        for hp_id, res in hp_dict.items():
                            raw_data_list.append({
                                'sample_ID': s_id,
                                'modeling_method': m_method,
                                'comm_method_name': c_name,
                                'hp_ID': hp_id,
                                'communities': res['communities']
                            })
        else:
            print("Warning: self.results not found or empty.")
            # Optionally, return an empty DataFrame or handle as needed
            return pd.DataFrame()

        if not raw_data_list:
            print("No data available to process into DataFrame.")
            return pd.DataFrame()

        results_df = pd.DataFrame(raw_data_list)
        return results_df


    def get_visualizer(self):
        return VisualizationMethods(self)

    
    def print_available_results(self):
        # print a summary of available results for visualization
        print("Available Results for Visualization:")
        print("="*50)
        
        for sample_id in self.results:
            print(f"\nSample: {sample_id}")
            
            # show adjacency matrices
            if sample_id in self.adjacencies:
                print(f"  Adjacency Matrices: {list(self.adjacencies[sample_id].keys())}")
            
            # show community detection results
            for modeling_method in self.results[sample_id]:
                print(f"  Network Method: {modeling_method}")
                for comm_method in self.results[sample_id][modeling_method]:
                    hp_sets = list(self.results[sample_id][modeling_method][comm_method].keys())
                    print(f"    Community Method: {comm_method} ({len(hp_sets)} parameter sets)")









# ------------------------------------------------------------
class VisualizationMethods:
    """Concise visualization methods for media bias network analysis."""
    
    def __init__(self, framework):
        self.framework = framework
        self.results = framework.results
        self.adjacencies = framework.adjacencies
        plt.style.use('default')
        sns.set_palette("husl")
    
    def _get_data(self, sample_id, modeling_method, comm_method=None, hp_id=None):
        """Helper to retrieve and validate data."""
        try:
            adj_matrix = self.adjacencies[sample_id][modeling_method]
            communities = None
            if comm_method and hp_id:
                communities = self.results[sample_id][modeling_method][comm_method][hp_id]['communities']
            return adj_matrix, communities
        except KeyError as e:
            print(f"Data not found: {e}")
            return None, None
    
    def _setup_plot(self, figsize, title):
        """Helper to setup plot with consistent styling."""
        plt.figure(figsize=figsize)
        plt.title(title)
        return plt.gca()
    
    def _save_or_show_matplotlib_plot(self, save_path):
        """Helper to save or show a matplotlib plot."""
        fig = plt.gcf()
        if not plt.get_fignums(): # No figures open/active
            return None

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig) # Close the figure after saving
        else:
            plt.show()
        return fig
    
    def plot_adjacency_heatmap(self, sample_id, modeling_method, figsize=(10, 8), save_path=None):
        """Plot adjacency matrix heatmap."""
        adj_matrix, _ = self._get_data(sample_id, modeling_method)
        if adj_matrix is None:
            return None
        
        self._setup_plot(figsize, f'Adjacency: {modeling_method} | {sample_id}')
        sns.heatmap(adj_matrix, cmap='viridis', square=True, 
                   cbar_kws={'label': 'Strength'})
        plt.tight_layout()
        return self._save_or_show_matplotlib_plot(save_path)
    
    def plot_network_graph(self, sample_id, modeling_method, layout='spring', 
                          threshold=0.1, figsize=(12, 10), save_path=None):
        """Plot network graph with nodes and edges."""
        adj_matrix, _ = self._get_data(sample_id, modeling_method)
        if adj_matrix is None:
            return None
        
        if isinstance(adj_matrix, pd.DataFrame):
            G = nx.from_pandas_adjacency(adj_matrix.where(adj_matrix >= threshold, 0))
        else:
            G = nx.from_numpy_array(np.where(adj_matrix >= threshold, adj_matrix, 0))
        
        self._setup_plot(figsize, f'Network: {modeling_method} | {sample_id}')
        pos = getattr(nx, f'{layout}_layout')(G) if hasattr(nx, f'{layout}_layout') else nx.spring_layout(G)
        nx.draw(G, pos, node_color='lightblue', node_size=300, 
               with_labels=True, font_size=8, edge_color='gray', alpha=0.7)
        plt.axis('off')
        return self._save_or_show_matplotlib_plot(save_path)
    
    def plot_community_network(self, sample_id, modeling_method, comm_method, hp_id,
                              layout='spring', figsize=(12, 10), save_path=None):
        """Plot network colored by communities."""
        adj_matrix, communities = self._get_data(sample_id, modeling_method, comm_method, hp_id)
        if adj_matrix is None or communities is None:
            return None
        
        G = nx.from_pandas_adjacency(adj_matrix.where(adj_matrix > 0.1, 0)) if isinstance(adj_matrix, pd.DataFrame) else nx.from_numpy_array(np.where(adj_matrix > 0.1, adj_matrix, 0))
        node_colors_indices = [communities.get(i, 0) if isinstance(communities, dict) else communities[i] for i in range(len(G.nodes()))]
        
        self._setup_plot(figsize, f'Communities: {comm_method} | {modeling_method}')
        pos = getattr(nx, f'{layout}_layout')(G) if hasattr(nx, f'{layout}_layout') else nx.spring_layout(G)
        
        unique_comms = list(set(node_colors_indices))
        palette = plt.cm.Set3(np.linspace(0, 1, len(unique_comms)))
        node_color_map = [palette[unique_comms.index(c)] for c in node_colors_indices]
        
        nx.draw(G, pos, node_color=node_color_map, node_size=400, 
               with_labels=True, font_size=8, edge_color='gray', alpha=0.7)
        plt.axis('off')
        return self._save_or_show_matplotlib_plot(save_path)
    
    def plot_community_distribution(self, figsize=(12, 6), save_path=None):
        """Plot community size distributions across methods."""
        data = []
        for sample_id in self.results:
            for modeling_method in self.results[sample_id]:
                for comm_method in self.results[sample_id][modeling_method]:
                    for hp_id in self.results[sample_id][modeling_method][comm_method]:
                        communities_data = self.results[sample_id][modeling_method][comm_method][hp_id]['communities']
                        comm_counts = Counter(communities_data.values() if isinstance(communities_data, dict) else communities_data)
                        for size_val in list(comm_counts.values()):
                            data.append({'Size': size_val, 'Method': f"{comm_method}_{hp_id}"})
        
        if not data:
            print("No community results found for distribution plot.")
            return None
        
        self._setup_plot(figsize, 'Community Size Distribution')
        df = pd.DataFrame(data)
        sns.boxplot(data=df, x='Method', y='Size')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return self._save_or_show_matplotlib_plot(save_path)
    

    def compare_adjacency_matrices(self, sample_id, methods=None, figsize=(15, 10), save_path=None):
        """Compare adjacency matrices side by side."""
        if sample_id not in self.adjacencies:
            print(f"No adjacencies for {sample_id}")
            return None
        
        methods_to_plot = methods or list(self.adjacencies[sample_id].keys())[:4]
        
        # Determine grid size (e.g., 2x2 for up to 4, 1xN for fewer)
        num_methods = len(methods_to_plot)
        if num_methods == 0:
            print("No methods specified or found for comparison.")
            return None
        
        cols = 2 if num_methods > 1 else 1
        rows = (num_methods + cols - 1) // cols # Calculate rows needed, ensuring it's at least 1
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()
        
        for i, method_name in enumerate(methods_to_plot):
            if i >= rows * cols: break # Ensure we don't exceed subplot capacity
            ax = axes_flat[i]
            if method_name in self.adjacencies[sample_id]:
                sns.heatmap(self.adjacencies[sample_id][method_name], ax=ax, 
                           cmap='viridis', cbar=True, square=True)
                ax.set_title(method_name)
            else:
                ax.set_title(f'{method_name} (Not found)')
                ax.axis('off') # Hide axis if data not found
        
        # Hide unused subplots
        for j in range(num_methods, rows * cols):
            axes_flat[j].set_visible(False)
        
        fig.suptitle(f'Matrix Comparison - {sample_id}')
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle
        return self._save_or_show_matplotlib_plot(save_path)
    
    def plot_interactive_comparison(self, sample_id, save_path=None):
        """Interactive method comparison using Plotly."""
        if sample_id not in self.results:
            print(f"No results found for sample {sample_id} for interactive comparison.")
            return None
        
        data = []
        for modeling_method in self.results[sample_id]:
            for comm_method in self.results[sample_id][modeling_method]:
                for hp_id in self.results[sample_id][modeling_method][comm_method]:
                    communities_data = self.results[sample_id][modeling_method][comm_method][hp_id]['communities']
                    comm_counts = Counter(communities_data.values() if isinstance(communities_data, dict) else communities_data)
                    sizes = list(comm_counts.values())
                    
                    data.append({
                        'Modeling': modeling_method,
                        'Community': comm_method,
                        'HP': hp_id,
                        'N_Communities': len(set(communities_data.values() if isinstance(communities_data, dict) else communities_data)),
                        'Avg_Size': np.mean(sizes) if sizes else 0,
                        'Max_Size': max(sizes) if sizes else 0
                    })
        
        if not data:
            print("No data to plot for interactive comparison.")
            return None
        
        df = pd.DataFrame(data)
        fig = px.scatter(df, x='N_Communities', y='Avg_Size', color='Community',
                        symbol='Modeling', size='Max_Size', hover_data=['HP'],
                        title=f'Method Comparison - {sample_id}')
        
        if save_path:
            if save_path.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
                fig.write_image(save_path)
            else:
                fig.write_html(save_path) # Default to HTML for interactive plots
        else:
            fig.show()
        return fig


    def summary(self):
        """Get concise summary of available data."""
        summary_dict = {
            'samples': len(self.results),
            'sample_ids': list(self.results.keys()),
            'adjacency_matrices': sum(len(adj) for adj in self.adjacencies.values()),
            'modeling_methods': len(set().union(*[list(self.adjacencies[s].keys()) for s in self.adjacencies])),
            'modeling_method_names': list(set().union(*[list(self.adjacencies[s].keys()) for s in self.adjacencies])),
            'community_methods': len(set().union(*[list(self.results[s][m].keys()) 
                                                  for s in self.results for m in self.results[s]])),
            'community_method_names': list(set().union(*[list(self.results[s][m].keys()) 
                                                  for s in self.results for m in self.results[s]])),
            'total_results': sum(len(self.results[s][m][c]) 
                               for s in self.results for m in self.results[s] for c in self.results[s][m])
        }
        return summary_dict