"""
Streamlined visualization tools for media bias network analysis
Focused on core research visualizations for publication and analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
import os
from typing import Optional, List, Dict
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from matplotlib.colors import LinearSegmentedColormap

from .analysis import ResultsAnalyzer

# =============================================================================
# CONSISTENT COLOR PALETTE SETUP
# =============================================================================

# define consistent color palette for all visualizations
COLORS = {
    # primary analysis colors
    'primary': '#2E86AB',      # blue for main data
    'secondary': '#A23B72',    # purple for secondary data
    'tertiary': '#F18F01',     # orange for tertiary/comparison
    'quaternary': '#C73E1D',   # red for exclusions/negative
    
    # specific semantic colors
    'exclusion_k1': '#C73E1D',        # red for k=1 exclusions
    'exclusion_k49': '#F18F01',       # orange for k=49 exclusions
    'frequency': '#2E86AB',           # blue for frequency data
    'entropy': '#A23B72',             # purple for entropy data
    'uncertainty_high': '#C73E1D',    # red for high uncertainty
    'uncertainty_med': '#F18F01',     # orange for medium uncertainty
    'uncertainty_low': '#2E86AB',     # blue for low uncertainty
    
    # statistical markers
    'mean': '#C73E1D',         # red for means
    'median': '#F18F01',       # orange for medians
    'quartiles': '#2E86AB',    # blue for quartiles
    
    # neutral colors
    'background': '#F7F7F7',   # light gray
    'grid': '#CCCCCC',         # gray for grids
    'text': '#333333'          # dark gray for text
}

# create color lists for multiple categories
CATEGORICAL_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3A6B35', '#F4B942', '#8E44AD', '#E67E22']
SEQUENTIAL_COLORS = ['#F7F7F7', '#BDD7E7', '#6BAED6', '#3182BD', '#08519C']  # light to dark blue
DIVERGING_COLORS = ['#C73E1D', '#F18F01', '#F7F7F7', '#6BAED6', '#2E86AB']  # red-orange-white-blue

# create custom colormaps
HEATMAP_CMAP = LinearSegmentedColormap.from_list('custom_heatmap', 
                                                ['#F7F7F7', '#F18F01', '#C73E1D'], N=256)
ENTROPY_CMAP = LinearSegmentedColormap.from_list('custom_entropy', 
                                                ['#F7F7F7', '#A23B72', '#2E86AB'], N=256)
DIVERGING_CMAP = LinearSegmentedColormap.from_list('custom_diverging', DIVERGING_COLORS, N=256)

# set style with consistent colors
plt.style.use('default')
plt.rcParams.update({
    'axes.prop_cycle': plt.cycler('color', CATEGORICAL_COLORS),
    'axes.facecolor': COLORS['background'],
    'figure.facecolor': 'white',
    'axes.edgecolor': COLORS['text'],
    'axes.labelcolor': COLORS['text'],
    'text.color': COLORS['text'],
    'xtick.color': COLORS['text'],
    'ytick.color': COLORS['text'],
    'grid.color': COLORS['grid'],
    'grid.alpha': 0.3
})

# set seaborn palette to match
sns.set_palette(CATEGORICAL_COLORS)


class Visualizer:
    """focused visualization methods for core research questions"""
    
    def __init__(self, analyzer: ResultsAnalyzer, output_dir: str = "figures"):
        self.analyzer = analyzer
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    # ===== HELPER METHODS =====
    
    def _setup_plot(self, figsize: tuple) -> None:
        """setup standard plot formatting"""
        plt.figure(figsize=figsize)
    
    def _finalize_plot(self, title: str, save: bool = True) -> None:
        """apply final formatting, save, and show plot"""
        plt.title(title)
        plt.tight_layout()
        if save:
            filename = self._sanitize_filename(title)
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.show()
        plt.close()
    
    def _print_stats(self, data: np.ndarray, name: str) -> None:
        """print summary statistics"""
        print(f"\n{name} summary:")
        print(f"mean: {data.mean():.3f}, std: {data.std():.3f}")
        print(f"min: {data.min():.3f}, max: {data.max():.3f}")
    
    def _check_data(self, data, message: str) -> bool:
        """check if data is available"""
        if data.empty if hasattr(data, 'empty') else not data:
            print(message)
            return False
        return True
    
    def _sanitize_filename(self, title: str) -> str:
        """convert plot title to a filesystem-safe filename"""
        safe = ''.join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in title).strip()
        return safe.replace(' ', '_').lower() + '.png'
    
    
    # ===== STABILITY ANALYSIS =====
    
    def plot_stability_ranking(self, top_n: int = 15, dataset: str = None,
                              figsize: tuple = (12, 8)) -> None:
        """plot ranking of most stable method combinations"""
        stability = self.analyzer.analyze_stability(dataset=dataset)
        
        if not self._check_data(stability, "no stability data available"):
            return
        
        top_methods = stability.head(top_n)
        self._setup_plot(figsize)
        
        # horizontal bar plot
        method_labels = [f"{row['network_method']}\n{row['community_method']}" 
                        for _, row in top_methods.iterrows()]
        bars = plt.barh(range(len(top_methods)), top_methods['stability_score'],
                       color='skyblue', alpha=0.8, edgecolor='black')
        
        plt.yticks(range(len(top_methods)), method_labels)
        plt.xlabel('Stability Score')
        plt.ylabel('Method Combinations')
        plt.xlim(0, 1)
        
        # add value labels
        for i, (bar, (_, row)) in enumerate(zip(bars, top_methods.iterrows())):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{row["stability_score"]:.3f}\n(n={row["count"]})',
                    ha='left', va='center', fontsize=9)
        
        title = f'Top {top_n} Most Stable Methods'
        if dataset:
            title += f' ({dataset} dataset)'
        self._finalize_plot(title)
        
        # print ranking
        print(f"\nTop {min(top_n, len(stability))} Most Stable Methods:")
        for i, (_, row) in enumerate(top_methods.iterrows(), 1):
            print(f"{i:2d}. {row['network_method']} + {row['community_method']}: "
                  f"{row['stability_score']:.3f}")
    
    
    # ===== SUMMARY VISUALIZATION =====
    
    def plot_modularity_analysis(self, figsize: tuple = (12, 8)) -> None:
        """analyze modularity distributions and relationships with separate figures"""
        df = self.analyzer.get_results()
        if not self._check_data(df, "no results available for modularity analysis"):
            return

        mod_data = df.dropna(subset=['modularity'])
        if not self._check_data(mod_data, "no modularity data available"):
            return

        # 1) modularity distribution
        self._setup_plot(figsize)
        mod_data['modularity'].hist(bins=30, alpha=0.7, color='skyblue')
        plt.axvline(mod_data['modularity'].mean(), color='red', linestyle='--',
                    label=f"mean: {mod_data['modularity'].mean():.3f}")
        plt.xlabel('modularity')
        plt.ylabel('frequency')
        plt.title('modularity distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        self._finalize_plot('modularity distribution')

        # 2) modularity vs k scatter
        self._setup_plot(figsize)
        scatter = plt.scatter(mod_data['n_communities'], mod_data['modularity'],
                              alpha=0.6, c=mod_data['n_communities'], cmap='viridis')
        plt.xlabel('number of communities (k)')
        plt.ylabel('modularity')
        plt.title('modularity vs k')
        plt.colorbar(scatter, label='k')
        plt.grid(True, alpha=0.3)
        self._finalize_plot('modularity vs k')

        # 3) modularity by k boxplot
        self._setup_plot(figsize)
        k_values = sorted(mod_data['n_communities'].unique())
        modularity_by_k = [mod_data[mod_data['n_communities'] == k]['modularity'].values for k in k_values]
        bp = plt.boxplot(modularity_by_k, labels=k_values, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        plt.xlabel('number of communities (k)')
        plt.ylabel('modularity')
        plt.title('modularity by k')
        plt.grid(True, alpha=0.3)
        self._finalize_plot('modularity by k')

        # 4) mean modularity by method heatmap
        self._setup_plot(figsize)
        pivot_table = mod_data.pivot_table(values='modularity', index='network_method',
                                           columns='community_method', aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis',
                    cbar_kws={'label': 'mean modularity'})
        self._finalize_plot('mean modularity by method')

        # print stats
        self._print_stats(mod_data['modularity'].values, 'modularity')
        correlation = mod_data['n_communities'].corr(mod_data['modularity'])
        print(f"k-modularity correlation: {correlation:.3f}")
    
    def plot_k_distribution_analysis(self, figsize: tuple = (12, 8)) -> None:
        """analyze distribution of community counts across methods with separate figures"""
        df = self.analyzer.get_results()
        if not self._check_data(df, "no results available for k analysis"):
            return

        # 1) k distribution histogram
        self._setup_plot(figsize)
        k_range = range(int(df['n_communities'].min()), int(df['n_communities'].max()) + 2)
        plt.hist(df['n_communities'], bins=k_range, alpha=0.7, edgecolor='black')
        plt.axvline(df['n_communities'].mean(), color='red', linestyle='--',
                    label=f"mean: {df['n_communities'].mean():.1f}")
        plt.xlabel('number of communities (k)')
        plt.ylabel('frequency')
        plt.title('k distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        self._finalize_plot('k distribution')

        # 2) k by network method boxplot
        self._setup_plot(figsize)
        network_methods = df['network_method'].unique()
        k_by_network = [df[df['network_method'] == method]['n_communities'] for method in network_methods]
        plt.boxplot(k_by_network, labels=network_methods)
        plt.ylabel('number of communities (k)')
        plt.title('k by network method')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        self._finalize_plot('k by network method')

        # 3) k by community method swarmplot
        self._setup_plot(figsize)
        sns.swarmplot(data=df, x='community_method', y='n_communities', hue='network_method', size=4, alpha=0.7)
        plt.ylabel('number of communities (k)')
        plt.xlabel('community method')
        plt.title('k by community method (swarmplot)')
        plt.xticks(rotation=45)
        plt.legend(title='network method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        self._finalize_plot('k by community method')

        # 4) k by network method (swarm colored by community method)
        self._setup_plot(figsize)
        sns.swarmplot(data=df, x='network_method', y='n_communities', hue='community_method', size=4, alpha=0.7)
        plt.ylabel('number of communities (k)')
        plt.xlabel('network method')
        plt.title('k by network method (swarmplot)')
        plt.xticks(rotation=45)
        plt.legend(title='community method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        self._finalize_plot('k by network method (swarmplot)')

        # print stats
        self._print_stats(df['n_communities'].values, 'k values')
        mode_k = df['n_communities'].mode().iloc[0]
        print(f"most common k: {mode_k} ({(df['n_communities'] == mode_k).sum()} occurrences)")
    
    def plot_k_modularity_relationship(self, figsize: tuple = (12, 8)) -> None:
        """analyze relationship between k and modularity across methods with separate figures"""
        df = self.analyzer.get_results()
        mod_data = df.dropna(subset=['modularity'])
        if not self._check_data(mod_data, "no modularity data for k-modularity analysis"):
            return

        # 1) k vs modularity scatter by community method
        self._setup_plot(figsize)
        community_methods = mod_data['community_method'].unique()
        for method in community_methods:
            method_data = mod_data[mod_data['community_method'] == method]
            plt.scatter(method_data['n_communities'], method_data['modularity'], alpha=0.7, label=method, s=30)
        plt.xlabel('number of communities (k)')
        plt.ylabel('modularity')
        plt.title('k vs modularity by community method')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        self._finalize_plot('k vs modularity by community method')

        # 2) modularity distribution by k boxplot
        self._setup_plot(figsize)
        k_values = sorted(mod_data['n_communities'].unique())
        modularity_by_k = [mod_data[mod_data['n_communities'] == k]['modularity'].values for k in k_values]
        plt.boxplot(modularity_by_k, labels=k_values)
        plt.xlabel('number of communities (k)')
        plt.ylabel('modularity')
        plt.title('modularity distribution by k')
        plt.grid(True, alpha=0.3)
        self._finalize_plot('modularity distribution by k')

        # print correlation statistics
        correlation = mod_data['n_communities'].corr(mod_data['modularity'])
        print(f"\nk-modularity correlation: {correlation:.3f}")