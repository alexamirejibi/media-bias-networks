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
    
    # ===== METHOD SIMILARITY =====
    
    def plot_method_similarity(self, sample_id: str, network_method: str,
                              metric: str = 'ari', figsize: tuple = (10, 8)) -> None:
        """plot heatmap showing agreement between community detection methods"""
        similarity_matrix = self.analyzer.calculate_method_similarity(sample_id, network_method, metric)
        
        if not self._check_data(similarity_matrix, f"no similarity data for {sample_id}, {network_method}"):
            return
        
        self._setup_plot(figsize)
        sns.heatmap(similarity_matrix, annot=True, cmap='RdYlBu', center=0.5,
                   square=True, linewidths=0.5, cbar_kws={'label': f'{metric.upper()} Score'})
        self._finalize_plot(f'Method Agreement ({metric.upper()})\n{sample_id} | {network_method}')
        
        # print summary stats
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        if mask.any():
            self._print_stats(similarity_matrix.values[mask], "Method agreement")
    
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
    
    def plot_method_comparison(self, figsize: tuple = (12, 8)) -> None:
        """plot comprehensive method performance comparison with separate figures"""
        performance = self.analyzer.compare_method_performance()
        if not self._check_data(performance, "no performance data available"):
            return

        # ---- average communities heat-map ----
        performance_reset = performance.reset_index()
        heatmap_data = performance_reset.pivot(index='network_method',
                                               columns='community_method',
                                               values='avg_communities')
        self._setup_plot(figsize)
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='Blues',
                    cbar_kws={'label': 'Avg Communities'})
        self._finalize_plot('Average Communities Found')

        # ---- stability heat-map ----
        stability = self.analyzer.analyze_stability()
        self._setup_plot(figsize)
        if self._check_data(stability, ""):
            stability_agg = stability.groupby(['network_method', 'community_method'])['stability_score'].mean().reset_index()
            stability_heatmap_data = stability_agg.pivot(index='network_method',
                                                         columns='community_method',
                                                         values='stability_score')
            sns.heatmap(stability_heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                        cbar_kws={'label': 'Stability Score'})
        else:
            plt.text(0.5, 0.5, 'No Stability\nData Available',
                     ha='center', va='center', fontsize=14)
        self._finalize_plot('Method Stability')
    
    # ===== OUTLET GROUPINGS =====
    
    def plot_outlet_cooccurrence(self, min_frequency: float = 0.5, dataset: str = None,
                                figsize: tuple = (12, 10)) -> None:
        """plot heatmap showing which outlets frequently cluster together"""
        cooccurrence = self.analyzer.outlet_clustering_frequency(
            min_frequency=min_frequency, dataset=dataset)
        
        if not self._check_data(cooccurrence, "no outlet cooccurrence data available"):
            return
        
        self._setup_plot(figsize)
        
        # cluster outlets by similarity
        distance_matrix = 1 - cooccurrence.values
        np.fill_diagonal(distance_matrix, 0)
        condensed_distances = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distances, method='ward')
        dendro = dendrogram(linkage_matrix, labels=cooccurrence.index, no_plot=True)
        
        # reorder matrix by clustering
        order = dendro['leaves']
        cooccurrence_ordered = cooccurrence.iloc[order, order]
        
        sns.heatmap(cooccurrence_ordered, annot=False, cmap='Reds', 
                   square=True, linewidths=0.1, cbar_kws={'label': 'Co-clustering Frequency'},
                   xticklabels=True, yticklabels=True)
        
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        
        title = f'Outlet Co-clustering Frequency (min: {min_frequency:.1%})'
        if dataset:
            title += f' - {dataset} dataset'
        self._finalize_plot(title)
        
        # print summary
        frequencies = cooccurrence.values[np.triu_indices_from(cooccurrence.values, k=1)]
        above_threshold = frequencies[frequencies >= min_frequency]
        
        print(f"\nOutlet Co-clustering Analysis:")
        print(f"Outlets analyzed: {len(cooccurrence)}")
        print(f"Pairs above {min_frequency:.1%} threshold: {len(above_threshold)}")
        if len(above_threshold) > 0:
            print(f"Mean frequency: {above_threshold.mean():.3f}")
            print(f"Max frequency: {above_threshold.max():.3f}")
    
    def plot_stable_outlet_groups(self, frequency_threshold: float = 0.7, 
                                 min_group_size: int = 2, figsize: tuple = (12, 8)) -> None:
        """visualize stable outlet communities as network graph"""
        stable_groups = self.analyzer.find_stable_outlet_groups(
            frequency_threshold=frequency_threshold, 
            min_group_size=min_group_size
        )
        
        if not self._check_data(stable_groups, f"no stable groups found with threshold {frequency_threshold:.1%}"):
            return
        
        # create network graph
        G = nx.Graph()
        colors = plt.cm.Set3(np.linspace(0, 1, len(stable_groups)))
        node_colors = {}
        
        for i, (group_name, outlets) in enumerate(stable_groups.items()):
            color = colors[i]
            for outlet in outlets:
                G.add_node(outlet)
                node_colors[outlet] = color
            
            # fully connect group members
            for j, outlet1 in enumerate(outlets):
                for outlet2 in outlets[j+1:]:
                    G.add_edge(outlet1, outlet2)
        
        self._setup_plot(figsize)
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        node_color_list = [node_colors[node] for node in G.nodes()]
        
        nx.draw(G, pos, node_color=node_color_list, node_size=800,
               with_labels=True, font_size=10, font_weight='bold',
               edge_color='gray', alpha=0.8, width=2)
        
        plt.axis('off')
        self._finalize_plot(f'Stable Outlet Groups (frequency ≥ {frequency_threshold:.1%}, size ≥ {min_group_size})')
        
        # print group details
        print(f"\nStable Outlet Groups Found ({len(stable_groups)}):")
        print("-" * 50)
        for i, (group_name, outlets) in enumerate(stable_groups.items(), 1):
            print(f"{i}. {group_name}: {len(outlets)} outlets")
            print(f"   Members: {', '.join(outlets)}\n")
    
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
    
    def plot_analysis_summary(self, figsize: tuple = (12, 8)) -> None:
        """generate and save a concise set of summary plots"""
        if not self._check_data(self.analyzer.results_df, "no results to visualize"):
            return

        # 1) top methods by communities found
        performance = self.analyzer.compare_method_performance()
        if not performance.empty:
            self._setup_plot(figsize)
            performance['avg_communities'].head(10).plot(kind='bar', color='lightblue', alpha=0.8)
            plt.title('Top 10 Methods by Communities Found')
            plt.ylabel('Average Communities')
            plt.xticks(rotation=45)
            self._finalize_plot('top 10 methods by communities found')

        # 2) stability ranking
        stability = self.analyzer.analyze_stability()
        if not stability.empty:
            self._setup_plot(figsize)
            stability.head(10)['stability_score'].plot(kind='bar', color='lightgreen', alpha=0.8)
            plt.title('Top 10 Most Stable Methods')
            plt.ylabel('Stability Score')
            plt.xticks(rotation=45)
            self._finalize_plot('top 10 most stable methods')

        # 3) dataset comparison
        datasets = self.analyzer.results_df.get('dataset', pd.Series()).unique()
        if len(datasets) > 1:
            self._setup_plot(figsize)
            dataset_summary = self.analyzer.results_df.groupby('dataset')['n_communities'].mean()
            dataset_summary.plot(kind='bar', color='orange', alpha=0.8)
            plt.title('Average Communities by Dataset')
            plt.ylabel('Average Communities')
            plt.xticks(rotation=0)
            self._finalize_plot('average communities by dataset')
        else:
            self._setup_plot((6, 4))
            plt.text(0.5, 0.5, 'Single Dataset\nAnalysis', ha='center', va='center', fontsize=14)
            self._finalize_plot('dataset analysis')

        # 4) outlet co-clustering distribution
        cooccurrence = self.analyzer.outlet_clustering_frequency(min_frequency=0.3)
        self._setup_plot(figsize)
        if not cooccurrence.empty:
            frequencies = cooccurrence.values[np.triu_indices_from(cooccurrence.values, k=1)]
            frequencies = frequencies[frequencies > 0]
            plt.hist(frequencies, bins=20, alpha=0.7, color='purple', edgecolor='black')
            plt.title('Distribution of Outlet Co-clustering')
            plt.xlabel('Co-clustering Frequency')
            plt.ylabel('Number of Outlet Pairs')
        else:
            plt.text(0.5, 0.5, 'No Outlet\nData Available', ha='center', va='center')
            plt.title('Outlet Analysis')
        self._finalize_plot('outlet co-clustering distribution')

        # print overall summary
        self._print_analysis_summary(stability)
    
    def _print_analysis_summary(self, stability: pd.DataFrame) -> None:
        """print overall analysis summary"""
        summary = self.analyzer.summary()
        print(f"\nANALYSIS SUMMARY")
        print("=" * 40)
        print(f"Total results: {summary['total_results']}")
        print(f"Methods tested: {summary['network_methods']} network × {summary['community_methods']} community")
        print(f"Average communities found: {summary['avg_communities']:.1f}")
        
        if not stability.empty:
            best_method = stability.iloc[0]
            print(f"Most stable method: {best_method['network_method']} + {best_method['community_method']} "
                  f"(score: {best_method['stability_score']:.3f})")
        
        stable_groups = self.analyzer.find_stable_outlet_groups(frequency_threshold=0.6)
        print(f"Stable outlet groups: {len(stable_groups)}")
        print("=" * 40)
    
    # ===== NETWORK VISUALIZATION METHODS =====
    
    def plot_significant_connections_network(self, high_pairs: List[Dict], low_pairs: List[Dict], 
                                           outlet_names: List[str], figsize: tuple = (16, 12)) -> None:
        """create network graph from significant connections"""
        print("\\n=== NETWORK GRAPH OF SIGNIFICANT CONNECTIONS ===")
        
        # create network graph from significant connections
        G = nx.Graph()
        
        # add all outlets as nodes
        for outlet in outlet_names:
            G.add_node(outlet)
        
        # add edges for significant connections
        high_edges = []
        low_edges = []
        
        # add significantly HIGH co-clustering pairs
        for pair in high_pairs:
            outlet1, outlet2 = pair['outlet1'], pair['outlet2']
            weight = pair['observed']
            p_val = pair['p_value']
            
            G.add_edge(outlet1, outlet2, weight=weight, p_value=p_val, 
                       edge_type='high', deviation=pair['deviation'])
            high_edges.append((outlet1, outlet2))
        
        # add significantly LOW co-clustering pairs  
        for pair in low_pairs:
            outlet1, outlet2 = pair['outlet1'], pair['outlet2']
            weight = pair['observed']
            p_val = pair['p_value']
            
            G.add_edge(outlet1, outlet2, weight=weight, p_value=p_val, 
                       edge_type='low', deviation=pair['deviation'])
            low_edges.append((outlet1, outlet2))
        
        print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        print(f"  Significantly HIGH co-clustering edges: {len(high_edges)}")
        print(f"  Significantly LOW co-clustering edges: {len(low_edges)}")
        
        # create the network visualization
        self._setup_plot(figsize)
        
        # use spring layout for better visualization
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=COLORS['primary'], 
                               node_size=800, alpha=0.8, edgecolors=COLORS['text'])
        
        # draw edges with different colors for high vs low significance
        if high_edges:
            nx.draw_networkx_edges(G, pos, edgelist=high_edges, 
                                  edge_color=COLORS['secondary'], width=2, alpha=0.7,
                                  label='Significantly HIGH co-clustering')
        
        if low_edges:
            nx.draw_networkx_edges(G, pos, edgelist=low_edges, 
                                  edge_color=COLORS['quaternary'], width=1, alpha=0.7,
                                  style='dashed', label='Significantly LOW co-clustering')
        
        # draw labels
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', 
                               font_color=COLORS['text'])
        
        plt.title('Network Graph of Statistically Significant Co-clustering Relationships\\n' +
                  f'({len(high_edges)} high-significance, {len(low_edges)} low-significance connections)',
                  fontsize=14, fontweight='bold', pad=20)
        
        # add legend
        if high_edges or low_edges:
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'significant_connections_network.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # analyze network properties
        print(f"\\nNetwork Analysis:")
        print(f"  Nodes (outlets): {G.number_of_nodes()}")
        print(f"  Edges (significant connections): {G.number_of_edges()}")
        print(f"  Network density: {nx.density(G):.3f}")
        print(f"  Connected components: {nx.number_connected_components(G)}")
        
        # identify highly connected outlets
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        
        print(f"\\nMost connected outlets (by degree centrality):")
        for outlet, centrality in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]:
            degree = G.degree(outlet)
            print(f"  {outlet}: {degree} connections (centrality: {centrality:.3f})")
        
        print(f"\\nMost influential outlets (by betweenness centrality):")
        for outlet, centrality in sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]:
            if centrality > 0:
                print(f"  {outlet}: {centrality:.3f}")
        
        return G
    
    def plot_high_clustering_network(self, high_pairs: List[Dict], outlet_names: List[str], 
                                   figsize: tuple = (16, 12)) -> None:
        """create focused network showing only high co-clustering relationships"""
        print("\\n=== FOCUSED GRAPH: HIGH CO-CLUSTERING RELATIONSHIPS ONLY ===")
        
        if not high_pairs:
            print("No significantly high co-clustering pairs found to visualize")
            return
            
        # create network with only high co-clustering relationships
        G_high_only = nx.Graph()
        
        # add all outlets as nodes first
        for outlet in outlet_names:
            G_high_only.add_node(outlet)
        
        # add only significantly HIGH co-clustering edges
        high_edges_only = []
        for pair in high_pairs:
            outlet1, outlet2 = pair['outlet1'], pair['outlet2']
            weight = pair['observed']
            p_val = pair['p_value']
            
            G_high_only.add_edge(outlet1, outlet2, weight=weight, p_value=p_val, 
                                 deviation=pair['deviation'])
            high_edges_only.append((outlet1, outlet2))
        
        print(f"High co-clustering network: {G_high_only.number_of_nodes()} nodes, {G_high_only.number_of_edges()} edges")
        
        # create layout optimized for this network
        self._setup_plot(figsize)
        
        # use different layout depending on network density
        if nx.density(G_high_only) > 0.3:
            pos_high = nx.kamada_kawai_layout(G_high_only)
        else:
            pos_high = nx.spring_layout(G_high_only, k=2, iterations=100, seed=42)
        
        # calculate node sizes based on degree (number of high-significance connections)
        degrees = dict(G_high_only.degree())
        node_sizes = [degrees[node] * 150 + 400 for node in G_high_only.nodes()]
        
        # create color mapping based on degree
        max_degree = max(degrees.values()) if degrees.values() else 1
        node_colors = [degrees[node] / max_degree for node in G_high_only.nodes()]
        
        # draw nodes with size and color based on connectivity
        nodes = nx.draw_networkx_nodes(G_high_only, pos_high, 
                                      node_size=node_sizes,
                                      node_color=node_colors,
                                      cmap=plt.cm.Blues,
                                      alpha=0.8, 
                                      edgecolors=COLORS['text'],
                                      linewidths=2)
        
        # draw edges with thickness based on co-clustering strength
        edge_weights = [G_high_only[u][v]['weight'] for u, v in high_edges_only]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [3 * (w / max_weight) + 1 for w in edge_weights]  # scale edge thickness
        
        nx.draw_networkx_edges(G_high_only, pos_high, edgelist=high_edges_only,
                              edge_color=COLORS['secondary'], width=edge_widths, 
                              alpha=0.7)
        
        # draw labels
        nx.draw_networkx_labels(G_high_only, pos_high, font_size=10, font_weight='bold', 
                               font_color=COLORS['text'])
        
        plt.title('Media Outlets with Statistically Higher Than Expected Co-clustering\\n' +
                  f'({len(high_edges_only)} significant relationships, α = 0.05)',
                  fontsize=16, fontweight='bold', pad=20)
        
        # add colorbar for node colors
        if max_degree > 1:
            cbar = plt.colorbar(nodes, shrink=0.8, pad=0.1)
            cbar.set_label('Number of High-Significance Connections', fontweight='bold')
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'high_coclustering_network.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # analyze the high co-clustering network
        print(f"\\nHigh Co-clustering Network Analysis:")
        print(f"  Network density: {nx.density(G_high_only):.3f}")
        print(f"  Connected components: {nx.number_connected_components(G_high_only)}")
        
        # identify most connected outlets in high co-clustering
        high_degree_centrality = nx.degree_centrality(G_high_only)
        
        print(f"\\nOutlets with most high-significance connections:")
        for outlet, centrality in sorted(high_degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]:
            degree = G_high_only.degree(outlet)
            if degree > 0:
                print(f"  {outlet}: {degree} high-significance connections")
        
        # community detection on high co-clustering network
        if G_high_only.number_of_edges() > 0:
            # remove isolated nodes (no edges)
            G_connected = G_high_only.copy()
            isolated_nodes = [node for node in G_connected.nodes() if G_connected.degree(node) == 0]
            G_connected.remove_nodes_from(isolated_nodes)
            
            print(f"Removed {len(isolated_nodes)} isolated nodes for community detection")
            
            try:
                from networkx.algorithms import community
                communities_high = community.greedy_modularity_communities(G_connected)
                modularity = community.modularity(G_connected, communities_high)
                
                print(f"\\nCommunity Detection Results:")
                print(f"  Found {len(communities_high)} communities (modularity: {modularity:.3f})")
                for i, comm in enumerate(communities_high, 1):
                    if len(comm) > 1:
                        print(f"  Community {i}: {', '.join(sorted(comm))}")
                
                # visualize communities (only connected nodes)
                self._setup_plot((14, 10))
                pos_comm = nx.spring_layout(G_connected, k=2, iterations=100, seed=42)
                
                # assign colors to communities
                colors = plt.cm.Set3(np.linspace(0, 1, len(communities_high)))
                node_colors = []
                for node in G_connected.nodes():
                    for i, comm in enumerate(communities_high):
                        if node in comm:
                            node_colors.append(colors[i])
                            break
                
                # draw network with community colors
                nx.draw_networkx_nodes(G_connected, pos_comm, node_color=node_colors, 
                                     node_size=600, alpha=0.8, edgecolors='black')
                nx.draw_networkx_edges(G_connected, pos_comm, alpha=0.6, edge_color='gray')
                nx.draw_networkx_labels(G_connected, pos_comm, font_size=9, font_weight='bold')
                
                plt.title(f'Community Structure in High Co-clustering Network\\n'
                         f'({len(communities_high)} communities, modularity = {modularity:.3f})',
                         fontweight='bold', fontsize=14)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'high_coclustering_communities.png'), 
                            dpi=300, bbox_inches='tight')
                plt.show()
                
            except Exception as e:
                print(f"Community detection failed: {e}")
        
        # show strongest co-clustering pairs
        print(f"\\nStrongest high co-clustering relationships:")
        high_pairs_sorted = sorted(high_pairs, key=lambda x: x['observed'], reverse=True)
        for i, pair in enumerate(high_pairs_sorted[:10], 1):
            print(f"  {i:2d}. {pair['outlet1']} ↔ {pair['outlet2']}: "
                  f"frequency={pair['observed']:.3f}, p={pair['p_value']:.6f}")
        
        return G_high_only 