"""
Streamlined visualization tools for media bias network analysis
Focused on core research visualizations for publication and analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
from typing import Optional, List, Dict
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from .analysis import ResultsAnalyzer

# set plotting defaults
plt.style.use('default')
sns.set_palette('husl')


class Visualizer:
    """focused visualization methods for core research questions"""
    
    def __init__(self, analyzer: ResultsAnalyzer):
        self.analyzer = analyzer
    
    # ===== HELPER METHODS =====
    
    def _setup_plot(self, figsize: tuple) -> None:
        """setup standard plot formatting"""
        plt.figure(figsize=figsize)
    
    def _finalize_plot(self, title: str) -> None:
        """apply final formatting and show plot"""
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
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
    
    def plot_method_comparison(self, figsize: tuple = (12, 12)) -> None:
        """plot comprehensive method performance comparison"""
        performance = self.analyzer.compare_method_performance()
        
        if not self._check_data(performance, "no performance data available"):
            return
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # communities heatmap
        performance_reset = performance.reset_index()
        heatmap_data = performance_reset.pivot(index='network_method', 
                                             columns='community_method', 
                                             values='avg_communities')
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='Blues', 
                   ax=axes[0], cbar_kws={'label': 'Avg Communities'})
        axes[0].set_title('Average Communities Found')
        
        # stability heatmap
        stability = self.analyzer.analyze_stability()
        if self._check_data(stability, ""):
            stability_agg = stability.groupby(['network_method', 'community_method'])['stability_score'].mean().reset_index()
            stability_heatmap_data = stability_agg.pivot(index='network_method',
                                                       columns='community_method',
                                                       values='stability_score')
            
            sns.heatmap(stability_heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                       ax=axes[1], cbar_kws={'label': 'Stability Score'})
            axes[1].set_title('Method Stability')
        else:
            axes[1].text(0.5, 0.5, 'No Stability\nData Available', 
                        ha='center', va='center', transform=axes[1].transAxes, fontsize=14)
            axes[1].set_title('Method Stability')
        
        plt.tight_layout()
        plt.show()
    
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
    
    def plot_modularity_analysis(self, figsize: tuple = (12, 16)) -> None:
        """analyze modularity distributions and relationships"""
        df = self.analyzer.get_results()
        
        if not self._check_data(df, "no results available for modularity analysis"):
            return
        
        mod_data = df.dropna(subset=['modularity'])
        if not self._check_data(mod_data, "no modularity data available"):
            return
        
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # overall modularity distribution
        mod_data['modularity'].hist(bins=30, alpha=0.7, color='skyblue', ax=axes[0])
        axes[0].axvline(mod_data['modularity'].mean(), color='red', linestyle='--', 
                         label=f'mean: {mod_data["modularity"].mean():.3f}')
        axes[0].set_xlabel('modularity')
        axes[0].set_ylabel('frequency')
        axes[0].set_title('modularity distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # modularity vs communities scatter
        scatter = axes[1].scatter(mod_data['n_communities'], mod_data['modularity'], 
                                   alpha=0.6, c=mod_data['n_communities'], cmap='viridis')
        axes[1].set_xlabel('number of communities (k)')
        axes[1].set_ylabel('modularity')
        axes[1].set_title('modularity vs k')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1])
        
        # modularity by k (box plots)
        k_values = sorted(mod_data['n_communities'].unique())
        modularity_by_k = [mod_data[mod_data['n_communities'] == k]['modularity'].values for k in k_values]
        bp = axes[2].boxplot(modularity_by_k, labels=k_values, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[2].set_xlabel('number of communities (k)')
        axes[2].set_ylabel('modularity')
        axes[2].set_title('modularity by k')
        axes[2].grid(True, alpha=0.3)
        
        # method combination heatmap
        pivot_table = mod_data.pivot_table(values='modularity', index='network_method', 
                                          columns='community_method', aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis', ax=axes[3])
        axes[3].set_title('mean modularity by method')
        
        plt.tight_layout()
        plt.show()
        
        # print stats
        self._print_stats(mod_data['modularity'].values, "modularity")
        correlation = mod_data['n_communities'].corr(mod_data['modularity'])
        print(f"k-modularity correlation: {correlation:.3f}")
    
    def plot_k_distribution_analysis(self, figsize: tuple = (12, 16)) -> None:
        """analyze distribution of community counts across methods"""
        df = self.analyzer.get_results()
        
        if not self._check_data(df, "no results available for k analysis"):
            return
        
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # k distribution histogram
        k_range = range(int(df['n_communities'].min()), int(df['n_communities'].max()) + 2)
        axes[0].hist(df['n_communities'], bins=k_range, alpha=0.7, edgecolor='black')
        axes[0].axvline(df['n_communities'].mean(), color='red', linestyle='--', 
                         label=f'mean: {df["n_communities"].mean():.1f}')
        axes[0].set_xlabel('number of communities (k)')
        axes[0].set_ylabel('frequency')
        axes[0].set_title('k distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # k by network method
        network_methods = df['network_method'].unique()
        k_by_network = [df[df['network_method'] == method]['n_communities'] for method in network_methods]
        axes[1].boxplot(k_by_network, labels=network_methods)
        axes[1].set_ylabel('number of communities (k)')
        axes[1].set_title('k by network method')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # k swarmplot by community method (colored by network method)
        sns.swarmplot(data=df, x='community_method', y='n_communities', 
                     hue='network_method', ax=axes[2], size=4, alpha=0.7)
        axes[2].set_ylabel('number of communities (k)')
        axes[2].set_xlabel('community method')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].set_title('k by community method (swarmplot)')
        axes[2].legend(title='network method', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2].grid(True, alpha=0.3)
        
        # k swarmplot by network method (colored by community method)
        sns.swarmplot(data=df, x='network_method', y='n_communities', 
                     hue='community_method', ax=axes[3], size=4, alpha=0.7)
        axes[3].set_ylabel('number of communities (k)')
        axes[3].set_xlabel('network method')
        axes[3].tick_params(axis='x', rotation=45)
        axes[3].set_title('k by network method (swarmplot)')
        axes[3].legend(title='community method', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # print stats
        self._print_stats(df['n_communities'].values, "k values")
        mode_k = df['n_communities'].mode().iloc[0]
        print(f"most common k: {mode_k} ({(df['n_communities'] == mode_k).sum()} occurrences)")
    
    def plot_k_modularity_relationship(self, figsize: tuple = (12, 8)) -> None:
        """analyze relationship between k and modularity across methods"""
        df = self.analyzer.get_results()
        mod_data = df.dropna(subset=['modularity'])
        
        if not self._check_data(mod_data, "no modularity data for k-modularity analysis"):
            return
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # k vs modularity by community method
        community_methods = mod_data['community_method'].unique()
        for method in community_methods:
            method_data = mod_data[mod_data['community_method'] == method]
            axes[0].scatter(method_data['n_communities'], method_data['modularity'], 
                             alpha=0.7, label=method, s=30)
        axes[0].set_xlabel('number of communities (k)')
        axes[0].set_ylabel('modularity')
        axes[0].set_title('k vs modularity by community method')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # modularity boxplot by k
        k_values = sorted(mod_data['n_communities'].unique())
        modularity_by_k = [mod_data[mod_data['n_communities'] == k]['modularity'].values for k in k_values]
        axes[1].boxplot(modularity_by_k, labels=k_values)
        axes[1].set_xlabel('number of communities (k)')
        axes[1].set_ylabel('modularity')
        axes[1].set_title('modularity distribution by k')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # print correlation statistics
        correlation = mod_data['n_communities'].corr(mod_data['modularity'])
        print(f"\nk-modularity correlation: {correlation:.3f}")
    
    def plot_analysis_summary(self, figsize: tuple = (12, 16)) -> None:
        """comprehensive summary plot combining all core analyses"""
        if not self._check_data(self.analyzer.results_df, "no results to visualize"):
            return
        
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # method performance
        performance = self.analyzer.compare_method_performance()
        if not performance.empty:
            performance['avg_communities'].head(10).plot(kind='bar', ax=axes[0], 
                                                        color='lightblue', alpha=0.8)
            axes[0].set_title('Top 10 Methods by Communities Found')
            axes[0].set_ylabel('Average Communities')
            axes[0].tick_params(axis='x', rotation=45)
        
        # stability ranking
        stability = self.analyzer.analyze_stability()
        if not stability.empty:
            stability.head(10)['stability_score'].plot(kind='bar', ax=axes[1], 
                                                      color='lightgreen', alpha=0.8)
            axes[1].set_title('Top 10 Most Stable Methods')
            axes[1].set_ylabel('Stability Score')
            axes[1].tick_params(axis='x', rotation=45)
        
        # dataset comparison
        datasets = self.analyzer.results_df.get('dataset', pd.Series()).unique()
        if len(datasets) > 1:
            dataset_summary = self.analyzer.results_df.groupby('dataset')['n_communities'].mean()
            dataset_summary.plot(kind='bar', ax=axes[2], color='orange', alpha=0.8)
            axes[2].set_title('Average Communities by Dataset')
            axes[2].set_ylabel('Average Communities')
            axes[2].tick_params(axis='x', rotation=0)
        else:
            axes[2].text(0.5, 0.5, 'Single Dataset\nAnalysis', 
                          ha='center', va='center', transform=axes[2].transAxes, fontsize=14)
            axes[2].set_title('Dataset Analysis')
        
        # outlet clustering summary
        cooccurrence = self.analyzer.outlet_clustering_frequency(min_frequency=0.3)
        if not cooccurrence.empty:
            frequencies = cooccurrence.values[np.triu_indices_from(cooccurrence.values, k=1)]
            frequencies = frequencies[frequencies > 0]
            
            axes[3].hist(frequencies, bins=20, alpha=0.7, color='purple', edgecolor='black')
            axes[3].set_title('Distribution of Outlet Co-clustering')
            axes[3].set_xlabel('Co-clustering Frequency')
            axes[3].set_ylabel('Number of Outlet Pairs')
        else:
            axes[3].text(0.5, 0.5, 'No Outlet\nData Available', 
                          ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title('Outlet Analysis')
        
        plt.tight_layout()
        plt.show()
        
        # print summary
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