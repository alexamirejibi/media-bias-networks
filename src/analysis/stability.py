"""
Stability analysis methods - method consistency, cross-sample validation, and robustness testing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
from collections import Counter
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform


class StabilityAnalyzer:
    """Specialized analyzer for method stability and consistency analysis."""
    
    def __init__(self, core_analyzer):
        self.core = core_analyzer
    
    def analyze_stability(self, dataset: str = None) -> pd.DataFrame:
        """analyze method stability across samples"""
        
        # filter by dataset if specified
        df = self.core.results_df if dataset is None else self.core.results_df[self.core.results_df['dataset'] == dataset]
        
        if df.empty:
            return pd.DataFrame()
        
        # group by method combination and compute stability metrics
        stability = df.groupby(['network_method', 'community_method', 'param_id'])[
            'n_communities'
        ].agg(['mean', 'std', 'count']).reset_index()
        
        # stability score: lower variance = more stable
        stability['stability_score'] = 1 / (1 + stability['std'])
        stability = stability.sort_values('stability_score', ascending=False)
        
        return stability.round(3)
    
    def method_consistency(self, network_method: str, community_method: str, 
                          param_id: str, metric: str = 'ari') -> Dict[str, float]:
        """measure consistency of a method across different samples"""
        
        # get all samples with this method combination
        filtered = self.core.get_results({
            'network_method': network_method,
            'community_method': community_method,
            'param_id': param_id
        })
        
        if len(filtered) < 2:
            return {'error': 'need at least 2 samples for consistency analysis'}
        
        samples = filtered['sample_id'].unique()
        similarities = []
        
        # compute pairwise similarities between samples
        for i, sample1 in enumerate(samples):
            for sample2 in samples[i + 1:]:
                part1 = self.core._get_partition(sample1, network_method, community_method, param_id)
                part2 = self.core._get_partition(sample2, network_method, community_method, param_id)
                
                if part1 and part2:
                    similarity = self.core._compare_partitions(part1, part2, metric)
                    if similarity is not None:
                        similarities.append(similarity)
        
        if not similarities:
            return {'error': 'no valid comparisons found'}
        
        return {
            'mean_consistency': round(np.mean(similarities), 3),
            'std_consistency': round(np.std(similarities), 3),
            'n_comparisons': len(similarities)
        }
            
    def analyze_exclusions(self, colors: dict, min_communities: int = 2, max_communities: int = 48):
        """analyze and visualize method exclusions before applying filter"""
        print("\\n=== EXCLUSION ANALYSIS ===")

        # get results before exclusion
        all_results = self.core.get_results()
        excluded_results = all_results[(all_results['n_communities'] == 1) | (all_results['n_communities'] >= 49)]

        print(f"Total results: {len(all_results)}")
        print(f"Excluded results (k=1 or k=n_outlets): {len(excluded_results)} ({len(excluded_results)/len(all_results)*100:.1f}%)")

        # analyze exclusion by method type
        def calculate_exclusion_fractions(results_df, method_col):
            """calculate fraction of results excluded for each method"""
            method_stats = []
            
            for method in results_df[method_col].unique():
                method_results = results_df[results_df[method_col] == method]
                total = len(method_results)
                k1_count = len(method_results[method_results['n_communities'] == 1])
                k49plus_count = len(method_results[method_results['n_communities'] >= 49])
                excluded_count = k1_count + k49plus_count
                
                method_stats.append({
                    'method': method,
                    'total': total,
                    'k1_count': k1_count,
                    'k49plus_count': k49plus_count,
                    'excluded_count': excluded_count,
                    'excluded_fraction': excluded_count / total if total > 0 else 0,
                    'k1_fraction': k1_count / total if total > 0 else 0,
                    'k49plus_fraction': k49plus_count / total if total > 0 else 0
                })
            
            return pd.DataFrame(method_stats)

        # analyze by network method
        print("\\n--- Exclusion by Network Method ---")
        network_exclusions = calculate_exclusion_fractions(all_results, 'network_method')
        print(network_exclusions[['method', 'total', 'excluded_count', 'excluded_fraction']].round(3))

        # analyze by community method
        print("\\n--- Exclusion by Community Method ---")
        community_exclusions = calculate_exclusion_fractions(all_results, 'community_method')
        print(community_exclusions[['method', 'total', 'excluded_count', 'excluded_fraction']].round(3))

        # create visualization
        self._visualize_exclusions(network_exclusions, community_exclusions, colors)

        # apply the exclusion filter
        print(f"\\nApplying exclusion filter: keeping {min_communities} ≤ k ≤ {max_communities}")
        before_count = len(self.core.results_df)
        self.core.exclude_results(min_communities, max_communities)
        after_count = len(self.core.results_df)
        
        print(f"Results after exclusion: {after_count} (removed {before_count - after_count})")
        print(f"Retention rate: {after_count/before_count*100:.1f}%")

    def _visualize_exclusions(self, network_exclusions, community_exclusions, colors):
        """Create visualization for exclusion analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. network method exclusions (stacked bar)
        ax1 = axes[0]
        methods_net = network_exclusions['method']
        k1_fractions_net = network_exclusions['k1_fraction']
        k49_fractions_net = network_exclusions['k49plus_fraction']
        
        x_pos_net = np.arange(len(methods_net))
        ax1.bar(x_pos_net, k1_fractions_net, color=colors['exclusion_k1'], alpha=0.8, label='k=1 exclusions')
        ax1.bar(x_pos_net, k49_fractions_net, bottom=k1_fractions_net, color=colors['exclusion_k49'], alpha=0.8, label='k≥49 exclusions')
        
        ax1.set_xlabel('Network Method')
        ax1.set_ylabel('Exclusion Fraction')
        ax1.set_title('Exclusions by Network Method', fontweight='bold')
        ax1.set_xticks(x_pos_net)
        ax1.set_xticklabels(methods_net, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. community method exclusions (stacked bar)
        ax2 = axes[1]
        methods_comm = community_exclusions['method']
        k1_fractions_comm = community_exclusions['k1_fraction']
        k49_fractions_comm = community_exclusions['k49plus_fraction']
        
        x_pos_comm = np.arange(len(methods_comm))
        ax2.bar(x_pos_comm, k1_fractions_comm, color=colors['exclusion_k1'], alpha=0.8, label='k=1 exclusions')
        ax2.bar(x_pos_comm, k49_fractions_comm, bottom=k1_fractions_comm, color=colors['exclusion_k49'], alpha=0.8, label='k≥49 exclusions')
        
        ax2.set_xlabel('Community Method')
        ax2.set_ylabel('Exclusion Fraction')
        ax2.set_title('Exclusions by Community Method', fontweight='bold')
        ax2.set_xticks(x_pos_comm)
        ax2.set_xticklabels(methods_comm, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('results/exclusion_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'network_exclusions': network_exclusions,
            'community_exclusions': community_exclusions
        }
    