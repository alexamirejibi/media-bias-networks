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
    
    def calculate_method_similarity(self, sample_id: str, network_method: str, 
                                   metric: str = 'ari') -> pd.DataFrame:
        """calculate pairwise similarity between all community detection methods"""
        
        # get all methods for this sample/network combination
        filtered = self.core.get_results({
            'sample_id': sample_id,
            'network_method': network_method
        })
        
        if filtered.empty:
            return pd.DataFrame()
        
        # create method identifiers
        methods = []
        for _, row in filtered.iterrows():
            method_id = f"{row['community_method']}_{row['param_id']}"
            methods.append((method_id, row['community_method'], row['param_id']))
        
        # initialize similarity matrix
        n_methods = len(methods)
        similarity_matrix = np.zeros((n_methods, n_methods))
        method_names = [m[0] for m in methods]
        
        # compute pairwise similarities
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                part1 = self.core._get_partition(sample_id, network_method, methods[i][1], methods[i][2])
                part2 = self.core._get_partition(sample_id, network_method, methods[j][1], methods[j][2])
                
                if part1 and part2:
                    similarity = self.core._compare_partitions(part1, part2, metric)
                    if similarity is not None:
                        similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
        
        # diagonal is 1 (perfect similarity with self)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return pd.DataFrame(similarity_matrix, index=method_names, columns=method_names)
    
    def compare_method_performance(self) -> pd.DataFrame:
        """compare average performance across methods"""
        if self.core.results_df.empty:
            return pd.DataFrame()
        
        performance = self.core.results_df.groupby(['network_method', 'community_method']).agg({
            'n_communities': ['mean', 'std'],
            'largest_community': 'mean'
        }).round(2)
        
        performance.columns = ['avg_communities', 'std_communities', 'avg_largest']
        return performance.sort_values('avg_communities', ascending=False)
    
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
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. network method exclusions (stacked bar)
        ax1 = axes[0, 0]
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
        ax2 = axes[0, 1]
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

        # 3. network method total counts
        ax3 = axes[1, 0]
        total_counts = network_exclusions['total']
        x_pos_net = np.arange(len(network_exclusions))
        ax3.bar(x_pos_net, total_counts, color=colors['frequency'], alpha=0.8)
        ax3.set_xlabel('Network Method')
        ax3.set_ylabel('Total Results')
        ax3.set_title('Total Results by Network Method', fontweight='bold')
        ax3.set_xticks(x_pos_net)
        ax3.set_xticklabels(network_exclusions['method'], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. community method total counts
        ax4 = axes[1, 1]
        total_counts = community_exclusions['total']
        x_pos_comm = np.arange(len(community_exclusions))
        ax4.bar(x_pos_comm, total_counts, color=colors['frequency'], alpha=0.8)
        ax4.set_xlabel('Community Method')
        ax4.set_ylabel('Total Results')
        ax4.set_title('Total Results by Community Method', fontweight='bold')
        ax4.set_xticks(x_pos_comm)
        ax4.set_xticklabels(community_exclusions['method'], rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('results/exclusion_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'network_exclusions': network_exclusions,
            'community_exclusions': community_exclusions
        }
    
    def outlet_clustering_frequency(self, min_frequency: float = 0.0, 
                                   normalize: bool = True) -> pd.DataFrame:
        """calculate how frequently outlets cluster together across all analyses"""
        
        if self.core.results_df.empty or self.core.outlet_names is None:
            return pd.DataFrame()
        
        n_outlets = len(self.core.outlet_names)
        cooccurrence_counts = np.zeros((n_outlets, n_outlets))
        total_analyses = 0
        
        # count co-occurrences across all clustering results
        for _, row in self.core.results_df.iterrows():
            communities = row['communities']
            if not communities:
                continue
                
            total_analyses += 1
            
            # check all pairs of outlets
            for i in range(n_outlets):
                for j in range(i + 1, n_outlets):
                    outlet_i_cluster = communities.get(i)
                    outlet_j_cluster = communities.get(j)
                    
                    if (outlet_i_cluster is not None and 
                        outlet_j_cluster is not None and 
                        outlet_i_cluster == outlet_j_cluster):
                        cooccurrence_counts[i, j] += 1
                        cooccurrence_counts[j, i] += 1
        
        # normalize to get frequencies
        if normalize and total_analyses > 0:
            frequencies = cooccurrence_counts / total_analyses
        else:
            frequencies = cooccurrence_counts
        
        # apply threshold
        frequencies = np.where(frequencies >= min_frequency, frequencies, 0)
        
        return pd.DataFrame(frequencies, index=self.core.outlet_names, columns=self.core.outlet_names)
    
    def validate_null_distribution(self, n_permutations: int = 5000, 
                                 clustering_index: int = 0, 
                                 random_state: Optional[int] = None,
                                 save_path: Optional[str] = None, 
                                 show_plot: bool = True) -> Dict[str, Any]:
        """validate analytical null distribution against empirical permutation test"""
        from collections import Counter
        from scipy.stats import norm
        import os
        
        print("\\n=== NULL DISTRIBUTION VALIDATION ===")
        
        if self.core.results_df.empty or self.core.outlet_names is None:
            return {'error': 'no results available for validation'}
        
        # get a specific clustering for validation
        if clustering_index >= len(self.core.results_df):
            clustering_index = 0
        
        test_clustering = self.core.results_df.iloc[clustering_index]
        communities = test_clustering['communities']
        
        if not communities:
            return {'error': 'selected clustering has no communities'}
        
        print(f"validating with clustering {clustering_index}: {test_clustering['sample_id']} | "
              f"{test_clustering['network_method']} | {test_clustering['community_method']}")
        
        # calculate analytical expectation
        n_outlets = len(self.core.outlet_names)
        cluster_sizes = Counter(communities.values())
        
        # analytical null for surprisal-weighted co-clustering frequency
        expected_x = 0.0
        expected_x2 = 0.0
        
        for size in cluster_sizes.values():
            if size > 1:
                # probability that a random pair falls in cluster of this size
                p = size * (size - 1) / (n_outlets * (n_outlets - 1))
                # surprisal weight for this cluster size
                weight = -np.log2(size / n_outlets)
                
                expected_x += p * weight
                expected_x2 += p * (weight ** 2)
        
        analytic_variance = expected_x2 - (expected_x ** 2)
        analytic_std = np.sqrt(analytic_variance)
        
        print(f"analytical expectation: mean={expected_x:.6f}, std={analytic_std:.6f}")
        
        # empirical validation via permutation
        rng = np.random.default_rng(random_state)
        samples = np.empty(n_permutations, dtype=float)
        
        print(f"running {n_permutations} permutations...")
        
        # test pair (first two outlets)
        i, j = 0, 1
        
        for perm in range(n_permutations):
            # create random partition with same cluster size distribution
            labels = np.full(n_outlets, -1, dtype=int)
            
            # assign outlets to clusters following original size distribution
            outlet_idx = 0
            for cluster_id, size in enumerate(cluster_sizes.values()):
                for _ in range(size):
                    if outlet_idx < n_outlets:
                        labels[outlet_idx] = cluster_id
                        outlet_idx += 1
            
            # fill remaining outlets with singleton clusters
            next_cluster_id = len(cluster_sizes)
            for idx in range(outlet_idx, n_outlets):
                labels[idx] = next_cluster_id
                next_cluster_id += 1
            
            # shuffle the labels
            rng.shuffle(labels)
            
            # calculate surprisal weight for test pair
            if labels[i] == labels[j]:
                cluster_size = np.sum(labels == labels[i])
                weight = -np.log2(cluster_size / n_outlets)
                samples[perm] = weight
            else:
                samples[perm] = 0.0
        
        # empirical statistics
        emp_mean = samples.mean()
        emp_std = samples.std(ddof=1)

        # histogram with analytic normal curve
        if show_plot or save_path:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(samples, bins=50, density=True, alpha=0.6, color='skyblue', 
                   edgecolor='black', label='Empirical Distribution')
            
            # overlay analytical normal distribution
            x_vals = np.linspace(samples.min(), samples.max(), 1000)
            ax.plot(x_vals, norm.pdf(x_vals, loc=expected_x, scale=analytic_std), 
                   'r-', linewidth=3, label=f'Analytical Normal\\n(μ={expected_x:.4f}, σ={analytic_std:.4f})')
            
            # add vertical lines for means
            ax.axvline(expected_x, color='red', linestyle='--', alpha=0.8, 
                      label=f'Analytical Mean: {expected_x:.4f}')
            ax.axvline(emp_mean, color='blue', linestyle='--', alpha=0.8,
                      label=f'Empirical Mean: {emp_mean:.4f}')
            
            ax.set_xlabel('Surprisal Weight', fontweight='bold')
            ax.set_ylabel('Density', fontweight='bold')
            ax.set_title(f'Null Distribution Validation\\nPair ({i}, {j}) - {n_permutations:,} Permutations', 
                        fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"validation plot saved to {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close(fig)

        # calculate validation metrics
        mean_error = abs(emp_mean - expected_x)
        std_error = abs(emp_std - analytic_std)
        mean_rel_error = mean_error / abs(expected_x) if expected_x != 0 else float('inf')
        std_rel_error = std_error / analytic_std if analytic_std != 0 else float('inf')

        print(f"validation results:")
        print(f"  empirical mean: {emp_mean:.6f} (error: {mean_error:.6f}, {mean_rel_error:.1%})")
        print(f"  empirical std:  {emp_std:.6f} (error: {std_error:.6f}, {std_rel_error:.1%})")

        # validation status
        validation_passed = (mean_rel_error < 0.05) and (std_rel_error < 0.05)
        print(f"  validation: {'✓ PASSED' if validation_passed else '✗ FAILED'} "
              f"(both errors < 5%: {mean_rel_error < 0.05 and std_rel_error < 0.05})")

        return {
            'pair': (i, j),
            'analytical_mean': expected_x,
            'analytical_std': analytic_std,
            'empirical_mean': emp_mean,
            'empirical_std': emp_std,
            'mean_error': mean_error,
            'std_error': std_error,
            'mean_rel_error': mean_rel_error,
            'std_rel_error': std_rel_error,
            'n_permutations': n_permutations,
            'validation_passed': validation_passed,
            'samples': samples
        }