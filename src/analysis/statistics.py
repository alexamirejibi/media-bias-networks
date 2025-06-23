"""
Statistical analysis methods - significance testing, null distributions, and hypothesis testing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, ttest_1samp
from statsmodels.stats.multitest import multipletests
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any, Tuple
import time
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap


class StatisticsAnalyzer:
    """Specialized analyzer for statistical significance testing."""
    
    def __init__(self, core_analyzer):
        self.core = core_analyzer
    
    def analyze_statistical_significance(self, frequency_matrix_weighted: pd.DataFrame, 
                                       colors: dict, n_permutations: int = 1000, 
                                       alpha: float = 0.05, use_joint_permutation_cov: bool = True, 
                                       cov_permutations: int = 5000) -> Dict[str, Any]:
        """perform statistical significance testing using permutation tests with surprisal weighting"""
        print("\n=== STATISTICAL SIGNIFICANCE TESTING ===")
        
        def partition_to_coclustering_matrix(partition, n_outlets):
            """convert partition to binary co-clustering matrix"""
            assignments = np.full(n_outlets, -1, dtype=int)
            for outlet_id, community_id in partition.items():
                if outlet_id < n_outlets:
                    assignments[outlet_id] = community_id
            
            valid_mask = assignments >= 0
            assignments_2d = assignments[:, np.newaxis]
            assignments_broadcast = assignments[np.newaxis, :]
            
            matrix = ((assignments_2d == assignments_broadcast) & 
                      valid_mask[:, np.newaxis] & 
                      valid_mask[np.newaxis, :]).astype(int)
            
            return matrix

        def generate_null_distribution(analyzer):
            """analytical pair-specific null (mean & std) with correct Var(\bar X)=Var(X)/K"""
            start = time.time()
            df = analyzer.get_results()
            n_out = len(analyzer.outlet_names)
            K = len(df)

            print("pair-specific analytical null (corrected variance)")
            print(f"processing {K} clustering results")

            mean_sum = np.zeros((n_out, n_out))
            var_sum  = np.zeros((n_out, n_out))

            for _, row in df.iterrows():
                comm = row['communities']
                if not comm:
                    continue
                sizes = Counter(comm.values())
                ex = ex2 = 0.0
                for sz in sizes.values():
                    if sz > 1:
                        p = sz*(sz-1)/(n_out*(n_out-1))
                        w = -np.log2(sz / n_out)
                        ex  += p * w
                        ex2 += p * w * w
                var_x = ex2 - ex**2
                for i in range(n_out):
                    if i not in comm: continue
                    for j in range(i+1, n_out):
                        if j not in comm: continue
                        mean_sum[i,j] += ex
                        mean_sum[j,i] += ex
                        var_sum[i,j]  += var_x
                        var_sum[j,i]  += var_x

            mean_mat = mean_sum / max(K,1)
            var_mat  = (var_sum / max(K,1)) / max(K,1)  # Var(mean) = Var(X)/K
            std_mat  = np.sqrt(np.maximum(var_mat, 1e-10))

            np.fill_diagonal(mean_mat, 0)
            np.fill_diagonal(std_mat, 0)

            tri = np.triu_indices(n_out,1)
            g_mean = float(mean_mat[tri].mean()) if tri[0].size else 0.0
            g_std  = float(std_mat[tri].mean()) if tri[0].size else 0.0

            print(f"done in {time.time()-start:.2f}s | global mean {g_mean:.6f} | global std {g_std:.6f}")

            return {
                'mean_matrix': mean_mat,
                'std_matrix': std_mat,
                'global_mean': g_mean,
                'global_std': g_std
            }

        # generate null distribution
        null_dist = generate_null_distribution(self.core)
        null_mean_matrix = null_dist['mean_matrix']
        null_std_matrix = null_dist['std_matrix']
        null_mean = null_dist['global_mean']
        null_std = null_dist['global_std']

        print(f"null distribution: mean={null_mean:.6f}, std={null_std:.6f}")

        # calculate z-scores and p-values for all pairs
        print("\ncalculating statistical significance for all outlet pairs...")
        
        observed_matrix = frequency_matrix_weighted.values
        n_outlets = len(frequency_matrix_weighted)
        
        # handle edge case of zero standard deviation
        safe_std_matrix = np.where(null_std_matrix > 1e-10, null_std_matrix, 1e-10)
        
        # calculate z-scores
        z_scores = (observed_matrix - null_mean_matrix) / safe_std_matrix
        
        # calculate two-tailed p-values (using standard normal distribution)
        from scipy.stats import norm
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        
        # set diagonal to 1 (no self-comparison)
        np.fill_diagonal(p_values, 1.0)
        np.fill_diagonal(z_scores, 0.0)
        
        # collect all valid off-diagonal p-values for multiple testing correction
        off_diag_mask = ~np.eye(n_outlets, dtype=bool)
        off_diag_p_values = p_values[off_diag_mask]
        
        print(f"testing {len(off_diag_p_values)} outlet pairs for significance")
        
        # benjamini-hochberg correction for multiple testing
        rejected, corrected_p_values, _, _ = multipletests(off_diag_p_values, 
                                                          alpha=alpha, 
                                                          method='fdr_bh')
        
        # reconstruct corrected p-value matrix
        corrected_p_matrix = np.ones((n_outlets, n_outlets))
        corrected_p_matrix[off_diag_mask] = corrected_p_values
        corrected_p_df = pd.DataFrame(corrected_p_matrix, 
                                    index=frequency_matrix_weighted.index,
                                    columns=frequency_matrix_weighted.columns)
        
        # identify significant pairs
        significant_mask = corrected_p_matrix < alpha
        np.fill_diagonal(significant_mask, False)  # exclude diagonal
        
        n_significant = np.sum(significant_mask)
        print(f"found {n_significant} significant pairs after FDR correction (α = {alpha})")
        
        # categorize significant pairs into high and low co-clustering
        high_pairs = []
        low_pairs = []
        
        for i in range(n_outlets):
            for j in range(i+1, n_outlets):  # upper triangle only
                if significant_mask[i, j]:
                    outlet1 = frequency_matrix_weighted.index[i]
                    outlet2 = frequency_matrix_weighted.index[j]
                    observed = observed_matrix[i, j]
                    expected = null_mean_matrix[i, j]
                    p_val = corrected_p_matrix[i, j]
                    z_score = z_scores[i, j]
                    deviation = observed - expected
                    
                    pair_info = {
                        'outlet1': outlet1,
                        'outlet2': outlet2,
                        'observed': observed,
                        'expected': expected,
                        'deviation': deviation,
                        'z_score': z_score,
                        'p_value': p_val
                    }
                    
                    if deviation > 0:  # higher than expected
                        high_pairs.append(pair_info)
                    else:  # lower than expected
                        low_pairs.append(pair_info)
        
        # sort by absolute deviation
        high_pairs.sort(key=lambda x: x['deviation'], reverse=True)
        low_pairs.sort(key=lambda x: x['deviation'])  # most negative first
        
        print(f"\nsignificant pairs breakdown:")
        print(f"  high co-clustering (more than expected): {len(high_pairs)}")
        print(f"  low co-clustering (less than expected): {len(low_pairs)}")
        
        # display top results
        if high_pairs:
            print(f"\ntop 10 significantly HIGH co-clustering pairs:")
            for i, pair in enumerate(high_pairs[:10], 1):
                print(f"  {i:2d}. {pair['outlet1']} ↔ {pair['outlet2']}: "
                      f"obs={pair['observed']:.3f}, exp={pair['expected']:.3f}, "
                      f"dev=+{pair['deviation']:.3f}, p={pair['p_value']:.6f}")
        
        if low_pairs:
            print(f"\ntop 10 significantly LOW co-clustering pairs:")
            for i, pair in enumerate(low_pairs[:10], 1):
                print(f"  {i:2d}. {pair['outlet1']} ↔ {pair['outlet2']}: "
                      f"obs={pair['observed']:.3f}, exp={pair['expected']:.3f}, "
                      f"dev={pair['deviation']:.3f}, p={pair['p_value']:.6f}")
        
        # create visualization
        self._visualize_significance_results(
            frequency_matrix_weighted, z_scores, significant_mask, 
            corrected_p_df, high_pairs, low_pairs, colors, alpha
        )
        
        return {
            'z_scores': z_scores,
            'p_values': p_values,
            'corrected_p_values': corrected_p_values,
            'corrected_p_df': corrected_p_df,
            'significant_mask': significant_mask,
            'high_pairs': high_pairs,
            'low_pairs': low_pairs,
            'n_significant': n_significant,
            'null_mean': null_mean,
            'null_std': null_std,
            'null_mean_matrix': null_mean_matrix,
            'null_std_matrix': null_std_matrix
        }

    def _visualize_significance_results(self, frequency_matrix, z_scores, significant_mask, 
                                      corrected_p_df, high_pairs, low_pairs, colors, alpha):
        """Create visualizations for significance testing results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Original frequency matrix
        heatmap_cmap = LinearSegmentedColormap.from_list('custom_heatmap', 
                                                       ['#F7F7F7', '#F18F01', '#C73E1D'], N=256)
        sns.heatmap(frequency_matrix, mask=frequency_matrix.values == 0, 
                   cmap=heatmap_cmap, square=True, ax=axes[0,0],
                   cbar_kws={'label': 'Co-clustering Frequency'})
        axes[0,0].set_title('Original Co-clustering Frequency Matrix', fontweight='bold')
        
        # 2. Z-scores heatmap
        diverging_cmap = LinearSegmentedColormap.from_list('custom_diverging', 
                                                         ['#C73E1D', '#F18F01', '#F7F7F7', '#6BAED6', '#2E86AB'], N=256)
        mask_z = np.eye(len(frequency_matrix), dtype=bool)
        sns.heatmap(z_scores, mask=mask_z, cmap=diverging_cmap, center=0, 
                   square=True, ax=axes[0,1], cbar_kws={'label': 'Z-score'})
        axes[0,1].set_title('Z-scores (Observed - Expected) / Std', fontweight='bold')
        
        # 3. Significance mask
        sns.heatmap(significant_mask.astype(int), cmap='RdBu_r', center=0.5, 
                   square=True, ax=axes[1,0], cbar_kws={'label': 'Significant (1) vs Non-significant (0)'})
        axes[1,0].set_title(f'Statistical Significance Mask (FDR α = {alpha})', fontweight='bold')
        
        # 4. Corrected p-values (log scale)
        log_p_values = -np.log10(corrected_p_df.values + 1e-10)  # add small value to avoid log(0)
        np.fill_diagonal(log_p_values, 0)  # set diagonal to 0
        
        sns.heatmap(log_p_values, cmap='viridis', square=True, ax=axes[1,1],
                   cbar_kws={'label': '-log10(corrected p-value)'})
        axes[1,1].set_title('Statistical Significance\n(Darker = More Significant)', fontweight='bold')
        
        for ax in axes.flat:
            ax.tick_params(axis='both', labelsize=8)
        
        plt.tight_layout()
        plt.savefig('results/statistical_significance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_significance_across_samples(self, alpha: float = 0.05, 
                                          min_sample_frac: float = 0.5,
                                          test: str = "auto") -> Dict[str, Any]:
        """Test significance across independent samples using binomial test."""
        print(f"\\n=== SIGNIFICANCE ACROSS SAMPLES ANALYSIS ===")
        
        # Get all unique samples
        df = self.core.get_results()
        samples = df['sample_id'].unique()
        n_samples = len(samples)
        
        print(f"Found {n_samples} independent samples")
        
        if n_samples < 2:
            return {'error': 'Need at least 2 samples for across-samples analysis'}
        
        # Calculate minimum required samples for significance
        min_samples = max(2, int(min_sample_frac * n_samples))
        print(f"Requiring relationships to appear in at least {min_samples}/{n_samples} samples ({min_sample_frac:.1%})")
        
        # Get frequency matrices for each sample
        sample_matrices = {}
        for sample_id in samples:
            print(f"Processing sample {sample_id}...")
            
            # Create temporary analyzer for this sample
            sample_results = df[df['sample_id'] == sample_id]
            
            if sample_results.empty:
                continue
                
            # Aggregate results for this sample using existing method
            sample_matrix = self.core._aggregate_clustering_results_with_surprisal(
                sample_results, method_name=f"sample_{sample_id}", use_surprisal_weighting=True
            )
            
            if sample_matrix is not None:
                sample_matrices[sample_id] = sample_matrix
        
        if len(sample_matrices) < min_samples:
            return {'error': f'Only {len(sample_matrices)} samples have valid matrices, need at least {min_samples}'}
        
        print(f"Successfully processed {len(sample_matrices)} samples")
        
        # For each outlet pair, count how many samples show co-clustering
        n_outlets = len(self.core.outlet_names)
        pair_counts = np.zeros((n_outlets, n_outlets), dtype=int)
        pair_coverage = np.zeros((n_outlets, n_outlets), dtype=int)  # how many samples have this pair
        
        for sample_id, matrix in sample_matrices.items():
            for i in range(n_outlets):
                for j in range(i+1, n_outlets):  # only upper triangle
                    pair_coverage[i, j] += 1
                    pair_coverage[j, i] += 1
                    
                    if matrix.iloc[i, j] > 0:  # co-clustered in this sample
                        pair_counts[i, j] += 1
                        pair_counts[j, i] += 1
        
        # Calculate observed and expected frequencies
        observed_freq = np.zeros((n_outlets, n_outlets))
        expected_freq = np.zeros((n_outlets, n_outlets))
        
        # Estimate overall co-clustering probability from aggregated data
        total_freq_matrix = self.core.aggregate_all_results(use_surprisal_weighting=True)
        off_diag_mask = ~np.eye(n_outlets, dtype=bool)
        overall_p = np.mean(total_freq_matrix.values[off_diag_mask] > 0)  # fraction of pairs that co-cluster
        
        print(f"Overall co-clustering probability: {overall_p:.3f}")
        
        for i in range(n_outlets):
            for j in range(i+1, n_outlets):
                if pair_coverage[i, j] >= min_samples:
                    n_obs = pair_counts[i, j]
                    n_trials = pair_coverage[i, j]
                    
                    observed_freq[i, j] = n_obs / n_trials
                    observed_freq[j, i] = observed_freq[i, j]
                    
                    expected_freq[i, j] = overall_p
                    expected_freq[j, i] = overall_p
        
        # Perform binomial tests
        from scipy.stats import binom_test
        
        high_pairs = []
        low_pairs = []
        masked_pairs = []
        p_values = np.ones((n_outlets, n_outlets))
        
        for i in range(n_outlets):
            for j in range(i+1, n_outlets):
                if pair_coverage[i, j] < min_samples:
                    # Insufficient coverage
                    masked_pairs.append({
                        'outlet1': self.core.outlet_names[i],
                        'outlet2': self.core.outlet_names[j],
                        'coverage': pair_coverage[i, j],
                        'min_required': min_samples
                    })
                    continue
                
                n_obs = pair_counts[i, j]
                n_trials = pair_coverage[i, j]
                obs_freq = observed_freq[i, j]
                exp_freq = expected_freq[i, j]
                
                # Two-tailed binomial test
                p_val = binom_test(n_obs, n_trials, exp_freq, alternative='two-sided')
                p_values[i, j] = p_val
                p_values[j, i] = p_val
                
                if p_val < alpha:
                    pair_info = {
                        'outlet1': self.core.outlet_names[i],
                        'outlet2': self.core.outlet_names[j],
                        'observed': obs_freq,
                        'expected': exp_freq,
                        'deviation': obs_freq - exp_freq,
                        'n_samples_coocurring': n_obs,
                        'n_samples_total': n_trials,
                        'p_value': p_val
                    }
                    
                    if obs_freq > exp_freq:
                        high_pairs.append(pair_info)
                    else:
                        low_pairs.append(pair_info)
        
        # Apply multiple testing correction
        off_diag_p = p_values[off_diag_mask]
        valid_mask = off_diag_p < 1.0  # exclude masked pairs
        
        if np.sum(valid_mask) > 0:
            valid_p = off_diag_p[valid_mask]
            rejected, corrected_p, _, _ = multipletests(valid_p, alpha=alpha, method='fdr_bh')
            
            # Update significance based on corrected p-values
            corrected_p_matrix = np.ones((n_outlets, n_outlets))
            off_diag_indices = np.where(off_diag_mask)
            valid_indices = np.where(valid_mask)[0]
            
            for idx, corr_p in zip(valid_indices, corrected_p):
                i, j = off_diag_indices[0][idx], off_diag_indices[1][idx]
                corrected_p_matrix[i, j] = corr_p
                corrected_p_matrix[j, i] = corr_p
            
            # Filter pairs by corrected p-values
            high_pairs = [p for p in high_pairs if corrected_p_matrix[
                self.core.outlet_names.index(p['outlet1']), 
                self.core.outlet_names.index(p['outlet2'])
            ] < alpha]
            
            low_pairs = [p for p in low_pairs if corrected_p_matrix[
                self.core.outlet_names.index(p['outlet1']), 
                self.core.outlet_names.index(p['outlet2'])
            ] < alpha]
        else:
            corrected_p_matrix = p_values.copy()
        
        # Sort results
        high_pairs.sort(key=lambda x: x['deviation'], reverse=True)
        low_pairs.sort(key=lambda x: x['deviation'])
        
        corrected_p_df = pd.DataFrame(corrected_p_matrix, 
                                    index=self.core.outlet_names,
                                    columns=self.core.outlet_names)
        
        return {
            'high_pairs': high_pairs,
            'low_pairs': low_pairs,
            'masked_pairs': masked_pairs,
            'p_adj_matrix': corrected_p_df,
            'n_samples': len(sample_matrices),
            'min_samples_required': min_samples,
            'overall_cooccurrence_prob': overall_p
        }

    def construct_validated_clustering(self, high_pairs: List[Dict], low_pairs: List[Dict], 
                                     null_mean: float, null_std: float, 
                                     n_clusters: Optional[int] = None) -> Optional[Dict]:
        """Construct clustering using only statistically validated relationships."""
        print("\\n=== CONSTRUCTING VALIDATED CLUSTERING ===")
        
        if not high_pairs and not low_pairs:
            print("No significant pairs found - cannot construct validated clustering")
            return None
        
        n_outlets = len(self.core.outlet_names)
        
        # Create signed similarity matrix
        # Start with null expectation as baseline
        similarity_matrix = np.full((n_outlets, n_outlets), null_mean)
        np.fill_diagonal(similarity_matrix, 1.0)  # perfect self-similarity
        
        # Add validated high relationships (attractive)
        for pair in high_pairs:
            i = self.core.outlet_names.index(pair['outlet1'])
            j = self.core.outlet_names.index(pair['outlet2'])
            # Use observed frequency as similarity
            similarity_matrix[i, j] = pair['observed']
            similarity_matrix[j, i] = pair['observed']
        
        # Add validated low relationships (repulsive)
        for pair in low_pairs:
            i = self.core.outlet_names.index(pair['outlet1'])
            j = self.core.outlet_names.index(pair['outlet2'])
            # Use observed frequency (which is lower than expected)
            similarity_matrix[i, j] = pair['observed']
            similarity_matrix[j, i] = pair['observed']
        
        print(f"Constructed signed similarity matrix using {len(high_pairs)} high and {len(low_pairs)} low pairs")
        
        # Convert to distance matrix for clustering
        # Map similarities to distances: high similarity -> low distance
        max_sim = similarity_matrix.max()
        distance_matrix = max_sim - similarity_matrix
        
        # Ensure distance matrix is valid (non-negative, symmetric, zero diagonal)
        distance_matrix = np.maximum(distance_matrix, 0)
        np.fill_diagonal(distance_matrix, 0)
        
        # Perform hierarchical clustering
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        try:
            condensed_distances = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Determine number of clusters
            if n_clusters is None:
                # Use elbow method or reasonable default
                n_clusters = min(6, max(2, len(high_pairs) // 3))
            
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Create communities dictionary
            communities = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                communities[label].append(self.core.outlet_names[i])
            
            print(f"Successfully constructed {len(communities)} validated communities")
            
            return {
                'linkage': linkage_matrix,
                'labels': cluster_labels,
                'communities': dict(communities),
                'distance_matrix': distance_matrix,
                'similarity_matrix': similarity_matrix,
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            print(f"Failed to construct validated clustering: {e}")
            return None