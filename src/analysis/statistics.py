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

# -----------------------------------------------------------------------------
# scipy compatibility utilities
# -----------------------------------------------------------------------------
# ``scipy.stats.binom_test`` was deprecated in SciPy 1.7 and removed entirely in
# SciPy 2.0.  Its replacement, ``scipy.stats.binomtest``, returns an object with
# a ``pvalue`` attribute.  To avoid littering the codebase with version checks
# (and to preserve the original API expected elsewhere in *statistics.py*), we
# define a thin wrapper that emulates the old behaviour when necessary.

try:
    # available in SciPy < 2.0
    from scipy.stats import binom_test  # type: ignore
except ImportError:  # pragma: no cover – we are on SciPy ≥ 2.0
    from scipy.stats import binomtest as _binomtest  # type: ignore

    def binom_test(k: int, n: int, p: float = 0.5, *, alternative: str = "two-sided") -> float:  # noqa: N802
        """Backwards-compatibility wrapper for the removed *binom_test* function.

        Parameters
        ----------
        k : int
            Number of successes.
        n : int
            Number of trials.
        p : float, optional
            Hypothesised probability of success (default is 0.5).
        alternative : {"two-sided", "greater", "less"}, optional
            The alternative hypothesis.

        Returns
        -------
        float
            The p-value of the binomial test.
        """

        return float(_binomtest(k, n, p, alternative=alternative).pvalue)

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
        """Evaluate statistical significance of outlet–pair co-clustering across *independent* samples.

        1) builds a surprisal-weighted co-clustering matrix **per sample**,
        2) derives an analytical null distribution for every sample (mean & variance),
        3) computes residuals across samples and applies a paired parametric/non-parametric
           test (t-test for S≥15, Wilcoxon otherwise),
        4) corrects p-values with Benjamini–Hochberg FDR and classifies significant
           pairs as *high* (more than expected) or *low* (less than expected).
        """
        print("\n=== SIGNIFICANCE ACROSS SAMPLES ===")

        # ------------------------------------------------------------------
        # basic input checks
        # ------------------------------------------------------------------
        if self.core.outlet_names is None:
            raise ValueError("outlet_names is not set")

        df = self.core.results_df
        if 'sample_id' not in df.columns:
            raise ValueError("sample_id column not found in results_df")

        unique_samples = df['sample_id'].unique()
        if len(unique_samples) < 2:
            raise ValueError(f"Need at least 2 distinct sample_id values, found {len(unique_samples)}")

        print(f"Processing {len(unique_samples)} samples")

        # ------------------------------------------------------------------
        # step B – aggregate within each sample
        # ------------------------------------------------------------------
        samples, X = self._stack_sample_matrices()
        S, n_outlets = len(samples), len(self.core.outlet_names)

        # ------------------------------------------------------------------
        # step C – analytical null (mean & variance) for every sample
        # ------------------------------------------------------------------
        MU, VAR = self._generate_null_for_samples(samples)

        # ------------------------------------------------------------------
        # step D – determine testable outlet pairs (coverage ≥ min_sample_frac)
        # ------------------------------------------------------------------
        testable, masked_pairs = self._create_outlet_mask(VAR, min_sample_frac, S)

        # ------------------------------------------------------------------
        # step E – residuals (observed – expected)
        # ------------------------------------------------------------------
        obs_mean = X.mean(axis=0)
        exp_mean = MU.mean(axis=0)
        Y = X - MU  # residuals per sample

        # ------------------------------------------------------------------
        # step F – choose statistical test
        # ------------------------------------------------------------------
        if test == "auto":
            test_method = "t" if S >= 15 else "wilcoxon"
        else:
            test_method = test
        print(f"Using {test_method} test for {S} samples")

        # ------------------------------------------------------------------
        # step G – compute p-values per pair
        # ------------------------------------------------------------------
        p_values = self._compute_pvalues_across_samples(Y, VAR, testable, test_method, S)

        # ------------------------------------------------------------------
        # step H – FDR correction
        # ------------------------------------------------------------------
        p_adj_matrix, significant_pairs = self._apply_fdr_correction(p_values, testable, alpha, n_outlets)

        # ------------------------------------------------------------------
        # step I – classify significant pairs
        # ------------------------------------------------------------------
        high_pairs, low_pairs = self._classify_significant_pairs(significant_pairs, obs_mean, exp_mean)

        # console summary
        print(f"\nRESULTS SUMMARY:")
        print(f"Samples analysed: {S}")
        print(f"Significantly HIGH pairs: {len(high_pairs)}")
        print(f"Significantly LOW  pairs: {len(low_pairs)}")
        print(f"Masked pairs: {len(masked_pairs)}")

        # ------------------------------------------------------------------
        # step J – visualisation
        # ------------------------------------------------------------------
        self._visualize_across_samples_results(p_adj_matrix, testable, alpha)

        # new: effect size matrix (mean residual across samples)
        effect_matrix = obs_mean - exp_mean

        return {
            'high_pairs': high_pairs,
            'low_pairs': low_pairs,
            'p_adj_matrix': p_adj_matrix,
            'n_samples': S,
            'masked_pairs': masked_pairs,
            'effect_matrix': pd.DataFrame(effect_matrix, index=self.core.outlet_names, columns=self.core.outlet_names),
        }

    # ------------------------------------------------------------------
    # helper methods for across-samples significance
    # ------------------------------------------------------------------

    def _stack_sample_matrices(self) -> Tuple[List[str], np.ndarray]:
        """Aggregate clustering results within each sample and stack into a 3-D array."""
        sample_groups = self.core.results_df.groupby('sample_id')
        samples: List[str] = []
        matrices: List[np.ndarray] = []

        for sample_id, group in sample_groups:
            print(f"Processing sample {sample_id}: {len(group)} runs")
            freq_matrix = self.core._aggregate_clustering_results_with_surprisal(
                group,
                method_name=f"sample_{sample_id}",
                use_surprisal_weighting=True,
            )
            if freq_matrix is not None:
                samples.append(sample_id)
                matrices.append(freq_matrix.values)
            else:
                print(f"Warning: Failed to aggregate sample {sample_id}")

        if not matrices:
            raise ValueError("No valid sample matrices generated")

        X = np.asarray(matrices)  # shape: (S, n, n)
        print(f"Stacked {len(samples)} sample matrices, shape: {X.shape}")
        return samples, X

    def _generate_null_for_samples(self, samples: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute analytical null mean & variance for each sample independently."""
        n_outlets = len(self.core.outlet_names)
        S = len(samples)
        MU = np.zeros((S, n_outlets, n_outlets))
        VAR = np.zeros((S, n_outlets, n_outlets))

        for s, sample_id in enumerate(samples):
            sample_df = self.core.results_df[self.core.results_df['sample_id'] == sample_id]
            K_s = len(sample_df)
            if K_s == 0:
                continue
            print(f"Computing null for sample {sample_id}: {K_s} runs")
            mean_sum = np.zeros((n_outlets, n_outlets))
            var_sum = np.zeros((n_outlets, n_outlets))
            for _, row in sample_df.iterrows():
                comm = row['communities']
                if not comm:
                    continue
                sizes = Counter(comm.values())
                ex = ex2 = 0.0
                for size in sizes.values():
                    if size > 1:
                        p = size * (size - 1) / (n_outlets * (n_outlets - 1))
                        w = -np.log2(size / n_outlets)
                        ex += p * w
                        ex2 += p * w * w
                var_x = ex2 - ex ** 2
                for i in range(n_outlets):
                    if i not in comm:
                        continue
                    for j in range(i + 1, n_outlets):
                        if j not in comm:
                            continue
                        mean_sum[i, j] += ex
                        mean_sum[j, i] += ex
                        var_sum[i, j] += var_x
                        var_sum[j, i] += var_x
            MU[s] = mean_sum / max(K_s, 1)
            VAR[s] = var_sum / max(K_s, 1)
        return MU, VAR

    def _create_outlet_mask(self, VAR: np.ndarray, min_sample_frac: float, S: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Determine which outlet pairs have sufficient coverage across samples.

        A pair is deemed *testable* if either
        1. it has strictly positive variance in at least ``min_sample_frac`` of the S samples, or
        2. it has zero variance across *all* samples (i.e. perfectly consistent) – such cases will
           be handled specially later on because they may still exhibit a systematic bias.
        """
        n_outlets = VAR.shape[1]
        testable = np.zeros((n_outlets, n_outlets), dtype=bool)
        masked_pairs: List[Tuple[int, int]] = []
        for i in range(n_outlets):
            for j in range(i + 1, n_outlets):
                var_ij = VAR[:, i, j]
                # coverage criterion
                valid_count = np.sum(var_ij > 0)
                var_all_zero = np.all(var_ij == 0)
                if (valid_count / S >= min_sample_frac) or var_all_zero:
                    testable[i, j] = testable[j, i] = True
                else:
                    masked_pairs.append((i, j))
        print(f"Testable pairs: {np.sum(testable) // 2}, Masked pairs: {len(masked_pairs)}")
        return testable, masked_pairs

    def _compute_pvalues_across_samples(self, Y: np.ndarray, VAR: np.ndarray, testable: np.ndarray,
                                        test_method: str, S: int) -> np.ndarray:
        """Compute p-values for each testable pair using selected test.

        Improvements over the original implementation:
        1. Weighted t-test for heteroscedastic residuals (when ``test_method == 't'``)
           – weights are inverse variances ``w_s = 1 / VAR``.
        2. Deterministic zero-variance pairs: if all ``VAR == 0`` and the mean residual is non-zero,
           we flag the pair as highly significant (p ≈ 0). If mean residual is exactly 0, p = 1.
        3. Permutation fallback for very small sample sizes (``len(residuals) < 5``).
        """
        import numpy as np  # local import to avoid polluting module namespace
        from scipy import stats

        def _permutation_pvalue(res, n_perm: int = 2000) -> float:
            """Exact two-sided permutation test for mean of signed residuals."""
            obs = abs(res.mean())
            count = 0
            for _ in range(n_perm):
                signs = np.random.choice([-1, 1], size=res.size)
                if abs((res * signs).mean()) >= obs:
                    count += 1
            return (count + 1) / (n_perm + 1)

        n_outlets = testable.shape[0]
        p_values = np.ones((n_outlets, n_outlets))
        rows, cols = np.where(np.triu(testable, k=1))
        for i, j in zip(rows, cols):
            residuals = Y[:, i, j]
            var_ij = VAR[:, i, j]

            # case A – fully deterministic across samples
            if np.all(var_ij == 0):
                mean_res = residuals.mean()
                p_val = 1.0 if np.isclose(mean_res, 0.0) else 1e-12
                p_values[i, j] = p_values[j, i] = p_val
                continue

            # select samples with positive variance for weighting
            valid_mask = var_ij > 0
            valid_res = residuals[valid_mask]
            valid_var = var_ij[valid_mask]

            if valid_res.size < 2:
                # not enough samples for a meaningful test
                continue

            try:
                if test_method == 'wilcoxon':
                    if np.all(valid_res == 0):
                        p_val = 1.0
                    elif valid_res.size < 5:
                        # permutation fallback
                        p_val = _permutation_pvalue(valid_res)
                    else:
                        _, p_val = wilcoxon(valid_res, alternative='two-sided')

                else:  # weighted t-test
                    if valid_res.size < 5:
                        # permutation fallback retains correct type-I error for tiny n
                        p_val = _permutation_pvalue(valid_res)
                    else:
                        # weights: inverse variances (avoid divide-by-zero)
                        weights = 1.0 / np.maximum(valid_var, 1e-12)
                        mu_w = np.sum(weights * valid_res) / np.sum(weights)
                        se_w = np.sqrt(1.0 / np.sum(weights))
                        if se_w == 0:
                            p_val = 1.0
                        else:
                            t_stat = mu_w / se_w
                            df = valid_res.size - 1  # conservative
                            p_val = 2 * stats.t.sf(abs(t_stat), df)

                p_values[i, j] = p_values[j, i] = p_val
            except Exception as e:
                print(f"Warning: test failed for pair ({i}, {j}): {e}")
        return p_values

    def _apply_fdr_correction(self, p_values: np.ndarray, testable: np.ndarray, alpha: float,
                              n_outlets: int) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
        """Benjamini–Hochberg FDR correction and extraction of significant pairs."""
        rows, cols = np.where(np.triu(testable, k=1))
        testable_p = p_values[rows, cols]
        if testable_p.size == 0:
            padj_df = pd.DataFrame(np.ones((n_outlets, n_outlets)),
                                   index=self.core.outlet_names,
                                   columns=self.core.outlet_names)
            return padj_df, []
        reject, p_adj, _, _ = multipletests(testable_p, alpha=alpha, method='fdr_bh')
        padj_mat = np.ones((n_outlets, n_outlets))
        padj_mat[rows, cols] = p_adj
        padj_mat[cols, rows] = p_adj
        padj_df = pd.DataFrame(padj_mat, index=self.core.outlet_names, columns=self.core.outlet_names)
        sig_pairs = [(rows[k], cols[k]) for k, r in enumerate(reject) if r]
        print(f"FDR correction: {len(sig_pairs)}/{len(testable_p)} pairs significant")
        return padj_df, sig_pairs

    def _classify_significant_pairs(self, significant_pairs: List[Tuple[int, int]],
                                    obs_mean: np.ndarray, exp_mean: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """Split significant pairs into HIGH and LOW based on deviation sign."""
        high_pairs: List[Dict[str, Any]] = []
        low_pairs: List[Dict[str, Any]] = []
        for i, j in significant_pairs:
            observed = obs_mean[i, j]
            expected = exp_mean[i, j]
            deviation = observed - expected
            entry = {
                'outlet1': self.core.outlet_names[i],
                'outlet2': self.core.outlet_names[j],
                'observed': observed,
                'expected': expected,
                'deviation': deviation
            }
            (high_pairs if deviation > 0 else low_pairs).append(entry)
        # sort by absolute deviation
        high_pairs.sort(key=lambda x: abs(x['deviation']), reverse=True)
        low_pairs.sort(key=lambda x: abs(x['deviation']),  reverse=True)
        return high_pairs, low_pairs

    def _visualize_across_samples_results(self, p_adj_matrix: pd.DataFrame, testable: np.ndarray, alpha: float):
        """Generate heatmaps visualising across-samples significance results."""
        try:
            log_p = -np.log10(p_adj_matrix.values)
            log_p[p_adj_matrix.values == 1.0] = 0.0
            plt.figure(figsize=(8, 6))
            sns.heatmap(log_p, square=True, cmap='viridis',
                        cbar_kws={'label': '-log₁₀(p-value)'},
                        xticklabels=self.core.outlet_names,
                        yticklabels=self.core.outlet_names)
            plt.title('Significance Across Samples\n(-log₁₀ adjusted p-values)', fontweight='bold')
            plt.tick_params(axis='both', labelsize=6)
            plt.tight_layout()
            plt.savefig('results/significance_across_samples_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()

            plt.figure(figsize=(8, 6))
            sns.heatmap(testable.astype(int), cmap='RdBu_r', center=0.5, square=True,
                        cbar_kws={'label': 'Testable (1) vs Masked (0)'},
                        xticklabels=self.core.outlet_names,
                        yticklabels=self.core.outlet_names)
            plt.title('Sample Coverage Mask', fontweight='bold')
            plt.tick_params(axis='both', labelsize=6)
            plt.tight_layout()
            plt.savefig('results/significance_across_samples_mask.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Warning: visualization failed: {e}")

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