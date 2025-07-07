"""
Temporal analysis methods - time-series analysis, window stability, and drift detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import Counter


class TemporalAnalyzer:
    """Specialized analyzer for temporal stability and time-series analysis."""
    
    def __init__(self, core_analyzer):
        self.core = core_analyzer
    
    def temporal_stability(self, window_ids: List[str], metric: str = 'ari', n_clusters: int = 6) -> pd.DataFrame:
        """compute pairwise similarity between windows created by run_temporal_experiment

        metric
        ------
        'corr' : pearson correlation between off-diagonal entries of the aggregated surprisal-weighted
                  co-clustering matrices for two windows
        'ari'  : adjusted rand index between ward-linkage clusterings (using `n_clusters`)
        """
        if not window_ids:
            return pd.DataFrame()

        # precompute aggregated matrices per window
        agg_matrices = {}
        for wid in window_ids:
            subset_df = self.core.get_results({'sample_id': wid})
            if subset_df.empty:
                continue
            mat = self.core._aggregate_clustering_results_with_surprisal(subset_df, wid, use_surprisal_weighting=True)
            if mat is not None:
                agg_matrices[wid] = mat

        wins = list(agg_matrices.keys())
        n = len(wins)
        if n == 0:
            return pd.DataFrame()

        sim = np.zeros((n, n))
        np.fill_diagonal(sim, 1.0)

        for i in range(n):
            for j in range(i + 1, n):
                m1, m2 = agg_matrices[wins[i]], agg_matrices[wins[j]]
                if metric == 'corr':
                    mask = ~np.eye(m1.shape[0], dtype=bool)
                    val = np.corrcoef(m1.values[mask], m2.values[mask])[0, 1]
                else:  # 'ari'
                    # convert to distance matrices and ward linkage
                    d1 = m1.values.max() - m1.values
                    d2 = m2.values.max() - m2.values
                    Z1 = linkage(squareform(d1, checks=False), method='ward')
                    Z2 = linkage(squareform(d2, checks=False), method='ward')
                    labels1 = fcluster(Z1, n_clusters, criterion='maxclust')
                    labels2 = fcluster(Z2, n_clusters, criterion='maxclust')
                    val = adjusted_rand_score(labels1, labels2)
                sim[i, j] = sim[j, i] = val

        return pd.DataFrame(sim, index=wins, columns=wins)
    
    def analyze_temporal_drift(self, window_ids: List[str], metric: str = 'ari') -> Dict[str, Any]:
        """analyze temporal drift patterns between consecutive windows"""
        
        if len(window_ids) < 2:
            return {'error': 'need at least 2 windows for drift analysis'}
        
        # get temporal stability matrix
        stability_matrix = self.temporal_stability(window_ids, metric=metric)
        
        if stability_matrix.empty:
            return {'error': 'could not compute temporal stability matrix'}
        
        # calculate consecutive window similarities
        consecutive_similarities = []
        for i in range(len(window_ids) - 1):
            if window_ids[i] in stability_matrix.index and window_ids[i+1] in stability_matrix.index:
                sim = stability_matrix.loc[window_ids[i], window_ids[i+1]]
                consecutive_similarities.append(sim)
        
        # calculate lag-based similarities
        lag_similarities = {}
        max_lag = min(5, len(window_ids) - 1)  # up to 5 lags
        
        for lag in range(1, max_lag + 1):
            lag_sims = []
            for i in range(len(window_ids) - lag):
                if (window_ids[i] in stability_matrix.index and 
                    window_ids[i + lag] in stability_matrix.index):
                    sim = stability_matrix.loc[window_ids[i], window_ids[i + lag]]
                    lag_sims.append(sim)
            
            if lag_sims:
                lag_similarities[lag] = {
                    'mean': np.mean(lag_sims),
                    'std': np.std(lag_sims),
                    'n_comparisons': len(lag_sims)
                }
        
        # overall temporal statistics
        off_diag_mask = ~np.eye(len(stability_matrix), dtype=bool)
        all_similarities = stability_matrix.values[off_diag_mask]
        
        return {
            'consecutive_similarities': consecutive_similarities,
            'consecutive_mean': np.mean(consecutive_similarities) if consecutive_similarities else np.nan,
            'consecutive_std': np.std(consecutive_similarities) if consecutive_similarities else np.nan,
            'lag_similarities': lag_similarities,
            'overall_mean': np.mean(all_similarities),
            'overall_std': np.std(all_similarities),
            'stability_matrix': stability_matrix,
            'metric': metric
        }
    
    
    def _estimate_joint_permutation_mean_std(self, n_permutations: int = 5000,
                                             random_state: Optional[int] = None,
                                             pair: Tuple[int, int] = (0, 1)) -> Tuple[float, float]:
        """estimate both the mean and std of X̄ via joint-permutation Monte-Carlo

        this is identical to `_estimate_joint_permutation_std` but also returns the
        empirical mean so that we can optionally replace the analytical mean with a
        simulation-based estimate.
        """
        
        if self.core.results_df.empty or self.core.outlet_names is None:
            return 0.0, 0.0

        n_outlets = len(self.core.outlet_names)
        if n_outlets < 2:
            return 0.0, 0.0

        i, j = pair
        if i == j or i >= n_outlets or j >= n_outlets:
            i, j = 0, 1  # fall back to first two outlets

        rng = np.random.default_rng(random_state)

        # pre-compute label arrays and weight maps for every clustering run
        label_arrays = []  # list[np.ndarray]
        weight_maps = []   # list[dict]

        for _, row in self.core.results_df.iterrows():
            communities = row['communities']
            labels = np.full(n_outlets, -1, dtype=int)
            if communities:
                for outlet_id, comm_id in communities.items():
                    if 0 <= outlet_id < n_outlets:
                        labels[outlet_id] = comm_id
            # assign unique singleton ids to missing outlets
            next_id = (labels.max() + 1) if labels.max() >= 0 else 0
            for idx in range(n_outlets):
                if labels[idx] == -1:
                    labels[idx] = next_id
                    next_id += 1
            # cluster sizes and corresponding weights
            unique, counts = np.unique(labels, return_counts=True)
            size_map = dict(zip(unique, counts))
            weight_map = {cid: -np.log2(sz / n_outlets) for cid, sz in size_map.items() if sz > 1}
            label_arrays.append(labels)
            weight_maps.append(weight_map)

        label_arrays = np.asarray(label_arrays)  # shape (K, n_outlets)
        K = label_arrays.shape[0]
        if K == 0:
            return 0.0, 0.0

        samples = np.empty(n_permutations, dtype=float)

        for perm in range(n_permutations):
            # permute independently within each clustering
            permuted_arrays = label_arrays.copy()
            for k in range(K):
                rng.shuffle(permuted_arrays[k])

            # calculate aggregated weight for the pair (i, j)
            total_weight = 0.0
            for k in range(K):
                labels = permuted_arrays[k]
                weight_map = weight_maps[k]
                
                if labels[i] == labels[j]:
                    cluster_id = labels[i]
                    weight = weight_map.get(cluster_id, 0.0)
                    total_weight += weight

            # average over all K clusterings
            samples[perm] = total_weight / K

        empirical_mean = float(samples.mean())
        empirical_std = float(samples.std(ddof=1))

        return empirical_mean, empirical_std
    
    def analyze_window_stability_trends(self, window_ids: List[str]) -> Dict[str, Any]:
        """analyze trends in stability across temporal windows"""
        
        if len(window_ids) < 3:
            return {'error': 'need at least 3 windows for trend analysis'}
        
        # get drift analysis
        drift_results = self.analyze_temporal_drift(window_ids)
        
        if 'error' in drift_results:
            return drift_results
        
        consecutive_sims = drift_results['consecutive_similarities']
        
        if not consecutive_sims:
            return {'error': 'no consecutive similarities computed'}
        
        # analyze trends
        x = np.arange(len(consecutive_sims))
        y = np.array(consecutive_sims)
        
        # linear trend
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        else:
            slope, intercept = 0, y[0] if len(y) > 0 else 0
            trend_direction = 'insufficient_data'
        
        # stability statistics
        stability_stats = {
            'mean_stability': np.mean(y),
            'std_stability': np.std(y),
            'min_stability': np.min(y),
            'max_stability': np.max(y),
            'trend_slope': slope,
            'trend_intercept': intercept,
            'trend_direction': trend_direction
        }
        
        # identify periods of low stability
        threshold = np.mean(y) - np.std(y)  # one std below mean
        unstable_periods = []
        
        for i, sim in enumerate(consecutive_sims):
            if sim < threshold:
                unstable_periods.append({
                    'window_transition': f"{window_ids[i]} -> {window_ids[i+1]}",
                    'similarity': sim,
                    'position': i
                })
        
        return {
            'stability_stats': stability_stats,
            'consecutive_similarities': consecutive_sims,
            'unstable_periods': unstable_periods,
            'trend_analysis': {
                'slope': slope,
                'direction': trend_direction,
                'strength': abs(slope)
            },
            'window_ids': window_ids
        }