


from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, mutual_info_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import entropy
from collections import Counter, defaultdict
import numpy as np


"""
NOTE
Add any metrics to this file, make sure that any helper/util functions are prefixed with _
This way only the metric functions are automatically called by all_metrics()
"""


def conf_mat(partition1, partition2):
    aligned2 = _align_labels(partition1, partition2)
    # Create confusion matrix between partition1 and partition2
    cm = confusion_matrix(partition1, aligned2)
    
    return cm


def ari(partition1, partition2):
    return adjusted_rand_score(partition1, partition2)


def nmi(partition1, partition2):
    return float(normalized_mutual_info_score(partition1, partition2))


def _align_labels(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize overlap
    label_map = {col: row for row, col in zip(row_ind, col_ind)}
    aligned = [int(label_map[label]) for label in pred_labels]
    return aligned


# def edit_distance(partition1, partition2):
#     aligned_partition2 = _align_labels(partition1, partition2)
#     return sum([1 for (t1,t2) in zip(partition1, aligned_partition2) if t1!=t2])

def edit_distance(partition1, partition2):
    cm = conf_mat(partition1, partition2)
   
    total_elements = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(len(cm)))
    incorrect = total_elements - correct

    return incorrect


# def norm_edit_distance(partition1, partition2):
#     """returns 1 if edit-distance is zero (no difference), returns 0 if edit-distance is n"""
#     n = len(partition1)
#     assert n==len(partition2), "partitions must be the same length"
#     return 1 - (edit_distance(partition1, partition2) / n)
    
def norm_edit_distance(partition1, partition2):
    """returns 1 if edit-distance is zero (no difference), returns 0 if edit-distance is n"""
    cm = conf_mat(partition1, partition2)
   
    total_elements = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(len(cm)))
    incorrect = total_elements - correct

    return 1 - (incorrect / total_elements)  # Proportion of correct assignments


def split_join_distance(partition1, partition2):
    def compute_partition_sets(partition):
        groups = defaultdict(set)
        for idx, label in enumerate(partition):
            groups[label].add(idx)
        return groups
    groups1 = compute_partition_sets(partition1)
    groups2 = compute_partition_sets(partition2)

    total = 0
    for g1 in groups1.values():
        total += min(len(g1 - g2) + len(g2 - g1) for g2 in groups2.values())
    for g2 in groups2.values():
        total += min(len(g2 - g1) + len(g1 - g2) for g1 in groups1.values())
    
    return total / 2  # since we double-counted


def all_metrics(partition1, partition2, verbose=0):
    metrics = {}
    prefix = "_"
    for name, func in globals().items():
        if (
            callable(func)
            and not name.startswith(prefix)
            and func.__module__ == __name__
            and name != 'all_metrics'
        ):
            res = func(partition1, partition2)
            if verbose: print(f"{name}: {res}",end='\n\n')
            metrics[name] = res

    return metrics
