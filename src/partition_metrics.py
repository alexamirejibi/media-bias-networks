


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


