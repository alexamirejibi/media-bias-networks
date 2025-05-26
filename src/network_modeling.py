"""
Given a co-occurence matrix, returns the adjacency matrix
"""

import numpy as np

def _cluster_data_to_cooccurrence(concatenated_df):
    co_occurrence_matrix = concatenated_df.dot(concatenated_df.T)
    co_occurrence_matrix = np.array(co_occurrence_matrix, dtype=float)
    return co_occurrence_matrix


def normalized_counts(cluster_data):
    """
    Normalize each row of the co-occurrence matrix so that it sums to 1.

    Args:
        cooc_matrix (np.ndarray): NxN co-occurrence matrix (non-negative).

    Returns:
        np.ndarray: NxN adjacency matrix with row-normalized values.
    """
    cooc_matrix = _cluster_data_to_cooccurrence(cluster_data)
    
    row_sums = cooc_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    return cooc_matrix / row_sums


def all_modeling_methods(verbose=0):
    modelings_methods = []
    prefix = "_"
    for name, func in globals().items():
        if (
            callable(func)
            and not name.startswith(prefix)
            and func.__module__ == __name__
            and name != 'all_modeling_methods'
        ):
            modelings_methods.append(func)

    if verbose: print('experimenting with:\n',[m.__name__ for m in modelings_methods])
    return modelings_methods


