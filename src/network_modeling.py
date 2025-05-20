"""
Given a co-occurence matrix, returns the adjacency matrix
"""

import numpy

def normalized_counts(cooc_matrix):
    """
    Normalize each row of the co-occurrence matrix so that it sums to 1.

    Args:
        cooc_matrix (np.ndarray): NxN co-occurrence matrix (non-negative).

    Returns:
        np.ndarray: NxN adjacency matrix with row-normalized values.
    """
    cooc_matrix = np.array(cooc_matrix, dtype=float)
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
