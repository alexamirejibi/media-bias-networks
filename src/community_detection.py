"""
Given an adjacency matrix, compute communities
"""

# import networkx.community.louvain_communities as community_louvain
import networkx as nx
import numpy as np


def louvain(adj_mat):
    """
    Detects communities using the Louvain algorithm.

    Args:
        adj_mat (np.ndarray): Symmetric adjacency matrix.

    Returns:
        ?[type]: Mapping of node to community ID.
    """
    G = nx.from_numpy_array(adj_mat)
    partition = nx.community.louvain_communities.best_partition(G)
    return partition


def all_comm_detection(verbose=0):
    comm_detection = []
    prefix = "_"
    for name, func in globals().items():
        if (
            callable(func)
            and not name.startswith(prefix)
            and func.__module__ == __name__
            and name != 'all_comm_detection'
        ):
            comm_detection.append(func)

    if verbose: print('experimenting with:\n',[m.__name__ for m in comm_detection])
    return comm_detection

