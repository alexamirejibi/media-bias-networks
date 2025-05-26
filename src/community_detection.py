"""
Given an adjacency matrix, compute communities
"""

# import networkx.community.louvain_communities as community_louvain
import networkx as nx
import numpy as np
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def louvain(adj_mat):
    """
    Detects communities using the Louvain algorithm.

    Args:
        adj_mat (np.ndarray): Symmetric adjacency matrix.

    Returns:
        ?[type]: Mapping of node to community ID.
    """
    # load the karate club graph
    # G = nx.karate_club_graph()
    G = nx.from_numpy_array(adj_mat)

    #first compute the best partition
    partition = community_louvain.best_partition(G)

    # TODO delete. This is from the docs example
    # # draw the graph
    # pos = nx.spring_layout(G)
    # # color the nodes according to their partition
    # cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    # nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
    #                     cmap=cmap, node_color=list(partition.values()))
    # nx.draw_networkx_edges(G, pos, alpha=0.5)
    # plt.show()
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

