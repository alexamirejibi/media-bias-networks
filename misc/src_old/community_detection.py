"""
Given an adjacency matrix, compute communities
"""

# import networkx.community.louvain_communities as community_louvain
import networkx as nx
import numpy as np
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import leidenalg as la
import igraph as ig
import infomap


def louvain(adj_mat,parameters):
    """
    Detects communities using the Louvain algorithm.

    Args:
        adj_mat (np.ndarray): Symmetric adjacency matrix.

    Returns:
        ?[type]: Mapping of node to community ID.
    """
    G = nx.from_numpy_array(adj_mat)
    resolution = parameters['resolution']
    partition = community_louvain.best_partition(G,resolution=resolution)
    return partition

def leiden(adj_mat,parameters):
    '''
    Detects communities using the Leiden algorithm.
    '''
    G = _convert_to_igraph(adj_mat)
    partition = la.find_partition(G, la.ModularityVertexPartition)
    
    # Convert partition to dictionary format like louvain
    partition_dict = {i: partition.membership[i] for i in range(len(partition.membership))}
    return partition_dict

def label_propagation(adj_mat, parameters):
    """
    detects communities using label propagation algorithm.
    fast method where nodes adopt most common label among neighbors.
    """
    G = nx.from_numpy_array(adj_mat)
    communities = nx.community.asyn_lpa_communities(G)
    
    # convert communities to dictionary format
    partition_dict = {}
    for i, community in enumerate(communities):
        for node in community:
            partition_dict[node] = i
    
    return partition_dict

def girvan_newman(adj_mat, parameters):
    """
    Detects communities using the Girvan-Newman algorithm.
    This method focuses on edge betweenness centrality.
    """
    G = nx.from_numpy_array(adj_mat)
    communities_generator = nx.community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    
    partition_dict = {}
    for i, community_set in enumerate(top_level_communities):
        for node in community_set:
            partition_dict[node] = i
            
    return partition_dict

def infomap_community(adj_mat, parameters):
    """
    Detects communities using the Infomap algorithm.
    This method is based on information theory and random walks.
    """
    # Convert adjacency matrix to igraph Graph
    
    G_ig = _convert_to_igraph(adj_mat)

    im = infomap.Infomap()
    for edge in G_ig.es:
        im.add_link(edge.source, edge.target, weight=edge['weight'])

    im.run()

    partition_dict = {}
    for node_id, community_id in im.get_modules().items():
        partition_dict[node_id] = community_id
        
    return partition_dict

def _convert_to_igraph(adj_mat):
    """
    Convert adjacency matrix to igraph Graph. Check for symmetry.
    """
    symmetric = np.allclose(adj_mat, adj_mat.T)
    if symmetric:
        G_ig = ig.Graph.Weighted_Adjacency(adj_mat.tolist(), mode="undirected")
    else:
        G_ig = ig.Graph.Weighted_Adjacency(adj_mat.tolist(), mode="directed")

    return G_ig

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



if __name__ == "__main__":
    adj_mat = np.array([[0, 1, 1, 0],
                        [1, 0, 1, 0],
                        [1, 1, 0, 1],
                        [0, 0, 1, 0]])
    # test all community detection methods
    print("\nTesting all community detection methods:")
    for method in all_comm_detection():
        result = method(adj_mat, parameters={'resolution': 1})
        # ensure result is a dict mapping node ids to community ids
        if not isinstance(result, dict):
            print(f"Warning: {method.__name__} returned {type(result)}, expected dict")
            continue
        print(f"{method.__name__}: {result}")