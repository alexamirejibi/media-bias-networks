"""
All network modeling and community detection methods using the plugin system
"""

import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

from .plugins import network_method, community_method


# ============================================================================
# NETWORK ADJACENCY METHODS
# ============================================================================

@network_method('cooccurrence')
def cooccurrence_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """direct co-occurrence counts between entities"""
    adj = data.dot(data.T)
    np.fill_diagonal(adj.values, 0)
    return adj


@network_method('jaccard')
def jaccard_similarity(data: pd.DataFrame) -> pd.DataFrame:
    """jaccard similarity between entities"""
    data_bool = data.values.astype(bool)
    n_entities = len(data)
    entities = data.index.tolist()
    adj = np.zeros((n_entities, n_entities), dtype=float)
    
    for i in range(n_entities):
        for j in range(i + 1, n_entities):
            entity_i, entity_j = data_bool[i], data_bool[j]
            intersection = np.sum(entity_i & entity_j)
            union = np.sum(entity_i | entity_j)
            
            if union > 0:
                jaccard_sim = intersection / union
                adj[i, j] = adj[j, i] = jaccard_sim
    
    return pd.DataFrame(adj, index=entities, columns=entities)


@network_method('dice')
def dice_coefficient(data: pd.DataFrame) -> pd.DataFrame:
    """dice coefficient between entities"""
    data_bool = data.values.astype(bool)
    n_entities = len(data)
    entities = data.index.tolist()
    adj = np.zeros((n_entities, n_entities), dtype=float)
    
    for i in range(n_entities):
        for j in range(i + 1, n_entities):
            entity_i, entity_j = data_bool[i], data_bool[j]
            intersection = np.sum(entity_i & entity_j)
            sum_sizes = np.sum(entity_i) + np.sum(entity_j)
            
            if sum_sizes > 0:
                dice_coeff = (2 * intersection) / sum_sizes
                adj[i, j] = adj[j, i] = dice_coeff
    
    return pd.DataFrame(adj, index=entities, columns=entities)


@network_method('cosine')
def cosine_similarity_method(data: pd.DataFrame) -> pd.DataFrame:
    """cosine similarity of raw vectors"""
    adj = cosine_similarity(data)
    np.fill_diagonal(adj, 0)
    return pd.DataFrame(adj, index=data.index, columns=data.index)


@network_method('correlation')
def correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """pearson correlation between entities"""
    adj = np.corrcoef(data)
    np.fill_diagonal(adj, 0)
    adj = np.nan_to_num(adj)  # replace nans with 0
    return pd.DataFrame(adj, index=data.index, columns=data.index)


@network_method('pmi')
def pointwise_mutual_information(data: pd.DataFrame) -> pd.DataFrame:
    """pointwise mutual information between entities"""
    n_sets = data.shape[1]
    data_bool = data.values.astype(bool)
    entities = data.index.tolist()
    
    entity_probs = data_bool.sum(axis=1) / n_sets
    cooccur_counts = np.dot(data_bool, data_bool.T)
    joint_probs = cooccur_counts / n_sets
    expected_probs = np.outer(entity_probs, entity_probs)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log(np.divide(joint_probs, expected_probs, 
                             out=np.zeros_like(joint_probs), 
                             where=(joint_probs > 0) & (expected_probs > 0)))
    
    adj = np.maximum(0, pmi)  # positive pmi
    np.fill_diagonal(adj, 0)
    return pd.DataFrame(adj, index=entities, columns=entities)


@network_method('lift')
def lift_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """lift (association rule strength) between entities"""
    n_sets = data.shape[1]
    data_bool = data.values.astype(bool)
    entities = data.index.tolist()
    
    entity_probs = data_bool.sum(axis=1) / n_sets
    cooccur_counts = np.dot(data_bool, data_bool.T)
    joint_probs = cooccur_counts / n_sets
    
    # lift = P(j|i) / P(j) = P(i,j) / (P(i) * P(j))
    expected_probs = np.outer(entity_probs, entity_probs)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        lift = np.divide(joint_probs, expected_probs,
                        out=np.zeros_like(joint_probs),
                        where=(expected_probs > 0))
    
    np.fill_diagonal(lift, 0)
    return pd.DataFrame(lift, index=entities, columns=entities)


@network_method('tfidf_similarity')
def tfidf_similarity(data: pd.DataFrame) -> pd.DataFrame:
    """tf-idf cosine similarity between entities"""
    from sklearn.feature_extraction.text import TfidfTransformer
    
    # treat entities as documents, clusters as terms
    tfidf = TfidfTransformer()
    tfidf_matrix = tfidf.fit_transform(data.values)
    
    # compute cosine similarity
    adj = cosine_similarity(tfidf_matrix.toarray())
    np.fill_diagonal(adj, 0)
    
    return pd.DataFrame(adj, index=data.index, columns=data.index)


@network_method('conditional_probability')
def conditional_probability_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """conditional probability P(j|i) between entities"""
    n_sets = data.shape[1]
    data_bool = data.values.astype(bool)
    entities = data.index.tolist()
    n_entities = len(entities)
    
    adj = np.zeros((n_entities, n_entities), dtype=float)
    
    for i in range(n_entities):
        entity_i_count = np.sum(data_bool[i])
        if entity_i_count > 0:
            for j in range(n_entities):
                if i != j:
                    intersection = np.sum(data_bool[i] & data_bool[j])
                    adj[i, j] = intersection / entity_i_count
    
    return pd.DataFrame(adj, index=entities, columns=entities)


# ============================================================================
# COMMUNITY DETECTION METHODS
# ============================================================================

@community_method('louvain', [
    {'resolution': 0.5}, 
    {'resolution': 1.0}, 
    {'resolution': 1.5}
])
def louvain_communities(adj_matrix: np.ndarray, params: Dict) -> Dict[int, int]:
    """louvain community detection with different resolutions"""
    import networkx as nx
    import community as community_louvain
    
    G = nx.from_numpy_array(adj_matrix)
    resolution = params.get('resolution', 1.0)
    partition = community_louvain.best_partition(G, resolution=resolution)
    return partition


@community_method('leiden', [{}])
def leiden_communities(adj_matrix: np.ndarray, params: Dict) -> Dict[int, int]:
    """leiden community detection"""
    import igraph as ig
    import leidenalg as la
    
    symmetric = np.allclose(adj_matrix, adj_matrix.T)
    mode = "undirected" if symmetric else "directed"
    G = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), mode=mode)
    
    partition = la.find_partition(G, la.ModularityVertexPartition)
    return {i: partition.membership[i] for i in range(len(partition.membership))}


@community_method('label_propagation', [{}])
def label_propagation_communities(adj_matrix: np.ndarray, params: Dict) -> Dict[int, int]:
    """asynchronous label propagation algorithm"""
    import networkx as nx
    
    G = nx.from_numpy_array(adj_matrix)
    communities = nx.community.asyn_lpa_communities(G)
    
    partition = {}
    for i, community in enumerate(communities):
        for node in community:
            partition[node] = i
    return partition


@community_method('girvan_newman', [{}])
def girvan_newman_communities(adj_matrix: np.ndarray, params: Dict) -> Dict[int, int]:
    """girvan-newman edge betweenness algorithm"""
    import networkx as nx
    
    G = nx.from_numpy_array(adj_matrix)
    communities_generator = nx.community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    
    partition = {}
    for i, community_set in enumerate(top_level_communities):
        for node in community_set:
            partition[node] = i
    return partition


@community_method('infomap', [{}])
def infomap_communities(adj_matrix: np.ndarray, params: Dict) -> Dict[int, int]:
    """infomap algorithm based on information theory"""
    import igraph as ig
    import infomap
    
    # checks symmetry
    mode = "undirected" if np.allclose(adj_matrix, adj_matrix.T) else "directed"
    if mode == "undirected":
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
        
    G_ig = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), mode=mode)
    
    im = infomap.Infomap()
    for edge in G_ig.es:
        im.add_link(edge.source, edge.target, weight=edge['weight'])
    
    im.run()
    return dict(im.get_modules())


@community_method('spectral_clustering', [
    {'n_clusters': 3}, 
    {'n_clusters': 5}, 
    {'n_clusters': 8}
])
def spectral_clustering_communities(adj_matrix: np.ndarray, params: Dict) -> Dict[int, int]:
    """spectral clustering community detection"""
    from sklearn.cluster import SpectralClustering
    
    n_clusters = params.get('n_clusters', 5)
    
    spectral = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='precomputed',
        random_state=42
    )
    
    cluster_labels = spectral.fit_predict(adj_matrix)
    return {i: int(label) for i, label in enumerate(cluster_labels)}


@community_method('threshold_components', [
    {'threshold': 0.1}, 
    {'threshold': 0.2}, 
    {'threshold': 0.3}
])
def threshold_based_communities(adj_matrix: np.ndarray, params: Dict) -> Dict[int, int]:
    """threshold-based connected components"""
    import networkx as nx
    
    threshold = params.get('threshold', 0.1)
    
    # threshold the matrix and find connected components
    thresholded = np.where(adj_matrix >= threshold, adj_matrix, 0)
    G = nx.from_numpy_array(thresholded)
    
    # get connected components
    components = list(nx.connected_components(G))
    
    # create partition dict
    partition = {}
    for comp_id, component in enumerate(components):
        for node in component:
            partition[node] = comp_id
    
    return partition


@community_method('k_cores', [
    {'k': 2}, 
    {'k': 3}
])
def k_core_communities(adj_matrix: np.ndarray, params: Dict) -> Dict[int, int]:
    """k-core based community detection"""
    import networkx as nx
    
    k = params.get('k', 2)
    
    G = nx.from_numpy_array(adj_matrix)
    
    # find k-core subgraph
    k_core = nx.k_core(G, k=k)
    
    # assign communities: nodes in k-core get community 0, others get community 1
    partition = {}
    k_core_nodes = set(k_core.nodes())
    
    for node in range(len(adj_matrix)):
        if node in k_core_nodes:
            partition[node] = 0  # core community
        else:
            partition[node] = 1  # periphery community
    
    return partition


@community_method('fast_greedy', [{}])
def fast_greedy_communities(adj_matrix: np.ndarray, params: Dict) -> Dict[int, int]:
    """fast greedy modularity optimization"""
    import igraph as ig
    
    symmetric = np.allclose(adj_matrix, adj_matrix.T)
    mode = "undirected" if symmetric else "directed"
    G = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), mode=mode)
    
    # fast greedy algorithm
    dendrogram = G.community_fastgreedy(weights='weight')
    communities = dendrogram.as_clustering()
    
    partition = {}
    for i, community in enumerate(communities):
        for node in community:
            partition[node] = i
    
    return partition


@community_method('walktrap', [{}])
def walktrap_communities(adj_matrix: np.ndarray, params: Dict) -> Dict[int, int]:
    """walktrap algorithm based on random walks"""
    import igraph as ig
    
    symmetric = np.allclose(adj_matrix, adj_matrix.T)
    mode = "undirected" if symmetric else "directed"
    G = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), mode=mode)
    
    # walktrap algorithm
    dendrogram = G.community_walktrap(weights='weight')
    communities = dendrogram.as_clustering()
    
    partition = {}
    for i, community in enumerate(communities):
        for node in community:
            partition[node] = i
    
    return partition 