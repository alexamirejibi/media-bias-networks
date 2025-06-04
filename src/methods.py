"""
All network modeling and community detection methods using the plugin system
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import itertools
import warnings
warnings.filterwarnings('ignore')

from .plugins import network_method, community_method


# ============================================================================
# PARAMETER COMBINATION HELPERS
# ============================================================================

def generate_param_combinations(**param_ranges) -> List[Dict]:
    """generate all combinations of parameters from ranges"""
    if not param_ranges:
        return [{}]
    
    keys = list(param_ranges.keys())
    values = list(param_ranges.values())
    
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


def single_param_range(param_name: str, values: List) -> List[Dict]:
    """generate parameter combinations for a single parameter"""
    return [{param_name: value} for value in values]


# ============================================================================
# NORMALIZATION HELPERS
# ============================================================================

def normalize_to_01_range(matrix: np.ndarray) -> np.ndarray:
    """normalize matrix to 0-1 range using min-max scaling"""
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val == min_val:
        return np.zeros_like(matrix)
    return (matrix - min_val) / (max_val - min_val)


def transform_correlation_to_similarity(matrix: np.ndarray) -> np.ndarray:
    """transform correlation matrix (-1 to 1) to similarity (0 to 1)"""
    # option 1: take absolute value to treat negative correlation as similarity
    # option 2: shift and scale: (x + 1) / 2
    # using option 2 to preserve the sign information in a similarity context
    return (matrix + 1) / 2


def transform_unbounded_to_01(matrix: np.ndarray) -> np.ndarray:
    """transform unbounded positive values to 0-1 range"""
    # use sigmoid-like transformation: x / (1 + x)
    return matrix / (1 + matrix)


def transform_lift_to_similarity(matrix: np.ndarray) -> np.ndarray:
    """transform lift values to 0-1 similarity (1 = independence baseline)"""
    # lift > 1 indicates positive association, lift < 1 indicates negative association
    # map this to similarity where higher lift -> higher similarity
    matrix_clipped = np.clip(matrix, 0, None)  # ensure non-negative
    # normalize using sigmoid centered at 1: 1 / (1 + exp(-(lift - 1)))
    return 1 / (1 + np.exp(-(matrix_clipped - 1)))


# ============================================================================
# NETWORK ADJACENCY METHODS
# ============================================================================

@network_method('cooccurrence')
def cooccurrence_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """direct co-occurrence counts between entities"""
    adj = data.dot(data.T)
    np.fill_diagonal(adj.values, 0)
    # normalize to 0-1 range
    adj_normalized = normalize_to_01_range(adj.values)
    return pd.DataFrame(adj_normalized, index=adj.index, columns=adj.columns)


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
    # transform from -1,1 to 0,1 range
    adj_transformed = transform_correlation_to_similarity(adj)
    return pd.DataFrame(adj_transformed, index=data.index, columns=data.index)


@network_method('correlation')
def correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """pearson correlation between entities"""
    adj = np.corrcoef(data)
    np.fill_diagonal(adj, 0)
    adj = np.nan_to_num(adj)  # replace nans with 0
    # transform from -1,1 to 0,1 range
    adj_transformed = transform_correlation_to_similarity(adj)
    return pd.DataFrame(adj_transformed, index=data.index, columns=data.index)


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
    # transform unbounded to 0-1 range
    adj_transformed = transform_unbounded_to_01(adj)
    return pd.DataFrame(adj_transformed, index=entities, columns=entities)


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
    # transform lift to 0-1 similarity
    adj_transformed = transform_lift_to_similarity(lift)
    return pd.DataFrame(adj_transformed, index=entities, columns=entities)


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

@community_method('louvain', single_param_range('resolution', [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]))
def louvain_communities(adj_matrix: np.ndarray, params: Dict) -> Dict[int, int]:
    """louvain community detection with different resolutions"""
    import networkx as nx
    import community as community_louvain
    
    # handle negative weights by shifting values to make minimum = 0
    # this preserves relative relationships while ensuring non-negative weights
    min_val = np.min(adj_matrix)
    if min_val < 0:
        adj_matrix_shifted = adj_matrix - min_val
    else:
        adj_matrix_shifted = adj_matrix.copy()
    
    G = nx.from_numpy_array(adj_matrix_shifted)
    resolution = params.get('resolution', 1.0)
    partition = community_louvain.best_partition(G, resolution=resolution)
    return partition


@community_method('leiden', [
    {'objective_function': 'modularity'}
] + generate_param_combinations(
    resolution=[0.01, 0.05, 0.1, 0.2],
    objective_function=['CPM']
))
def leiden_communities(adj_matrix: np.ndarray, params: Dict) -> Dict[int, int]:
    """leiden community detection"""
    import igraph as ig
    import leidenalg as la
    
    # handle negative weights by shifting values to make minimum = 0
    min_val = np.min(adj_matrix)
    if min_val < 0:
        adj_matrix = adj_matrix - min_val
    
    # ensure perfect symmetry for undirected graphs (handles floating point precision issues)
    symmetric = np.allclose(adj_matrix, adj_matrix.T, rtol=1e-10, atol=1e-12)
    if symmetric:
        # make perfectly symmetric by averaging with transpose
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
        mode = "undirected"
    else:
        mode = "directed"
    
    G = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), mode=mode)
    
    # get objective function and parameters
    objective_function = params.get('objective_function', 'modularity')
    resolution = params.get('resolution', 0.1)
    
    if objective_function == 'modularity':
        partition = la.find_partition(G, la.ModularityVertexPartition)
    elif objective_function == 'CPM':
        partition = la.find_partition(G, la.CPMVertexPartition,
                                    resolution_parameter=resolution)
    else:
        partition = la.find_partition(G, la.ModularityVertexPartition)
    
    return {i: partition.membership[i] for i in range(len(partition.membership))}


@community_method('label_propagation', [{}])
def label_propagation_communities(adj_matrix: np.ndarray, params: Dict) -> Dict[int, int]:
    """asynchronous label propagation algorithm"""
    import networkx as nx
    
    seed = params.get('seed', None)
    G = nx.from_numpy_array(adj_matrix)
    
    communities = nx.community.asyn_lpa_communities(G, seed=seed)
    
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


@community_method('infomap', generate_param_combinations(
    num_trials=[1, 5, 10],
    two_level=[False, True]
))
def infomap_communities(adj_matrix: np.ndarray, params: Dict) -> Dict[int, int]:
    """infomap algorithm"""
    import igraph as ig
    import infomap
    
    # debug: check matrix properties
    print(f"infomap input matrix shape: {adj_matrix.shape}")
    print(f"infomap input matrix range: [{np.min(adj_matrix):.6f}, {np.max(adj_matrix):.6f}]")
    print(f"infomap input matrix non-zero elements: {np.count_nonzero(adj_matrix)}/{adj_matrix.size}")
    
    # checks symmetry
    mode = "undirected" if np.allclose(adj_matrix, adj_matrix.T) else "directed"
    if mode == "undirected":
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
    
    # scale weights to make them more meaningful for infomap
    # infomap works better with larger weight values
    if np.max(adj_matrix) > 0:
        adj_matrix = adj_matrix * 100  # scale up small weights
    
    # remove very small weights that might be noise
    threshold = np.max(adj_matrix) * 0.01  # 1% of max weight
    adj_matrix = np.where(adj_matrix < threshold, 0, adj_matrix)
    
    print(f"infomap after preprocessing range: [{np.min(adj_matrix):.6f}, {np.max(adj_matrix):.6f}]")
    print(f"infomap after preprocessing non-zero: {np.count_nonzero(adj_matrix)}/{adj_matrix.size}")
        
    G_ig = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), mode=mode)
    
    # get parameters
    num_trials = params.get('num_trials', 10)
    seed = params.get('seed', None)
    two_level = params.get('two_level', False)
    
    im = infomap.Infomap()
    if seed is not None:
        im.seed = seed
    if two_level:
        im.two_level = True
        
    # add links with weights
    n_links_added = 0
    for edge in G_ig.es:
        weight = edge['weight']
        if weight > 0:  # only add edges with positive weight
            im.add_link(edge.source, edge.target, weight=weight)
            n_links_added += 1
    
    print(f"infomap added {n_links_added} links to infomap")
    
    if n_links_added == 0:
        print("warning: no links added to infomap, returning single community")
        return {i: 0 for i in range(adj_matrix.shape[0])}
    
    im.run(num_trials=num_trials)
    
    # Extract community assignments
    partition = {}
    for node in im.nodes:
        partition[node.node_id] = node.module_id
    
    n_communities = len(set(partition.values()))
    print(f"infomap found {n_communities} communities")
    
    return partition


@community_method('spectral_clustering', generate_param_combinations(
    n_clusters=[2, 3, 4, 5, 6, 8],
    eigen_solver=['arpack'],
    assign_labels=['kmeans']
) + generate_param_combinations(
    n_clusters=[3, 5, 8],
    eigen_solver=['lobpcg'],
    assign_labels=['kmeans']
) + generate_param_combinations(
    n_clusters=[3, 5, 8],
    eigen_solver=['arpack'],
    assign_labels=['discretize']
))
def spectral_clustering_communities(adj_matrix: np.ndarray, params: Dict) -> Dict[int, int]:
    """spectral clustering community detection"""
    from sklearn.cluster import SpectralClustering
    
    n_clusters = params.get('n_clusters', 5)
    eigen_solver = params.get('eigen_solver', None)
    assign_labels = params.get('assign_labels', 'kmeans')
    
    spectral = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='precomputed',
        eigen_solver=eigen_solver,
        assign_labels=assign_labels,
        random_state=42
    )
    
    cluster_labels = spectral.fit_predict(adj_matrix)
    return {i: int(label) for i, label in enumerate(cluster_labels)}


@community_method('threshold_components', single_param_range('threshold', [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]))
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


@community_method('k_cores', single_param_range('k', [1, 2, 3, 4, 5]))
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


@community_method('fast_greedy', single_param_range('weights', [True, False]))
def fast_greedy_communities(adj_matrix: np.ndarray, params: Dict) -> Dict[int, int]:
    """fast greedy modularity optimization"""
    import igraph as ig
    
    symmetric = np.allclose(adj_matrix, adj_matrix.T)
    mode = "undirected" if symmetric else "directed"
    G = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), mode=mode)
    
    # get parameters
    use_weights = params.get('weights', True)
    weights = 'weight' if use_weights else None
    
    # fast greedy algorithm
    dendrogram = G.community_fastgreedy(weights=weights)
    communities = dendrogram.as_clustering()
    
    partition = {}
    for i, community in enumerate(communities):
        for node in community:
            partition[node] = i
    
    return partition


@community_method('walktrap', generate_param_combinations(
    steps=[2, 3, 4, 5, 6, 8],
    weights=[True]
) + generate_param_combinations(
    steps=[3, 4, 5],
    weights=[False]
))
def walktrap_communities(adj_matrix: np.ndarray, params: Dict) -> Dict[int, int]:
    """walktrap algorithm based on random walks"""
    import igraph as ig
    
    symmetric = np.allclose(adj_matrix, adj_matrix.T)
    mode = "undirected" if symmetric else "directed"
    G = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), mode=mode)
    
    # get parameters
    steps = params.get('steps', 4)
    use_weights = params.get('weights', True)
    weights = 'weight' if use_weights else None
    
    # walktrap algorithm
    dendrogram = G.community_walktrap(weights=weights, steps=steps)
    communities = dendrogram.as_clustering()
    
    partition = {}
    for i, community in enumerate(communities):
        for node in community:
            partition[node] = i
    
    return partition 