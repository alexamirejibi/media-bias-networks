import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

import time

class EntityAdjacencyMatrixMethods:
    """
    Optimized methods for creating square adjacency matrices between entities from set-based data.
    Input: DataFrame where rows are entities, columns are sets, values are binary (0/1).
    Output: Square adjacency matrices (entity × entity) showing relationships.
    
    All methods have been optimized using:
    - Vectorized NumPy operations instead of nested loops
    - Proper handling of division by zero with np.errstate
    - Sparse matrix operations for memory efficiency
    - Existing optimized libraries (sklearn, scipy)
    """
    
    def __init__(self):
        self.df = None
        self.entities = None
        self.n_entities = None

    def set_data(self, cluster_df):
        self.df = cluster_df.astype(int)  # entities × sets
        self.entities = cluster_df.index.tolist()
        self.n_entities = len(self.entities)
        
    def method_1_cooccurrence_matrix(self):
        """
        Direct co-occurrence counts as adjacency matrix.
        A[i,j] = number of sets containing both entity i and entity j.
        """
        adjacency = self.df.dot(self.df.T)  # entities × entities
        np.fill_diagonal(adjacency.values, 0)  # remove self-connections
        return adjacency
    
    def method_2_jaccard_similarity(self):
        """
        Jaccard similarity implementation with explicit set operations.
        A[i,j] = |sets(i) ∩ sets(j)| / |sets(i) ∪ sets(j)|
        """
        # Convert to boolean numpy array
        data = self.df.values.astype(bool)
        n_entities = len(self.entities)
        
        # Initialize adjacency matrix
        adjacency = np.zeros((n_entities, n_entities), dtype=float)
        
        # Compute pairwise Jaccard similarities
        for i in range(n_entities):
            for j in range(i + 1, n_entities):  # Only compute upper triangle
                # Get boolean vectors for entities i and j
                entity_i = data[i]
                entity_j = data[j]
                
                # Compute intersection and union
                intersection_count = np.sum(entity_i & entity_j)
                union_count = np.sum(entity_i | entity_j)
                
                # Compute Jaccard similarity
                if union_count > 0:
                    jaccard_sim = intersection_count / union_count
                    adjacency[i, j] = jaccard_sim
                    adjacency[j, i] = jaccard_sim  # Make symmetric
        
        # Debug: Check result statistics
        non_zero_count = np.count_nonzero(adjacency)
        # total_elements = adjacency.size - len(adjacency) if n_entities > 0 else 0 # exclude diagonal
        total_elements = n_entities * (n_entities -1) # more robust way to get pairs
        
        if total_elements > 0:
            sparsity = 1 - (non_zero_count / total_elements)
            print(f"Jaccard matrix sparsity: {sparsity:.3f} ({non_zero_count}/{total_elements} non-zero)")
        else:
            print("Jaccard matrix: Not enough elements to calculate sparsity (or only 1 entity).")

        return pd.DataFrame(adjacency, index=self.entities, columns=self.entities)
    
    def method_3_dice_coefficient(self):
        """
        Dice coefficient between entities with explicit set operations.
        A[i,j] = 2 * |sets(i) ∩ sets(j)| / (|sets(i)| + |sets(j)|)
        """
        # Convert to boolean numpy array
        data = self.df.values.astype(bool)
        n_entities = len(self.entities)
        
        # Initialize adjacency matrix
        adjacency = np.zeros((n_entities, n_entities), dtype=float)
        
        # Compute pairwise Dice coefficients
        for i in range(n_entities):
            for j in range(i + 1, n_entities):  # Only compute upper triangle
                # Get boolean vectors for entities i and j
                entity_i = data[i]
                entity_j = data[j]
                
                # Compute intersection and individual set sizes
                intersection_count = np.sum(entity_i & entity_j)
                size_i = np.sum(entity_i)
                size_j = np.sum(entity_j)
                sum_sizes = size_i + size_j
                
                # Compute Dice coefficient
                if sum_sizes > 0:
                    dice_coeff = (2 * intersection_count) / sum_sizes
                    adjacency[i, j] = dice_coeff
                    adjacency[j, i] = dice_coeff  # Make symmetric
        
        # Debug: Check result statistics
        non_zero_count = np.count_nonzero(adjacency)
        total_elements = n_entities * (n_entities - 1)  # more robust way to get pairs
        
        if total_elements > 0:
            sparsity = 1 - (non_zero_count / total_elements)
            print(f"Dice matrix sparsity: {sparsity:.3f} ({non_zero_count}/{total_elements} non-zero)")
        else:
            print("Dice matrix: Not enough elements to calculate sparsity (or only 1 entity).")
            
        return pd.DataFrame(adjacency, index=self.entities, columns=self.entities)
    
    def method_4_cosine_similarity_embeddings(self, method='nmf', n_components=10):
        """
        Cosine similarity between entity embeddings.
        First creates embeddings, then computes pairwise cosine similarity.
        """
        # get embeddings using specified method - entities are rows
        if method == 'nmf':
            from sklearn.decomposition import NMF
            nmf = NMF(n_components=n_components, random_state=42, max_iter=500)
            embeddings = nmf.fit_transform(self.df)  # entities × components
        elif method == 'svd':
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            embeddings = svd.fit_transform(self.df)  # entities × components
        elif method == 'raw':
            embeddings = self.df.values # entities × sets
        else:
            raise ValueError("Method must be 'nmf', 'svd', or 'raw'")
        
        # compute cosine similarity
        adjacency = cosine_similarity(embeddings)
        np.fill_diagonal(adjacency, 0)
        
        return pd.DataFrame(adjacency, index=self.entities, columns=self.entities)
    
    def method_5_correlation_matrix(self):
        """
        Pearson correlation between entity occurrence patterns.
        A[i,j] = correlation between entity i and entity j across all sets.
        """
        adjacency = np.corrcoef(self.df)  # entities × entities correlation
        np.fill_diagonal(adjacency, 0)
        # replace NaNs with 0 (happens when entity appears in 0 or all sets)
        adjacency = np.nan_to_num(adjacency)
        
        return pd.DataFrame(adjacency, index=self.entities, columns=self.entities)
    
    def method_6_pmi_matrix(self):
        """
        Pointwise Mutual Information between entities (optimized vectorized version).
        A[i,j] = log(P(i,j) / (P(i) * P(j)))
        """
        n_sets = self.df.shape[1]  # number of sets
        data = self.df.values.astype(bool)
        
        # Calculate probabilities - how often each entity appears
        entity_probs = data.sum(axis=1) / n_sets  # P(entity)
        
        # Compute co-occurrence counts (intersection)
        cooccur_counts = np.dot(data, data.T)
        joint_probs = cooccur_counts / n_sets
        
        # Compute expected probabilities: P(i) * P(j)
        expected_probs = np.outer(entity_probs, entity_probs)
        
        # Compute PMI with proper handling of zeros
        with np.errstate(divide='ignore', invalid='ignore'):
            pmi = np.log(np.divide(joint_probs, expected_probs, 
                                 out=np.zeros_like(joint_probs), 
                                 where=(joint_probs > 0) & (expected_probs > 0)))
        
        # Use positive PMI and remove self-connections
        adjacency = np.maximum(0, pmi)
        np.fill_diagonal(adjacency, 0)
        
        return pd.DataFrame(adjacency, index=self.entities, columns=self.entities)
    
    def method_7_lift_matrix(self):
        """
        Lift (association rule strength) between entities (optimized vectorized version).
        A[i,j] = P(j|i) / P(j) = confidence / expected_confidence
        """
        n_sets = self.df.shape[1]
        data = self.df.values.astype(bool)
        
        # Calculate entity probabilities
        entity_probs = data.sum(axis=1) / n_sets  # P(entity)
        entity_counts = data.sum(axis=1)  # count of sets each entity appears in
        
        # Compute co-occurrence counts
        cooccur_counts = np.dot(data, data.T)
        
        # Compute confidence: P(j|i) = count(i,j) / count(i)
        with np.errstate(divide='ignore', invalid='ignore'):
            confidence = np.divide(cooccur_counts, entity_counts[:, np.newaxis], 
                                 out=np.zeros_like(cooccur_counts, dtype=float), 
                                 where=entity_counts[:, np.newaxis] > 0)
        
        # Compute lift: confidence / P(j)
        with np.errstate(divide='ignore', invalid='ignore'):
            lift = np.divide(confidence, entity_probs[np.newaxis, :], 
                           out=np.zeros_like(confidence), 
                           where=entity_probs[np.newaxis, :] > 0)
        
        # Subtract 1 so independence = 0, and ensure non-negative
        adjacency = np.maximum(0, lift - 1)
        np.fill_diagonal(adjacency, 0)
        
        return pd.DataFrame(adjacency, index=self.entities, columns=self.entities)
    
    def method_8_tfidf_similarity(self):
        """
        TF-IDF based similarity between entities (optimized version).
        Treats each entity as a 'document' of sets it appears in.
        """
        # Create documents for each entity more efficiently
        data = self.df.values.astype(bool)
        set_names = self.df.columns.astype(str)
        
        entity_documents = []
        for i in range(len(self.entities)):
            # Get indices where entity appears
            set_indices = np.where(data[i])[0]
            # Convert to set names and join
            entity_documents.append(' '.join(set_names[set_indices]))
        
        # Compute TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(entity_documents)
        
        # Compute cosine similarity
        adjacency = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(adjacency, 0)
        
        return pd.DataFrame(adjacency, index=self.entities, columns=self.entities)
    
    def method_9_conditional_probability(self):
        """
        Conditional probability matrix (optimized vectorized version).
        A[i,j] = P(j|i) = P(entity j appears | entity i appears)
        """
        data = self.df.values.astype(bool)
        
        # Compute co-occurrence counts
        cooccur_counts = np.dot(data, data.T)
        
        # Compute entity counts (how many sets each entity appears in)
        entity_counts = data.sum(axis=1)
        
        # Compute conditional probabilities: P(j|i) = count(i,j) / count(i)
        with np.errstate(divide='ignore', invalid='ignore'):
            adjacency = np.divide(cooccur_counts, entity_counts[:, np.newaxis], 
                                out=np.zeros_like(cooccur_counts, dtype=float), 
                                where=entity_counts[:, np.newaxis] > 0)
        
        # Remove self-connections
        np.fill_diagonal(adjacency, 0)
        
        return pd.DataFrame(adjacency, index=self.entities, columns=self.entities)
    
    def method_10_symmetric_conditional_prob(self):
        """
        Symmetric conditional probability.
        A[i,j] = (P(j|i) + P(i|j)) / 2
        """
        cond_prob = self.method_9_conditional_probability()
        adjacency = (cond_prob + cond_prob.T) / 2
        
        return adjacency
        
    def get_adjacency_matrix(self, method='cooccurrence', normalize=False, **kwargs):
        """
        Get adjacency matrix using specified method.
        
        Args:
            method: One of ['cooccurrence', 'jaccard', 'dice', 'cosine_nmf', 'cosine_svd', 
                           'cosine_raw', 'correlation', 'pmi', 'lift', 'tfidf', 'conditional', 
                           'symmetric_conditional', 'sparse_cooccurrence']
            normalize: If True, apply min-max normalization to the matrix.
            **kwargs: Additional arguments for specific methods
        
        Returns:
            pd.DataFrame: Square adjacency matrix (entity × entity)
        """
        if method == 'cooccurrence':
            adj_matrix = self.method_1_cooccurrence_matrix()
        elif method == 'jaccard':
            adj_matrix = self.method_2_jaccard_similarity()
        elif method == 'dice':
            adj_matrix = self.method_3_dice_coefficient()
        elif method == 'cosine_nmf':
            adj_matrix = self.method_4_cosine_similarity_embeddings('nmf', **kwargs)
        elif method == 'cosine_svd':
            adj_matrix = self.method_4_cosine_similarity_embeddings('svd', **kwargs)
        elif method == 'cosine_raw':
            adj_matrix = self.method_4_cosine_similarity_embeddings('raw', **kwargs)
        elif method == 'correlation':
            adj_matrix = self.method_5_correlation_matrix()
        elif method == 'pmi':
            adj_matrix = self.method_6_pmi_matrix()
        elif method == 'lift':
            adj_matrix = self.method_7_lift_matrix()
        elif method == 'tfidf':
            adj_matrix = self.method_8_tfidf_similarity()
        elif method == 'conditional':
            adj_matrix = self.method_9_conditional_probability()
        elif method == 'symmetric_conditional':
            adj_matrix = self.method_10_symmetric_conditional_prob()
        else:
            raise ValueError(f"Unknown method: {method}")

        if normalize:
            # from sklearn.preprocessing import StandardScaler
            from sklearn.preprocessing import MinMaxScaler
            
            # Get the shape and index/columns for reconstruction
            original_shape = adj_matrix.shape
            original_index = adj_matrix.index
            original_columns = adj_matrix.columns
            
            # Flatten the matrix for StandardScaler
            values_flat = adj_matrix.values.flatten().reshape(-1, 1)
            
            scaler = MinMaxScaler()
            normalized_values = scaler.fit_transform(values_flat)

            # multiply by 10
            normalized_values = normalized_values * 10
            
            # Reshape back to original matrix shape
            normalized_matrix = normalized_values.reshape(original_shape)
            
            # Shift everything up so the minimum value is 0
            min_val = normalized_matrix.min()
            normalized_matrix = normalized_matrix - min_val
            
            # Reconstruct DataFrame with original index and columns
            adj_matrix = pd.DataFrame(normalized_matrix, index=original_index, columns=original_columns)
            
        return adj_matrix
    
    def all_modeling_methods(self, normalize=True):
        """
        Generate all adjacency matrices.
        
        Args:
            normalize: If True, normalize all adjacency matrices.
        
        Returns:
            dict: All adjacency matrices
        """
        methods = {
            'cooccurrence': 'Direct co-occurrence counts',
            'jaccard': 'Jaccard similarity coefficient',
            'dice': 'Dice coefficient',
            # 'cosine_nmf': 'Cosine similarity of NMF embeddings',
            # 'cosine_svd': 'Cosine similarity of SVD embeddings',
            'cosine_raw': 'Cosine similarity of raw embeddings',
            'correlation': 'Pearson correlation',
            'pmi': 'Positive Pointwise Mutual Information',
            'lift': 'Association rule lift',
            'tfidf': 'TF-IDF cosine similarity',
            'conditional': 'Conditional probability P(j|i)',
            'symmetric_conditional': 'Symmetric conditional probability',
            'sparse_cooccurrence': 'Sparse matrix co-occurrence (memory efficient)'
        }
        
        results = {}
        
        for method_name in methods:
            try:
                start = time.time()
                print(f"creating adjacency matrix with {method_name}",end=' ')
                adj_matrix = self.get_adjacency_matrix(method_name, normalize=normalize)

                # adj_matrix = (adj_matrix.subtract(adj_matrix.min(axis=1), axis=0).divide(adj_matrix.max(axis=1) - adj_matrix.min(axis=1), axis=0))

                results[method_name] = adj_matrix
                end = time.time()
                print(f'{round(end-start,2)} seconds')
            except Exception as e:
                print(f"Error with {method_name}: {e}")
                
        return results


# def all_modeling_methods(verbose=0):
#     modelings_methods = []
#     prefix = "_"
#     for name, func in globals().items():
#         if (
#             callable(func)
#             and not name.startswith(prefix)
#             and func.__module__ == __name__
#             and name != 'all_modeling_methods'
#         ):
#             modelings_methods.append(func)

#     if verbose: print('experimenting with:\n',[m.__name__ for m in modelings_methods])
#     return modelings_methods
