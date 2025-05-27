import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

class EntityAdjacencyMatrixMethods:
    """
    Methods for creating square adjacency matrices between entities from set-based data.
    Input: DataFrame where rows are entities, columns are sets, values are binary (0/1).
    Output: Square adjacency matrices (entity × entity) showing relationships.
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
        Jaccard similarity between entities.
        A[i,j] = |sets(i) ∩ sets(j)| / |sets(i) ∪ sets(j)|
        """
        adjacency = np.zeros((self.n_entities, self.n_entities))
        
        for i in range(self.n_entities):
            for j in range(i+1, self.n_entities):
                sets_i = set(self.df.iloc[i][self.df.iloc[i] == 1].index)
                sets_j = set(self.df.iloc[j][self.df.iloc[j] == 1].index)
                
                union_size = len(sets_i.union(sets_j))
                if union_size > 0:
                    jaccard = len(sets_i.intersection(sets_j)) / union_size
                    adjacency[i, j] = jaccard
                    adjacency[j, i] = jaccard
        
        return pd.DataFrame(adjacency, index=self.entities, columns=self.entities)
    
    def method_3_dice_coefficient(self):
        """
        Dice coefficient between entities.
        A[i,j] = 2 * |sets(i) ∩ sets(j)| / (|sets(i)| + |sets(j)|)
        """
        adjacency = np.zeros((self.n_entities, self.n_entities))
        
        for i in range(self.n_entities):
            for j in range(i+1, self.n_entities):
                sets_i = set(self.df.iloc[i][self.df.iloc[i] == 1].index)
                sets_j = set(self.df.iloc[j][self.df.iloc[j] == 1].index)
                
                if len(sets_i) + len(sets_j) > 0:
                    dice = 2 * len(sets_i.intersection(sets_j)) / (len(sets_i) + len(sets_j))
                    adjacency[i, j] = dice
                    adjacency[j, i] = dice
        
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
        else:
            raise ValueError("Method must be 'nmf' or 'svd'")
        
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
        Pointwise Mutual Information between entities.
        A[i,j] = log(P(i,j) / (P(i) * P(j)))
        """
        n_sets = self.df.shape[1]  # number of sets
        adjacency = np.zeros((self.n_entities, self.n_entities))
        
        # calculate probabilities - how often each entity appears
        entity_probs = self.df.sum(axis=1) / n_sets  # P(entity)
        
        for i in range(self.n_entities):
            for j in range(i+1, self.n_entities):
                # p(entity_i, entity_j) = co-occurrence probability
                cooccur_count = (self.df.iloc[i] * self.df.iloc[j]).sum()
                joint_prob = cooccur_count / n_sets
                
                if joint_prob > 0 and entity_probs.iloc[i] > 0 and entity_probs.iloc[j] > 0:
                    pmi = np.log(joint_prob / (entity_probs.iloc[i] * entity_probs.iloc[j]))
                    adjacency[i, j] = max(0, pmi)  # use positive PMI
                    adjacency[j, i] = adjacency[i, j]
        
        return pd.DataFrame(adjacency, index=self.entities, columns=self.entities)
    
    def method_7_lift_matrix(self):
        """
        Lift (association rule strength) between entities.
        A[i,j] = P(j|i) / P(j) = confidence / expected_confidence
        """
        n_sets = self.df.shape[1]
        adjacency = np.zeros((self.n_entities, self.n_entities))
        
        entity_probs = self.df.sum(axis=1) / n_sets  # P(entity)
        
        for i in range(self.n_entities):
            for j in range(self.n_entities):
                if i != j and entity_probs.iloc[j] > 0:
                    # P(j|i) = P(i,j) / P(i)
                    cooccur_count = (self.df.iloc[i] * self.df.iloc[j]).sum()
                    entity_i_count = self.df.iloc[i].sum()
                    
                    if entity_i_count > 0:
                        confidence = cooccur_count / entity_i_count
                        lift = confidence / entity_probs.iloc[j]
                        adjacency[i, j] = max(0, lift - 1)  # subtract 1 so independence = 0
        
        return pd.DataFrame(adjacency, index=self.entities, columns=self.entities)
    
    def method_8_tfidf_similarity(self):
        """
        TF-IDF based similarity between entities.
        Treats each entity as a 'document' of sets it appears in.
        """
        # create documents for each entity (sets they appear in)
        entity_documents = []
        for entity in self.entities:
            sets_with_entity = self.df[self.df[entity] == 1].index.astype(str)
            entity_documents.append(' '.join(sets_with_entity))
        
        # compute TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(entity_documents)
        
        # compute cosine similarity
        adjacency = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(adjacency, 0)
        
        return pd.DataFrame(adjacency, index=self.entities, columns=self.entities)
    
    def method_9_conditional_probability(self):
        """
        Conditional probability matrix.
        A[i,j] = P(j|i) = P(entity j appears | entity i appears)
        """
        adjacency = np.zeros((self.n_entities, self.n_entities))
        
        for i in range(self.n_entities):
            for j in range(self.n_entities):
                if i != j:
                    entity_i_count = self.df.iloc[i].sum()
                    if entity_i_count > 0:
                        cooccur_count = (self.df.iloc[i] * self.df.iloc[j]).sum()
                        adjacency[i, j] = cooccur_count / entity_i_count
        
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
                           'correlation', 'pmi', 'lift', 'tfidf', 'conditional', 'symmetric_conditional']
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
            min_val = adj_matrix.values.min()
            max_val = adj_matrix.values.max()
            if max_val > min_val:  # avoid division by zero if all values are the same
                adj_matrix = (adj_matrix - min_val) / (max_val - min_val)
            elif max_val == min_val and max_val != 0: # if all values are same but not zero, normalize to 1
                 adj_matrix = adj_matrix / max_val
            # if all values are 0, it remains 0, which is fine.
            
        return adj_matrix
    
    def all_modeling_methods(self, normalize=False):
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
            'cosine_nmf': 'Cosine similarity of NMF embeddings',
            'cosine_svd': 'Cosine similarity of SVD embeddings',
            'correlation': 'Pearson correlation',
            'pmi': 'Positive Pointwise Mutual Information',
            'lift': 'Association rule lift',
            'tfidf': 'TF-IDF cosine similarity',
            'conditional': 'Conditional probability P(j|i)',
            'symmetric_conditional': 'Symmetric conditional probability'
        }
        
        results = {}
        
        for method_name in methods:
            try:
                adj_matrix = self.get_adjacency_matrix(method_name, normalize=normalize)
                results[method_name] = adj_matrix
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
