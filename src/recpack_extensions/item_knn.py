from recpack.algorithms import ItemKNN
from recpack.algorithms.util import to_binary
from recpack.util import get_top_K_values
from scipy.sparse import csr_matrix, diags
import numpy as np
from sklearn.preprocessing import Normalizer

def compute_jaccard_similarity(X: csr_matrix) -> csr_matrix:
    """Compute the jaccard similarity between the items in the matrix.
    
    J(i, j) = count(i ∩ j) / count(i ∪ j)
    """
    X_binary = to_binary(X)
    cooccurence = X_binary.T @ X_binary
    
    # Degree of each item (marginal counts)
    degree = np.array(X_binary.sum(axis=0)).flatten()
    
    rows, cols = cooccurence.nonzero()
    data = cooccurence.data
    
    # J = C / (D1 + D2 - C)
    denom = degree[rows] + degree[cols] - data
    # Avoid division by zero
    new_data = np.zeros_like(data, dtype=float)
    nz = denom > 0
    new_data[nz] = data[nz] / denom[nz]
    
    jaccard = csr_matrix((new_data, (rows, cols)), shape=cooccurence.shape)
    jaccard.setdiag(0)
    return jaccard

def compute_lift_similarity(X: csr_matrix) -> csr_matrix:
    """Compute the lift similarity between the items in the matrix.
    
    Lift(i, j) = P(i ∩ j) / (P(i) * P(j)) = (count(i ∩ j) * N) / (count(i) * count(j))
    """
    X_binary = to_binary(X)
    n_users = X_binary.shape[0]
    cooccurence = X_binary.T @ X_binary
    
    degree = np.array(X_binary.sum(axis=0)).flatten()
    
    rows, cols = cooccurence.nonzero()
    data = cooccurence.data
    
    # Lift = (coocc * N) / (deg1 * deg2)
    denom = degree[rows] * degree[cols]
    new_data = np.zeros_like(data, dtype=float)
    nz = denom > 0
    new_data[nz] = (data[nz] * n_users) / denom[nz]
    
    lift = csr_matrix((new_data, (rows, cols)), shape=cooccurence.shape)
    lift.setdiag(0)
    return lift

class ItemKNNExtended(ItemKNN):
    """
    Extended ItemKNN with support for 'jaccard', 'lift', and 'pearson' similarity metrics.
    """
    SUPPORTED_SIMILARITIES = ["cosine", "conditional_probability", "jaccard", "lift", "pearson"]
    
    def _fit(self, X: csr_matrix) -> None:
        if self.similarity in ["jaccard", "lift", "pearson"]:
            from recpack.algorithms.nearest_neighbour import (
                compute_pearson_similarity, 
                compute_conditional_probability,
                compute_cosine_similarity
            )
            transformer = Normalizer(norm="l1", copy=False)

            if self.normalize_X:
                X = transformer.transform(X)

            if self.similarity == "jaccard":
                item_similarities = compute_jaccard_similarity(X)
            elif self.similarity == "lift":
                item_similarities = compute_lift_similarity(X)
            elif self.similarity == "pearson":
                item_similarities = compute_pearson_similarity(X)
            
            item_similarities = get_top_K_values(item_similarities, K=self.K)

            if self.normalize_sim:
                item_similarities = transformer.transform(item_similarities)

            self.similarity_matrix_ = item_similarities
        else:
            # Fallback to original ItemKNN fit for cosine and conditional_probability
            super()._fit(X)
