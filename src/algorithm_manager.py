from recpack.algorithms import ItemKNN, Popularity
import numpy as np
from scipy.sparse import csr_matrix

def train_algorithms(matrix, algos_names, k=20):
    models = {}
    for name in algos_names:
        if name == "Popularity":
            model = Popularity()
        elif "ItemKNN" in name:
            similarity = "cosine" if "Cosine" in name else "jaccard"
            model = ItemKNN(K=k, similarity=similarity)
        # Add more mappings if needed
            
        model.fit(matrix)
        models[name] = model
    return models

def predict_for_mock_user(models, history_items, num_total_items, top_k=10):
    """
    Generates predictions for a mock user defined by history_items.
    history_items: list of item indices.
    """
    # Create user interaction vector
    rows = [0] * len(history_items)
    cols = history_items
    data = [1.0] * len(history_items)
    
    mock_user_matrix = csr_matrix((data, (rows, cols)), shape=(1, num_total_items))
    
    results = {}
    
    for name, model in models.items():
        try:
            scores = model.predict(mock_user_matrix)
             # dense array for this single user
            user_scores = scores.toarray()[0]
            
            # Set history items to -inf to avoid recommending them
            # Ensure indices are within bounds of user_scores
            valid_history = [i for i in history_items if i < len(user_scores)]
            user_scores[valid_history] = -np.inf
            
            top_indices = np.argsort(user_scores)[::-1][:top_k]
            top_scores = user_scores[top_indices]
            
            results[name] = list(zip(top_indices, top_scores))
        except Exception as e:
            results[name] = f"Error during prediction: {str(e)}"
            
    return results
