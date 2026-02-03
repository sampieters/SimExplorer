from .constants import ALGORITHMS
import inspect
import numpy as np
from scipy.sparse import csr_matrix

def get_algorithm_class(name):
    return ALGORITHMS.get(name)

def get_algorithm_args(name):
    """
    Returns a dict of argument names and a dict with {default, type} for the algorithm's __init__.
    """
    cls = get_algorithm_class(name)
    if not cls:
        return {}
    
    sig = inspect.signature(cls.__init__)
    args = {}
    for param_name, param in sig.parameters.items():
        if param_name in ['self', 'X']: # X is usually passed in fit or if pre-calculated
            continue
        
        default = param.default if param.default is not inspect.Parameter.empty else None
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else None
        
        args[param_name] = {
            'default': default,
            'type': annotation
        }
        
    return args

def get_concise_name(name, params):
    """
    Generates a concise display name for an algorithm based on its parameters.
    """
    defaults = get_algorithm_args(name)
    relevant_params = []
    
    # Sort params for stable naming
    sorted_params = sorted(params.items())
    
    for p, val in sorted_params:
        # If the parameter differs from the default, or if it's a 'core' parameter
        if p in defaults and val != defaults[p]['default']:
            # Shorten common parameter names for display
            short_p = p.replace('similarity', 'sim').replace('pop_discount', 'pop')
            relevant_params.append(f"{short_p}={val}")
    
    if relevant_params:
        return f"{name} [{', '.join(relevant_params)}]"
    else:
        return name

import streamlit as st
from recpack.matrix import InteractionMatrix

@st.cache_resource
def _train_single_algorithm_cached(df, item_ix, user_ix, algo_name, params_tuple):
    """
    Caches the trained model object.
    """
    matrix = InteractionMatrix(df, item_ix, user_ix)
    cls = get_algorithm_class(algo_name)
    params = dict(params_tuple)
    
    model = cls(**params)
    model.fit(matrix)
    return model

def train_algorithms(matrix, algorithm_configs):
    """
    matrix: The InteractionMatrix (we'll extract the DF from it to use as a cache key)
    algorithm_configs: List of dicts, e.g., [{"name": "ItemKNN", "params": {"K": 20, "similarity": "cosine"}}]
    """
    models = {}
    df = matrix._df # InteractionMatrix stores the dataframe internally in _df
    item_ix = InteractionMatrix.ITEM_IX
    user_ix = InteractionMatrix.USER_IX
    
    for config in algorithm_configs:
        name = config['name']
        params = config.get('params', {})
        
        cls = get_algorithm_class(name)
        if not cls:
            continue
            
        # Explicit Casting base on signature
        sig = inspect.signature(cls.__init__)
        for param_name, param in sig.parameters.items():
            if param_name in params and param.annotation is not inspect.Parameter.empty:
                anno = param.annotation
                val = params[param_name]
                try:
                    anno_str = str(anno).lower()
                    if 'int' in anno_str:
                        params[param_name] = int(val)
                    elif 'float' in anno_str:
                        params[param_name] = float(val)
                    elif 'bool' in anno_str:
                        if isinstance(val, str):
                            params[param_name] = val.lower() in ['true', '1', 't', 'y', 'yes']
                        else:
                            params[param_name] = bool(val)
                except (ValueError, TypeError):
                    pass
        
        # Convert params to a stable hashable format
        params_tuple = tuple(sorted(params.items()))
        
        model = _train_single_algorithm_cached(df, item_ix, user_ix, name, params_tuple)
        
        # Create a concise display name
        display_name = get_concise_name(name, params)
            
        # Ensure name is unique in the local dict if multiple identical configs are added
        base_display_name = display_name
        counter = 1
        while display_name in models:
            display_name = f"{base_display_name} #{counter}"
            counter += 1
            
        models[display_name] = model
        
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
