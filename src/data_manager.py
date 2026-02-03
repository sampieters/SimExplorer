from .constants import DATASETS

def get_dataset_class(name):
    return DATASETS.get(name)

import streamlit as st

@st.cache_data
def _cached_load_df(name):
    cls = DATASETS.get(name)
    if not cls:
        raise ValueError(f"Dataset {name} not found.")
    
    dataset = cls()
    if hasattr(dataset, 'fetch_dataset'):
        dataset.fetch_dataset()
    elif hasattr(dataset, 'fetch'):
        dataset.fetch()
        
    df = dataset._load_dataframe()
    
    # Extract important metadata for the session
    metadata = {
        'ITEM_IX': getattr(dataset, 'ITEM_IX', None),
        'USER_IX': getattr(dataset, 'USER_IX', None),
        'TIMESTAMP_IX': getattr(dataset, 'TIMESTAMP_IX', None),
        'item_mapping': getattr(dataset, 'item_mapping_', None),
        'user_mapping': getattr(dataset, 'user_mapping_', None)
    }
    
    return df, metadata

def load_dataset_as_df(name):
    """
    Instantiates, fetches, and returns the RAW dataframe and a placeholder dataset object.
    Utilizes st.cache_data for speed.
    """
    df, metadata = _cached_load_df(name)
    
    # Create a dummy object to satisfy API expectations
    from types import SimpleNamespace
    dataset_obj = SimpleNamespace(**metadata)
    
    return df, dataset_obj

def load_dataset(name):
    # Existing load_dataset for backward compatibility if needed, 
    # but we should probably prefer the DF approach for the pipeline.
    df, ds = load_dataset_as_df(name)
    from recpack.matrix import InteractionMatrix
    return InteractionMatrix(df, ds.ITEM_IX, ds.USER_IX)
