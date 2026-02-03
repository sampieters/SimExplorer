from .constants import DATASETS

def get_dataset_class(name):
    return DATASETS.get(name)

def load_dataset_as_df(name):
    """
    Instantiates, fetches, and returns the RAW dataframe and the dataset object.
    """
    cls = get_dataset_class(name)
    if not cls:
        raise ValueError(f"Dataset {name} not found.")
    
    dataset = cls()
    
    if hasattr(dataset, 'fetch_dataset'):
        dataset.fetch_dataset()
    elif hasattr(dataset, 'fetch'):
        dataset.fetch()
        
    df = dataset._load_dataframe()
    return df, dataset

def load_dataset(name):
    # Existing load_dataset for backward compatibility if needed, 
    # but we should probably prefer the DF approach for the pipeline.
    df, ds = load_dataset_as_df(name)
    from recpack.matrix import InteractionMatrix
    return InteractionMatrix(df, ds.ITEM_IX, ds.USER_IX)
