from .constants import FILTERS
import inspect

def get_filter_class(name):
    return FILTERS.get(name)

def get_filter_args(name):
    """
    Returns a dict of argument names and a dict with {default, type} for the filter's __init__.
    """
    cls = get_filter_class(name)
    if not cls:
        return {}
    
    sig = inspect.signature(cls.__init__)
    args = {}
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
        
        default = param.default if param.default is not inspect.Parameter.empty else None
        # Try to get type from annotation
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else None
        
        args[param_name] = {
            'default': default,
            'type': annotation
        }
        
    return args

import streamlit as st

@st.cache_data
def _apply_filter_cached(df, filter_name, params_tuple):
    """
    Internal cached function. params_tuple is a tuple of (key, value) pairs.
    """
    cls = get_filter_class(filter_name)
    kwargs = dict(params_tuple)
    filter_instance = cls(**kwargs)
    return filter_instance.apply(df)

def apply_filter(df, filter_name, dataset, **kwargs):
    """
    Applies a single filter to the dataframe, utilizing caching.
    """
    cls = get_filter_class(filter_name)
    if not cls:
        raise ValueError(f"Filter {filter_name} not found")
        
    sig = inspect.signature(cls.__init__)
    
    # Auto-fill ix if present in dataset and expected by filter
    if 'item_ix' in sig.parameters and 'item_ix' not in kwargs:
        val = getattr(dataset, 'ITEM_IX', None)
        if val: kwargs['item_ix'] = val
            
    if 'user_ix' in sig.parameters and 'user_ix' not in kwargs:
        val = getattr(dataset, 'USER_IX', None)
        if val: kwargs['user_ix'] = val
            
    # Explicit Casting based on signature
    for param_name, param in sig.parameters.items():
        if param_name in kwargs and param.annotation is not inspect.Parameter.empty:
            anno = param.annotation
            val = kwargs[param_name]
            try:
                anno_str = str(anno).lower()
                if 'int' in anno_str:
                    kwargs[param_name] = int(val)
                elif 'float' in anno_str:
                    kwargs[param_name] = float(val)
                elif 'bool' in anno_str:
                    if isinstance(val, str):
                        kwargs[param_name] = val.lower() in ['true', '1', 't', 'y', 'yes']
                    else:
                        kwargs[param_name] = bool(val)
            except (ValueError, TypeError):
                pass 
            
    # Convert kwargs to a stable hashable format for caching
    params_tuple = tuple(sorted(kwargs.items()))
    
    return _apply_filter_cached(df, filter_name, params_tuple)
