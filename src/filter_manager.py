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

def apply_filter(df, filter_name, dataset, **kwargs):
    """
    Applies a single filter to the dataframe.
    """
    cls = get_filter_class(filter_name)
    if not cls:
        raise ValueError(f"Filter {filter_name} not found")
        
    # Auto-fill ix if present in dataset and expected by filter
    sig = inspect.signature(cls.__init__)
    
    if 'item_ix' in sig.parameters and 'item_ix' not in kwargs:
        if hasattr(dataset, 'ITEM_IX'):
            kwargs['item_ix'] = dataset.ITEM_IX
            
    if 'user_ix' in sig.parameters and 'user_ix' not in kwargs:
         if hasattr(dataset, 'USER_IX'):
            kwargs['user_ix'] = dataset.USER_IX
            
    # Explicit Casting based on signature
    for param_name, param in sig.parameters.items():
        if param_name in kwargs and param.annotation is not inspect.Parameter.empty:
            anno = param.annotation
            val = kwargs[param_name]
            try:
                # Handle types (both direct and string representations)
                anno_str = str(anno).lower()
                if 'int' in anno_str:
                    kwargs[param_name] = int(val)
                elif 'float' in anno_str:
                    kwargs[param_name] = float(val)
                elif 'bool' in anno_str:
                    # Handle string "true"/"false" if they somehow ended up here
                    if isinstance(val, str):
                        kwargs[param_name] = val.lower() in ['true', '1', 't', 'y', 'yes']
                    else:
                        kwargs[param_name] = bool(val)
            except (ValueError, TypeError):
                pass # Fallback to original value if casting fails
            
    filter_instance = cls(**kwargs)
    return filter_instance.apply(df)
