import pandas as pd
import os
import zipfile
import streamlit as st

def load_item_metadata(dataset_name, dataset_obj):
    """
    Returns a dict mapping item IDs to titles for known datasets.
    """
    metadata = {}
    path = dataset_obj.path if hasattr(dataset_obj, 'path') else 'data'
    
    try:
        if "MovieLens100K" in dataset_name:
            # RecPack might have zip in 'data'
            zip_path = os.path.join(path, "ml-100k.zip")
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as z:
                    # u.item: movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western |
                    with z.open('ml-100k/u.item') as f:
                        df = pd.read_csv(f, sep='|', header=None, encoding='latin-1', usecols=[0, 1])
                        metadata = dict(zip(df[0], df[1]))
            
        elif "MovieLens1M" in dataset_name:
            zip_path = os.path.join(path, "ml-1m.zip")
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as z:
                    # MovieID::Title::Genres
                    with z.open('ml-1m/movies.dat') as f:
                        df = pd.read_csv(f, sep='::', header=None, engine='python', encoding='latin-1', usecols=[0, 1])
                        metadata = dict(zip(df[0], df[1]))

        elif "MovieLens10M" in dataset_name:
            zip_path = os.path.join(path, "ml-10m.zip")
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as z:
                    # MovieID::Title::Genres
                    target = [n for n in z.namelist() if n.endswith('movies.dat')]
                    if target:
                        with z.open(target[0]) as f:
                            df = pd.read_csv(f, sep='::', header=None, engine='python', encoding='latin-1', usecols=[0, 1])
                            metadata = dict(zip(df[0], df[1]))

        elif "MovieLens25M" in dataset_name:
            zip_path = os.path.join(path, "ml-25m.zip")
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as z:
                    # movieId,title,genres
                    target = [n for n in z.namelist() if n.endswith('movies.csv')]
                    if target:
                        with z.open(target[0]) as f:
                            df = pd.read_csv(f)
                            metadata = dict(zip(df['movieId'], df['title']))

    except Exception as e:
        st.warning(f"Could not load metadata for {dataset_name}: {e}")
        
    return metadata
