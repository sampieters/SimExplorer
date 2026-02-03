import streamlit as st
import pandas as pd
import numpy as np
from src.constants import DATASETS, FILTERS, ALGORITHMS
from src.data_manager import load_dataset_as_df
from src.filter_manager import get_filter_args, apply_filter
from src.algorithm_manager import train_algorithms, predict_for_mock_user, get_algorithm_args, get_concise_name
from src.ui_components import (
    display_global_metrics, plot_popularity_distribution, 
    plot_similarity_distribution, plot_interaction_graph, plot_similarity_graph,
    display_similar_items, plot_similarity_heatmap
)
from recpack.matrix import InteractionMatrix

from src.metadata_manager import load_item_metadata

st.set_page_config(page_title="SimExp - RecPack Explorer", layout="wide")
st.title("🔍 RecPack Similarity Explorer")

# --- Session State Init ---
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = None
if "dataset_obj" not in st.session_state:
    st.session_state.dataset_obj = None
if "item_metadata" not in st.session_state:
    st.session_state.item_metadata = {}
if "active_filters" not in st.session_state:
    st.session_state.active_filters = [] 
if "active_algorithms" not in st.session_state:
    st.session_state.active_algorithms = []
if "models" not in st.session_state:
    st.session_state.models = {}

# ==========================================
# 1. Dataset Selection & Loading
# ==========================================
st.sidebar.header("1. Data")
selected_dataset = st.sidebar.selectbox("Select Dataset", list(DATASETS.keys()))

if st.sidebar.button("Load Dataset (Raw)"):
    with st.spinner(f"Loading {selected_dataset}..."):
        try:
            df, ds = load_dataset_as_df(selected_dataset)
            st.session_state.raw_df = df
            st.session_state.filtered_df = df
            st.session_state.dataset_obj = ds
            
            # Load metadata
            st.session_state.item_metadata = load_item_metadata(selected_dataset, ds)
            
            # Clear downstream
            st.session_state.models = {}
            st.success(f"Loaded {selected_dataset} with {len(df)} interactions.")
        except Exception as e:
            st.error(f"Error loading {selected_dataset}: {e}")

# ==========================================
# 2. Filter Builder
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("2. Filters")

# Add New Filter UI
filter_type = st.sidebar.selectbox("Filter Type", list(FILTERS.keys()))
# Dynamic Arguments

args_info = get_filter_args(filter_type)
arg_inputs = {}

for arg, info in args_info.items():
    if arg in ["item_ix", "user_ix"]:
        continue
        
    default = info['default']
    arg_type = info['type']
    arg_type_str = str(arg_type).lower()
    
    # Determine widget based on type or default value
    if 'bool' in arg_type_str or isinstance(default, bool):
        arg_inputs[arg] = st.sidebar.checkbox(f"{arg}", value=bool(default) if default is not None else False)
    elif 'int' in arg_type_str or isinstance(default, int):
        # If default is None but type is int, we probably need a number input
        val = int(default) if default is not None else 0
        arg_inputs[arg] = st.sidebar.number_input(f"{arg}", value=val, step=1)
    elif 'float' in arg_type_str or isinstance(default, float):
        val = float(default) if default is not None else 0.0
        arg_inputs[arg] = st.sidebar.number_input(f"{arg}", value=val, step=0.1)
    else:
        val = str(default) if default is not None else ""
        arg_inputs[arg] = st.sidebar.text_input(f"{arg}", value=val)

if st.sidebar.button("Add Filter"):
    st.session_state.active_filters.append({
        "name": filter_type,
        "params": arg_inputs
    })
    st.success(f"Added {filter_type}")

# List Active Filters
if st.session_state.active_filters:
    st.sidebar.subheader("Active Pipeline")
    for idx, f in enumerate(st.session_state.active_filters):
        col_rem, col_info = st.sidebar.columns([1, 4])
        if col_rem.button("❌", key=f"del_{idx}"):
            st.session_state.active_filters.pop(idx)
            st.rerun()
        col_info.markdown(f"**{f['name']}**")
        if f['params']:
            col_info.caption(f"{f['params']}")
        
    if st.sidebar.button("Apply Pipeline"):
        if st.session_state.raw_df is None:
            st.sidebar.warning("Load a dataset first!")
        else:
            with st.spinner("Applying filters..."):
                current_df = st.session_state.raw_df
                for step in st.session_state.active_filters:
                    current_df = apply_filter(current_df, step['name'], st.session_state.dataset_obj, **step['params'])
                st.session_state.filtered_df = current_df
                st.success("Pipeline Applied!")
else:
    st.sidebar.info("No filters active. Using raw data.")

# ==========================================
# 3. Algorithms
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("3. Algorithms")

# Add New Algorithm UI
algo_type = st.sidebar.selectbox("Algorithm", list(ALGORITHMS.keys()))

algo_args_info = get_algorithm_args(algo_type)
algo_arg_inputs = {}

for arg, info in algo_args_info.items():
    default = info['default']
    arg_type = info['type']
    arg_type_str = str(arg_type).lower()
    
    # Determine widget based on type or default value
    if 'bool' in arg_type_str or isinstance(default, bool):
        algo_arg_inputs[arg] = st.sidebar.checkbox(f"Alg: {arg}", value=bool(default) if default is not None else False)
    elif 'int' in arg_type_str or isinstance(default, int):
        val = int(default) if default is not None else 0
        algo_arg_inputs[arg] = st.sidebar.number_input(f"Alg: {arg}", value=val, step=1)
    elif 'float' in arg_type_str or isinstance(default, float):
        val = float(default) if default is not None else 0.0
        algo_arg_inputs[arg] = st.sidebar.number_input(f"Alg: {arg}", value=val, step=0.1)
    elif arg == 'similarity' and algo_type == 'ItemKNN':
            # Special case for ItemKNN similarity strings
            algo_arg_inputs[arg] = st.sidebar.selectbox(f"Alg: {arg}", ["cosine", "jaccard", "pearson", "lift", "conditional_probability"])
    else:
        val = str(default) if default is not None else ""
        algo_arg_inputs[arg] = st.sidebar.text_input(f"Alg: {arg}", value=val)

if st.sidebar.button("Add Algorithm"):
    st.session_state.active_algorithms.append({
        "name": algo_type,
        "params": algo_arg_inputs
    })
    st.success(f"Added {algo_type}")

# List Active Algorithms
if st.session_state.active_algorithms:
    st.sidebar.subheader("Active Algorithms")
    for idx, a in enumerate(st.session_state.active_algorithms):
        col_rem_a, col_info_a = st.sidebar.columns([1, 4])
        if col_rem_a.button("❌", key=f"del_algo_{idx}"):
            st.session_state.active_algorithms.pop(idx)
            st.rerun()
        concise_algo_name = get_concise_name(a['name'], a['params'])
        col_info_a.markdown(f"**{concise_algo_name}**")
        
    if st.sidebar.button("Train Models"):
        if st.session_state.filtered_df is None:
            st.error("No data available.")
        else:
            with st.spinner("Training models..."):
                try:
                    ds = st.session_state.dataset_obj
                    matrix = InteractionMatrix(st.session_state.filtered_df, ds.ITEM_IX, ds.USER_IX)
                    models = train_algorithms(matrix, st.session_state.active_algorithms)
                    st.session_state.models = models
                    st.success("Training Complete!")
                except Exception as e:
                    st.error(f"Training Error: {e}")
else:
    st.sidebar.info("Add algorithms to train.")

# ==========================================
# MAIN AREA DISPLAY
# ==========================================

st.header("Global Metrics")
if st.session_state.filtered_df is not None:
    ds = st.session_state.dataset_obj
    fm = InteractionMatrix(st.session_state.filtered_df, ds.ITEM_IX, ds.USER_IX)
    rm = InteractionMatrix(st.session_state.raw_df, ds.ITEM_IX, ds.USER_IX) if st.session_state.raw_df is not None else None
    
    deltas = {}
    if rm:
        def get_avg_pop(m):
            counts = np.array(m.values.sum(axis=0)).flatten()
            return np.mean(counts) if len(counts) > 0 else 0

        deltas = {
            'users': fm.num_active_users - rm.num_active_users,
            'items': fm.num_active_items - rm.num_active_items,
            'interactions': fm.num_interactions - rm.num_interactions,
            'sparsity': (100 * (1 - fm.density)) - (100 * (1 - rm.density)),
            'avg_pop': get_avg_pop(fm) - get_avg_pop(rm)
        }
        
    display_global_metrics(fm, "Processed Matrix", deltas)
    st.plotly_chart(plot_popularity_distribution(fm), width='stretch')
    
    with st.expander("🕸️ Interaction Network Explorer"):
        col_g1, col_g2, col_g3 = st.columns(3)
        
        # Calculate max degrees for sensible defaults
        item_degrees = np.array(fm.values.sum(axis=0)).flatten()
        user_degrees = np.array(fm.values.sum(axis=1)).flatten()
        max_item_deg = int(item_degrees.max()) if len(item_degrees) > 0 else 1
        
        min_i_deg = col_g1.number_input("Min Item Degree (In)", value=max_item_deg, step=1)
        min_u_deg = col_g2.number_input("Min User Degree (Out)", value=1, step=1)
        max_n = col_g3.number_input("Max Nodes", value=100, step=10)
        
        plot_interaction_graph(fm, item_names=st.session_state.item_metadata, min_item_degree=min_i_deg, min_user_degree=min_u_deg, max_nodes=max_n)
else:
    st.info("Start by loading a dataset from the sidebar.")

if st.session_state.models:
    st.divider()
    st.header("Algorithm Insights")
    
    # 1. Per-Algorithm Visualizations (Columns)
    algo_cols = st.columns(len(st.session_state.models))
    ds = st.session_state.dataset_obj
    fm = InteractionMatrix(st.session_state.filtered_df, ds.ITEM_IX, ds.USER_IX)
    item_metadata = st.session_state.item_metadata

    for idx, (name, model) in enumerate(st.session_state.models.items()):
        with algo_cols[idx]:
            st.subheader(f"📊 {name}")
            
            # Heatmap
            fig_heat = plot_similarity_heatmap(model, fm, item_names=item_metadata)
            if fig_heat:
                st.markdown("**Similarity Heatmap**")
                st.plotly_chart(fig_heat, width='stretch')
            else:
                st.info("Heatmap not available.")
            
            # Distribution
            fig_dist = plot_similarity_distribution(model, name)
            if fig_dist:
                st.markdown("**Score Distribution**")
                st.plotly_chart(fig_dist, width='stretch')

            # Similarity Network
            with st.expander(f"🕸️ {name} Network", expanded=False):
                col_s1, col_s2 = st.columns(2)
                sim_thresh = col_s1.slider("Threshold", 0.0, 1.0, 0.4, key=f"thresh_{name}")
                sim_deg = col_s2.number_input("Min Degree", 0, 100, 0, key=f"deg_{name}")
                sim_max_n = st.number_input("Max Nodes", 10, 200, 50, key=f"nodes_{name}")
                
                plot_similarity_graph(
                    model, 
                    item_names=item_metadata, 
                    min_similarity=sim_thresh, 
                    min_degree=sim_deg,
                    max_nodes=sim_max_n,
                    suffix=name
                )

    # 2. Comparison Tools (Full Width)
    st.divider()
    
    # New Similarity Explorer
    with st.expander("🔎 Similarity Explorer (Item-to-Item)"):
        st.write("Compare which items are most similar according to different algorithms.")
        # Get active item indices for the dropdown
        active_items = sorted(list(fm.active_items))
        
        selected_item = st.selectbox(
            "Select an Item to explore its neighbors:", 
            active_items[:2000],
            format_func=lambda x: f"{item_metadata.get(x, f'Item {x}')} (ID: {x})"
        )
        
        from src.ui_components import display_similar_items
        display_similar_items(st.session_state.models, selected_item, matrix=fm, item_names=item_metadata)

if st.session_state.models and st.session_state.filtered_df is not None:
    st.divider()
    st.header("Mock User & Recommendations")
    
    ds = st.session_state.dataset_obj
    fm = InteractionMatrix(st.session_state.filtered_df, ds.ITEM_IX, ds.USER_IX)
    
    # Use indices_in to get valid item IDs that exist in the matrix
    active_items = sorted(list(fm.active_items))
    item_metadata = st.session_state.item_metadata
    
    selected_history = st.multiselect(
        "Select items for user history:", 
        active_items[:2000],
        format_func=lambda x: f"{item_metadata.get(x, f'Item {x}')} (ID: {x})"
    )
    
    if st.button("Get Recommendations"):
        if not selected_history:
            st.warning("Select at least one item.")
        else:
            # num_total_items must be fm.shape[1] (max_id + 1) to match model expectations
            results = predict_for_mock_user(st.session_state.models, selected_history, fm.shape[1])
            
            # Get popularity for context
            pop = np.array(fm.values.sum(axis=0)).flatten()
            
            r_cols = st.columns(len(results))
            for idx, (algo_name, recs) in enumerate(results.items()):
                with r_cols[idx]:
                    st.markdown(f"**{algo_name}**")
                    if isinstance(recs, str):
                        st.error(recs)
                    else:
                        for rank, (item_id, score) in enumerate(recs, 1):
                            popularity = int(pop[item_id]) if item_id < len(pop) else 0
                            item_name = item_metadata.get(item_id, f"Item {item_id}")
                            st.write(f"{rank}. **{item_name}** ({score:.4f}, Pop: {popularity})")
