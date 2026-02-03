import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os

def display_global_metrics(matrix, title="Dataset Metrics", deltas=None):
    st.markdown(f"#### {title}")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate current values
    sparsity = 100 * (1 - matrix.density)
    interactions_per_item = np.array(matrix.values.sum(axis=0)).flatten()
    avg_pop = np.mean(interactions_per_item) if len(interactions_per_item) > 0 else 0

    def get_delta_color(val, inverse=False):
        if val == 0:
            return "off"
        if inverse:
            return "inverse"
        return "normal"

    with col1:
        d = deltas.get('users') if deltas else None
        st.metric("Total Users", matrix.num_active_users, delta=d, delta_color=get_delta_color(d))
    with col2:
        d = deltas.get('items') if deltas else None
        st.metric("Total Items", matrix.num_active_items, delta=d, delta_color=get_delta_color(d))
    with col3:
        d = deltas.get('interactions') if deltas else None
        st.metric("Interactions", matrix.num_interactions, delta=d, delta_color=get_delta_color(d))
    with col4:
        d = deltas.get('sparsity') if deltas else None
        st.metric("Sparsity", f"{sparsity:.2f}%", delta=f"{d:.2f}%" if d is not None else None, delta_color=get_delta_color(d, inverse=True))
    with col5:
        d = deltas.get('avg_pop') if deltas else None
        st.metric("Avg Popularity", f"{avg_pop:.2f}", delta=f"{d:.2f}" if d is not None else None, delta_color=get_delta_color(d))

def plot_popularity_distribution(matrix):
    # Ensure we use the sparse matrix values
    interactions_per_item = np.array(matrix.values.sum(axis=0)).flatten()
    sorted_popularity = np.sort(interactions_per_item)[::-1]
    
    fig = px.line(
        y=sorted_popularity, 
        x=range(len(sorted_popularity)), 
        labels={'x': 'Items (Sorted by Popularity)', 'y': 'Interaction Count'}, 
        title="Item Popularity Distribution (Long Tail)"
    )
    return fig

def plot_similarity_distribution(model, model_name):
    sim_matrix = getattr(model, 'similarity_matrix_', getattr(model, 'similarity_matrix', None))
    if sim_matrix is not None:
        # RecPack similarity_matrix is usually a sparse matrix
        sim_values = sim_matrix.data
        if len(sim_values) == 0:
            return None
        fig = px.histogram(
            sim_values, 
            nbins=30, 
            title=f"Similarity Score Distribution - {model_name}",
            labels={'value': 'Similarity Score'}
        )
        return fig
    return None

def plot_similarity_heatmap(model, matrix, item_names=None):
    """
    Plots a high-density heatmap of the similarity matrix, sorted by popularity.
    Labels are hidden to provide a clean birds-eye view of clusters.
    """
    sim_matrix = getattr(model, 'similarity_matrix_', getattr(model, 'similarity_matrix', None))
    if sim_matrix is None:
        return None
    
    # Get popularity (interaction counts)
    interactions_per_item = np.array(matrix.values.sum(axis=0)).flatten()
    num_items = sim_matrix.shape[0]
    
    if len(interactions_per_item) > num_items:
        interactions_per_item = interactions_per_item[:num_items]
    elif len(interactions_per_item) < num_items:
        interactions_per_item = np.pad(interactions_per_item, (0, num_items - len(interactions_per_item)))

    # Use a safe maximum to prevent browser/memory crashes
    MAX_HEATMAP_SIZE = 2000
    all_sorted_indices = np.argsort(interactions_per_item)[::-1]
    top_indices = all_sorted_indices[:MAX_HEATMAP_SIZE]
    
    # Extract submatrix
    sub_sim = sim_matrix[top_indices, :][:, top_indices].toarray()
    
    # Labels only for hover tooltips
    labels = [f"{item_names.get(idx, f'Item {idx}')} (Pop: {int(interactions_per_item[idx])})" for idx in top_indices]
    
    fig = go.Figure(data=go.Heatmap(
        z=sub_sim,
        x=labels,
        y=labels,
        colorscale='Viridis',
        hoverongaps=False,
        showscale=True,
        hovertemplate="<b>X:</b> %{x}<br><b>Y:</b> %{y}<br><b>Sim:</b> %{z:.4f}<extra></extra>")
    )
    
    fig.update_layout(
        title=f"Similarity Matrix (Top {len(top_indices)} items sorted by popularity)",
        xaxis_title="",
        yaxis_title="",
        height=700,
        xaxis=dict(showticklabels=False, fixedrange=False),
        yaxis=dict(showticklabels=False, fixedrange=False),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def display_similar_items(models, item_id, matrix=None, item_names=None, top_k=10):
    title_str = item_names.get(item_id, f"Item {item_id}") if item_names else f"Item {item_id}"
    st.markdown(f"#### Most Similar to {title_str}")
    
    # Calculate popularity if matrix is provided
    pop_map = {}
    if matrix is not None:
        # matrix.values is CSR. sum along axis 0 gives item popularity
        pop = np.array(matrix.values.sum(axis=0)).flatten()
        pop_map = {i: p for i, p in enumerate(pop)}

    cols = st.columns(len(models))
    for idx, (name, model) in enumerate(models.items()):
        with cols[idx]:
            st.markdown(f"**{name}**")
            sim_matrix = getattr(model, 'similarity_matrix_', getattr(model, 'similarity_matrix', None))
            if sim_matrix is not None:
                if item_id < sim_matrix.shape[0]:
                    row = sim_matrix.getrow(item_id)
                    indices = row.indices
                    data = row.data
                    
                    sorted_idx = np.argsort(data)[::-1][:top_k]
                    
                    for i in sorted_idx:
                        sim_item = indices[i]
                        score = data[i]
                        pop_info = f", Pop: {int(pop_map[sim_item])}" if sim_item in pop_map else ""
                        item_display = item_names.get(sim_item, f"Item {sim_item}") if item_names else f"Item {sim_item}"
                        st.write(f"- {item_display} (Sim: {score:.4f}{pop_info})")
                else:
                    st.warning("Item not in similarity matrix")
            else:
                st.write("N/A (No similarity matrix)")

def plot_network_graph(nodes, edges, node_types=None, title="Network Graph", height="500px", suffix=""):
    """
    Renders a pyvis network graph in Streamlit with transparent background.
    """
    net = Network(height=height, width="100%", bgcolor="rgba(0,0,0,0)", font_color="#666666", notebook=False)
    
    for node_id, label in nodes.items():
        color = "#97c2fc" 
        if node_types and node_id in node_types:
            if node_types[node_id] == 'user':
                color = "#fb7e81" 
            elif node_types[node_id] == 'item':
                color = "#7be141" 
        
        net.add_node(node_id, label=label, color=color)
        
    for source, target, weight in edges:
        net.add_edge(source, target, value=weight, title=f"Weight: {weight:.4f}")
        
    tmp_path = f"tmp_network_{suffix}.html"
    net.save_graph(tmp_path)
    
    with open(tmp_path, 'r', encoding='utf-8') as f:
        html = f.read()
    
    html = html.replace('background-color: white;', 'background-color: rgba(0,0,0,0);')
    html = html.replace('background-color: #ffffff;', 'background-color: rgba(0,0,0,0);')
    
    components.html(html, height=int(height[:-2]) + 50)
    os.remove(tmp_path)

def plot_interaction_graph(matrix, item_names=None, min_item_degree=5, min_user_degree=1, max_nodes=100):
    """
    Builds an interaction graph (Users <-> Items).
    """
    item_degrees = np.array(matrix.values.sum(axis=0)).flatten()
    user_degrees = np.array(matrix.values.sum(axis=1)).flatten()
    
    valid_items = np.where(item_degrees >= min_item_degree)[0]
    valid_users = np.where(user_degrees >= min_user_degree)[0]
    
    if len(valid_items) > max_nodes // 2:
        valid_items = valid_items[np.argsort(item_degrees[valid_items])[::-1][:max_nodes // 2]]
    
    if len(valid_users) > max_nodes // 2:
        valid_users = valid_users[np.argsort(user_degrees[valid_users])[::-1][:max_nodes // 2]]
        
    item_set = set(valid_items)
    user_set = set(valid_users)
    
    rows, cols = matrix.values.nonzero()
    
    nodes = {}
    edges = []
    node_types = {}
    
    for r, c in zip(rows, cols):
        if r in user_set and c in item_set:
            u_id = f"U_{r}"
            i_id = f"I_{c}"
            
            if i_id not in nodes:
                nodes[i_id] = item_names.get(c, f"Item {c}") if item_names else f"Item {c}"
                node_types[i_id] = 'item'
            
            if u_id not in nodes:
                nodes[u_id] = f"User {r}"
                node_types[u_id] = 'user'
            
            edges.append((u_id, i_id, 1))
                
    if not edges:
        st.warning("No nodes found with the selected degree/limit filters.")
        return
        
    plot_network_graph(nodes, edges, node_types, title="Interaction Network (Users & Items)", suffix="interactions")

def plot_similarity_graph(model, item_names=None, min_similarity=0.3, min_degree=0, max_nodes=50, suffix=""):
    """
    Builds a similarity graph (Items <-> Items).
    """
    sim_matrix = getattr(model, 'similarity_matrix_', getattr(model, 'similarity_matrix', None))
    if sim_matrix is None:
        st.warning("No similarity matrix available for this algorithm.")
        return
        
    rows, cols = sim_matrix.nonzero()
    data = sim_matrix.data
    
    significant_edges = []
    node_degree_counts = {}
    
    for r, c, val in zip(rows, cols, data):
        if r != c and val >= min_similarity:
            significant_edges.append((r, c, val))
            node_degree_counts[r] = node_degree_counts.get(r, 0) + 1
            node_degree_counts[c] = node_degree_counts.get(c, 0) + 1
            
    nodes = {}
    edges = []
    node_types = {}
    
    filtered_edges = [e for e in significant_edges if node_degree_counts.get(e[0], 0) >= min_degree and node_degree_counts.get(e[1], 0) >= min_degree]
    
    edge_count = 0
    for r, c, val in filtered_edges:
        i1 = f"I_{r}"
        i2 = f"I_{c}"
        
        if i1 not in nodes and len(nodes) < max_nodes:
            nodes[i1] = item_names.get(r, f"Item {r}") if item_names else f"Item {r}"
            node_types[i1] = 'item'
        
        if i2 not in nodes and len(nodes) < max_nodes:
            nodes[i2] = item_names.get(c, f"Item {c}") if item_names else f"Item {c}"
            node_types[i2] = 'item'
            
        if i1 in nodes and i2 in nodes:
            edges.append((i1, i2, val))
            edge_count += 1
            if len(nodes) >= max_nodes and edge_count > max_nodes * 2:
                break
            
    if not edges:
        st.warning("No similarities found with the selected filters.")
        return

    plot_network_graph(nodes, edges, node_types=node_types, title="Item Similarity Network", suffix=f"sim_{suffix}")
