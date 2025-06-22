from src.load_data import load_data
from src.load_graphs import load_graphs_from_file
from src.predict import predict_colors
import sys
import time
import networkx as nx
from contextlib import contextmanager
from src.load_model import load_model
from src.model import DeepGCN
from src.graph_reducer import GraphReducer

# Store the original stdout
original_stdout = sys.stdout

@contextmanager
def redirect_stdout(new_stdout):
    """Context manager to temporarily redirect stdout"""
    sys.stdout = new_stdout
    try:
        yield
    finally:
        sys.stdout = original_stdout

# Function to convert adjacency list to NetworkX graph
def adj_list_to_nx_graph(adj_list):
    """Convert adjacency list to NetworkX graph"""
    G = nx.Graph()
    for node, neighbors in enumerate(adj_list):
        G.add_node(node)
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    return G

# Function to convert NetworkX graph to adjacency list
def nx_graph_to_adj_list(graph):
    """Convert NetworkX graph to adjacency list"""
    adj_list = []
    nodes = sorted(graph.nodes())
    
    # Create a mapping from node labels to indices
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    for node in nodes:
        neighbors = [node_to_idx[neighbor] for neighbor in graph.neighbors(node)]
        adj_list.append(neighbors)
    
    return adj_list

# Configuration
graphs_file = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/networkx_graphs.pkl'
output_file = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/hybrid.txt'
model_path = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/model_parameters/gcn_model.pth'
names_file = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/graph_names.pkl'
colors_file = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/graph_colors.pkl'

# Open output file for writing
output_stream = open(output_file, 'w', buffering=1)

# Load model silently
with redirect_stdout(None):
    model = load_model(model_path, DeepGCN, 32, 20)

# Load graphs silently
with redirect_stdout(None):
    adj_lists, names, colors = load_graphs_from_file(graphs_file, names_file, colors_file)

# Process each graph
for i, adj_list in enumerate(adj_lists):
    if(i<68):
        continue
    graph_name = names[i] if i < len(names) else f"Graph_{i}"
    graph_known_colors = colors[i] if i < len(colors) else "Unknown"
    
    # Convert adjacency list to NetworkX graph
    overall_start_time = time.time()
    nx_graph = adj_list_to_nx_graph(adj_list)
    
    # Get original graph stats
    original_nodes = len(nx_graph.nodes())
    original_edges = len(nx_graph.edges())
    
    # Only print when coloring process starts
    print(f"Graph {i+1}/{len(adj_lists)}: {graph_name}", file=output_stream)
    print(f"  - Nodes: {original_nodes}", file=output_stream)
    print(f"  - Edges: {original_edges}", file=output_stream)
    print(f"  - Known colors: {graph_known_colors}", file=output_stream)
    
    # Perform graph reduction silently
    reduction_start_time = time.time()
    with redirect_stdout(None):
        reducer = GraphReducer(nx_graph)
        reduced_graph = reducer.reduce_graph()
    
    reduction_time = time.time() - reduction_start_time
    
    # Get reduced graph stats
    reduced_nodes = len(reduced_graph.nodes())
    reduced_edges = len(reduced_graph.edges())
    
    # Calculate reduction percentages
    node_reduction_pct = ((original_nodes - reduced_nodes) / original_nodes) * 100
    edge_reduction_pct = ((original_edges - reduced_edges) / original_edges) * 100
    
    print(f"  - Reduced nodes: {reduced_nodes} ({node_reduction_pct:.1f}% reduction)", file=output_stream)
    print(f"  - Reduced edges: {reduced_edges} ({edge_reduction_pct:.1f}% reduction)", file=output_stream)
    print(f"  - Reduction time: {reduction_time:.2f} seconds", file=output_stream)
    
    # Convert reduced graph back to adjacency list
    reduced_adj_list = nx_graph_to_adj_list(reduced_graph)
    
    # Predict colors on reduced graph
    coloring_start_time = time.time()
    with redirect_stdout(None):
        num_colors = predict_colors(model, reduced_adj_list)
    
    coloring_time = time.time() - coloring_start_time
    total_time = time.time() - overall_start_time
    
    print(f"  - Coloring time: {coloring_time:.2f} seconds", file=output_stream)
    print(f"  - Colors required: {num_colors}", file=output_stream)
    print(f"  - Total execution time: {total_time:.2f} seconds", file=output_stream)
    print("", file=output_stream)

output_stream.close()
print(f"Results saved to {output_file}")