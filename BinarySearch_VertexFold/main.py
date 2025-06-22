from src.load_model import load_model
from src.model import DeepGCN
from src.predict import predict_colors
from src.load_graphs import load_graphs_from_file
from k_core_red import k_reduction
from large_graphs import process_graph
from large_graphs import get_reduced, get_graphs
from src.graph_reducer import GraphReducer
import sys
import time

graph_file = '/Users/prishajain/Desktop/MTP/BinarySearch_VertexFold/networkx_graphs.pkl'
output_file = '/Users/prishajain/Desktop/MTP/BinarySearch_VertexFold/output.txt'
names_file = '/Users/prishajain/Desktop/MTP/BinarySearch_VertexFold/graph_names.pkl'
colors_file = '/Users/prishajain/Desktop/MTP/BinarySearch_VertexFold/graph_colors.pkl'
model_path = '/Users/prishajain/Desktop/MTP/BinarySearch_VertexFold/model_parameters/gcn_model.pth'

g,names,colors = load_graphs_from_file(graph_file,names_file,colors_file)
graphs = {name: graph for graph, name in zip(g, names)}
# graphs = get_graphs()
# graphs = get_reduced(graphs)
hidden_dim = 32
num_layers = 20
model = load_model(model_path, DeepGCN, hidden_dim, num_layers)

# Convert NetworkX graphs to adjacency lists before calling predict_colors
def convert_to_adj_list(graph):
    """Convert a NetworkX graph to an adjacency list"""
    
    # Create a properly indexed adjacency list
    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    num_nodes = len(nodes)
    adj_list = [[] for _ in range(num_nodes)]
    
    for node in nodes:
        idx = node_to_idx[node]
        for neighbor in graph.neighbors(node):
            adj_list[idx].append(node_to_idx[neighbor])
            
    return adj_list

with open('temp.txt', 'w', buffering=1) as f:
    sys.stdout = f
    with open(output_file, 'w', buffering=1) as file:
        for graph_name, graph in graphs.items():
            start_time = time.time()
            # Convert NetworkX graph to adjacency list
            adj_list = convert_to_adj_list(graph)
            
            # Call predict_colors with adjacency list instead of NetworkX graph
            # file.write(f"{graph_name}: predicted colors = {predict_colors(model, adj_list)}\n")
            k = graph.number_of_nodes()
            l = 2
            r = k
            min_colors = k

            
            while l <= r:
                mid = (l + r) // 2
                
                new_graph = graph.copy()
                new_graph = k_reduction(new_graph, mid)
                new_graph = process_graph(new_graph, graph_name)
                # reducer = GraphReducer(new_graph)
                # reduced_graph = reducer.reduce_graph()
                new_adj_list = convert_to_adj_list(new_graph)
                new_colors = predict_colors(model, new_adj_list)
                
                
                if new_colors <= mid:
                    min_colors = mid
                    r = mid - 1
                    
                else:
                    l = mid+1

            end_time = time.time()
            execution_time = end_time - start_time
            file.write(f"Final colors required for {graph_name} = {min_colors}\n")
            file.write(f"Total execution time: {execution_time:.2f} seconds\n\n")