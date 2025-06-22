import pickle
import networkx as nx
import os
from src.load_graphs import load_graphs_from_file


# File paths (adjust these paths as needed)
# graph_file   = '/Users/prishajain/Desktop/MTP/GCP_optimalMIS/networkx_graphs.pkl'
# names_file   = '/Users/prishajain/Desktop/MTP/GCP_optimalMIS/graph_names.pkl'
# colors_file  = '/Users/prishajain/Desktop/MTP/GCP_optimalMIS/graph_colors.pkl'
# output_file  = '/Users/prishajain/Desktop/MTP/GCP_optimalMIS/updated_graphs.pkl'

# # Load the graphs, names, and colors from file
# graphs, names, colors = load_graphs_from_file(graph_file, names_file, colors_file)


def vertex_fold(graph):
    """
    Perform one vertex folding reduction on the given graph, if possible.
    Uses the minimum node ID as the representative for the combined vertex.
    
    Parameters:
        graph (nx.Graph): A NetworkX undirected graph.
    
    Returns:
        bool: True if a fold was performed; False otherwise.
    """
    # Check if graph is empty
    if len(graph.nodes) == 0:
        return False
    
    # Process pendant vertices (degree 1)
    for v in list(graph.nodes()):
        if graph.degree(v) == 1:
            nbrs = list(graph.neighbors(v))
            u = nbrs[0]
            
            # Use the minimum node ID as the representative
            rep_node = min(u, v)
            
            # Get all external neighbors (excluding u and v)
            external_nbrs = set()
            for node in set(graph.neighbors(u)) | set(graph.neighbors(v)):
                if node not in {u, v}:
                    external_nbrs.add(node)
            
            # Remove both nodes
            graph.remove_node(u)
            if v in graph:  # Check if v still exists (might have been removed with u)
                graph.remove_node(v)
            
            # Add the representative node back
            graph.add_node(rep_node)
            
            # Connect the representative to all external neighbors
            for nbr in external_nbrs:
                graph.add_edge(rep_node, nbr)
                
            return True
            
    # Process degree-2 vertices
    for v in list(graph.nodes()):
        if graph.degree(v) == 2:
            nbrs = list(graph.neighbors(v))
            if len(nbrs) != 2:
                continue  # Safety check
            u, w = nbrs
            # Only fold if the two neighbors are not directly connected
            if graph.has_edge(u, w):
                continue
                
            # Use the minimum node ID as the representative
            rep_node = min(u, v, w)

            # Determine all external neighbors from u, v, and w (excluding u, v, w)
            external_nbrs = set()
            for node in set(graph.neighbors(u)) | set(graph.neighbors(v)) | set(graph.neighbors(w)):
                if node not in {u, v, w}:
                    external_nbrs.add(node)

            # Remove the original vertices
            graph.remove_node(u)
            if v in graph:
                graph.remove_node(v)
            if w in graph:
                graph.remove_node(w)
                
            # Add the representative node back
            graph.add_node(rep_node)
            
            # Connect the representative to all external neighbors
            for nbr in external_nbrs:
                graph.add_edge(rep_node, nbr)
                
            # A fold was performed; return True to indicate the graph changed.
            return True
            
    # Remove isolated vertices
    for v in list(graph.nodes()):
        if graph.degree(v) == 0:
            graph.remove_node(v)
            if len(graph.nodes) > 0:  # Only return True if there are still nodes left
                return True
                
    # No eligible vertex was found for folding.
    return False


# # Process each graph until no more folding is possible
# for i, G in enumerate(graphs):
#     print(f"Graph {i}: has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
#     reduction_performed = True
#     while reduction_performed:
#         reduction_performed = vertex_fold(G)
#     print(f"Graph {i}: reduced to {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# # Save the updated graphs (along with names and colors) to a new pickle file.
# graphs_dict = {i: graphs[i] for i in range(len(graphs))}
# names_dict = {i: names[i] for i in range(len(names))}
# colors_dict = {i: colors[i] for i in range(len(colors))}

# with open(output_file, 'wb') as f:
#     pickle.dump(graphs_dict, f)

# with open(names_file, 'wb') as f:
#     pickle.dump(names_dict, f)

# with open(colors_file, 'wb') as f:
#     pickle.dump(colors_dict, f)


# print("Vertex folding reduction complete. Updated graphs saved to:", output_file)
# graphs, names, colors = load_graphs_from_file(output_file, names_file, colors_file)