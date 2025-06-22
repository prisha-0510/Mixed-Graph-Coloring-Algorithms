# from src.load_data import load_data
# from src.load_graphs import load_graphs_from_file
# from src.predict import predict_colors
# import sys
# import time
# import copy
# import numpy as np
# from contextlib import contextmanager
# from src.load_model import load_model
# from src.model import DeepGCN

# # Store the original stdout
# original_stdout = sys.stdout

# @contextmanager
# def redirect_stdout(new_stdout):
#     """Context manager to temporarily redirect stdout"""
#     sys.stdout = new_stdout
#     try:
#         yield
#     finally:
#         sys.stdout = original_stdout

# # Add debug print function that prints to both console and file
# def debug_print(message, file=None):
#     print(message)
#     if file:
#         print(message, file=file)
#         file.flush()  # Ensure it's written immediately

# def k_reduction(adj_list, k):
#     """
#     Remove vertices with degree < k from the original graph (non-iterative).
    
#     Args:
#         adj_list (list): Adjacency list representation of the graph
#         k (int): Degree threshold
    
#     Returns:
#         tuple: (new_adj_list, vertex_map) where vertex_map maps new indices to original indices
#     """
#     # Calculate degrees for all vertices in the original graph
#     degrees = [len(neighbors) for neighbors in adj_list]
    
#     # Identify vertices to keep (those with degree >= k)
#     keep_vertices = [v for v, deg in enumerate(degrees) if deg >= k]
#     vertex_map = {new_idx: old_idx for new_idx, old_idx in enumerate(keep_vertices)}
    
#     # Create new adjacency list with only the kept vertices
#     new_adj_list = []
#     for old_idx in keep_vertices:
#         # Get neighbors that are kept
#         valid_neighbors = [n for n in adj_list[old_idx] if n in keep_vertices]
        
#         # Map old indices to new indices
#         new_neighbors = []
#         for old_neighbor in valid_neighbors:
#             new_neighbor = keep_vertices.index(old_neighbor)
#             new_neighbors.append(new_neighbor)
        
#         new_adj_list.append(new_neighbors)
    
#     return new_adj_list, vertex_map

# def binary_search_coloring(model, adj_list):
#     """
#     Implements binary search to find the minimum number of colors required.
    
#     Args:
#         model: Trained model for coloring prediction
#         adj_list (list): Adjacency list representation of the graph
    
#     Returns:
#         int: Minimum number of colors required
#     """
#     # Initial range
#     l = 2  # Minimum possible colors (at least 2 for any non-trivial graph)
    
#     # Get initial upper bound
#     debug_print("Computing initial upper bound...")
#     r = predict_colors(model, adj_list)
#     debug_print(f"Initial upper bound: {r}")
    
#     min_colors = float('inf')
    
#     # Store info for reporting
#     iterations = []
    
#     # Binary search
#     while l <= r:
#         mid = (l + r) // 2
#         debug_print(f"Trying k={mid}...")
        
#         # Apply k-reduction
#         reduced_adj_list, _ = k_reduction(adj_list, mid)
        
#         # If the reduced graph is empty, all vertices had degree < k
#         if not reduced_adj_list:
#             debug_print(f"Reduced graph is empty, colors required ≤ {mid}")
#             colors_required = 0  # Empty graph needs 0 colors
#         else:
#             # Predict colors for reduced graph
#             colors_required = predict_colors(model, reduced_adj_list)
        
#         debug_print(f"k={mid}, reduced graph colors={colors_required}")
        
#         # Store iteration info
#         iterations.append({
#             'k': mid,
#             'colors_required': colors_required
#         })
        
#         if colors_required <= mid:
#             min_colors = mid
#             debug_print(f"Found valid coloring with {mid} colors")
#             r = mid - 1
#         else:
#             debug_print(f"Need more than {mid} colors")
#             l = mid + 1
    
#     # If we didn't find a solution, return the upper bound
#     if min_colors == float('inf'):
#         min_colors = r
    
#     return min_colors, iterations

# # Main execution
# if __name__ == "__main__":
#     # Configuration
#     graphs_file = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/networkx_graphs.pkl'
#     bin_output_file = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/bin.txt'
#     model_path = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/model_parameters/gcn_model.pth'
#     names_file = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/graph_names.pkl'
#     colors_file = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/graph_colors.pkl'
    
#     # Load model
#     hidden_dim = 32
#     num_layers = 20
#     model = load_model(model_path, DeepGCN, hidden_dim, num_layers)
    
#     # Load graphs
#     load_start_time = time.time()
#     with redirect_stdout(original_stdout):
#         adj_lists, names, colors = load_graphs_from_file(graphs_file, names_file, colors_file)
#     print(f"Loaded {len(adj_lists)} graphs in {time.time() - load_start_time:.2f} seconds")
    
#     # Open output file for writing
#     bin_output_stream = open(bin_output_file, 'w', buffering=1)
#     print("Starting binary search graph processing...", file=bin_output_stream)
    
#     # Process each graph
#     for i, adj_list in enumerate(adj_lists):
#         graph_name = names[i] if i < len(names) else f"Graph {i+1}"
#         print(f"\nProcessing graph {i+1}/{len(adj_lists)}: {graph_name}", file=bin_output_stream)
        
#         # Run binary search with original stdout
#         bs_start_time = time.time()
#         with redirect_stdout(original_stdout):
#             best_colors, iterations = binary_search_coloring(model, adj_list)
        
#         processing_time = time.time() - bs_start_time
        
#         # Write only final results to file
#         print(f"  - Total execution time: {processing_time:.2f} seconds", file=bin_output_stream)
#         print(f"  - Minimum colors required: {best_colors}", file=bin_output_stream)
    
#     bin_output_stream.close()
#     print(f"Binary search processing complete. Results saved to {bin_output_file}")
from src.load_data import load_data
from src.load_graphs import load_graphs_from_file
from src.predict import predict_colors
import sys
import time
import copy
import numpy as np
import heapq
from contextlib import contextmanager
from src.load_model import load_model
from src.model import DeepGCN

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

# Add debug print function that prints to both console and file
def debug_print(message, file=None):
    print(message)
    if file:
        print(message, file=file)
        file.flush()  # Ensure it's written immediately

def iterative_k_reduction(adj_list, k):
    """
    Iteratively remove vertices with degree < k from the graph.
    
    Args:
        adj_list (list): Adjacency list representation of the graph
        k (int): Degree threshold
    
    Returns:
        tuple: (new_adj_list, vertex_map) where vertex_map maps new indices to original indices
    """
    # Calculate initial degrees for all vertices
    degrees = [len(neighbors) for neighbors in adj_list]
    
    # Initialize priority queue with (degree, vertex) pairs
    priority_queue = [(degrees[v], v) for v in range(len(adj_list))]
    heapq.heapify(priority_queue)
    
    # Set to track removed vertices
    removed = set()
    
    # Process vertices in order of increasing degree
    while priority_queue and priority_queue[0][0] < k:
        degree, vertex = heapq.heappop(priority_queue)
        
        # Skip if already processed or degree has changed
        if vertex in removed or degree != degrees[vertex]:
            continue
        
        # Mark vertex as removed
        removed.add(vertex)
        
        # Update degrees of neighbors
        for neighbor in adj_list[vertex]:
            if neighbor not in removed:
                # Decrease the degree of the neighbor
                degrees[neighbor] -= 1
                # Add updated degree to priority queue
                heapq.heappush(priority_queue, (degrees[neighbor], neighbor))
    
    # Create mapping from new indices to original indices
    # Get remaining vertices by considering all vertices not in the removed set
    kept_vertices = [v for v in range(len(adj_list)) if v not in removed]
    vertex_map = {new_idx: old_idx for new_idx, old_idx in enumerate(kept_vertices)}
    
    # Map from old indices to new indices
    old_to_new = {old_idx: new_idx for new_idx, old_idx in vertex_map.items()}
    
    # Create new adjacency list with only the kept vertices
    new_adj_list = []
    for old_idx in kept_vertices:
        # Get neighbors that are kept and map to new indices
        new_neighbors = [old_to_new[n] for n in adj_list[old_idx] if n not in removed]
        new_adj_list.append(new_neighbors)
    
    return new_adj_list, vertex_map

# Use the iterative k-reduction as the main k_reduction function
def k_reduction(adj_list, k):
    """
    Remove vertices with degree < k from the original graph.
    
    Args:
        adj_list (list): Adjacency list representation of the graph
        k (int): Degree threshold
    
    Returns:
        tuple: (new_adj_list, vertex_map) where vertex_map maps new indices to original indices
    """
    return iterative_k_reduction(adj_list, k)

def binary_search_coloring(model, adj_list):
    """
    Implements binary search to find the minimum number of colors required.
    
    Args:
        model: Trained model for coloring prediction
        adj_list (list): Adjacency list representation of the graph
    
    Returns:
        int: Minimum number of colors required
    """
    # Initial range
    l = 2  # Minimum possible colors (at least 2 for any non-trivial graph)
    
    # Get initial upper bound
    debug_print("Computing initial upper bound...")
    r = predict_colors(model, adj_list)
    debug_print(f"Initial upper bound: {r}")
    
    min_colors = float('inf')
    
    # Store info for reporting
    iterations = []
    
    # Binary search
    while l <= r:
        mid = (l + r) // 2
        debug_print(f"Trying k={mid}...")
        
        # Apply k-reduction
        reduced_adj_list, vertex_map = k_reduction(adj_list, mid)
        
        # If the reduced graph is empty, all vertices had degree < k
        if not reduced_adj_list:
            debug_print(f"Reduced graph is empty, colors required ≤ {mid}")
            colors_required = 0  # Empty graph needs 0 colors
        else:
            # Predict colors for reduced graph
            colors_required = predict_colors(model, reduced_adj_list)
        
        debug_print(f"k={mid}, reduced graph colors={colors_required}")
        
        # Store iteration info
        iterations.append({
            'k': mid,
            'colors_required': colors_required
        })
        
        if colors_required <= mid:
            min_colors = mid
            debug_print(f"Found valid coloring with {mid} colors")
            r = mid - 1
        else:
            debug_print(f"Need more than {mid} colors")
            l = mid + 1
    
    # If we didn't find a solution, return the upper bound
    if min_colors == float('inf'):
        min_colors = r
    
    return min_colors, iterations

# Main execution
if __name__ == "__main__":
    # Configuration
    graphs_file = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/networkx_graphs.pkl'
    bin_output_file = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/bin.txt'
    model_path = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/model_parameters/gcn_model.pth'
    names_file = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/graph_names.pkl'
    colors_file = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/graph_colors.pkl'
    
    # Load model
    hidden_dim = 32
    num_layers = 20
    model = load_model(model_path, DeepGCN, hidden_dim, num_layers)
    
    # Load graphs
    load_start_time = time.time()
    with redirect_stdout(original_stdout):
        adj_lists, names, colors = load_graphs_from_file(graphs_file, names_file, colors_file)
    print(f"Loaded {len(adj_lists)} graphs in {time.time() - load_start_time:.2f} seconds")
    
    # Open output file for writing
    bin_output_stream = open(bin_output_file, 'w', buffering=1)
    print("Starting binary search graph processing...", file=bin_output_stream)
    
    # Process each graph
    for i, adj_list in enumerate(adj_lists):
        graph_name = names[i] if i < len(names) else f"Graph {i+1}"
        print(f"\nProcessing graph {i+1}/{len(adj_lists)}: {graph_name}", file=bin_output_stream)
        
        # Run binary search with original stdout
        bs_start_time = time.time()
        with redirect_stdout(original_stdout):
            best_colors, iterations = binary_search_coloring(model, adj_list)
        
        processing_time = time.time() - bs_start_time
        
        # Write only final results to file
        print(f"  - Total execution time: {processing_time:.2f} seconds", file=bin_output_stream)
        print(f"  - Minimum colors required: {best_colors}", file=bin_output_stream)
    
    bin_output_stream.close()
    print(f"Binary search processing complete. Results saved to {bin_output_file}")