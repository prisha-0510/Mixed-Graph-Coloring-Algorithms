from src.predict import predict_mis
from src.predict_single import predict_mis_single
import random

def find_colors(graph,model):
    """
    Colors a graph by repeatedly extracting maximum independent sets.
    Selection criteria: Choose MIS that leaves largest MIS in residual graph.
    
    Args:
        graph (networkx.Graph): Input graph to be colored
        
    Returns:
        int: Number of colors used
        dict: Color assignments for each vertex (1-based color indices)
    """
    def get_residual_graph(G, mis_nodes):
        """Returns a copy of G with MIS nodes removed"""
        G_copy = G.copy()
        G_copy.remove_nodes_from(mis_nodes)
        return G_copy
        
    def get_mis_nodes(solution):
        """Extract nodes that are in the MIS (labeled 1) from solution dict"""
        return [node for node, label in solution.items() if label == 1]
    
    def find_best_mis(model,mis_solutions, current_graph):
        """
        Find the MIS that leaves the largest MIS in residual graph.
        Returns the chosen MIS solution dict.
        """
        best_size = 0
        best_candidates = []
        
        # Evaluate each MIS solution
        for mis_sol in mis_solutions:
            mis_nodes = get_mis_nodes(mis_sol)
            residual = get_residual_graph(current_graph, mis_nodes)
            
            # Find size of maximum MIS in residual graph
            if len(residual.nodes()) > 0:           
                mis = predict_mis_single(model,residual)
                residual_size = sum(1 for v in mis.values() if v == 1)                                       
            else:
                residual_size = 0
                return mis_sol
                
            # Update best candidates
            if residual_size > best_size:
                best_size = residual_size
                best_candidates = [mis_sol]
            elif residual_size == best_size:
                best_candidates.append(mis_sol)
        
        # Randomly choose from best candidates
        return random.choice(best_candidates) if best_candidates else None
    
    # Initialize
    current_graph = graph.copy()
    colors_used = 0
    color_assignments = {}
    

    # Main coloring loop
    while current_graph.number_of_nodes() > 0:
        # Get all possible MIS solutions for current graph
        mis_solutions = predict_mis(model,current_graph)

        # Find best MIS to remove
        chosen_mis = find_best_mis(model,mis_solutions, current_graph)
        # Increment color counter
        colors_used += 1
        
        # Assign current color to chosen MIS nodes
        mis_nodes = get_mis_nodes(chosen_mis)
        for node in mis_nodes:
            color_assignments[node] = colors_used
            
        # Update graph
        current_graph = get_residual_graph(current_graph, mis_nodes)
    return colors_used

