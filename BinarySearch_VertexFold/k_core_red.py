def k_reduction(graph, k):
    """
    Iteratively remove vertices with degree < k until all remaining vertices have degree ≥ k.
    
    Parameters:
    -----------
    graph : networkx.Graph
        The input graph to be reduced
    k : int
        The minimum degree threshold
    
    Returns:
    --------
    networkx.Graph
        A new graph where all vertices have degree ≥ k or an empty graph if all vertices are removed
    """
    # Create a copy of the graph to avoid modifying the original
    G = graph.copy()
    
    # Continue until no more vertices need to be removed
    vertices_removed = True
    
    while vertices_removed and len(G.nodes) > 0:  # Check if graph is not empty
        vertices_removed = False
        
        # Find all vertices with degree < k
        vertices_to_remove = [node for node in G.nodes() if G.degree(node) < k]
        
        # If we found any vertices to remove
        if vertices_to_remove:
            # Remove the vertices
            G.remove_nodes_from(vertices_to_remove)
            vertices_removed = True
    
    return G