def vertex_fold(graph):
    """
    Perform one vertex folding reduction on the given graph, if possible.
    
    Parameters:
        graph (nx.Graph): A NetworkX undirected graph.
    
    Returns:
        bool: True if a fold was performed; False otherwise.
    """
    # Check if graph is empty
    if len(graph.nodes) == 0:
        return False
        
    next_id = max(graph.nodes) + 1
    
    # Process pendant vertices (degree 1)
    for v in list(graph.nodes()):
        if graph.degree(v) == 1:
            nbrs = list(graph.neighbors(v))
            u = nbrs[0]
            new_v = next_id
            next_id += 1
            if new_v in graph:
                continue
            graph.add_node(new_v)
            external_nbrs = set()
            for node in set(graph.neighbors(u)) | set(graph.neighbors(v)):
                if node not in {u, v}:
                    external_nbrs.add(node)
            for nbr in external_nbrs:
                graph.add_edge(new_v, nbr)

            graph.remove_node(u)
            if v in graph:
                graph.remove_node(v)
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
            # Create a new vertex name representing the folded group
            new_v = next_id
            next_id += 1
            # (Optional step: skip if the new folded vertex already exists)
            if new_v in graph:
                continue

            # Add new vertex to the graph
            graph.add_node(new_v)

            # Determine all external neighbors from u, v, and w (excluding u, v, w)
            external_nbrs = set()
            for node in set(graph.neighbors(u)) | set(graph.neighbors(v)) | set(graph.neighbors(w)):
                if node not in {u, v, w}:
                    external_nbrs.add(node)

            # Connect the new folded vertex to all these external neighbors
            for nbr in external_nbrs:
                graph.add_edge(new_v, nbr)

            # Remove the original vertices as they have been folded
            graph.remove_node(u)
            if v in graph:
                graph.remove_node(v)
            graph.remove_node(w)
            # A fold was performed; return True to indicate the graph changed.
            return True
            
        # Remove isolated vertices
        if graph.degree(v) == 0:
            graph.remove_node(v)
            if len(graph.nodes) > 0:  # Only return True if there are still nodes left
                return True
                
    # No eligible vertex was found for folding.
    return False
