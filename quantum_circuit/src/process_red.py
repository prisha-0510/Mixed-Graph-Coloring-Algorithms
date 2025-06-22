from reduction import vertex_fold

def process_graph(graph):
    """
    Process a graph by repeatedly applying vertex folding until no more reductions are possible.
    
    Parameters:
        graph (nx.Graph): A NetworkX undirected graph.
        name (str): Name of the graph for logging purposes.
        
    Returns:
        nx.Graph: The reduced graph.
    """
    # Check if graph is empty
    if len(graph.nodes) == 0:
        return graph
        
    reduction_performed = True
    while reduction_performed:
        reduction_performed = vertex_fold(graph)
        # Break if graph becomes empty during reduction
        if len(graph.nodes) == 0:
            break
            
    return graph

