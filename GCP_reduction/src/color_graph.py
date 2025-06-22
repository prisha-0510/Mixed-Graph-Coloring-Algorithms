import random

def rlf(graph):
    """
    Implements the Recursive Largest First (RLF) graph coloring algorithm.

    Parameters:
        graph (networkx.Graph): A NetworkX graph to be colored.


    Returns:
        tuple: (color_count, coloring) where:
            - color_count (int): The number of colors required to color the graph.
            - coloring (dict): A dictionary mapping each node to its assigned color (1-indexed).
    """
    # Make a copy of the input graph to avoid modifying the original
    working_graph = graph.copy()
    color_count = 0  # Number of colors used
    coloring = {}    # Dictionary to store node-to-color mapping
    
    all_nodes = set(graph.nodes)  # Keep track of all original nodes
    
    while working_graph.nodes:
        # Increment color count for a new color
        color_count += 1

        # Initialize the set of uncolored nodes at the start of this iteration
        uncolored_nodes = set(working_graph.nodes)

        # Initialize V1 (uncolored nodes not adjacent to any colored node)
        V1 = set()
        # Pick the node with the maximal degree in the current graph
        start_node = max(uncolored_nodes, key=lambda x: working_graph.degree(x))
        V1.add(start_node)

        # Assign current color to the starting node
        colored_nodes = {start_node}
        uncolored_nodes.remove(start_node)
        
        # Assign the current color to this node in our coloring dictionary
        coloring[start_node] = color_count

        while True:
            # V2: Uncolored nodes adjacent to at least one colored node
            V2 = {node for node in uncolored_nodes if any(neigh in colored_nodes for neigh in working_graph.neighbors(node))}

            # Vt: Uncolored nodes not adjacent to any colored node
            Vt = uncolored_nodes - V2

            if not Vt:
                break
                
            # Create a subgraph induced by Vt
            subgraph_Vt = working_graph.subgraph(Vt)

            # Select a node in Vt with maximal degree in the subgraph induced by V2
            degrees = {node: sum(1 for neigh in graph.neighbors(node) if neigh in V2) for node in Vt}

            # Find the maximum degree
            max_degree = max(degrees.values(), default=0)

            # Collect all nodes with the maximum degree
            candidates = [node for node, degree in degrees.items() if degree == max_degree]
            next_node = min(candidates, key=lambda x: subgraph_Vt.degree(x), default=None)

            if next_node is None:
                break

            # Add the node to V1, mark it as colored, and remove it from uncolored_nodes
            V1.add(next_node)
            colored_nodes.add(next_node)
            uncolored_nodes.remove(next_node)
            
            # Assign the current color to this node in our coloring dictionary
            coloring[next_node] = color_count

        # Remove all nodes in V1 from the working graph
        working_graph.remove_nodes_from(V1)
    
    # Ensure all original nodes have a color assigned (in case the input graph was not connected)
    for node in all_nodes:
        if node not in coloring:
            color_count += 1
            coloring[node] = color_count

    return color_count, coloring

def rnd(graph):
    """
    Implements the Randomly Ordered Sequential (RND) graph coloring algorithm.

    Parameters:
        graph (networkx.Graph): A NetworkX graph to be colored.

    Returns:
        int: The number of colors required to color the graph.
    """
    if graph.number_of_nodes() == 0:
        return 0  
    # Step 1: Randomly order the nodes
    nodes = list(graph.nodes())
    random.shuffle(nodes)  # Randomly shuffle the nodes
    
    # Step 2: Initialize color assignments (empty initially)
    color_map = {}
    
    # Step 3: Assign colors to each node
    for node in nodes:
        # Find the colors already assigned to the neighbors
        neighbor_colors = {color_map[neighbor] for neighbor in graph.neighbors(node) if neighbor in color_map}
        
        # Assign the lowest possible color that hasn't been used by any neighbor
        color = 1
        while color in neighbor_colors:
            color += 1
        
        # Assign the chosen color to the node
        color_map[node] = color
    
    # The number of colors used is the maximum color assigned
    return max(color_map.values())

def lf(graph):
    if graph.number_of_nodes() == 0:
        return 0 
    nodes = sorted(graph.nodes(), key=lambda node: graph.degree(node), reverse=True)
    color_map = {}
    
    # Step 3: Assign colors to each node
    for node in nodes:
        # Find the colors already assigned to the neighbors
        neighbor_colors = {color_map[neighbor] for neighbor in graph.neighbors(node) if neighbor in color_map}
        
        # Assign the lowest possible color that hasn't been used by any neighbor
        color = 1
        while color in neighbor_colors:
            color += 1
        
        # Assign the chosen color to the node
        color_map[node] = color
    
    # The number of colors used is the maximum color assigned
    return max(color_map.values())

def sl(graph):
    """
    Implements the Smallest Last (SL) graph coloring algorithm.

    Parameters:
        graph (networkx.Graph): A NetworkX graph to be colored.

    Returns:
        int: The number of colors required to color the graph.
    """
    if graph.number_of_nodes() == 0:
        return 0 
    # Step 1: Create a copy of the graph to avoid modifying the original graph
    working_graph = graph.copy()

    # Step 2: Initialize the list to store the ordering
    node_ordering = []

    # Step 3: Recursively find the smallest degree node and remove it from the graph
    while working_graph.nodes:
        # Find the node with the smallest degree
        smallest_degree_node = min(working_graph.nodes, key=lambda node: working_graph.degree(node))

        # Add the node to the ordering
        node_ordering.append(smallest_degree_node)

        # Remove the node from the graph
        working_graph.remove_node(smallest_degree_node)

    node_ordering.reverse()

    # Step 4: Initialize color assignments
    color_map = {}

    # Step 5: Assign colors to the nodes based on the ordering
    for node in node_ordering:
        # Find the colors already assigned to the neighbors
        neighbor_colors = {color_map[neighbor] for neighbor in graph.neighbors(node) if neighbor in color_map}

        # Assign the lowest possible color that hasn't been used by any neighbor
        color = 1
        while color in neighbor_colors:
            color += 1

        # Assign the chosen color to the node
        color_map[node] = color

    # The number of colors used is the maximum color assigned
    return max(color_map.values())

# def mis_single_step_greedy(graph):
#     """
#     Computes the coloring of a graph by repeatedly finding and removing the Maximum Independent Set (MIS).
#     Each time an MIS is found, it is assigned a new color.

#     Parameters:
#         graph (networkx.Graph): The graph for which to compute the coloring.

#     Returns:
#         int: The number of colors used to color the graph.
#     """
    # num_colors = 0  # Number of colors used
    # current_graph = graph.copy()  # Create a copy of the original graph to avoid modifying the original

    # while current_graph.nodes:
    #     # Find the Maximum Independent Set (MIS) using Gurobi
    #     mis_nodes = greedy_mis_algorithm(current_graph)
        
    #     # Increment the number of colors used
    #     num_colors += 1
        
    #     # Remove the nodes in the MIS from the current graph
    #     mis_nodes_in_graph = [node for node, in_mis in mis_nodes.items() if in_mis == 1]
    #     current_graph.remove_nodes_from(mis_nodes_in_graph)

    # return num_colors

def dsatur(graph):
    """
    Implements the DSatur algorithm for graph coloring.

    Parameters:
        graph (networkx.Graph): An undirected graph.

    Returns:
        dict: A dictionary mapping each vertex to its assigned color.
    """
    if graph.number_of_nodes() == 0:
        return 0 
    # Initialize data structures
    colors = {}  # Maps nodes to their assigned color
    saturation_degree = {node: 0 for node in graph.nodes()}  # Saturation degree of each node (number of colored neighbors)
    uncolored_vertices = set(graph.nodes())

    # Precompute the degree of each vertex
    degree = dict(graph.degree())

    while uncolored_vertices:
        # Find the vertex with the maximum saturation degree
        # Break ties by selecting the vertex with the highest degree in the uncolored subgraph
        v = max(
            uncolored_vertices,
            key=lambda node: (saturation_degree[node], degree[node])
        )

        # Determine the smallest available color not used by neighbors of v
        neighbor_colors = {colors[neighbor] for neighbor in graph.neighbors(v) if neighbor in colors}
        color = 1
        while color in neighbor_colors:
            color += 1

        # Assign the chosen color to the vertex
        colors[v] = color
        uncolored_vertices.remove(v)

        # Update the saturation degree of the neighbors of v
        for neighbor in graph.neighbors(v):
            if neighbor in uncolored_vertices:
                neighbor_colors = {colors[n] for n in graph.neighbors(neighbor) if n in colors}
                saturation_degree[neighbor] = len(neighbor_colors)

    return max(colors.values())
