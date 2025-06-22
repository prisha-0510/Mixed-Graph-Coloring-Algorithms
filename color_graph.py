import random
import copy
from gurobi_mis import gurobi_mis
from classical_greedy import greedy_mis_algorithm
import networkx as nx
import numpy as np

def rlf(graph):
    """
    Implements the Recursive Largest First (RLF) graph coloring algorithm.

    Parameters:
        graph (networkx.Graph): A NetworkX graph to be colored.

    Returns:
        int: The number of colors required to color the graph.
    """
    
    # Make a copy of the input graph to avoid modifying the original
    working_graph = graph.copy()
    color_count = 0  # Number of colors used

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

        # Remove all nodes in V1 from the working graph
        working_graph.remove_nodes_from(V1)

    return color_count

def rnd(graph):
    """
    Implements the Randomly Ordered Sequential (RND) graph coloring algorithm.

    Parameters:
        graph (networkx.Graph): A NetworkX graph to be colored.

    Returns:
        int: The number of colors required to color the graph.
    """
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

def mis_single_step(graph):
    """
    Computes the coloring of a graph by repeatedly finding and removing the Maximum Independent Set (MIS).
    Each time an MIS is found, it is assigned a new color.

    Parameters:
        graph (networkx.Graph): The graph for which to compute the coloring.

    Returns:
        int: The number of colors used to color the graph.
    """
    num_colors = 0  # Number of colors used
    current_graph = graph.copy()  # Create a copy of the original graph to avoid modifying the original

    while current_graph.nodes:
        # Find the Maximum Independent Set (MIS) using Gurobi
        mis_nodes = gurobi_mis(current_graph)
        
        # Increment the number of colors used
        num_colors += 1
        
        # Remove the nodes in the MIS from the current graph
        mis_nodes_in_graph = [node for node, in_mis in mis_nodes.items() if in_mis == 1]
        current_graph.remove_nodes_from(mis_nodes_in_graph)

    return num_colors

def mis_single_step_greedy(graph):
    """
    Computes the coloring of a graph by repeatedly finding and removing the Maximum Independent Set (MIS).
    Each time an MIS is found, it is assigned a new color.

    Parameters:
        graph (networkx.Graph): The graph for which to compute the coloring.

    Returns:
        int: The number of colors used to color the graph.
    """
    num_colors = 0  # Number of colors used
    current_graph = graph.copy()  # Create a copy of the original graph to avoid modifying the original

    while current_graph.nodes:
        # Find the Maximum Independent Set (MIS) using Gurobi
        mis_nodes = greedy_mis_algorithm(current_graph)
        
        # Increment the number of colors used
        num_colors += 1
        
        # Remove the nodes in the MIS from the current graph
        mis_nodes_in_graph = [node for node, in_mis in mis_nodes.items() if in_mis == 1]
        current_graph.remove_nodes_from(mis_nodes_in_graph)

    return num_colors

def dsatur(graph):
    """
    Implements the DSatur algorithm for graph coloring.

    Parameters:
        graph (networkx.Graph): An undirected graph.

    Returns:
        dict: A dictionary mapping each vertex to its assigned color.
    """
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


def new_alg(G):
    """
    Find the number of colors needed for proper coloring using NumPy optimizations.
    
    Parameters:
    G (networkx.Graph): Input graph
    
    Returns:
    int: Number of colors needed
    """
    # Convert NetworkX graph to NumPy adjacency matrix for faster operations
    adj_matrix = nx.to_numpy_array(G, dtype=int)
    n = adj_matrix.shape[0]
    
    # Initialize active vertices
    active_vertices = set(range(n))
    
    # Call the main recursive function
    colors = process_graph(adj_matrix, active_vertices)
    
    return colors

def process_graph(adj_matrix, active_vertices):
    """
    Process the graph and find the number of colors needed.
    
    Parameters:
    adj_matrix (numpy.ndarray): Adjacency matrix of the graph
    active_vertices (set): Currently active vertices
    
    Returns:
    int: Number of colors needed
    """
    if not active_vertices:
        return 0
    
    # Find connected components (more efficiently with NumPy)
    components = find_components(adj_matrix, active_vertices)
    
    if len(components) > 1:
        # Process each component separately and take the maximum
        max_component_colors = 0
        for component in components:
            component_colors = process_graph(adj_matrix, set(component))
            max_component_colors = max(max_component_colors, component_colors)
        return max_component_colors
    
    # Calculate degree vector once for efficiency
    degree_vector = np.sum(adj_matrix, axis=1)
    
    # Check if graph is an independent set (all degrees are 0)
    if np.sum(degree_vector[list(active_vertices)]) == 0:
        return 1
    
    # Check if graph is a clique (all nodes have degree = n-1 for active vertices)
    n = len(active_vertices)
    is_clique = all(degree_vector[i] == n-1 for i in active_vertices)
    if is_clique and n > 0:
        return n
    
    # Step 1: Find the highest degree node
    max_degree = max(degree_vector[i] for i in active_vertices)
    highest_degree_nodes = [node for node in active_vertices if degree_vector[node] == max_degree]
    highest_degree_node = random.choice(highest_degree_nodes)
    
    # Step 2: Get S1 and S2 more efficiently
    S1 = np.where(adj_matrix[highest_degree_node] == 1)[0]
    S1 = set(S1) & active_vertices  # Ensure S1 only has active vertices
    
    S2 = set()
    for node in S1:
        neighbors = np.where(adj_matrix[node] == 1)[0]
        S2.update(set(neighbors) & active_vertices - S1)
    
    # Step 3: Calculate independent sets I1 and I2
    # Sort S1 and S2 by degree
    S1_sorted = sorted(list(S1), key=lambda i: degree_vector[i], reverse=True)
    S2_sorted = sorted(list(S2), key=lambda i: degree_vector[i], reverse=True)
    
    # Build I1
    I1 = set()
    S1_remaining = set(S1_sorted)
    while S1_remaining:
        current = max(S1_remaining, key=lambda i: degree_vector[i])
        I1.add(current)
        S1_remaining.remove(current)
        # Find and remove neighbors efficiently
        neighbors = set(np.where(adj_matrix[current] == 1)[0]) & S1_remaining
        S1_remaining -= neighbors
    
    # Build I2
    I2 = set()
    S2_remaining = set(S2_sorted)
    while S2_remaining:
        current = max(S2_remaining, key=lambda i: degree_vector[i])
        I2.add(current)
        S2_remaining.remove(current)
        # Find and remove neighbors efficiently
        neighbors = set(np.where(adj_matrix[current] == 1)[0]) & S2_remaining
        S2_remaining -= neighbors
    
    # Handle case where we're stuck in a loop
    if not I1 and not I2:
        active_vertices.remove(highest_degree_node)
        # Zero out connections for this node
        adj_matrix[highest_degree_node, :] = 0
        adj_matrix[:, highest_degree_node] = 0
        return 1 + process_graph(adj_matrix, active_vertices)
    
    # Step 4: Collapse nodes in I1 and I2
    # Get the next available node IDs (use the max node ID + offset for clarity)
    next_node_id = adj_matrix.shape[0]
    collapsed_nodes = []
    
    # Handle I1 collapsing
    if I1:
        # Create a representative node for I1
        I1_rep = min(I1)  # Use the minimum ID from I1 as representative
        
        # For all other nodes in I1, add their edges to the representative
        for node in I1:
            if node != I1_rep:
                # Add edges to the representative node
                adj_matrix[I1_rep] = np.logical_or(adj_matrix[I1_rep], adj_matrix[node]).astype(int)
                adj_matrix[:, I1_rep] = np.logical_or(adj_matrix[:, I1_rep], adj_matrix[:, node]).astype(int)
                
                # Remove the node from active set
                active_vertices.remove(node)
                
                # Zero out all connections for this node
                adj_matrix[node, :] = 0
                adj_matrix[:, node] = 0
        
        # Avoid self-loops
        adj_matrix[I1_rep, I1_rep] = 0
        collapsed_nodes.append(I1_rep)
    
    # Handle I2 collapsing
    if I2:
        # Create a representative node for I2
        I2_rep = min(I2)  # Use the minimum ID from I2 as representative
        
        # For all other nodes in I2, add their edges to the representative
        for node in I2:
            if node != I2_rep:
                # Add edges to the representative node
                adj_matrix[I2_rep] = np.logical_or(adj_matrix[I2_rep], adj_matrix[node]).astype(int)
                adj_matrix[:, I2_rep] = np.logical_or(adj_matrix[:, I2_rep], adj_matrix[:, node]).astype(int)
                
                # Remove the node from active set
                active_vertices.remove(node)
                
                # Zero out all connections for this node
                adj_matrix[node, :] = 0
                adj_matrix[:, node] = 0
        
        # Avoid self-loops
        adj_matrix[I2_rep, I2_rep] = 0
        collapsed_nodes.append(I2_rep)
    
    # Step 5: Find and remove nodes with maximum degree
    # Find components after collapsing
    components = find_components(adj_matrix, active_vertices)
    
    total_colors = 0
    remaining_active = set(active_vertices)
    
    for component in components:
        component_set = set(component)
        comp_size = len(component_set)
        
        if comp_size == 0:
            continue
            
        # Find nodes with maximum degree in this component
        max_degree_nodes = []
        for node in component_set:
            # Count connections only to nodes in this component
            component_degree = sum(adj_matrix[node, other] for other in component_set)
            if component_degree == comp_size - 1:
                max_degree_nodes.append(node)
        
        # Remove max degree nodes from active_vertices
        for node in max_degree_nodes:
            remaining_active.remove(node)
            # Zero out connections
            adj_matrix[node, :] = 0
            adj_matrix[:, node] = 0
        
        # Process remaining nodes in this component
        if component_set - set(max_degree_nodes):
            # Recursive call on remaining nodes in component
            remaining_colors = process_graph(adj_matrix, component_set - set(max_degree_nodes))
            component_colors = len(max_degree_nodes) + remaining_colors
        else:
            component_colors = len(max_degree_nodes)
            
        total_colors = max(total_colors, component_colors)
    
    # Update active_vertices with remaining active nodes
    active_vertices.clear()
    active_vertices.update(remaining_active)
    
    # Remove isolated nodes
    isolated_nodes = [node for node in active_vertices if np.sum(adj_matrix[node]) == 0]
    for node in isolated_nodes:
        active_vertices.remove(node)
    
    return total_colors

def find_components(adj_matrix, active_vertices):
    """
    Find connected components using NumPy for efficiency.
    
    Parameters:
    adj_matrix (numpy.ndarray): Adjacency matrix
    active_vertices (set): Set of active vertices
    
    Returns:
    list: List of connected components (each component is a list of vertices)
    """
    if not active_vertices:
        return []
        
    active_list = list(active_vertices)
    n = len(active_list)
    
    if n == 0:
        return []
    if n == 1:
        return [active_list]
    
    # Create node mapping for efficient lookups
    node_map = {node: i for i, node in enumerate(active_list)}
    rev_map = {i: node for node, i in node_map.items()}
    
    # Create a submatrix of the active vertices
    submatrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            node_i = active_list[i]
            node_j = active_list[j]
            if adj_matrix[node_i, node_j] == 1:
                submatrix[i, j] = 1
                submatrix[j, i] = 1
    
    # Find components using DFS
    visited = [False] * n
    components = []
    
    for i in range(n):
        if not visited[i]:
            component = []
            stack = [i]
            
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    component.append(rev_map[node])
                    
                    # Find all unvisited neighbors
                    neighbors = np.where(submatrix[node] == 1)[0]
                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            stack.append(neighbor)
            
            components.append(component)
    
    return components