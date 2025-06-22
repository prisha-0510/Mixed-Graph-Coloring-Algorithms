import torch
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from reduction import vertex_fold

def load_graph(dataset_name):
    dataset = Planetoid(root=f'./data/{dataset_name}', name=dataset_name)
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)
    print(f"Loaded Planetoid graph {dataset_name}, type: {type(G)}")
    return G

def load_snap_graph(graph_name):
    G = nx.read_edgelist(f'./data/{graph_name}.txt', nodetype=int)
    print(f"Loaded Snap graph {graph_name}, type: {type(G)}")
    return G

def process_graph(graph, name):
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
        print(f"{name}: Graph is empty, no processing needed.")
        return graph
        
    print(f"{name}: has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    reduction_performed = True
    while reduction_performed:
        reduction_performed = vertex_fold(graph)
        # Break if graph becomes empty during reduction
        if len(graph.nodes) == 0:
            print(f"{name}: Graph became empty during reduction.")
            break
            
    print(f"{name}: reduced to {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph
def get_graphs():
    cora_graph = load_graph("Cora")
    pubmed_graph = load_graph("PubMed")
    citeseer_graph = load_graph("CiteSeer")
    wiki_vote_graph = load_snap_graph("wiki-Vote")
    ego_facebook_graph = load_snap_graph("ego-Facebook")

    graphs = {"Cora":cora_graph,
              "PubMed":pubmed_graph,
              "CiteSeer":citeseer_graph,
              "wiki-Vote":wiki_vote_graph,
              "ego-Facebook":ego_facebook_graph}
    return graphs
def get_reduced(graphs):
    processed_graphs = {}
    for graph_name,graph in graphs.items():
        processed_graphs[graph_name] = process_graph(graph,graph_name)
    
    return processed_graphs
