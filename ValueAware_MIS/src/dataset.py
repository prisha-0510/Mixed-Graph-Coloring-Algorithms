import torch
from torch.utils.data import Dataset
import networkx as nx

class ValueAwareGraphDataset(Dataset):
    def __init__(self, graph_data):
        self.graph_data = graph_data

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):
        data = self.graph_data[idx]
        graph = data['reduced_graph']
        mis_solutions = data['mis_solutions']

        # Create adjacency matrix
        adjacency_matrix = torch.tensor(nx.adjacency_matrix(graph).todense(), dtype=torch.float32)
        
        # Create degree matrix
        degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))

        # Extract select and nonselect values from nodes
        nodes = sorted(graph.nodes())
        select_values = torch.tensor([graph.nodes[n]['select_value'] for n in nodes], dtype=torch.float32)
        nonselect_values = torch.tensor([graph.nodes[n]['nonselect_value'] for n in nodes], dtype=torch.float32)
        
        # Stack them for convenience in training
        node_features = torch.stack([select_values, nonselect_values], dim=1)

        # Create solution labels
        labels = torch.stack([
            torch.tensor([mis.get(n, 0) for n in nodes], dtype=torch.float32)
            for mis in mis_solutions
        ])  # Shape: [num_mis_solutions, num_nodes]

        return adjacency_matrix, degree_matrix, node_features, labels