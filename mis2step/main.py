from src.load_data import load_data
from src.colors import find_colors
from src.predict import predict_mis
from src.load_model import load_model
from src.model import DeepGCN
import sys
from src.load_graphs import load_graphs_from_file

graphs_file = '/Users/prishajain/Desktop/MTP/mis2step/networkx_graphs.pkl'
model_path = '/Users/prishajain/Desktop/MTP/mis2step/model_parameters/gcn_model.pth'
output_file = '/Users/prishajain/Desktop/MTP/mis2step/output.txt'
names_file = '/Users/prishajain/Desktop/MTP/mis2step/graph_names.pkl'
colors_file = '/Users/prishajain/Desktop/MTP/mis2step/graph_colors.pkl'


graphs,names,colors = load_graphs_from_file(graphs_file,names_file,colors_file)
hidden_dim = 16
num_layers = 20
model = load_model(model_path, DeepGCN, hidden_dim, num_layers)

with open('temp.txt', 'w', buffering=1) as f:
    sys.stdout = f
    with open(output_file, 'w', buffering=1) as file:

        for i,graph in enumerate(graphs):
            file.write("Colors required = " + str(find_colors(graph,model))+"\n")

