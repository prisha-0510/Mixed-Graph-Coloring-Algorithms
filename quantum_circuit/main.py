from src.load_graphs import load_graphs
from src.predict import predict_colors
from src.load_model import load_model
from src.model import DeepGCN

model_path = '/Users/prishajain/Desktop/MTP/quantum_circuit/model_parameters/gcn_model.pth'
graphs_dict = load_graphs('/Users/prishajain/Desktop/MTP/quantum_circuit/qasm_graphs.pkl')
output_file = '/Users/prishajain/Desktop/MTP/quantum_circuit/output.txt'
hidden_dim = 32
num_layers = 20
model = load_model(model_path, DeepGCN, hidden_dim, num_layers)
with open(output_file, 'w', buffering=1) as file:
    for name,g in graphs_dict.items():
        colors, coloring = predict_colors(g,model)
        file.write(name+"\n")
        file.write("predicted depth = "+str(colors)+"\n")