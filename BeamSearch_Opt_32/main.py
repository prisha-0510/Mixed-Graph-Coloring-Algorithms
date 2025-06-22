from src.load_data import load_data
from src.load_graphs import load_graphs_from_file
from src.predict import predict_colors
import sys
import time
from contextlib import contextmanager
from src.load_model import load_model
from src.model import DeepGCN

# Store the original stdout
original_stdout = sys.stdout

@contextmanager
def redirect_stdout(new_stdout):
    """Context manager to temporarily redirect stdout"""
    sys.stdout = new_stdout
    try:
        yield
    finally:
        sys.stdout = original_stdout

# Add debug print function that prints to both console and file
def debug_print(message, file=None):
    print(message)
    if file:
        print(message, file=file)
        file.flush()  # Ensure it's written immediately

# Configuration
graphs_file = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/networkx_graphs.pkl'
output_file = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/output.txt'
model_path = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/model_parameters/gcn_model.pth'
names_file = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/graph_names.pkl'
colors_file = '/Users/prishajain/Desktop/MTP/BeamSearch_Opt_32/graph_colors.pkl'


start_time = time.time()
hidden_dim = 32
num_layers = 20
model = load_model(model_path, DeepGCN, hidden_dim, num_layers)

start_time = time.time()
with redirect_stdout(original_stdout):
        adj_lists, names, colors = load_graphs_from_file(graphs_file, names_file, colors_file)
print(f"Loaded {len(adj_lists)} graphs in {time.time() - start_time:.2f} seconds")
    


# Open output file for writing
output_stream = open(output_file, 'w', buffering=1)
print("Starting graph processing...", file=output_stream)

# Process each graph
for i, adj_list in enumerate(adj_lists):
    # Predict colors with original stdout
    start_time = time.time()
    with redirect_stdout(original_stdout):
        num_colors = predict_colors(model, adj_list)
    processing_time = time.time() - start_time
    
    # Write both execution time and colors to file
    print(f"  - Execution time: {processing_time:.2f} seconds", file=output_stream)
    print(f"  - Colors required: {num_colors}", file=output_stream)

output_stream.close()