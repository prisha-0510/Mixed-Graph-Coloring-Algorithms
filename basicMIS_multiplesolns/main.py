from src.load_data import load_data
from src.load_labels import load_labels
from src.train import train_model
from sklearn.model_selection import train_test_split
from src.load_graphs import load_graphs_from_file
import sys

if __name__ == "__main__":
    graph_file = '/Users/prishajain/Desktop/MTP/basicMIS_multiplesolns/graphs.pickle'
    label_file = '/Users/prishajain/Desktop/MTP/basicMIS_multiplesolns/graph_labels.pickle'
    output_file = '/Users/prishajain/Desktop/MTP/basicMIS_multiplesolns/output.txt'
    save_path = '/Users/prishajain/Desktop/MTP/basicMIS_multiplesolns/model_parameters/gcn_model.pth'
    # names_file = '/Users/prishajain/Desktop/MTP/basicMIS_multiplesolns/graph_names.pkl'
    # colors_file = '/Users/prishajain/Desktop/MTP/basicMIS_multiplesolns/graph_colors.pkl'

    graphs = load_data(graph_file)

    sys.stdout = open(output_file, 'w', buffering=1) 
    labels_dict = load_labels(label_file,graphs)
    
    # X_train, X_test, y_train, y_test = train_test_split(graphs, labels_dict, test_size=0.2, random_state=42)
    
    # Model parameters
    hidden_dim = 32
    num_layers = 20
    epochs = 200
    learning_rate = 0.0001
    
    train_model(graphs, labels_dict, hidden_dim, num_layers, epochs, learning_rate, save_path)
    # test_model(X_test,save_path,hidden_dim,num_layers)
    sys.stdout.close()
