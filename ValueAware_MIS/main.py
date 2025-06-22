import os
import random
import torch
import pickle

from src import config
from src.logger import Logger
from src.train_model import train_value_aware_model
from src.evaluate_model import evaluate_model
from src.load_model import load_value_aware_model

def load_data(filepath):
    with open(filepath, 'rb') as f:
        raw_data = pickle.load(f)


    processed_data = []
    for graph_result in raw_data:
        original_graph = graph_result.get('original_graph')
        solutions = graph_result.get('solutions', [])

        if not solutions:
            continue

        reduced_graph = solutions[0].get('reduced_graph')
        if not reduced_graph:
            continue

        mis_solutions = [sol['mis_solution'] for sol in solutions if 'mis_solution' in sol]
        if not mis_solutions:
            continue

        processed_data.append({
            'reduced_graph': reduced_graph, 
            'mis_solutions': mis_solutions, 
            'original_graph': original_graph
        })

    return processed_data

def main():
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(config.TEST_DATA_PATH), exist_ok=True)
    
    logger = Logger(config.OUTPUT_FILE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.log(f"Device: {device}")
    logger.log(f"Config: Hidden Dim={config.HIDDEN_DIM}, Layers={config.NUM_LAYERS}, Maps={config.NUM_MAPS}")
    
    # Set random seeds for reproducibility
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.log("Loading data...")
    data = load_data(config.REDUCED_LABELS_FILE)
    random.shuffle(data)
    logger.log(f"Loaded {len(data)} graph datasets")

    split = int(0.8 * len(data))
    train_data = data[:split]
    test_data = data[split:]
    logger.log(f"Split: {len(train_data)} training, {len(test_data)} testing")

    # Train model
    train_model = True  # Set to False to skip training
    if train_model:
        logger.log("Starting model training...")
        model = train_value_aware_model(train_data, config, logger, device)
    else:
        logger.log("Skipping training, loading saved model...")

    # Save test data
    with open(config.TEST_DATA_PATH, 'wb') as f:
        pickle.dump(test_data, f)
    logger.log(f"Saved test data to {config.TEST_DATA_PATH}")

    # Load saved model for evaluation
    logger.log("Loading model for evaluation...")
    model = load_value_aware_model(
        config.MODEL_SAVE_PATH, 
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        num_maps=config.NUM_MAPS, 
        device=device
    )
    
    # Run evaluation
    logger.log("Starting evaluation...")
    evaluate_model(model, test_data, config, logger, device)
    
    logger.log("All done!")
    logger.close()

if __name__ == "__main__":
    main()