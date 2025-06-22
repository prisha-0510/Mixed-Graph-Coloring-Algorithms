import os
import pytket
from pytket.qasm import circuit_from_qasm

def calculate_pytket_depths(folder_path):
    """
    Calculate circuit depths for all QASM files in a folder using pytket.
    
    Args:
        folder_path: Path to the folder containing QASM files
        
    Returns:
        A dictionary with circuit names as keys and depths as values
    """
    results = {}
    
    # Get all QASM files in the directory
    qasm_files = [f for f in os.listdir(folder_path) if f.endswith('.qasm')]
    
    print(f"Found {len(qasm_files)} QASM files")
    
    for qasm_file in sorted(qasm_files):
        file_path = os.path.join(folder_path, qasm_file)
        circuit_name = os.path.splitext(qasm_file)[0]
        
        try:
            # Load the QASM file into pytket circuit with increased maxwidth
            circuit = circuit_from_qasm(file_path, maxwidth=200)
            
            # Get circuit depth
            depth = circuit.depth()
            
            # Store result
            results[circuit_name] = depth
            
            print(f"{circuit_name}: depth = {depth}")
            
        except Exception as e:
            print(f"Error processing {circuit_name}: {e}")
    
    return results

if __name__ == "__main__":
    # Set the path to your QASM_graphs folder
    qasm_folder = "/Users/prishajain/Desktop/MTP/quantum_circuit/QASM_graphs"
    
    # Calculate depths
    depths = calculate_pytket_depths(qasm_folder)
    
    # Print a summary
    print("\nSummary of Circuit Depths:")
    print("-" * 50)
    for name, depth in sorted(depths.items()):
        print(f"{name:<30} {depth}")