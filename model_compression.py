import torch
import torch.nn as nn
import os

# Model Architecture (same as in main.py)
class TaxiFareModel(nn.Module):
    """
    Deep Neural Network for taxi fare prediction
    """
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
        super(TaxiFareModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer (single neuron for regression)
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x).squeeze()

def compress_model():
    """Compress the taxi fare model using quantization"""
    
    # Model parameters (should match training)
    input_size = 14  # Number of features
    
    # Create model instance
    model = TaxiFareModel(input_size=input_size)
    
    # Load trained weights
    model_path = 'best_taxi_fare_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return False
    
    try:
        # Load the state dict
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        print("Original model loaded successfully")
        
        # Get original model size
        original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"Original model size: {original_size:.2f} MB")
        
        # Quantize to int8 (can reduce size by 75%)
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        # Save compressed model
        compressed_path = 'model_compressed.pth'
        torch.save(quantized_model.state_dict(), compressed_path)
        
        # Get compressed model size
        compressed_size = os.path.getsize(compressed_path) / (1024 * 1024)  # MB
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        print(f"Compressed model saved as '{compressed_path}'")
        print(f"Compressed model size: {compressed_size:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"Error during compression: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting model compression...")
    success = compress_model()
    if success:
        print("Model compression completed successfully!")
    else:
        print("Model compression failed!")