import os
import torch
import numpy as np

class SignPredictor:
    def __init__(self, models_dir):
        print("[INFO] Loading Inclubate AI Engine...")
        # 1. Load the TorchScript models
        self.static_model = torch.jit.load(os.path.join(models_dir, 'static_traced.pt'))
        self.dynamic_model = torch.jit.load(os.path.join(models_dir, 'dynamic_traced.pt'))
        
        # 2. Load the label maps into memory
        with open(os.path.join(models_dir, 'class_labels.txt'), 'r') as f:
            self.static_labels = f.read().splitlines()
        with open(os.path.join(models_dir, 'dynamic_labels.txt'), 'r') as f:
            self.dynamic_labels = f.read().splitlines()
            
    def predict_static(self, landmarks_array):
        """Expects a flat NumPy array of 126 coordinates for a single frame."""
        # Convert NumPy array to PyTorch tensor and add batch dimension
        tensor_data = torch.tensor(landmarks_array, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = self.static_model(tensor_data)
            predicted_idx = torch.argmax(output, dim=1).item()
            
        return self.static_labels[predicted_idx]

    def predict_dynamic(self, sequence_array):
        """Expects a NumPy array of shape (1, 30, 126) directly from SequenceBuffer."""
        # Removed .unsqueeze(0) because SequenceBuffer already adds the batch dimension
        tensor_data = torch.tensor(sequence_array, dtype=torch.float32)
        
        with torch.no_grad():
            output = self.dynamic_model(tensor_data)
            predicted_idx = torch.argmax(output, dim=1).item()
            
        return self.dynamic_labels[predicted_idx]