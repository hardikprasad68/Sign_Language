"""
mlp_static.py — Module 4.1: PyTorch Static MLP Classifier
=========================================================
Responsibility: Defines the PyTorch multi-layer perceptron (MLP) model structure,
loads trained checkpoints, and provides live inference predictions.

PHASE 2:
- You can perform hyperparameter sweeps (adjust hidden layers, learning rates, epochs) here.
- You can add class weight distributions to improve minority class predictions.
"""

import os
import torch
import torch.nn as nn
import numpy as np


class StaticMLP(nn.Module):
    """
    Multi-Layer Perceptron for static hand gesture landmark classification.
    Input dimensions: 126 features (2 hands * 21 landmarks * 3 coordinates).
    """
    def __init__(self, input_dim: int = 126, num_classes: int = 10, dropout_prob: float = 0.3):
        super(StaticMLP, self).__init__()
        
        # Dense stack
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """Returns raw logits (softmax is applied externally or via CrossEntropyLoss)."""
        return self.network(x)


class StaticClassifier:
    """Wrapper class used during runtime inside main.py to predict live hand signs."""
    
    def __init__(self, model_path: str, class_labels: list[str], input_dim: int = 126):
        self.device = torch.device("cpu")  # Force CPU for real-time inference latency consistency
        self.class_labels = class_labels
        self.input_dim = input_dim
        
        # Initialize model architecture
        self.model = StaticMLP(input_dim=self.input_dim, num_classes=len(self.class_labels))
        
        if os.path.exists(model_path):
            try:
                # Load saved parameters
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"[INFO] Loaded trained PyTorch MLP classifier from: {model_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load model weights: {e}. Falling back to default initialization.")
        else:
            print(f"[WARNING] Trained model weight file not found at: {model_path}. Predictor will run with random weights.")
            
        self.model.to(self.device)
        self.model.eval()

    def predict_with_confidence(self, flat_features: list[float]) -> tuple[str, float]:
        """
        Runs model forward pass and returns (class_label, confidence_score).
        
        Args:
            flat_features: A 126-dimensional flat list of floats.
        """
        # If all landmarks are 0.0 (no hand detected), skip prediction
        if not flat_features or np.all(np.array(flat_features) == 0.0):
            return "", 0.0

        try:
            tensor_input = torch.tensor(flat_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor_input)
                probabilities = torch.softmax(logits, dim=1)
                
                max_prob, max_idx = torch.max(probabilities, dim=1)
                
                conf = float(max_prob.item())
                predicted_idx = int(max_idx.item())
                
                if predicted_idx < len(self.class_labels):
                    return self.class_labels[predicted_idx], conf
                return "Unknown", conf
        except Exception as e:
            print(f"[ERROR] Live inference step failed: {e}")
            return "Error", 0.0

    def predict(self, flat_features: list[float]) -> str:
        """Returns the predicted class label string."""
        label, _ = self.predict_with_confidence(flat_features)
        return label
