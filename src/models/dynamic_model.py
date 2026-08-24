"""
dynamic_model.py — Track B: Dynamic (Word-Level) Gesture Classifier Model v1
=============================================================================
Responsibility: Defines the PyTorch GRU sequence model structure for word-level sign language recognition over landmark frames.

Features:
- Input sequence: (batch_size, 30 frames, 126 features per frame).
- Architecture: 2-layer Gated Recurrent Unit (GRU) + LayerNorm + Dropout + Linear Head.
- Motion energy gate: Filters out stationary/static hands and empty sequences to prevent false positive word predictions.
"""

import os
import torch
import torch.nn as nn
import numpy as np


class DynamicGRU(nn.Module):
    """
    Sequence classifier using GRU over landmark coordinate vectors.
    """
    def __init__(self, input_dim: int = 126, hidden_dim: int = 128, num_layers: int = 2, num_classes: int = 15, dropout_prob: float = 0.3):
        super(DynamicGRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        gru_out_dim = hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        final_step = gru_out[:, -1, :]
        logits = self.fc(final_step)
        return logits


class DynamicClassifier:
    """Wrapper class used during runtime to predict dynamic word signs."""
    
    def __init__(self, model_path: str, class_labels: list[str], input_dim: int = 126, sequence_length: int = 30):
        self.device = torch.device("cpu")
        self.class_labels = class_labels
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        
        self.model = DynamicGRU(
            input_dim=self.input_dim,
            hidden_dim=128,
            num_layers=2,
            num_classes=len(self.class_labels),
            dropout_prob=0.3
        )
        
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"[INFO] Loaded trained Dynamic GRU classifier from: {model_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load Dynamic GRU weights from {model_path}: {e}")
        else:
            print(f"[WARNING] Dynamic model file not found at: {model_path}. Predictor will run with random weights.")
            
        self.model.to(self.device)
        self.model.eval()

    def predict_with_confidence(self, sequence_arr: np.ndarray, motion_threshold: float = 0.004) -> tuple[str, float]:
        """
        Runs model forward pass on a sequence of shape (1, sequence_length, input_dim).
        Filters out low-motion stationary frames before classifying.

        The motion gate uses mean per-frame per-feature displacement, which is
        scale-independent and works correctly regardless of sequence length or
        feature dimensionality.
        """
        if sequence_arr is None or sequence_arr.size == 0:
            return "", 0.0

        if len(sequence_arr.shape) == 2:
            sequence_arr = np.expand_dims(sequence_arr, axis=0)

        # 1. Motion Energy Gate: mean per-frame per-feature displacement
        diffs = np.diff(sequence_arr[0], axis=0)  # shape: (seq_len-1, features)
        mean_motion = float(np.mean(np.abs(diffs)))
        
        # If hand is static/stationary or mostly empty, do not force a dynamic word prediction
        if mean_motion < motion_threshold:
            return "", 0.0

        try:
            tensor_input = torch.tensor(sequence_arr, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor_input)
                probabilities = torch.softmax(logits, dim=1)
                
                max_prob, max_idx = torch.max(probabilities, dim=1)
                
                conf = float(max_prob.item())
                predicted_idx = int(max_idx.item())
                
                # Lower threshold since the stabilizer already applies confidence filtering
                if conf < 0.30:
                    return "", conf

                if predicted_idx < len(self.class_labels):
                    return self.class_labels[predicted_idx], conf
                return "Unknown", conf
        except Exception as e:
            print(f"[ERROR] Dynamic live inference step failed: {e}")
            return "Error", 0.0

    def predict(self, sequence_arr: np.ndarray) -> str:
        label, _ = self.predict_with_confidence(sequence_arr)
        return label
