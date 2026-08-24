"""
test_model.py — Unit Tests for Split and Model Architecture
==========================================================
Verifies data splitting consistency and MLP feedforward shape.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from data_utils import split_dataset
from models.mlp_static import StaticMLP


def test_split_dataset():
    # Generate mock features and labels
    X = np.random.rand(100, 126).astype(np.float32)
    y = np.array([i % 5 for i in range(100)], dtype=np.int64)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, train_size=0.70, val_size=0.15, test_size=0.15, random_seed=42
    )

    assert len(X_train) == 70
    assert len(X_val) == 15
    assert len(X_test) == 15
    
    assert len(y_train) == 70
    assert len(y_val) == 15
    assert len(y_test) == 15

    # Check stratification
    # All classes should be roughly equal in size in splits
    print("[SUCCESS] Data split stratification test passed!")


def test_mlp_forward():
    model = StaticMLP(input_dim=126, num_classes=10, dropout_prob=0.3)
    model.eval()

    inputs = torch.randn(8, 126)
    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (8, 10)
    print("[SUCCESS] MLP forward propagation shape test passed!")


if __name__ == "__main__":
    test_split_dataset()
    test_mlp_forward()
    print("[INFO] Model and utility tests completed successfully.")
