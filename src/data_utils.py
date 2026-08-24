"""
data_utils.py — Module 3.1: Data Loading & Splitting Utilities
=============================================================
This module provides shared methods to load landmark CSV datasets,
harmonize coordinate counts, and perform stratified train/val/test splits.

PHASE 2:
- You can add custom sample filtering or outlier removal criteria in load_dataset().
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from preprocessing import normalize_flat_vector


def load_dataset(csv_path: str, target_features: int = 126) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Loads features and labels from the landmark CSV file.
    Automatically handles rows with 63 features (single-hand Phase 0 format)
    by zero-padding them to target_features (126 features).

    Args:
        csv_path: Path to the landmark CSV file.
        target_features: Expected input feature count (usually 126).

    Returns:
        X (np.ndarray): Coordinate features of shape (num_samples, target_features).
        y (np.ndarray): Label index integers of shape (num_samples,).
        classes (list[str]): Sorted list of unique class label strings.
    """
    df = pd.read_csv(csv_path)
    
    # Identify labels
    if "label" not in df.columns:
        raise ValueError("[ERROR] CSV file does not contain a 'label' column.")
    
    labels = df["label"].values
    unique_classes = sorted(list(set(labels)))
    class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}
    y = np.array([class_to_idx[l] for l in labels], dtype=np.int64)

    # Filter coordinate columns (columns starting with x, y, z)
    coord_cols = [col for col in df.columns if col.startswith(('x', 'y', 'z'))]
    raw_x = df[coord_cols].values

    # Perform shape harmonization
    num_samples = raw_x.shape[0]
    num_cols = raw_x.shape[1]

    if num_cols == target_features:
        X = raw_x.astype(np.float32)
    elif num_cols < target_features:
        # Pad with zeros
        print(f"[INFO] Harmonizing dataset: Padding {num_cols} features to {target_features} features.")
        padded_x = np.zeros((num_samples, target_features), dtype=np.float32)
        padded_x[:, :num_cols] = raw_x
        X = padded_x
    else:
        # Truncate
        print(f"[INFO] Harmonizing dataset: Truncating {num_cols} features to {target_features} features.")
        X = raw_x[:, :target_features].astype(np.float32)

    # Ensure every sample uses the same wrist-centered, scale-normalized
    # representation as live inference (idempotent on already-normalized rows).
    normalized_rows = [normalize_flat_vector(row.tolist()) for row in X]
    X = np.array(normalized_rows, dtype=np.float32)

    return X, y, unique_classes


def split_dataset(
    X: np.ndarray, 
    y: np.ndarray, 
    train_size: float = 0.70, 
    val_size: float = 0.15, 
    test_size: float = 0.15,
    random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs a unified stratified train/val/test split.

    Args:
        X: Feature array.
        y: Label index array.
        train_size: Ratio for training set.
        val_size: Ratio for validation set.
        test_size: Ratio for test set.
        random_seed: Fixed seed for reproducibility.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test arrays.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-5, "Splits must sum to 1.0"

    # First split to extract train set
    test_val_ratio = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        train_size=train_size, 
        random_state=random_seed, 
        stratify=y
    )

    # Second split to separate val and test sets
    test_relative_ratio = test_size / test_val_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        train_size=(1.0 - test_relative_ratio), 
        random_state=random_seed, 
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
