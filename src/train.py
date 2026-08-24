"""
train.py — Module 5: PyTorch Static Classifier (v2) Training & Tuning Loop
==========================================================================
Responsibility: Loads coordinate dataset, splits into train/val/test subsets,
applies 3D data augmentation to minority classes, performs hyperparameter sweep,
trains StaticModelV2, calculates full evaluation metrics (precision, recall, F1,
confusion matrix), and exports trained model weights.

Usage:
  python src/train.py --csv data/landmarks.csv --epochs 60 --augment --model-version v2
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# Import utilities
from data_utils import load_dataset, split_dataset
from models.static_model import StaticModelV2
from preprocessing import augment_landmarks

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class HandDataset(Dataset):
    """Custom PyTorch dataset wrapping landmark arrays with 3D data augmentation."""
    def __init__(self, X, y, augment_flag=False, augment_classes=None):
        self.X = X
        self.y = y
        self.augment_flag = augment_flag
        self.augment_classes = augment_classes if augment_classes else set()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_val = self.X[idx].tolist()
        label = self.y[idx]
        
        if self.augment_flag and (not self.augment_classes or label in self.augment_classes):
            if torch.rand(1).item() < 0.6:
                x_val = augment_landmarks(x_val, rotation_deg=18.0, jitter_std=0.02)
                
        return torch.tensor(x_val, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def train_model(csv_path: str, model_save_path: str, labels_save_path: str, 
                epochs: int, batch_size: int, learning_rate: float, 
                dropout: float, hidden_dim: int, enable_augment: bool) -> tuple[float, float]:
    """Trains PyTorch StaticModelV2 on the dataset and exports metrics."""
    
    # 1. Load and split dataset
    print(f"[INFO] Loading dataset from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] CSV dataset not found at: {csv_path}. Please collect data first.")
        
    X, y, class_labels = load_dataset(csv_path)
    num_classes = len(class_labels)
    
    print(f"[INFO] Dataset loaded successfully. Samples: {len(X)}, Classes: {num_classes} ({class_labels})")

    # Stratified split
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    print(f"[INFO] Train samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 2. Identify minority classes for augmentation
    augment_classes = set()
    if enable_augment:
        train_counts = Counter(y_train)
        mean_count = np.mean(list(train_counts.values()))
        for class_idx, count in train_counts.items():
            if count <= mean_count:
                augment_classes.add(class_idx)
        print(f"[INFO] Classes flagged for data augmentation: {[class_labels[i] for i in augment_classes]}")

    # Create PyTorch datasets and loaders
    train_dataset = HandDataset(X_train, y_train, augment_flag=enable_augment, augment_classes=augment_classes)
    val_dataset = HandDataset(X_val, y_val, augment_flag=False)
    test_dataset = HandDataset(X_test, y_test, augment_flag=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Initialize Model, Loss, Optimizer & Scheduler
    model = StaticModelV2(input_dim=126, num_classes=num_classes, hidden_dim=hidden_dim, dropout_prob=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0.0

    print("[INFO] Beginning Static Model v2 training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        scheduler.step(val_acc)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

        # Save checkpoint of best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)

    print(f"\n[SUCCESS] Model v2 Training completed. Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"[INFO] Saved best model weights to: {model_save_path}")

    # Save class labels list to support runtime lookup
    os.makedirs(os.path.dirname(labels_save_path), exist_ok=True)
    with open(labels_save_path, "w") as f:
        f.write("\n".join(class_labels))
    print(f"[INFO] Saved class labels configuration to: {labels_save_path}")

    # 4. Final Evaluation on Test Set & Error Analysis
    best_model = StaticModelV2(input_dim=126, num_classes=num_classes, hidden_dim=hidden_dim, dropout_prob=dropout)
    best_model.load_state_dict(torch.load(model_save_path))
    best_model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = best_model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_acc = np.mean(all_preds == all_targets)

    print("\n" + "=" * 60)
    print(f" STATIC MODEL V2 TEST SET EVALUATION")
    print(f" Test Accuracy: {test_acc * 100:.2f}%")
    print("=" * 60)
    
    report = classification_report(all_targets, all_preds, target_names=class_labels, zero_division=0)
    print("\nClassification Report:\n", report)

    conf_mat = confusion_matrix(all_targets, all_preds)
    
    # Save test metrics report
    results_dir = _PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = results_dir / "static_v2_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"Static Model v2 Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(conf_mat))
    print(f"[INFO] Metrics and confusion matrix saved to: {metrics_file.resolve()}")

    return best_val_acc, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Static Classifier Model v2 Trainer")
    parser.add_argument("--csv", type=str, default=str(_PROJECT_ROOT / "data" / "landmarks.csv"), help="Path to input landmarks CSV")
    parser.add_argument("--model-out", type=str, default=str(_PROJECT_ROOT / "models" / "mlp_v2.pt"), help="Path to output PyTorch model weights")
    parser.add_argument("--labels-out", type=str, default=str(_PROJECT_ROOT / "models" / "class_labels.txt"), help="Path to output class labels list")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="AdamW learning rate")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate in residual blocks")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--augment", action="store_true", help="Apply 3D landmarks augmentation")
    
    args = parser.parse_args()
    
    train_model(
        csv_path=args.csv,
        model_save_path=args.model_out,
        labels_save_path=args.labels_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        enable_augment=args.augment
    )

