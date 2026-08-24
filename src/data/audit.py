"""
audit.py — Module 6: Dataset Imbalance Auditor
==============================================
Responsibility: Inspects the merged dataset CSV, prints sample counts per class, 
and flags any class that falls below a target representation threshold (e.g., 50 samples).

Usage:
  python src/data/audit.py --csv data/landmarks.csv --threshold 50
"""

import argparse
import os
import pandas as pd


def audit_dataset(csv_path: str, threshold: int) -> None:
    """Performs dataset auditing and imbalance checks."""
    if not os.path.exists(csv_path):
        print(f"[ERROR] Landmark CSV file not found at: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV dataset: {e}")
        return

    if "label" not in df.columns:
        print("[ERROR] CSV file does not contain a 'label' column.")
        return

    total_samples = len(df)
    class_counts = df["label"].value_counts()
    unique_classes = sorted(class_counts.index.tolist())

    print("\n" + "=" * 60)
    print(f" DATASET AUDIT REPORT: {os.path.basename(csv_path)}")
    print(f" Total Samples: {total_samples}")
    print(f" Unique Classes: {len(unique_classes)}")
    print("=" * 60)

    print(f"{'Class Sign':<15} | {'Sample Count':<12} | {'Status':<15}")
    print("-" * 50)

    underrepresented = []
    for cls in unique_classes:
        count = class_counts[cls]
        status = "OK"
        if count < threshold:
            status = f"WARNING (<{threshold})"
            underrepresented.append((cls, count))
        print(f"{cls:<15} | {count:<12} | {status:<15}")

    print("-" * 50)
    if underrepresented:
        print(f"[ALERT] {len(underrepresented)} classes are underrepresented:")
        for cls, count in underrepresented:
            print(f"  - Class '{cls}': Only {count} samples (Target: {threshold})")
        print("\n[RECOMMENDATION] Collect more samples for these classes or enable --augment in train.py.")
    else:
        print("[SUCCESS] All classes are well-represented (above threshold).")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit landmark dataset class balance")
    parser.add_argument("--csv", type=str, default="data/landmarks.csv", help="Path to input landmarks CSV")
    parser.add_argument("--threshold", type=int, default=50, help="Target minimum samples per class")
    
    args = parser.parse_args()
    audit_dataset(args.csv, args.threshold)
