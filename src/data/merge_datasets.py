"""
merge_datasets.py — Module 5.1: Dataset Processing & Merging Utility
=====================================================================
Responsibility: Runs MediaPipe over raw images from a public dataset directory 
(e.g., Kaggle ASL Alphabet), extracts coordinate landmarks, and merges them 
with self-recorded CSV logs.

Usage:
  python src/data/merge_datasets.py --public-dir data/raw/asl_alphabet --self-csv data/landmarks.csv --out data/landmarks.csv
"""

import argparse
import csv
import os
import time
from pathlib import Path
import cv2
import pandas as pd

# Import our landmark and preprocessing utilities
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from landmarks import get_hand_landmarks
from preprocessing import prepare_input_vector
from data_logger import save_landmark_sample, count_label_samples


def process_public_images(public_dir: str, out_csv: str) -> dict[str, int]:
    """
    Scans a directory of sign language images organized by folder labels (e.g. A, B, C),
    processes each image using MediaPipe Hands, extracts 126-dimensional features,
    and appends them to the CSV.

    Returns:
        A dictionary of {class_label: count_extracted}
    """
    counts = {}
    path = Path(public_dir)
    if not path.exists():
        print(f"[WARNING] Public dataset directory not found at: {path.resolve()}")
        print("[INFO] Skipping public image processing. Only self-recorded logs will be used.")
        return counts

    # Search for subfolders (each folder name represents a label)
    subfolders = [f for f in path.iterdir() if f.is_dir()]
    if not subfolders:
        # Fallback to checking if images are in the root with label in the filename
        print(f"[WARNING] No subdirectories found in: {path.resolve()}")
        return counts

    print(f"[INFO] Processing public dataset images in: {path.resolve()}")
    print("[INFO] Running MediaPipe Hands on images...")

    # We import MediaPipe local tools to process static images cleanly
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = str(Path(__file__).resolve().parent.parent.parent / "models" / "hand_landmarker.task")
    if not os.path.exists(model_path):
        print(f"[ERROR] MediaPipe task model not found at: {model_path}")
        return counts

    # For processing static images, we use IMAGE running mode (not VIDEO)
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    total_images_processed = 0

    for folder in subfolders:
        label = folder.name
        counts[label] = 0
        
        # Scan for images
        image_files = [f for f in folder.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        print(f"  -> Processing class '{label}' ({len(image_files)} images found)...")

        # Limit to first 200 images per class for validation speed during Phase 1
        for img_path in image_files[:200]:
            try:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_image)

                if result.hand_landmarks:
                    # Format as 126 coordinates flat vector (pads missing second hand with 0)
                    flat_vector = prepare_input_vector(result.hand_landmarks)
                    
                    # Log to CSV
                    save_landmark_sample(
                        flat_features=flat_vector,
                        label=label,
                        csv_path=out_csv,
                        session_tag="source_public_dataset"
                    )
                    counts[label] += 1
                    total_images_processed += 1
            except Exception as e:
                # Silently skip individual bad images
                continue

    print(f"[SUCCESS] Extracted landmarks from {total_images_processed} public images.")
    return counts


def create_data_readme(counts_public: dict[str, int], counts_self: dict[str, int], readme_path: str) -> None:
    """Writes metadata statistics for the combined dataset to data/README.md."""
    path = Path(readme_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    all_classes = sorted(list(set(list(counts_public.keys()) + list(counts_self.keys()))))
    
    table_rows = []
    total_p = 0
    total_s = 0
    
    for cls in all_classes:
        p_count = counts_public.get(cls, 0)
        s_count = counts_self.get(cls, 0)
        total_p += p_count
        total_s += s_count
        table_rows.append(f"| {cls} | {p_count} | {s_count} | {p_count + s_count} |")

    content = f"""# Hand Landmark Sign Dataset Documentation

This directory contains the landmark coordinate datasets for the Real-Time Sign Language Translator.

## Schema Specifications
Recorded in `landmarks.csv`. Each row contains **129 columns**:
1.  **Columns 0 to 125 (`x0,y0,z0` to `x41,y41,z41`):** Flat coordinate positions of up to two hands (wrist-normalized, scale-normalized, padded with `0.0` for missing hands).
2.  **Column 126 (`label`):** Gesture target sign (e.g. `A`, `Peace`, `Thumbs Up`).
3.  **Column 127 (`timestamp`):** Unix timestamp of sample insertion.
4.  **Column 128 (`session_tag`):** Session categorization tag.

## Dataset Class Distribution
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

| Class Sign | Public Dataset Count | Self-Recorded Count | Combined Total |
| :--- | :--- | :--- | :--- |
{chr(10).join(table_rows)}
| **Total** | **{total_p}** | **{total_s}** | **{total_p + total_s}** |

## Source References
*   **Public Dataset:** ASL Alphabet (processed image bounding boxes converted to keypoint lists).
*   **Self-Recorded Dataset:** Camera frame logs captured via `data_logger.py`.
"""
    with open(path, "w") as f:
        f.write(content)
    print(f"[INFO] Wrote dataset documentation file: {path.resolve()}")


def merge_pipelines(public_dir: str, self_csv: str, out_csv: str) -> None:
    """Coordinates public extraction and database merging."""
    
    # 1. Initialize output CSV if it doesn't exist
    from data_logger import _ensure_csv_exists
    _ensure_csv_exists(out_csv)

    # 2. Compile pre-merge counts for self-recorded data
    self_counts = {}
    if os.path.exists(self_csv):
        try:
            df = pd.read_csv(self_csv)
            if "label" in df.columns:
                self_counts = df[df["session_tag"] != "source_public_dataset"]["label"].value_counts().to_dict()
        except Exception:
            pass

    # 3. Process and append public image landmarks
    public_counts = process_public_images(public_dir, out_csv)

    # 4. Generate documentation README
    readme_path = str(Path(out_csv).parent / "README.md")
    create_data_readme(public_counts, self_counts, readme_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract landmarks from public image files and merge datasets")
    parser.add_argument("--public-dir", type=str, default="data/raw/asl_alphabet", help="Path to public dataset images")
    parser.add_argument("--self-csv", type=str, default="data/landmarks.csv", help="Path to self-recorded landmarks CSV")
    parser.add_argument("--out", type=str, default="data/landmarks.csv", help="Path to output merged CSV")
    
    args = parser.parse_args()
    merge_pipelines(args.public_dir, args.self_csv, args.out)
