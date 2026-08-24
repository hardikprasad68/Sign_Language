"""
augment_dynamic_data.py — Offline Data Augmentation for Dynamic Gesture Sequences
==================================================================================
Reads every .npy file in the input directory, applies randomized augmentations
(the same 5 used in train_dynamic.py: speed variation, frame dropout, Gaussian
noise, horizontal mirror, scale jitter), and writes augmented copies back to the
same directory with an 'aug_' prefix.

This expands the training set without requiring additional recording sessions.

Usage:
    python scripts/augment_dynamic_data.py --input data/dynamic --copies 10
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np


def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """
    Resample a variable-length sequence to a fixed target_length using
    linear interpolation along the time axis.
    """
    num_frames, num_features = sequence.shape

    if num_frames == target_length:
        return sequence.copy()

    if num_frames == 0:
        return np.zeros((target_length, num_features), dtype=np.float32)

    if num_frames == 1:
        return np.tile(sequence[0], (target_length, 1)).astype(np.float32)

    old_indices = np.linspace(0, num_frames - 1, num_frames)
    new_indices = np.linspace(0, num_frames - 1, target_length)

    resampled = np.zeros((target_length, num_features), dtype=np.float32)
    for feat_idx in range(num_features):
        resampled[:, feat_idx] = np.interp(new_indices, old_indices, sequence[:, feat_idx])

    return resampled


def augment_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Apply randomized augmentations to a raw variable-length sequence.
    Returns an augmented sequence with the SAME variable length as the output
    (not resampled to a fixed length — that happens during training).

    Augmentations applied:
      1. Temporal speed variation: stretch/compress raw frames by ±25%.
      2. Random frame dropout: drop up to 15% of frames.
      3. Coordinate Gaussian noise (σ=0.01).
      4. Horizontal mirroring (50% probability) — flips x-coordinates.
      5. Small scale jitter (90%–110%).
    """
    num_frames, num_features = sequence.shape

    if num_frames < 2:
        return sequence.copy()

    # 1. Temporal speed variation: resample to a random intermediate length
    speed_factor = np.random.uniform(0.75, 1.25)
    intermediate_len = max(3, int(num_frames * speed_factor))
    seq = resample_sequence(sequence, intermediate_len)

    # 2. Random frame dropout (drop up to 15% of frames, keep at least 3)
    if seq.shape[0] > 4:
        num_drop = max(0, int(seq.shape[0] * np.random.uniform(0.0, 0.15)))
        if num_drop > 0 and seq.shape[0] - num_drop >= 3:
            keep_indices = sorted(
                np.random.choice(seq.shape[0], seq.shape[0] - num_drop, replace=False)
            )
            seq = seq[keep_indices]

    # 3. Coordinate Gaussian noise
    noise = np.random.normal(0, 0.01, size=seq.shape).astype(np.float32)
    seq = seq + noise

    # 4. Horizontal mirroring (50% chance) — flip x-coordinates
    #    Landmarks are stored as (x, y, z) triplets. x is at indices 0, 3, 6, ...
    if np.random.random() < 0.5:
        for i in range(0, num_features, 3):
            seq[:, i] = -seq[:, i]

    # 5. Scale jitter
    scale = np.random.uniform(0.90, 1.10)
    seq = seq * scale

    return seq


def main():
    parser = argparse.ArgumentParser(
        description="Offline data augmentation for dynamic gesture sequences"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to directory containing .npy sequence files"
    )
    parser.add_argument(
        "--copies", type=int, default=10,
        help="Number of augmented copies to generate per original sequence (default: 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: None)"
    )

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir.resolve()}")
        sys.exit(1)

    # Find all ORIGINAL .npy files (exclude previously augmented ones)
    all_npy = list(input_dir.glob("*.npy"))
    original_files = [f for f in all_npy if not f.stem.startswith("aug_")]

    if not original_files:
        print(f"[ERROR] No original .npy files found in {input_dir.resolve()}")
        sys.exit(1)

    # Remove any existing augmented files to avoid stacking
    existing_aug = [f for f in all_npy if f.stem.startswith("aug_")]
    if existing_aug:
        print(f"[INFO] Removing {len(existing_aug)} existing augmented files...")
        for f in existing_aug:
            f.unlink()

    print(f"[INFO] Found {len(original_files)} original sequences in {input_dir.resolve()}")
    print(f"[INFO] Generating {args.copies} augmented copies per original...")

    total_generated = 0
    errors = 0

    for fpath in original_files:
        try:
            arr = np.load(fpath).astype(np.float32)

            if arr.ndim != 2:
                print(f"[WARNING] Skipping {fpath.name}: not 2D (shape={arr.shape})")
                errors += 1
                continue

            if arr.shape[0] < 3:
                print(f"[WARNING] Skipping {fpath.name}: too short ({arr.shape[0]} frames)")
                errors += 1
                continue

            for copy_idx in range(args.copies):
                augmented = augment_sequence(arr)

                # Build output filename: aug_<copy_idx>_<original_name>.npy
                out_name = f"aug_{copy_idx}_{fpath.stem}.npy"
                out_path = input_dir / out_name

                np.save(out_path, augmented)
                total_generated += 1

        except Exception as e:
            print(f"[WARNING] Error processing {fpath.name}: {e}")
            errors += 1

    print(f"\n[SUCCESS] Augmentation complete!")
    print(f"  Original files: {len(original_files)}")
    print(f"  Augmented copies generated: {total_generated}")
    print(f"  Total sequences now: {len(original_files) + total_generated}")
    if errors:
        print(f"  Skipped/errored: {errors}")


if __name__ == "__main__":
    main()
