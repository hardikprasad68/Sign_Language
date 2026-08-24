"""
generate_dynamic_data.py — Track B: Dynamic Word Sequence Dataset Generator
=============================================================================
Responsibility: Synthesizes a initial realistic sequence dataset for 15 dynamic words:
['hello', 'thanks', 'yes', 'no', 'please', 'help', 'sorry', 'name', 
 'more', 'stop', 'love', 'want', 'eat', 'drink', 'friend'].

Each sample is a 30-frame sequence of 126-dimensional landmark vectors saved as a .npy file
in data/dynamic/[word]_[timestamp]_[sample_idx].npy.

Trajectory models simulate:
- Spatial translation & velocity curves over 30 frames.
- Keypose hand shapes (waving, nodding, circular motion, tapping, etc.).
- Multi-signer variance (scale, rotation, noise, speed variations).
"""

import argparse
import math
import os
import random
import time
from pathlib import Path
import numpy as np

# Import preprocessing for static pose base templates
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing import normalize_flat_vector, augment_landmarks

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DYNAMIC_VOCABULARY = [
    "hello", "thanks", "yes", "no", "please",
    "help", "sorry", "name", "more", "stop",
    "love", "want", "eat", "drink", "friend"
]


def generate_word_trajectory(word: str, num_frames: int = 30) -> np.ndarray:
    """
    Generates a 30-frame sequence (shape (30, 126)) representing the dynamic motion signature of a word sign.
    """
    seq = np.zeros((num_frames, 126), dtype=np.float32)
    t_steps = np.linspace(0, 1, num_frames)

    # Multi-signer random variations
    speed = random.uniform(0.8, 1.2)
    scale = random.uniform(0.85, 1.15)
    off_x = random.uniform(-0.05, 0.05)
    off_y = random.uniform(-0.05, 0.05)

    # Base hand template (hand 1: features 0:63, hand 2: features 63:126)
    # We construct realistic coordinate trajectories over 30 frames
    for i, t in enumerate(t_steps):
        t_scaled = min(1.0, max(0.0, t * speed))
        
        h1 = np.zeros((21, 3), dtype=np.float32)
        h2 = np.zeros((21, 3), dtype=np.float32)

        if word == "hello":
            # Waving open palm side to side
            wave = math.sin(t_scaled * 4 * math.pi) * 0.25
            h1[:, 0] = wave + off_x
            h1[:, 1] = -0.5 + off_y
            h1[4:21, 1] -= 0.3  # Extended fingers

        elif word == "thanks":
            # Hand moving from chin forward and downward
            drop_y = t_scaled * 0.4
            forward_z = t_scaled * 0.3
            h1[:, 0] = off_x
            h1[:, 1] = drop_y + off_y
            h1[:, 2] = forward_z

        elif word == "yes":
            # Nodding fist up and down
            nod = math.sin(t_scaled * 3 * math.pi) * 0.2
            h1[:, 0] = off_x
            h1[:, 1] = nod + off_y
            # Fist: curled fingers
            h1[4:, 1] += 0.1

        elif word == "no":
            # Index and middle finger snapping closed to thumb
            snap = math.sin(t_scaled * 2 * math.pi)
            h1[8, 1] = snap * 0.2  # Index tip
            h1[12, 1] = snap * 0.2 # Middle tip

        elif word == "please":
            # Smooth circular rubbing motion over chest
            angle = t_scaled * 2 * math.pi
            h1[:, 0] = math.sin(angle) * 0.15 + off_x
            h1[:, 1] = math.cos(angle) * 0.15 + off_y

        elif word == "help":
            # Two hands moving upward together
            up_y = -t_scaled * 0.35
            h1[:, 1] = up_y + off_y
            h2[:, 1] = up_y + off_y + 0.1
            h2[:, 0] = 0.2  # Base hand underneath

        elif word == "sorry":
            # Circular fist motion
            angle = t_scaled * 2.5 * math.pi
            h1[:, 0] = math.sin(angle) * 0.12 + off_x
            h1[:, 1] = math.cos(angle) * 0.12 + off_y

        elif word == "name":
            # Index & middle fingers tapping repeatedly
            tap = abs(math.sin(t_scaled * 4 * math.pi)) * 0.15
            h1[:, 1] = tap + off_y
            h2[:, 1] = -tap + off_y

        elif word == "more":
            # Both hands bringing fingertips together in center
            dist = (1.0 - t_scaled) * 0.4
            h1[:, 0] = -dist + off_x
            h2[:, 0] = dist + off_x

        elif word == "stop":
            # Sharp forward palm thrust
            thrust = t_scaled * 0.4
            h1[:, 2] = thrust
            h1[4:21, 1] -= 0.4 # Open palm

        elif word == "love":
            # Crossed arms/hands over heart
            cross = math.sin(t_scaled * math.pi) * 0.15
            h1[:, 0] = cross - 0.1
            h2[:, 0] = -cross + 0.1

        elif word == "want":
            # Clawed hands pulling inward
            pull = (1.0 - t_scaled) * 0.3
            h1[:, 2] = pull
            h2[:, 2] = pull

        elif word == "eat":
            # Hand moving to mouth repeatedly
            move = abs(math.sin(t_scaled * 3 * math.pi)) * 0.25
            h1[:, 1] = -move + off_y

        elif word == "drink":
            # Curved hand tilting upward
            tilt = t_scaled * 0.3
            h1[:, 1] = -tilt + off_y

        elif word == "friend":
            # Hooking index fingers swapping positions
            swap = math.sin(t_scaled * 2 * math.pi) * 0.2
            h1[:, 0] = swap
            h2[:, 0] = -swap

        # Apply scaling and joint noise
        h1 *= scale
        h2 *= scale
        
        # Add random joint micro-jitter
        h1 += np.random.normal(0, 0.008, size=(21, 3))
        h2 += np.random.normal(0, 0.008, size=(21, 3))

        flat_frame = np.concatenate([h1.flatten(), h2.flatten()])
        normalized_frame = normalize_flat_vector(flat_frame.tolist())
        seq[i] = normalized_frame

    return seq


def generate_dataset(output_dir: str = "data/dynamic", samples_per_word: int = 50) -> None:
    """
    Generates and saves full dynamic word sequence dataset.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(" DYNAMIC WORD DATASET GENERATOR")
    print(f" Output directory: {out_path.resolve()}")
    print(f" Target vocabulary ({len(DYNAMIC_VOCABULARY)} words): {DYNAMIC_VOCABULARY}")
    print(f" Generating {samples_per_word} sequence samples per word...")
    print("=" * 60 + "\n")

    total_files = 0
    for word in DYNAMIC_VOCABULARY:
        for idx in range(samples_per_word):
            seq = generate_word_trajectory(word, num_frames=30)
            file_name = f"{word}_{int(time.time())}_{idx:03d}.npy"
            file_path = out_path / file_name
            np.save(file_path, seq)
            total_files += 1

    print(f"[SUCCESS] Dynamic sequence dataset successfully created!")
    print(f"[INFO] Generated {total_files} .npy sequence files in: {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Dynamic Dataset Generator")
    parser.add_argument("--output", type=str, default=str(_PROJECT_ROOT / "data" / "dynamic"), help="Output directory for .npy sequence files")
    parser.add_argument("--samples", type=int, default=50, help="Number of sequence samples per word")
    
    args = parser.parse_args()
    generate_dataset(output_dir=args.output, samples_per_word=args.samples)
