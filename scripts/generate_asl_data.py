"""
generate_asl_data.py — Synthetic ASL Alphabet Landmark Data Generator
=====================================================================
Generates realistic MediaPipe-style hand landmark coordinates for ASL
alphabet letters A–Y (excluding J and Z which require motion).

Each letter is modeled as a specific finger configuration (curled/extended/
partially bent) with anatomically correct joint chains. Random jitter,
rotation, and scale variation are applied to create diverse training samples.

Usage:
  python generate_asl_data.py --samples 300 --output data/landmarks.csv
"""

import csv
import math
import random
import time
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import normalize_coords_tuples
from labels import ASL_STATIC_LETTERS

NUM_LANDMARKS = 21
TOTAL_COORDS = 126  # 2 hands * 21 * 3

# ── Anatomical base hand model ─────────────────────────────────────────────
# Each landmark is defined relative to the wrist (0,0,0).
# MediaPipe hand landmarks are normalized 0–1, but since we normalize
# to wrist-relative coords, we work in a local coordinate system.
#
# Finger joint indices:
#   Thumb:  1(CMC), 2(MCP), 3(IP), 4(TIP)
#   Index:  5(MCP), 6(PIP), 7(DIP), 8(TIP)
#   Middle: 9(MCP), 10(PIP), 11(DIP), 12(TIP)
#   Ring:   13(MCP), 14(PIP), 15(DIP), 16(TIP)
#   Pinky:  17(MCP), 18(PIP), 19(DIP), 20(TIP)

# Base positions for a fully open hand (palm facing camera, right hand)
BASE_OPEN_HAND = [
    (0.0, 0.0, 0.0),       # 0: Wrist
    (-0.08, -0.06, -0.02),  # 1: Thumb CMC
    (-0.14, -0.12, -0.03),  # 2: Thumb MCP
    (-0.18, -0.18, -0.02),  # 3: Thumb IP
    (-0.21, -0.23, -0.01),  # 4: Thumb TIP
    (-0.05, -0.22, 0.0),    # 5: Index MCP
    (-0.06, -0.34, 0.0),    # 6: Index PIP
    (-0.06, -0.42, 0.0),    # 7: Index DIP
    (-0.06, -0.48, 0.0),    # 8: Index TIP
    (0.0, -0.23, 0.0),      # 9: Middle MCP
    (0.0, -0.36, 0.0),      # 10: Middle PIP
    (0.0, -0.44, 0.0),      # 11: Middle DIP
    (0.0, -0.50, 0.0),      # 12: Middle TIP
    (0.05, -0.22, 0.0),     # 13: Ring MCP
    (0.05, -0.33, 0.0),     # 14: Ring PIP
    (0.06, -0.40, 0.0),     # 15: Ring DIP
    (0.06, -0.45, 0.0),     # 16: Ring TIP
    (0.10, -0.20, 0.0),     # 17: Pinky MCP
    (0.10, -0.28, 0.0),     # 18: Pinky PIP
    (0.10, -0.33, 0.0),     # 19: Pinky DIP
    (0.10, -0.37, 0.0),     # 20: Pinky TIP
]


def _curl_finger(landmarks, mcp_idx, curl_amount=0.85):
    """Curl a finger by rotating PIP, DIP, TIP towards the palm."""
    mcp = landmarks[mcp_idx]
    pip = landmarks[mcp_idx + 1]
    dip = landmarks[mcp_idx + 2]
    tip = landmarks[mcp_idx + 3]

    # Curling moves the tip/dip/pip upward (toward palm) and forward (z)
    dx_pip = pip[0] - mcp[0]
    dy_pip = pip[1] - mcp[1]

    curl_y_factor = curl_amount
    curl_z = 0.04 * curl_amount

    # PIP curls partially
    new_pip = (mcp[0] + dx_pip * 0.3, mcp[1] + dy_pip * (1 - curl_y_factor * 0.6), mcp[2] + curl_z * 0.5)
    # DIP curls more
    new_dip = (mcp[0] + dx_pip * 0.2, mcp[1] + dy_pip * (1 - curl_y_factor * 0.3), mcp[2] + curl_z * 0.8)
    # TIP curls most — folds back toward palm
    new_tip = (mcp[0] + dx_pip * 0.1, mcp[1] + dy_pip * (1 - curl_y_factor * 0.1), mcp[2] + curl_z)

    landmarks[mcp_idx + 1] = new_pip
    landmarks[mcp_idx + 2] = new_dip
    landmarks[mcp_idx + 3] = new_tip


def _curl_thumb(landmarks, curl_amount=0.85):
    """Curl the thumb across the palm."""
    cmc = landmarks[1]
    # When curled, thumb folds across fingers
    landmarks[2] = (cmc[0] + 0.02 * curl_amount, cmc[1] - 0.06, cmc[2] + 0.03 * curl_amount)
    landmarks[3] = (cmc[0] + 0.06 * curl_amount, cmc[1] - 0.08, cmc[2] + 0.04 * curl_amount)
    landmarks[4] = (cmc[0] + 0.09 * curl_amount, cmc[1] - 0.09, cmc[2] + 0.04 * curl_amount)


def _extend_thumb_out(landmarks):
    """Extend thumb outward to the side."""
    landmarks[1] = (-0.08, -0.06, -0.02)
    landmarks[2] = (-0.15, -0.10, -0.03)
    landmarks[3] = (-0.20, -0.14, -0.02)
    landmarks[4] = (-0.24, -0.17, -0.01)


def _extend_thumb_up(landmarks):
    """Extend thumb straight up."""
    landmarks[1] = (-0.08, -0.06, -0.02)
    landmarks[2] = (-0.12, -0.14, -0.03)
    landmarks[3] = (-0.13, -0.22, -0.02)
    landmarks[4] = (-0.14, -0.28, -0.01)


def _partially_bend_finger(landmarks, mcp_idx, bend=0.4):
    """Partially bend a finger (not fully curled, not fully extended)."""
    _curl_finger(landmarks, mcp_idx, curl_amount=bend)


def _generate_letter_landmarks(letter: str) -> list[tuple]:
    """
    Generate anatomically-plausible landmark coordinates for a given ASL letter.
    Returns a list of 21 (x, y, z) tuples.
    """
    # Start with a copy of the open hand
    lm = [tuple(p) for p in BASE_OPEN_HAND]

    if letter == 'A':
        # Fist with thumb alongside (thumb extended slightly to the side)
        _curl_finger(lm, 5)   # index
        _curl_finger(lm, 9)   # middle
        _curl_finger(lm, 13)  # ring
        _curl_finger(lm, 17)  # pinky
        _extend_thumb_up(lm)

    elif letter == 'B':
        # All four fingers extended straight up, thumb curled across palm
        _curl_thumb(lm, 0.9)

    elif letter == 'C':
        # Curved hand — all fingers partially bent in a C shape
        _partially_bend_finger(lm, 5, 0.4)
        _partially_bend_finger(lm, 9, 0.4)
        _partially_bend_finger(lm, 13, 0.4)
        _partially_bend_finger(lm, 17, 0.4)
        # Thumb curves to face fingers
        landmarks_thumb_c = (-0.12, -0.14, -0.01)
        lm[2] = (-0.12, -0.10, -0.02)
        lm[3] = (-0.13, -0.14, -0.01)
        lm[4] = (-0.12, -0.17, 0.0)

    elif letter == 'D':
        # Index extended, other fingers curled, thumb touches middle finger
        _curl_finger(lm, 9)
        _curl_finger(lm, 13)
        _curl_finger(lm, 17)
        _curl_thumb(lm, 0.7)

    elif letter == 'E':
        # All fingers curled, fingertips touching thumb
        _curl_finger(lm, 5, 0.7)
        _curl_finger(lm, 9, 0.7)
        _curl_finger(lm, 13, 0.7)
        _curl_finger(lm, 17, 0.7)
        _curl_thumb(lm, 0.5)

    elif letter == 'F':
        # Index and thumb form a circle, other three fingers extended
        _curl_finger(lm, 5, 0.6)
        _curl_thumb(lm, 0.6)

    elif letter == 'G':
        # Index pointing to the side, thumb parallel, other fingers curled
        _curl_finger(lm, 9)
        _curl_finger(lm, 13)
        _curl_finger(lm, 17)
        # Index points sideways
        lm[6] = (-0.14, -0.24, 0.0)
        lm[7] = (-0.20, -0.25, 0.0)
        lm[8] = (-0.26, -0.25, 0.0)

    elif letter == 'H':
        # Index and middle pointing sideways, others curled
        _curl_finger(lm, 13)
        _curl_finger(lm, 17)
        _curl_thumb(lm, 0.8)
        # Both point sideways
        lm[6] = (-0.14, -0.24, 0.0)
        lm[7] = (-0.20, -0.25, 0.0)
        lm[8] = (-0.26, -0.25, 0.0)
        lm[10] = (-0.08, -0.25, 0.0)
        lm[11] = (-0.14, -0.26, 0.0)
        lm[12] = (-0.20, -0.26, 0.0)

    elif letter == 'I':
        # Pinky extended, all others curled
        _curl_finger(lm, 5)
        _curl_finger(lm, 9)
        _curl_finger(lm, 13)
        _curl_thumb(lm, 0.9)

    elif letter == 'K':
        # Index and middle extended in V, thumb between them
        _curl_finger(lm, 13)
        _curl_finger(lm, 17)
        _extend_thumb_up(lm)

    elif letter == 'L':
        # L-shape: index up, thumb out to side
        _curl_finger(lm, 9)
        _curl_finger(lm, 13)
        _curl_finger(lm, 17)
        _extend_thumb_out(lm)

    elif letter == 'M':
        # Three fingers (index, middle, ring) over thumb, pinky curled
        _curl_finger(lm, 5, 0.7)
        _curl_finger(lm, 9, 0.7)
        _curl_finger(lm, 13, 0.7)
        _curl_finger(lm, 17, 0.9)
        _curl_thumb(lm, 0.95)

    elif letter == 'N':
        # Two fingers (index, middle) over thumb
        _curl_finger(lm, 5, 0.7)
        _curl_finger(lm, 9, 0.7)
        _curl_finger(lm, 13, 0.9)
        _curl_finger(lm, 17, 0.9)
        _curl_thumb(lm, 0.95)

    elif letter == 'O':
        # All fingers curl to meet thumb forming an O
        _partially_bend_finger(lm, 5, 0.55)
        _partially_bend_finger(lm, 9, 0.55)
        _partially_bend_finger(lm, 13, 0.55)
        _partially_bend_finger(lm, 17, 0.55)
        lm[2] = (-0.10, -0.10, -0.02)
        lm[3] = (-0.08, -0.14, -0.01)
        lm[4] = (-0.05, -0.16, 0.0)

    elif letter == 'P':
        # Like K but pointing down
        _curl_finger(lm, 13)
        _curl_finger(lm, 17)
        # Rotate hand downward
        for i in range(21):
            x, y, z = lm[i]
            lm[i] = (x, -y * 0.3 + 0.1, z)

    elif letter == 'Q':
        # Like G but pointing down
        _curl_finger(lm, 9)
        _curl_finger(lm, 13)
        _curl_finger(lm, 17)
        # Point downward
        for i in range(21):
            x, y, z = lm[i]
            lm[i] = (x, -y * 0.3 + 0.1, z)

    elif letter == 'R':
        # Index and middle crossed, others curled
        _curl_finger(lm, 13)
        _curl_finger(lm, 17)
        _curl_thumb(lm, 0.8)
        # Cross middle over index
        lm[10] = (-0.04, -0.36, -0.01)
        lm[11] = (-0.05, -0.44, -0.01)
        lm[12] = (-0.05, -0.50, -0.01)

    elif letter == 'S':
        # Fist with thumb over fingers
        _curl_finger(lm, 5)
        _curl_finger(lm, 9)
        _curl_finger(lm, 13)
        _curl_finger(lm, 17)
        _curl_thumb(lm, 0.95)

    elif letter == 'T':
        # Fist with thumb between index and middle
        _curl_finger(lm, 5, 0.85)
        _curl_finger(lm, 9)
        _curl_finger(lm, 13)
        _curl_finger(lm, 17)
        # Thumb pokes between index and middle
        lm[2] = (-0.06, -0.10, 0.03)
        lm[3] = (-0.04, -0.16, 0.04)
        lm[4] = (-0.02, -0.18, 0.04)

    elif letter == 'U':
        # Index and middle extended together, others curled
        _curl_finger(lm, 13)
        _curl_finger(lm, 17)
        _curl_thumb(lm, 0.8)
        # Bring index and middle close together
        lm[6] = (-0.03, -0.34, 0.0)
        lm[7] = (-0.03, -0.42, 0.0)
        lm[8] = (-0.03, -0.48, 0.0)

    elif letter == 'V':
        # Peace sign — index and middle spread, others curled
        _curl_finger(lm, 13)
        _curl_finger(lm, 17)
        _curl_thumb(lm, 0.8)

    elif letter == 'W':
        # Index, middle, ring extended and spread, pinky curled
        _curl_finger(lm, 17)
        _curl_thumb(lm, 0.8)

    elif letter == 'X':
        # Index hooked (partially bent), others curled
        _curl_finger(lm, 9)
        _curl_finger(lm, 13)
        _curl_finger(lm, 17)
        _curl_thumb(lm, 0.8)
        _partially_bend_finger(lm, 5, 0.5)

    elif letter == 'Y':
        # Thumb and pinky extended, others curled (shaka/hang loose)
        _curl_finger(lm, 5)
        _curl_finger(lm, 9)
        _curl_finger(lm, 13)
        _extend_thumb_out(lm)

    return lm


def _apply_variation(landmarks, rotation_deg=15, jitter_std=0.012, scale_range=(0.85, 1.15)):
    """Apply random rotation, jitter, and scale to create training variations."""
    # Random 2D rotation
    angle = math.radians(random.uniform(-rotation_deg, rotation_deg))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Random scale
    scale = random.uniform(scale_range[0], scale_range[1])

    varied = []
    for x, y, z in landmarks:
        # Rotate
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a

        # Scale
        rx *= scale
        ry *= scale
        rz = z * scale

        # Jitter
        rx += random.gauss(0, jitter_std)
        ry += random.gauss(0, jitter_std)
        rz += random.gauss(0, jitter_std * 0.5)

        varied.append((rx, ry, rz))

    return varied


def _landmarks_to_flat_126(landmarks):
    """Convert 21 landmarks to a normalized 126-element flat list (pad second hand with zeros)."""
    normalized_hand = normalize_coords_tuples(landmarks)
    rounded = [round(v, 6) for v in normalized_hand]
    rounded.extend([0.0] * 63)
    return rounded


def _build_csv_header():
    cols = []
    for i in range(42):
        cols += [f"x{i}", f"y{i}", f"z{i}"]
    cols.append("label")
    cols.append("timestamp")
    cols.append("session_tag")
    return cols


def generate_dataset(output_csv: str, samples_per_letter: int = 300):
    """Generate synthetic ASL landmark data and write to CSV."""
    static_letters = ASL_STATIC_LETTERS

    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_build_csv_header())

        for letter in static_letters:
            base_landmarks = _generate_letter_landmarks(letter)

            for i in range(samples_per_letter):
                varied = _apply_variation(base_landmarks)
                flat = _landmarks_to_flat_126(varied)

                row = flat + [letter, round(time.time(), 4), "synthetic_asl_generator"]
                writer.writerow(row)
                total_written += 1

            print(f"  Generated {samples_per_letter} samples for letter '{letter}'")

    print(f"\n[SUCCESS] Generated {total_written} total samples across {len(static_letters)} letters.")
    print(f"[INFO] Saved to: {path.resolve()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic ASL alphabet landmark data")
    parser.add_argument("--samples", type=int, default=300, help="Samples per letter (default: 300)")
    parser.add_argument("--output", type=str, default="data/landmarks.csv", help="Output CSV path")
    args = parser.parse_args()

    generate_dataset(args.output, args.samples)
