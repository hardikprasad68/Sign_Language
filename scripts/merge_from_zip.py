"""
merge_from_zip.py — Extract landmarks straight out of a zipped image dataset
=============================================================================
Same job as src/data/merge_datasets.py, but reads images directly from a
.zip archive in memory instead of requiring you to extract tens of
thousands of tiny files to disk first (which is what freezes Windows).

Usage (run from the repo root, with your venv activated):

  python scripts/merge_from_zip.py \
      --zip "C:\\Users\\you\\Downloads\\archive.zip" \
      --self-csv data/landmarks.csv \
      --out data/landmarks_real.csv \
      --limit 200

--limit caps how many images per letter it processes (200 is plenty and
keeps runtime reasonable; the Kaggle set has ~3000 images per letter).
"""

import argparse
import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import prepare_input_vector
from data_logger import save_landmark_sample, _ensure_csv_exists

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _build_detector():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = str(Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task")
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
    return mp.Image, mp.ImageFormat, vision.HandLandmarker.create_from_options(options)


def process_zip(zip_path: str, out_csv: str, limit_per_class: int = 200, only_letters: set | None = None) -> dict:
    """
    Reads images directly out of the zip (no extraction to disk), groups
    them by their immediate parent folder name inside the zip (used as the
    label), extracts landmarks, and appends real samples to out_csv.

    If only_letters is given, all other letters are skipped entirely —
    useful for topping up just a few weak letters without reprocessing
    the whole dataset.
    """
    mp_Image, mp_ImageFormat, detector = _build_detector()
    counts: dict[str, int] = {}
    total = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        entries = [
            e for e in zf.namelist()
            if e.lower().endswith(IMAGE_EXTS) and not e.endswith("/")
        ]
        print(f"[INFO] Found {len(entries)} image entries inside the zip.")

        for entry in entries:
            # Label = the folder the image file sits directly inside (e.g. .../A/hand1.jpg -> "A")
            label = Path(entry).parent.name
            if not label or len(label) > 3:
                # Skip weird top-level entries that aren't per-letter folders
                continue

            if only_letters is not None and label not in only_letters:
                continue

            if counts.get(label, 0) >= limit_per_class:
                continue

            try:
                raw_bytes = zf.read(entry)
                frame = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Mirror to match the live webcam pipeline's orientation convention
                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp_Image(image_format=mp_ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_image)

                if result.hand_landmarks:
                    flat_vector = prepare_input_vector(result.hand_landmarks)
                    save_landmark_sample(
                        flat_features=flat_vector,
                        label=label,
                        csv_path=out_csv,
                        session_tag="source_public_zip",
                    )
                    counts[label] = counts.get(label, 0) + 1
                    total += 1
                    if total % 200 == 0:
                        print(f"[INFO] Processed {total} images so far...")

            except Exception:
                continue

    print(f"[SUCCESS] Extracted real landmarks from {total} images across {len(counts)} classes.")
    for label in sorted(counts):
        print(f"  {label}: {counts[label]}")
    return counts


def merge_with_self_csv(self_csv: str, out_csv: str) -> None:
    """Append existing real self-recorded rows (skipping synthetic) into out_csv."""
    if not Path(self_csv).exists():
        return
    df = pd.read_csv(self_csv)
    if "session_tag" not in df.columns:
        return
    real_rows = df[df["session_tag"] != "synthetic_asl_generator"]
    if real_rows.empty:
        return
    real_rows.to_csv(out_csv, mode="a", header=False, index=False)
    print(f"[INFO] Appended {len(real_rows)} existing real self-recorded rows from {self_csv}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract landmarks directly from a zipped image dataset")
    parser.add_argument("--zip", type=str, required=True, help="Path to the .zip file (not extracted)")
    parser.add_argument("--self-csv", type=str, default="data/landmarks.csv", help="Existing landmarks CSV with real self-recorded rows")
    parser.add_argument("--out", type=str, default="data/landmarks_real.csv", help="Output CSV path")
    parser.add_argument("--limit", type=int, default=200, help="Max images to process per letter")
    parser.add_argument("--letters", nargs="*", default=None,
                         help="Only process these specific letters (e.g. --letters M E U). "
                              "Omit to process all letters as before.")
    parser.add_argument("--skip-self-merge", action="store_true",
                         help="Skip appending self-csv rows — use this when --out already contains "
                              "your full merged dataset and you're just topping up a few letters, "
                              "to avoid duplicating rows you already have.")

    args = parser.parse_args()

    _ensure_csv_exists(args.out)
    only_letters = set(l.upper() for l in args.letters) if args.letters else None
    process_zip(args.zip, args.out, args.limit, only_letters=only_letters)
    if not args.skip_self_merge:
        merge_with_self_csv(args.self_csv, args.out)

    print(f"\n[DONE] Real-data training CSV ready at: {args.out}")
