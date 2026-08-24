"""
qa_checkpoint.py — Phase 0 Automated QA Script
===============================================
Runs non-interactive checks that can be verified without a human
staring at the camera. For the full visual QA checklist, see the
implementation plan.

Checks:
  1. All pipeline modules import without error
  2. Classifier self-test (9 gesture rules)
  3. CSV schema validation (if landmark_data.csv exists)
  4. Webcam can be opened and a frame read
  5. MediaPipe can process a frame and return landmarks (or empty list)

Usage:
    python scripts/qa_checkpoint.py
"""

import csv
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

PASS = 0
FAIL = 0


def _result(label: str, passed: bool, detail: str = ""):
    global PASS, FAIL
    if passed:
        PASS += 1
    else:
        FAIL += 1
    icon = "✓" if passed else "✗"
    msg = f"  [{icon}] {label}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)


def test_imports():
    print("\n[1/5] Module Imports")
    for name in ["capture", "detection", "classifier", "logger", "main"]:
        try:
            __import__(name)
            _result(f"import {name}", True)
        except Exception as e:
            _result(f"import {name}", False, str(e))


def test_classifier():
    print("\n[2/5] Classifier Self-Test")
    from classifier import predict, get_finger_states

    class LM:
        def __init__(self, x, y, z=0.0):
            self.x, self.y, self.z = x, y, z

    def make(extended: list[bool]):
        """Build 21 fake landmarks given [thumb, index, middle, ring, pinky] bools."""
        lms = [LM(0.4, 0.9), LM(0.5, 0.85)]  # wrist, thumb CMC
        # Thumb (MCP=2, IP=3, TIP=4)
        lms.append(LM(0.55, 0.80))
        lms.append(LM(0.60, 0.76))
        lms.append(LM(0.65 if extended[0] else 0.52, 0.72))
        # Four fingers
        for i, ext in enumerate(extended[1:]):
            bx = 0.4 + i * 0.07
            pip_y = 0.65
            tip_y = pip_y - 0.15 if ext else pip_y + 0.05
            lms.append(LM(bx, pip_y + 0.1))   # MCP
            lms.append(LM(bx, pip_y))          # PIP
            lms.append(LM(bx, pip_y - 0.07))   # DIP
            lms.append(LM(bx, tip_y))           # TIP
        return lms

    cases = [
        ([True, True, True, True, True], "Open Hand"),
        ([False, False, False, False, False], "Fist"),
        ([False, True, False, False, False], "Pointing"),
        ([False, True, True, False, False], "Peace ✌"),
        ([True, False, False, False, False], "Thumbs Up 👍"),
    ]

    for fingers, expected in cases:
        result = predict(make(fingers))
        _result(f"predict({expected})", result == expected,
                f"got '{result}'" if result != expected else "")

    # Edge case: None input
    result = predict(None)
    _result("predict(None) → empty string", result == "", f"got '{result}'")

    # Edge case: empty list
    result = predict([])
    _result("predict([]) → empty string", result == "", f"got '{result}'")


def test_csv_schema():
    print("\n[3/5] CSV Schema Validation")
    csv_path = Path(_PROJECT_ROOT) / "landmark_data.csv"
    if not csv_path.exists():
        _result("landmark_data.csv exists", False,
                "no CSV found (this is OK if you haven't saved any samples yet)")
        return

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        _result("CSV has 64 columns", len(header) == 64,
                f"found {len(header)} columns")
        _result("Last column is 'label'", header[-1] == "label",
                f"last column is '{header[-1]}'")
        _result("First column is 'x0'", header[0] == "x0",
                f"first column is '{header[0]}'")

        # Check first data row if it exists
        try:
            row = next(reader)
            _result("Has at least 1 data row", True)
            # Verify numeric values in first 63 columns
            try:
                vals = [float(v) for v in row[:63]]
                in_range = all(
                    -2.0 <= v <= 2.0 for v in vals  # z can be slightly negative
                )
                _result("Landmark values are numeric and in range", in_range)
            except ValueError:
                _result("Landmark values are numeric", False)
        except StopIteration:
            _result("Has at least 1 data row", False, "header only")


def test_webcam():
    print("\n[4/5] Webcam Access")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            _result("Webcam opens", True)
            _result("Frame readable", ret and frame is not None,
                    f"{frame.shape[1]}×{frame.shape[0]}" if ret and frame is not None else "")
        else:
            _result("Webcam opens", False, "check permissions")
    except Exception as e:
        _result("Webcam test", False, str(e))


def test_mediapipe():
    print("\n[5/5] MediaPipe Processing")
    try:
        import cv2
        import numpy as np
        from detection import process_frame_for_landmarks

        # Create a blank frame (no hand — should return empty list, not crash)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        results, landmarks = process_frame_for_landmarks(blank)
        _result("MediaPipe processes blank frame without crash", True)
        _result("Returns empty landmarks for blank frame",
                isinstance(landmarks, list) and len(landmarks) == 0,
                f"got {len(landmarks)} hands")

        # Try with webcam frame if available
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                results, landmarks = process_frame_for_landmarks(frame)
                _result("MediaPipe processes webcam frame without crash", True,
                        f"detected {len(landmarks)} hand(s)")
        else:
            _result("Webcam frame test", False, "skipped — no camera")

    except Exception as e:
        _result("MediaPipe test", False, str(e))


def main():
    print("=" * 60)
    print("  Phase 0 QA Checkpoint — Automated Tests")
    print("=" * 60)

    test_imports()
    test_classifier()
    test_csv_schema()
    test_webcam()
    test_mediapipe()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"  Results: {PASS}/{total} passed, {FAIL}/{total} failed")
    if FAIL == 0:
        print("  ✓ PHASE 0 CHECKPOINT PASSED")
    else:
        print("  ✗ PHASE 0 CHECKPOINT — ISSUES FOUND (review above)")
    print("=" * 60)

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
