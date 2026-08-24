"""
landmarks.py — Module 2: Hand Landmark Detection Wrapper
=========================================================
Responsibility: Run MediaPipe Hand Landmarker on each frame, extract 21
landmarks per hand, and draw the hand skeleton on the frame.

Uses the MediaPipe Tasks API.

PHASE 2:
- Swap in holistic models (face, pose, hands combined) if needed.
"""

import os
import time
from pathlib import Path
import threading
import cv2
import numpy as np

# ── Model path ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_PATH_ROOT = _SCRIPT_DIR.parent / "models" / "hand_landmarker.task"
_MODEL_PATH_LOCAL = _SCRIPT_DIR / "models" / "hand_landmarker.task"

_MODEL_PATH = str(_MODEL_PATH_ROOT) if _MODEL_PATH_ROOT.exists() else str(_MODEL_PATH_LOCAL)

if not os.path.exists(_MODEL_PATH):
    os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
    raise FileNotFoundError(
        f"[ERROR] Hand landmarker model file not found. Please download it and place it at:\n"
        f"  {_MODEL_PATH}\n\n"
        "You can download it with:\n"
        "  python -c \"import urllib.request; urllib.request.urlretrieve("
        "'https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task', "
        f"'{_MODEL_PATH}')\""
    )

# ── Lazy / Async Initialization state for HandLandmarker ───────────────────────
_detector = None
_detector_loading = False
_detector_ready = False
_detector_lock = threading.Lock()

# Cached mediapipe module reference (set once during background loading,
# avoids per-frame module lookup overhead in get_hand_landmarks)
_mp_module = None

# Frame counter for VIDEO mode timestamps
_frame_counter = 0


def _load_detector_in_background() -> None:
    """Asynchronously imports MediaPipe and initializes the landmarker."""
    global _detector, _detector_ready, _mp_module
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        # Suppress TensorFlow logging warnings during import/run
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = vision.HandLandmarker.create_from_options(options)
        
        with _detector_lock:
            _detector = detector
            _mp_module = mp
            _detector_ready = True
            
        print("[INFO] MediaPipe Hand Landmarker initialized successfully in background thread.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize MediaPipe Hand Landmarker: {e}")


def extract_landmarks_to_list(hand_landmarks) -> list[float]:
    """
    Extracts coordinates from a single MediaPipe hand landmarks list into a flat list of 63 floats.

    Args:
        hand_landmarks: List of 21 landmark objects (with .x, .y, .z).

    Returns:
        A list of 63 floats (x0, y0, z0, ..., x20, y20, z20).
    """
    flat = []
    for lm in hand_landmarks:
        flat.extend([lm.x, lm.y, lm.z])
    return flat


def get_hand_landmarks(frame):
    """
    Run MediaPipe Hand Landmarker on a single BGR frame and return landmarks.
    Initializes the landmarker in a background thread upon first call if not ready.

    Args:
        frame: A BGR uint8 numpy array (as returned by cv2.VideoCapture.read).

    Returns:
        A list of detected hands, where each hand is a list of 21 objects.
        Returns an empty list if no hands are detected (or if detector is loading).
    """
    global _frame_counter, _detector_loading, _detector_ready, _detector
    _frame_counter += 1

    # Check if the detector is ready
    with _detector_lock:
        ready = _detector_ready
        detector = _detector
        mp = _mp_module

    if not ready:
        if not _detector_loading:
            _detector_loading = True
            print("[INFO] Starting background initialization of MediaPipe Hand Landmarker...")
            threading.Thread(target=_load_detector_in_background, daemon=True).start()
        return []

    # MediaPipe Tasks expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # VIDEO mode requires monotonically increasing timestamps
    timestamp_ms = int(_frame_counter * (1000 / 30))  # assume ~30fps for timestamps
    result = detector.detect_for_video(mp_image, timestamp_ms)

    if not result.hand_landmarks:
        return []

    return result.hand_landmarks


# ── Hand skeleton connections (same as the old mp.solutions.hands) ────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),       # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),     # Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky + palm
]

# Drawing colours
_LANDMARK_COLOR = (0, 255, 0)     # green dots
_CONNECTION_COLOR = (255, 255, 255)  # white lines
_LANDMARK_RADIUS = 4
_CONNECTION_THICKNESS = 2
_NO_HAND_TEXT = "No hand detected"


def _draw_loading_notice(frame) -> None:
    """Overlay a prominent 'Loading hand detector...' notice on the frame."""
    h, w = frame.shape[:2]
    text = "Loading Hand Detector (MediaPipe)..."
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = 10
    y = h - 15
    cv2.rectangle(
        frame,
        (x - 4, y - text_h - 4),
        (x + text_w + 4, y + 4),
        (30, 30, 90),
        -1,
    )
    cv2.putText(
        frame, text,
        (x, y),
        font, scale,
        (150, 150, 255),
        thickness, cv2.LINE_AA,
    )


def _draw_no_hand_notice(frame) -> None:
    """Overlay a small 'No hand detected' notice on the frame."""
    h, w = frame.shape[:2]
    (text_w, text_h), _ = cv2.getTextSize(
        _NO_HAND_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    )
    x = 10
    y = h - 15
    cv2.rectangle(
        frame,
        (x - 4, y - text_h - 4),
        (x + text_w + 4, y + 4),
        (30, 30, 30),
        -1,
    )
    cv2.putText(
        frame, _NO_HAND_TEXT,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (100, 100, 255),
        1, cv2.LINE_AA,
    )


def _draw_hand_skeleton(frame, hand_landmarks) -> None:
    """Draw the 21 landmarks and connections for one hand onto the frame."""
    h, w = frame.shape[:2]

    # Convert normalised coords to pixel coords
    points = []
    for lm in hand_landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        points.append((px, py))

    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx],
                     _CONNECTION_COLOR, _CONNECTION_THICKNESS, cv2.LINE_AA)

    # Draw landmark dots
    for px, py in points:
        cv2.circle(frame, (px, py), _LANDMARK_RADIUS, _LANDMARK_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), _LANDMARK_RADIUS, (0, 0, 0), 1, cv2.LINE_AA)


def draw_landmarks_on_frame(frame, landmarks_list) -> None:
    """
    Draw hand skeletons for all detected hands onto the frame (in-place).
    If the detector is still initializing, displays a loading notice.
    """
    with _detector_lock:
        ready = _detector_ready

    if not ready:
        _draw_loading_notice(frame)
        return

    if not landmarks_list:
        _draw_no_hand_notice(frame)
        return

    for hand_landmarks in landmarks_list:
        _draw_hand_skeleton(frame, hand_landmarks)


def process_frame_for_landmarks(frame):
    """
    Wrapper mapping: returns (landmarks_list, landmarks_list)
    """
    landmarks_list = get_hand_landmarks(frame)
    return landmarks_list, landmarks_list


if __name__ == "__main__":
    from capture import open_camera, compute_fps, draw_fps

    try:
        cap = open_camera(0)
    except RuntimeError as e:
        print(e)
        raise SystemExit(1)

    prev_time = time.time()
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        landmarks_list, _ = process_frame_for_landmarks(frame)
        draw_landmarks_on_frame(frame, landmarks_list)

        fps, prev_time = compute_fps(prev_time)
        draw_fps(frame, fps)

        cv2.imshow("Sign Language Translator — Detection Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
