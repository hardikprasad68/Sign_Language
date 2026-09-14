"""
wlasl_to_sequences.py — Build real dynamic-word training sequences from WLASL
==============================================================================
Extracts real landmark sequences for your 15 dynamic words directly from the
WLASL dataset (Kaggle: "wlasl-processed" by risangbaskoro), instead of using
the synthetic sine-wave generator or recording yourself.

WLASL zip contents expected (this is the standard "wlasl-processed" layout):
  videos/<video_id>.mp4
  WLASL_v0.3.json   (gloss -> list of video instances with frame_start/end)

This script does NOT extract the whole zip (12k videos, several GB) — it only
pulls out the small number of videos matching your target word list.

Usage:
  python scripts/wlasl_to_sequences.py \
      --zip "C:\\Users\\you\\Downloads\\wlasl-processed.zip" \
      --out-dir data/dynamic_real \
      --words hello thanks yes no please help sorry name more stop love want eat drink friend

If a target word isn't found under that exact gloss, the script prints the
closest matches from WLASL's vocabulary so you can adjust --word-map.
"""

import argparse
import json
import zipfile
import tempfile
import shutil
import time
from pathlib import Path
from difflib import get_close_matches

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocessing import prepare_input_vector

# Common gloss mismatches worth checking manually if a word isn't found directly.
SUGGESTED_ALIASES = {
    "thanks": ["thank you", "thanks"],
}


def _build_detector():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = str(Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task")
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.Image, mp.ImageFormat, vision.HandLandmarker.create_from_options(options)


def find_json_in_zip(zf: zipfile.ZipFile) -> str:
    candidates = [n for n in zf.namelist() if n.lower().endswith(".json") and "wlasl" in n.lower()]
    if not candidates:
        raise FileNotFoundError("Could not find a WLASL_v*.json file inside the zip.")
    return candidates[0]


def find_video_entry(zf: zipfile.ZipFile, video_id: str) -> str | None:
    for ext in (".mp4", ".mkv", ".webm", ".avi"):
        candidates = [n for n in zf.namelist() if n.endswith(f"{video_id}{ext}")]
        if candidates:
            return candidates[0]
    return None


_last_ts = [0]


def _next_timestamp_ms() -> int:
    """Returns a timestamp guaranteed to be strictly greater than the last one returned."""
    now = int(time.time() * 1000)
    if now <= _last_ts[0]:
        now = _last_ts[0] + 1
    _last_ts[0] = now
    return now


def extract_sequence_from_video(video_path: str,
                                 mp_Image, mp_ImageFormat, detector) -> np.ndarray:
    """
    Runs MediaPipe over the whole clip and returns a (T, 126) real landmark sequence.

    Note: WLASL's frame_start/frame_end metadata refers to positions in the
    original, longer source video. The Kaggle "processed" package already
    ships pre-trimmed per-instance clips, so we deliberately do NOT apply
    that trim here — doing so discards almost every frame.
    """
    cap = cv2.VideoCapture(video_path)
    frames_out = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror to match the live webcam pipeline's orientation convention
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_Image(image_format=mp_ImageFormat.SRGB, data=rgb)
        # Use real wall-clock time (ms) as the timestamp. The detector is reused
        # across hundreds of separate video files, and MediaPipe's VIDEO mode
        # requires timestamps to strictly increase for the detector's entire
        # lifetime — a per-video counter that resets to 0 each time breaks that.
        ts = _next_timestamp_ms()
        result = detector.detect_for_video(mp_image, ts)

        if result.hand_landmarks:
            flat_vector = prepare_input_vector(result.hand_landmarks)
            frames_out.append(flat_vector)

    cap.release()
    if len(frames_out) < 5:
        return None
    return np.array(frames_out, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract real dynamic-word sequences from WLASL")
    parser.add_argument("--zip", type=str, required=True, help="Path to the WLASL dataset zip")
    parser.add_argument("--out-dir", type=str, default="data/dynamic_real", help="Where to save .npy sequences")
    parser.add_argument("--words", nargs="+", required=True, help="Target word list (must match your dynamic_labels.txt)")
    parser.add_argument("--max-per-word", type=int, default=40, help="Max video instances to process per word")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mp_Image, mp_ImageFormat, detector = _build_detector()

    with zipfile.ZipFile(args.zip, "r") as zf:
        json_entry = find_json_in_zip(zf)
        print(f"[INFO] Reading gloss index from: {json_entry}")
        data = json.loads(zf.read(json_entry))

        gloss_to_entry = {item["gloss"].lower(): item for item in data}
        all_glosses = list(gloss_to_entry.keys())

        tmpdir = tempfile.mkdtemp(prefix="wlasl_")
        try:
            for word in args.words:
                word_l = word.lower()
                candidates = [word_l] + SUGGESTED_ALIASES.get(word_l, [])
                match = next((c for c in candidates if c in gloss_to_entry), None)

                if not match:
                    close = get_close_matches(word_l, all_glosses, n=5)
                    print(f"[WARN] '{word}' not found in WLASL gloss list. Closest matches: {close}")
                    print(f"       Skipping '{word}' — add a mapping in SUGGESTED_ALIASES if one of these fits.")
                    continue

                entry = gloss_to_entry[match]
                instances = entry["instances"][: args.max_per_word]
                print(f"[INFO] '{word}' -> gloss '{match}': {len(instances)} instances to try")

                saved = 0
                for inst in instances:
                    video_id = inst["video_id"]
                    zip_video_path = find_video_entry(zf, video_id)
                    if zip_video_path is None:
                        continue

                    local_path = zf.extract(zip_video_path, path=tmpdir)

                    seq = extract_sequence_from_video(
                        local_path, mp_Image, mp_ImageFormat, detector
                    )
                    if seq is not None:
                        out_path = out_dir / f"{word_l}_{video_id}.npy"
                        np.save(out_path, seq)
                        saved += 1

                    # Windows sometimes briefly locks a freshly-closed video file
                    # (antivirus scan / delayed handle release) — retry a couple
                    # times, then just leave it for the final rmtree cleanup.
                    for attempt in range(3):
                        try:
                            Path(local_path).unlink(missing_ok=True)
                            break
                        except PermissionError:
                            time.sleep(0.5)

                print(f"  -> saved {saved} real sequences for '{word}'")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n[DONE] Real dynamic-word sequences saved to: {out_dir.resolve()}")
    print("Next: python src/train_dynamic.py --data-dir " + str(out_dir))


if __name__ == "__main__":
    main()
