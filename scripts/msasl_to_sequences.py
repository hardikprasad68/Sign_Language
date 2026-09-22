"""
msasl_to_sequences.py — Extract real landmark sequences from MS-ASL
=====================================================================
MS-ASL (Microsoft's word-level ASL dataset) is distributed as JSON files
listing YouTube URLs + timestamps, NOT bundled video files like WLASL was.
This script downloads just the relevant few-second clip for each matching
instance (not the whole YouTube video) using yt-dlp, then extracts real
landmark sequences the same way as wlasl_to_sequences.py.

REQUIRES: yt-dlp and ffmpeg installed and on PATH.
  pip install yt-dlp
  ffmpeg: download from ffmpeg.org (Windows builds) and add to PATH,
          or `winget install ffmpeg`.

Because these are individual YouTube videos, expect some failures
(deleted/private/region-blocked videos) — the script skips those and
reports how many succeeded.

Usage:
  python scripts/msasl_to_sequences.py \
      --json MSASL_train.json MSASL_val.json MSASL_test.json \
      --out-dir data/dynamic_real \
      --words friend hello help love more stop \
      --max-per-word 60
"""

import argparse
import json
import subprocess
import tempfile
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocessing import prepare_input_vector


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


_last_ts = [0]


def _next_timestamp_ms() -> int:
    now = int(time.time() * 1000)
    if now <= _last_ts[0]:
        now = _last_ts[0] + 1
    _last_ts[0] = now
    return now


def download_clip(url: str, start: float, end: float, out_path: str) -> bool:
    """Downloads just the [start, end] segment of a YouTube video using yt-dlp."""
    section = f"*{start}-{end}"
    cmd = [
        "yt-dlp",
        "--download-sections", section,
        "--force-keyframes-at-cuts",
        "-f", "mp4/best",
        "-o", out_path,
        "--quiet", "--no-warnings",
        url,
    ]
    try:
        result = subprocess.run(cmd, timeout=90, capture_output=True)
        return result.returncode == 0 and Path(out_path).exists()
    except Exception:
        return False


def extract_sequence_from_video(video_path: str, mp_Image, mp_ImageFormat, detector):
    cap = cv2.VideoCapture(video_path)
    frames_out = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # match live webcam mirror convention
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_Image(image_format=mp_ImageFormat.SRGB, data=rgb)
        ts = _next_timestamp_ms()
        result = detector.detect_for_video(mp_image, ts)
        if result.hand_landmarks:
            frames_out.append(prepare_input_vector(result.hand_landmarks))
    cap.release()
    if len(frames_out) < 5:
        return None
    return np.array(frames_out, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract real sequences from MS-ASL via yt-dlp")
    parser.add_argument("--json", nargs="+", required=True, help="Path(s) to MS-ASL JSON file(s)")
    parser.add_argument("--out-dir", type=str, default="data/dynamic_real")
    parser.add_argument("--words", nargs="+", required=True)
    parser.add_argument("--max-per-word", type=int, default=60)
    args = parser.parse_args()

    if shutil.which("yt-dlp") is None:
        print("[ERROR] yt-dlp not found on PATH. Install it with: pip install yt-dlp")
        sys.exit(1)
    if shutil.which("ffmpeg") is None:
        print("[ERROR] ffmpeg not found on PATH. yt-dlp needs it to cut video segments.")
        print("        Install with: winget install ffmpeg   (or download from ffmpeg.org)")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for jf in args.json:
        with open(jf, "r", encoding="utf-8") as f:
            entries.extend(json.load(f))

    target_words = set(w.lower() for w in args.words)
    by_word = {w: [] for w in target_words}
    for e in entries:
        text = e.get("text", "").lower()
        if text in by_word:
            by_word[text].append(e)

    mp_Image, mp_ImageFormat, detector = _build_detector()
    tmpdir = tempfile.mkdtemp(prefix="msasl_")

    try:
        for word, instances in by_word.items():
            instances = instances[: args.max_per_word]
            print(f"[INFO] '{word}': {len(instances)} candidate clips found in MS-ASL JSON")
            saved, failed = 0, 0

            for i, inst in enumerate(instances):
                clip_path = str(Path(tmpdir) / f"{word}_{i}.mp4")
                ok = download_clip(inst["url"], inst["start_time"], inst["end_time"], clip_path)
                if not ok:
                    failed += 1
                    continue

                seq = extract_sequence_from_video(clip_path, mp_Image, mp_ImageFormat, detector)
                Path(clip_path).unlink(missing_ok=True)

                if seq is not None:
                    out_path = out_dir / f"{word}_msasl_{i}.npy"
                    np.save(out_path, seq)
                    saved += 1
                else:
                    failed += 1

                if (i + 1) % 10 == 0:
                    print(f"  ... {i+1}/{len(instances)} attempted (saved={saved}, failed={failed})")

            print(f"  -> '{word}': saved {saved}, failed/skipped {failed}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n[DONE] Real MS-ASL sequences saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
