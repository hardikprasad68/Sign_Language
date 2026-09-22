"""
caption_overlay.py — Draws a live subtitle/caption bar directly onto video
frames, the same way closed captions appear on a video.

Used for "Mode 1: in-person live captions" — pointing the camera at someone
signing and seeing their translated words appear as subtitles on the video
itself, not just in a separate side panel.
"""

import cv2
import textwrap


def draw_caption_overlay(frame, sentence_text: str, live_label: str = "",
                          max_chars_per_line: int = 40, font_scale: float = 0.9,
                          max_lines: int = 3):
    """
    Draws a semi-transparent caption bar at the bottom of the frame showing
    the accumulated sentence, plus a small live-detection indicator above it
    if a sign is currently being held/recognized.

    Modifies frame in place and also returns it for convenience.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    line_height = 32
    padding = 14

    # Show only the tail end of the sentence so captions don't grow forever
    display_text = sentence_text.strip()[-200:] if sentence_text else ""

    lines = []
    if display_text:
        wrapped = textwrap.wrap(display_text, width=max_chars_per_line)
        lines = wrapped[-max_lines:]  # keep most recent lines only

    # Live indicator (currently-held sign, before it's confirmed into the sentence)
    live_line = f"[{live_label}]" if live_label else ""

    total_lines = len(lines) + (1 if live_line else 0)
    if total_lines == 0:
        return frame

    bar_height = total_lines * line_height + padding * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    alpha = 0.55
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    y = h - bar_height + padding + line_height - 8
    for line in lines:
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        x = max(10, (w - text_size[0]) // 2)
        cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height

    if live_line:
        text_size = cv2.getTextSize(live_line, font, font_scale * 0.85, thickness)[0]
        x = max(10, (w - text_size[0]) // 2)
        cv2.putText(frame, live_line, (x, y), font, font_scale * 0.85, (120, 220, 255), thickness, cv2.LINE_AA)

    return frame
