"""
dummy_classifier.py — Module 4: Rule-Based ASL Alphabet Classifier
==================================================================
Responsibility: Predict an ASL letter label from a hand's 21 landmark points.

Used as a fallback when no trained MLP model is available. Uses finger
extension heuristics mapped to ASL static alphabet letters (A–Y).
"""

import math


def _is_finger_extended(landmarks, tip_idx: int, pip_idx: int) -> bool:
    """Return True if the finger is extended (tip is higher on screen than PIP joint)."""
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def _is_finger_curled(landmarks, tip_idx: int, mcp_idx: int) -> bool:
    """Return True if the finger tip is close to or below its MCP joint."""
    return landmarks[tip_idx].y >= landmarks[mcp_idx].y - 0.02


def _is_thumb_extended(landmarks) -> bool:
    """Evaluate thumb extension by checking lateral x displacement relative to wrist."""
    wrist_x = landmarks[0].x
    cmc_x = landmarks[1].x
    tip_x = landmarks[4].x
    ip_x = landmarks[3].x

    if wrist_x < cmc_x:
        return tip_x > ip_x
    return tip_x < ip_x


def _finger_spread(landmarks, tip_a: int, tip_b: int) -> float:
    """Horizontal distance between two fingertip landmarks."""
    return abs(landmarks[tip_a].x - landmarks[tip_b].x)


def get_finger_states(landmarks) -> dict[str, bool]:
    """Check extension states of all five fingers."""
    return {
        "thumb": _is_thumb_extended(landmarks),
        "index": _is_finger_extended(landmarks, tip_idx=8, pip_idx=6),
        "middle": _is_finger_extended(landmarks, tip_idx=12, pip_idx=10),
        "ring": _is_finger_extended(landmarks, tip_idx=16, pip_idx=14),
        "pinky": _is_finger_extended(landmarks, tip_idx=20, pip_idx=18),
    }


def _count_extended(states: dict[str, bool]) -> int:
    return sum(1 for k in ("index", "middle", "ring", "pinky") if states[k])


def predict(landmarks) -> str:
    """
    Classify an ASL static letter from raw MediaPipe landmark data.
    Rules are ordered from most specific to least specific patterns.
    """
    if not landmarks or len(landmarks) < 21:
        return ""

    states = get_finger_states(landmarks)
    thumb = states["thumb"]
    index = states["index"]
    middle = states["middle"]
    ring = states["ring"]
    pinky = states["pinky"]
    extended_count = _count_extended(states)

    # I — only pinky extended
    if pinky and not index and not middle and not ring and not thumb:
        return "I"

    # Y — thumb and pinky extended (shaka)
    if thumb and pinky and not index and not middle and not ring:
        return "Y"

    # L — index and thumb extended, others curled
    if index and thumb and not middle and not ring and not pinky:
        return "L"

    # V — index and middle extended (peace sign)
    if index and middle and not ring and not pinky:
        index_middle_spread = _finger_spread(landmarks, 8, 12)
        if index_middle_spread < 0.06:
            return "U"  # U — index and middle together
        return "V"

    # W — index, middle, ring extended
    if index and middle and ring and not pinky:
        return "W"

    # B — four fingers extended, thumb curled across palm
    if not thumb and index and middle and ring and pinky:
        return "B"

    # D — index extended, others curled
    if index and not middle and not ring and not pinky:
        return "D"

    # A — all fingers curled, thumb extended alongside
    if thumb and extended_count == 0:
        return "A"

    # S — all fingers curled including thumb (closed fist)
    if not thumb and extended_count == 0:
        return "S"

    # C — partially bent (2-3 fingers partially extended with gaps)
    if extended_count >= 2 and not (index and middle and ring):
        partially_bent = sum(
            1 for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]
            if not _is_finger_extended(landmarks, tip, mcp - 1)
            and not _is_finger_curled(landmarks, tip, mcp)
        )
        if partially_bent >= 2:
            return "C"

    # O — all fingertips curled toward thumb (check proximity)
    if extended_count == 0 or (extended_count <= 1 and not index):
        thumb_tip = landmarks[4]
        fingertips = [landmarks[i] for i in (8, 12, 16, 20)]
        avg_dist = sum(
            math.sqrt((ft.x - thumb_tip.x) ** 2 + (ft.y - thumb_tip.y) ** 2)
            for ft in fingertips
        ) / 4
        if avg_dist < 0.12:
            return "O"

    # K — index and middle in V with thumb between (approximate)
    if index and middle and thumb and not ring and not pinky:
        return "K"

    # X — index hooked (partially bent), others curled
    if not middle and not ring and not pinky and not _is_finger_extended(landmarks, 8, 6):
        if not _is_finger_curled(landmarks, 8, 5):
            return "X"

    # R — index and middle crossed (middle tip left of index tip)
    if index and middle and not ring and not pinky:
        if landmarks[12].x < landmarks[8].x:
            return "R"

    # T — fist with thumb between index and middle
    if not index and not middle and not ring and not pinky and thumb:
        return "T"

    # M/N — fingers over thumb (all curled, thumb tucked)
    if extended_count == 0 and not thumb:
        return "M"

    # H — index and middle pointing sideways
    if index and middle and not ring and not pinky:
        if abs(landmarks[8].x - landmarks[0].x) > 0.08 and abs(landmarks[8].y - landmarks[0].y) < 0.06:
            return "H"

    # G — index pointing sideways, others curled
    if index and not middle and not ring and not pinky:
        if abs(landmarks[8].x - landmarks[0].x) > 0.08 and abs(landmarks[8].y - landmarks[0].y) < 0.06:
            return "G"

    # Fallback: best guess from finger count
    if extended_count == 4 and thumb:
        return "B"
    if extended_count == 3:
        return "W"
    if extended_count == 2:
        return "V"
    if extended_count == 1:
        return "D"
    if extended_count == 0 and thumb:
        return "A"
    if extended_count == 0:
        return "S"

    return ""
