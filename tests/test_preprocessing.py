"""
test_preprocessing.py — Unit Tests for Preprocessing
===================================================
Verifies coordinate centering relative to wrist and scaling.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import normalize_hand_landmarks, prepare_input_vector


class DummyLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def test_wrist_normalization():
    # Construct 21 mock landmarks. Wrist is (0.5, 0.5, 0.1). Middle MCP is (0.5, 0.2, 0.1).
    # Scaling distance = sqrt(0.0 + 0.09 + 0.0) = 0.3
    landmarks = [DummyLandmark(0.5, 0.5, 0.1)]  # wrist (0)
    for i in range(8):
        landmarks.append(DummyLandmark(0.4, 0.4, 0.1))
    landmarks.append(DummyLandmark(0.5, 0.2, 0.1))  # middle mcp (9)
    for i in range(11):
        landmarks.append(DummyLandmark(0.4, 0.4, 0.1))

    normalized = normalize_hand_landmarks(landmarks)
    
    assert len(normalized) == 63
    # Wrist (idx 0) should be exactly (0.0, 0.0, 0.0)
    assert abs(normalized[0]) < 1e-6
    assert abs(normalized[1]) < 1e-6
    assert abs(normalized[2]) < 1e-6

    # Middle MCP (idx 9) should be (0.0, -0.3/0.3, 0.0) = (0.0, -1.0, 0.0)
    assert abs(normalized[9*3]) < 1e-6
    assert abs(normalized[9*3 + 1] - (-1.0)) < 1e-6
    assert abs(normalized[9*3 + 2]) < 1e-6
    
    print("[SUCCESS] Wrist normalization and scaling tests passed!")


def test_prepare_input_vector():
    landmarks = [DummyLandmark(0.5, 0.5, 0.1)]
    for i in range(20):
        landmarks.append(DummyLandmark(0.4, 0.4, 0.1))

    # Single hand list -> should pad second hand with zeros
    vector = prepare_input_vector([landmarks])
    assert len(vector) == 126
    # Second hand (index 63 to 125) should be all 0.0
    for val in vector[63:]:
        assert val == 0.0

    print("[SUCCESS] Input vector padding tests passed!")


if __name__ == "__main__":
    test_wrist_normalization()
    test_prepare_input_vector()
    print("[INFO] All preprocessing tests completed successfully.")
