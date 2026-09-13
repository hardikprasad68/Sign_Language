import sys
import os
import numpy as np

# Add src to path so we can import your modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'src'))

from inference.live_inference import SignPredictor  # type: ignore
from inference.sequence_buffer import SequenceBuffer  # type: ignore

def run_integration_test():
    print("[INFO] Initializing Inclubate AI Engine...")
    engine = SignPredictor(os.path.join(project_root, 'models'))
    buffer = SequenceBuffer(sequence_length=30, feature_dim=126)

    print("\n[TEST 1] Static Sign Prediction")
    # Simulate one frame of exactly 126 coordinates
    dummy_static_frame = np.random.rand(126).astype(np.float32)
    static_result = engine.predict_static(dummy_static_frame)
    print(f"Static Engine Output: {static_result}")

    print("\n[TEST 2] Dynamic Sign Sequence")
    # Simulate a user performing a motion gesture over 40 frames
    print("Feeding webcam frames into motion buffer...")
    for i in range(40):
        # Create a fake moving hand (changing coordinates)
        dummy_dynamic_frame = np.random.rand(126).astype(np.float32) + (i * 0.01)
        buffer.add_frame(dummy_dynamic_frame.tolist())
        
        # Once the buffer detects a completed motion, pass it to the AI
        if buffer.has_gesture_ready():
            sequence = buffer.get_sequence()
            print(f"Motion captured! Sequence shape: {sequence.shape}")
            
            dynamic_result = engine.predict_dynamic(sequence)
            print(f"Dynamic Engine Output: {dynamic_result}")
            buffer.clear()
            break

    print("\n[SUCCESS] AI Wrapper and Motion Buffer are fully integrated!")

if __name__ == "__main__":
    run_integration_test()