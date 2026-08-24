import cv2
import time
import numpy as np

print("--- Testing Physical Webcams ---")
for idx in range(4):
    for backend_name, backend in [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]:
        try:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                continue
            
            # Read 15 warmup frames
            means = []
            for _ in range(15):
                ret, frame = cap.read()
                if ret and frame is not None:
                    means.append(float(np.mean(frame)))
                time.sleep(0.03)
            
            cap.release()
            if means:
                max_mean = max(means)
                last_mean = means[-1]
                print(f"Index {idx} [{backend_name}]: max_mean={max_mean:.2f}, last_mean={last_mean:.2f}, frames_read={len(means)}")
            else:
                print(f"Index {idx} [{backend_name}]: Opened but failed to read frames")
        except Exception as e:
            print(f"Index {idx} [{backend_name}]: Error {e}")
