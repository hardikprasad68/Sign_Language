"""Quick camera diagnostic — tests indices 0-3 with both MSMF and DSHOW backends."""
import cv2
import time

results = []

for idx in range(4):
    for backend_name, backend in [("MSMF", cv2.CAP_ANY), ("DSHOW", cv2.CAP_DSHOW)]:
        print(f"Testing index {idx} with {backend_name}...", end=" ", flush=True)
        start = time.time()
        try:
            cap = cv2.VideoCapture(idx, backend)
            opened = cap.isOpened()
            read_ok = False
            if opened:
                ret, frame = cap.read()
                read_ok = ret and frame is not None
                if read_ok:
                    h, w = frame.shape[:2]
                    results.append(f"  [OK] Index {idx} + {backend_name}: {w}x{h}")
                else:
                    results.append(f"  [FAIL] Index {idx} + {backend_name}: opened=True but read() failed")
            else:
                results.append(f"  [SKIP] Index {idx} + {backend_name}: isOpened()=False")
            cap.release()
        except Exception as e:
            results.append(f"  [ERROR] Index {idx} + {backend_name}: {e}")
        elapsed = time.time() - start
        print(f"done ({elapsed:.1f}s)")

print("\n" + "=" * 55)
print(" CAMERA DIAGNOSTIC RESULTS")
print("=" * 55)
for r in results:
    print(r)
print("=" * 55)
