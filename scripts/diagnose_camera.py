import cv2
import sys
import subprocess
import time

print("=" * 60)
print(" COMPREHENSIVE WINDOWS CAMERA DIAGNOSTIC TOOL")
print("=" * 60)

# 1. Enumerate PnP Camera Devices via PowerShell / WMI
print("\n[1] Querying Windows PnP Video Devices via PowerShell...")
try:
    ps_cmd = "Get-PnPDevice -Class Camera,Image | Select-Object FriendlyName, Status, InstanceId | Format-Table -AutoSize"
    res = subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, text=True, timeout=10)
    print(res.stdout if res.stdout else "No PnP devices returned.")
except Exception as e:
    print(f"PnP query error: {e}")

# 2. Test OpenCV Camera Indices across Backends
print("\n[2] Testing OpenCV VideoCapture Indices (0 to 5) across Backends...")
backends = [
    ("CAP_DSHOW (DirectShow)", cv2.CAP_DSHOW),
    ("CAP_MSMF (Media Foundation)", cv2.CAP_MSMF),
    ("CAP_ANY (Default)", cv2.CAP_ANY),
]

found_cameras = []

for idx in range(6):
    for name, b_id in backends:
        try:
            print(f"Testing Index {idx} with {name}...", end=" ", flush=True)
            cap = cv2.VideoCapture(idx, b_id)
            if not cap.isOpened():
                print("isOpened=False")
                continue
            
            # Read frame
            ret, frame = cap.read()
            if ret and frame is not None:
                mean_val = float(frame.mean())
                shape_str = f"{frame.shape}"
                print(f"SUCCESS! ret=True, shape={shape_str}, mean={mean_val:.2f}")
                if mean_val > 1.0:
                    found_cameras.append((idx, b_id, name, mean_val))
            else:
                print("isOpened=True but cap.read() failed")
            
            cap.release()
        except Exception as ex:
            print(f"Exception: {ex}")

print("\n" + "=" * 60)
if found_cameras:
    print(f"WORKING PHYSICAL CAMERAS FOUND ({len(found_cameras)}):")
    for idx, b_id, name, mean_val in found_cameras:
        print(f"  -> Index {idx} [{name}] — Pixel Mean Brightness: {mean_val:.2f}")
else:
    print("NO WORKING OPTICAL CAMERAS RETURNED A NON-BLACK FRAME.")
print("=" * 60)
