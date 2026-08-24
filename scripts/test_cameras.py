import cv2

def scan_cameras():
    print("Scanning camera indices 0 to 5...")
    found_any = False
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  [SUCCESS] Camera index {i}: Opened successfully and read a frame of shape {w}x{h}")
                found_any = True
            else:
                print(f"  [WARNING] Camera index {i}: Opened, but failed to read a frame (could be in use or virtual)")
            cap.release()
        else:
            print(f"  [INFO] Camera index {i}: Could not open")
    
    if not found_any:
        print("\nNo working cameras returned a valid frame.")
        print("Please check if your webcam is plugged in, or if it is currently being used by another application (e.g. Chrome, Discord, Zoom).")
        print("You can also try restarting your computer to release any locked camera resources.")

if __name__ == "__main__":
    scan_cameras()
