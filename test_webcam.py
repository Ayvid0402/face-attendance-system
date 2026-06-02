# test_webcam.py - find DroidCam
import cv2
import time

print("Searching for DroidCam...\n")

for index in [0, 1, 2, 3]:
    for backend, name in [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MSMF")]:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            time.sleep(1)
            ret, frame = cap.read()
            if ret:
                print(f"  Camera {index} with {name} works! ✅")
                cv2.imwrite(f"src/camera_{index}_{name}.jpg", frame)
            else:
                print(f"  Camera {index} with {name} opened but cant read ❌")
            cap.release()
        else:
            print(f"  Camera {index} with {name} not found ❌")

print("\nDone!")