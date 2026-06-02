# webcam_attendance.py
import cv2
import numpy as np
import pickle
import os
import sys
import time

sys.path.append("src")
from face_recognizer import load_encodings, recognize_faces, draw_results

UNKNOWN_FACES_FOLDER = "unknown_faces"
FRAME_SKIP = 3

def run_attendance_system():

    print("Starting Real-Time Attendance System...")
    print("Press Q to quit\n")

    known_encodings, known_names = load_encodings()

    if len(known_encodings) == 0:
        print("No encodings found! Run encode_faces.py first!")
        return

    # using IP Webcam app on phone
    print("Connecting to IP Webcam...")
    cap = cv2.VideoCapture("http://192.168.1.33:8080/video")

    # warmup
    print("Warming up camera...")
    for i in range(5):
        cap.read()
        time.sleep(0.1)
        print(f"  Warming up {i+1}/5")

    if not cap.isOpened():
        print("ERROR: Cannot connect to IP Webcam!")
        print("Make sure IP Webcam app is running on your phone!")
        return

    ret, test_frame = cap.read()
    if not ret:
        print("ERROR: Cannot read from IP Webcam!")
        return

    print("IP Webcam connected successfully!")
    print("Look at your phone camera!\n")

    frame_count = 0
    results = []
    fps_start = time.time()
    fps = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame!")
            break

        frame_count += 1

        if frame_count % FRAME_SKIP == 0:
            results = recognize_faces(frame, known_encodings, known_names)

            fps_end = time.time()
            fps = round(1 / (fps_end - fps_start + 0.001), 1)
            fps_start = fps_end

        frame = draw_results(frame, results)

        for result in results:
            if result["name"] == "Unknown":
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                unknown_path = os.path.join(
                    UNKNOWN_FACES_FOLDER,
                    f"unknown_{timestamp}.jpg"
                )
                if not os.path.exists(unknown_path):
                    top, right, bottom, left = result["location"]
                    face_img = frame[top:bottom, left:right]
                    if face_img.size > 0:
                        cv2.imwrite(unknown_path, face_img)

        cv2.putText(
            frame, f"FPS: {fps}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )

        cv2.putText(
            frame, "Press Q to quit",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )

        cv2.putText(
            frame, f"Faces: {len(results)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )

        cv2.imshow("Face Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nQ pressed — stopping system...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("System stopped!")
    print(f"Total frames processed: {frame_count}")


if __name__ == "__main__":
    run_attendance_system()