# video_attendance.py
# This file processes a video file instead of webcam
# Recognizes faces frame by frame and marks attendance
# Usage: python src/video_attendance.py

import cv2
import numpy as np
import pickle
import os
import sys
import time

sys.path.append("src")
from face_recognizer import load_encodings, recognize_faces, draw_results
from attendance_logger import mark_attendance

# ── Setup ─────────────────────────────────────────────────────
VIDEO_PATH = "test_video.mp4"      # put your video file here
OUTPUT_PATH = "src/output_video.avi"  # processed video saved here
FRAME_SKIP = 5                     # process every 5th frame

def process_video(video_path):
    """
    Process a video file for face recognition
    Works exactly like webcam but reads from file
    """

    print(f"Processing video: {video_path}\n")

    # ── Step 1: Load encodings ────────────────────────────────
    known_encodings, known_names = load_encodings()

    if len(known_encodings) == 0:
        print("No encodings found! Run encode_faces.py first!")
        return

    # ── Step 2: Open video file ───────────────────────────────
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {video_path}")
        print("Make sure the video file exists in the project folder!")
        return

    # get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info:")
    print(f"  Total frames : {total_frames}")
    print(f"  FPS          : {fps}")
    print(f"  Resolution   : {width}x{height}\n")

    # ── Step 3: Setup output video writer ─────────────────────
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # ── Step 4: Process frames ────────────────────────────────
    frame_count = 0
    results = []
    people_detected = set()
    start_time = time.time()

    print("Processing frames...")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("\nVideo processing complete!")
            break

        frame_count += 1

        # show progress every 30 frames
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

        # process every Nth frame
        if frame_count % FRAME_SKIP == 0:
            results = recognize_faces(frame, known_encodings, known_names)

            # track who was detected
            for result in results:
                if result["name"] != "Unknown":
                    people_detected.add(result["name"])

        # draw results on frame
        frame = draw_results(frame, results)

        # add frame number
        cv2.putText(
            frame, f"Frame: {frame_count}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )

        # write to output video
        out.write(frame)

        # show frame
        cv2.imshow("Video Attendance Processing", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nQ pressed — stopping!")
            break

    # ── Step 5: Cleanup ───────────────────────────────────────
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # ── Step 6: Summary ───────────────────────────────────────
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)

    print(f"\n{'='*40}")
    print(f"VIDEO PROCESSING SUMMARY")
    print(f"{'='*40}")
    print(f"Total frames processed : {frame_count}")
    print(f"Processing time        : {processing_time} seconds")
    print(f"People detected        : {people_detected}")
    print(f"Output saved to        : {OUTPUT_PATH}")
    print(f"{'='*40}")


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":

    # check if video file exists
    if not os.path.exists(VIDEO_PATH):
        print(f"Video file '{VIDEO_PATH}' not found!")
        print("Please add a video file named 'test_video.mp4' to the project folder!")
        print("\nHow to get a test video:")
        print("  1. Record a short video on your phone")
        print("  2. Transfer to laptop")
        print("  3. Rename it to 'test_video.mp4'")
        print("  4. Put it in face_attendance_system folder")
    else:
        process_video(VIDEO_PATH)