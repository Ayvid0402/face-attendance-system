# face_recognizer.py
# This file loads our saved encodings and recognizes faces
# It compares new faces against our known face database

import face_recognition
import cv2
import numpy as np
import pickle
import os
import sys

sys.path.append("src")
from attendance_logger import mark_attendance
from database import mark_attendance_db
from preprocessor import preprocess_frame

# ── Setup ─────────────────────────────────────────────────────
ENCODINGS_FILE = "src/encodings.pkl"
UNKNOWN_FACES_FOLDER = "unknown_faces"
TOLERANCE = 0.5

def load_encodings():
    """
    Load saved face encodings from pickle file
    """
    if not os.path.exists(ENCODINGS_FILE):
        print("ERROR: No encodings file found!")
        print("Please run encode_faces.py first!")
        return [], []

    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded {len(data['names'])} face encodings")
    print(f"Known people: {data['names']}")
    return data["encodings"], data["names"]


def recognize_faces(frame, known_encodings, known_names):
    """
    Takes a single video frame and recognizes all faces in it

    Steps:
    1. Preprocess frame (resize, RGB, normalize)
    2. Find face locations
    3. Encode faces to 128 numbers
    4. Compare with known encodings
    5. Return results
    """

    # ── Step 1: Preprocess frame ──────────────────────────────
    # resize to 25%, convert BGR to RGB, normalize
    rgb_small_frame = preprocess_frame(frame, scale=0.25)

    # ── Step 2: Find all faces in frame ───────────────────────
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    results = []

    # ── Step 3: Loop through each found face ──────────────────
    for face_encoding, face_location in zip(face_encodings, face_locations):

        # compare face against all known faces
        matches = face_recognition.compare_faces(
            known_encodings,
            face_encoding,
            tolerance=TOLERANCE
        )

        # calculate distance to every known face
        face_distances = face_recognition.face_distance(
            known_encodings,
            face_encoding
        )

        name = "Unknown"
        confidence = 0.0

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]

            if matches[best_match_index]:
                raw_name = known_names[best_match_index]
                name = raw_name.split("_")[0]

                # convert distance to confidence percentage
                confidence = round((1 - best_distance) * 100, 1)

                # mark attendance in CSV
                mark_attendance(name)

                # mark attendance in SQLite database
                mark_attendance_db(name, confidence)

        # scale face location back to original size
        # we resized to 25% so multiply by 4
        top, right, bottom, left = face_location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        results.append({
            "name": name,
            "confidence": confidence,
            "location": (top, right, bottom, left)
        })

    return results


def draw_results(frame, results):
    """
    Draws bounding boxes and labels on the frame
    Green box = known person
    Red box = unknown person
    """
    for result in results:
        name = result["name"]
        confidence = result["confidence"]
        top, right, bottom, left = result["location"]

        if name == "Unknown":
            color = (0, 0, 255)    # red for unknown
            label = "Unknown"
        else:
            color = (0, 255, 0)    # green for known
            label = f"{name} ({confidence}%)"

        # draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # draw filled rectangle for label background
        cv2.rectangle(
            frame,
            (left, bottom - 35),
            (right, bottom),
            color, cv2.FILLED
        )

        # draw name and confidence text
        cv2.putText(
            frame, label,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6, (255, 255, 255), 1
        )

    return frame


# ── Test with a single image ──────────────────────────────────
if __name__ == "__main__":
    print("Testing face recognizer...\n")

    known_encodings, known_names = load_encodings()

    test_image_path = "known_faces/divya_1.jpeg"

    if os.path.exists(test_image_path):
        frame = cv2.imread(test_image_path)
        frame = cv2.resize(frame, (600, 800))

        results = recognize_faces(frame, known_encodings, known_names)
        frame = draw_results(frame, results)

        print(f"\nResults:")
        for r in results:
            print(f"  Name       : {r['name']}")
            print(f"  Confidence : {r['confidence']}%")
            print(f"  Location   : {r['location']}")

        cv2.imwrite("src/test_result.jpg", frame)
        print(f"\nResult image saved to src/test_result.jpg")
    else:
        print(f"Test image not found!")