# encode_faces.py
import face_recognition
import os
import pickle
import cv2
import numpy as np

print("Starting face encoding...\n")

known_faces_folder = "known_faces"
encodings_file = "src/encodings.pkl"

known_encodings = []
known_names = []

print(f"Reading photos from '{known_faces_folder}' folder...\n")

for filename in os.listdir(known_faces_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):

        name = os.path.splitext(filename)[0]
        photo_path = os.path.join(known_faces_folder, filename)
        print(f"  Processing: {filename}  →  Name: {name}")

        image_bgr = cv2.imread(photo_path)
        image_bgr = cv2.resize(image_bgr, (600, 800))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)

        print(f"  Shape : {image_rgb.shape}")
        print(f"  Dtype : {image_rgb.dtype}")

        face_locations = face_recognition.face_locations(image_rgb)
        print(f"  Faces found: {len(face_locations)}")

        if len(face_locations) == 0:
            print(f"  WARNING: No face found - skipping!")
            continue

        encoding = face_recognition.face_encodings(image_rgb, face_locations)[0]

        known_encodings.append(encoding)
        known_names.append(name)
        print(f"  Encoded successfully!\n")

print("Saving encodings to file...")
data = {"encodings": known_encodings, "names": known_names}

with open(encodings_file, "wb") as f:
    pickle.dump(data, f)

print(f"\nDone!")
print(f"Total faces encoded : {len(known_names)}")
print(f"Names registered    : {known_names}")
print(f"Saved to            : {encodings_file}")