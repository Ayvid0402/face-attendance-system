# mask_detection.py
# Fast mask detection using skin color analysis

import cv2
import face_recognition
import numpy as np

MASK_COLOR = (0, 255, 255)    # yellow for mask
NO_MASK_COLOR = (0, 255, 0)   # green for no mask

def check_mask_by_skin(face_img):
    """
    Check mask using skin color detection
    Lower half of face should show skin
    If no skin detected → mask is covering it!
    """

    if face_img is None or face_img.size == 0:
        return False

    # get lower half of face
    h, w = face_img.shape[:2]
    lower_face = face_img[h//2:, :]

    # convert to HSV
    hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)

    # skin color range
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    skin_pixels = cv2.countNonZero(skin_mask)
    total_pixels = lower_face.shape[0] * lower_face.shape[1]

    if total_pixels == 0:
        return False

    skin_percentage = (skin_pixels / total_pixels) * 100

    # print for debugging
    print(f"  Skin %: {skin_percentage:.1f}%")

    # if less than 8% skin → mask detected
    if skin_percentage < 8:
        return True
    return False


def detect_mask_in_frame(frame):
    """
    Detect masks in frame
    """
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

    face_locations = face_recognition.face_locations(rgb_frame)

    results = []

    for face_location in face_locations:
        top, right, bottom, left = face_location

        # scale back
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        face_img = frame[top:bottom, left:right]

        if face_img.size == 0:
            continue

        has_mask = check_mask_by_skin(face_img)

        if has_mask:
            color = MASK_COLOR
            label = "Mask Detected!"
            status = "MASK"
        else:
            color = NO_MASK_COLOR
            label = "No Mask"
            status = "NO MASK"

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(
            frame,
            (left, top - 35),
            (right, top),
            color, cv2.FILLED
        )
        cv2.putText(
            frame, label,
            (left + 6, top - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5, (255, 255, 255), 1
        )

        results.append({
            "status": status,
            "has_mask": has_mask,
            "location": (top, right, bottom, left)
        })

    return frame, results


def run_mask_detection():
    """
    Run mask detection using IP Webcam
    """
    print("Starting Face Mask Detection...")
    print("Press Q to quit\n")

    cap = cv2.VideoCapture("http://192.168.1.35:8080/video")

    if not cap.isOpened():
        print("ERROR: Cannot connect to camera!")
        return

    frame_count = 0
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 5 == 0:
            frame, results = detect_mask_in_frame(frame)

        cv2.putText(
            frame, "Mask Detection — Press Q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )

        cv2.imshow("Face Mask Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    run_mask_detection()