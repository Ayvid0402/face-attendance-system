# anti_spoofing.py
# Detects if face is real or fake (photo/screen)
# Method: Eye Aspect Ratio (EAR) for blink detection
# Real face blinks → fake face doesn't blink!

import cv2
import numpy as np
from scipy.spatial import distance
import face_recognition

# ── Constants ─────────────────────────────────────────────────
EAR_THRESHOLD = 0.25    # below this = eye is closed (blinking)
BLINK_FRAMES = 2        # eye must be closed for this many frames
BLINKS_NEEDED = 1       # need this many blinks to confirm real face
CHECK_FRAMES = 200      # check within this many frames

def eye_aspect_ratio(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR)
    
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    
    When eye is OPEN  → EAR is high (~0.3)
    When eye is CLOSED → EAR is low (~0.1)
    
    Args:
        eye_points: 6 landmark points around eye
    
    Returns:
        EAR value (float)
    """
    # vertical distances
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])

    # horizontal distance
    C = distance.euclidean(eye_points[0], eye_points[3])

    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear


def get_eye_points(face_landmarks):
    """
    Extract eye landmark points from face landmarks
    face_recognition gives us 68 facial landmarks
    We use points 36-41 for left eye and 42-47 for right eye
    """
    left_eye = face_landmarks.get('left_eye', [])
    right_eye = face_landmarks.get('right_eye', [])
    return left_eye, right_eye


class AntiSpoofing:
    """
    Anti-spoofing detector using blink detection
    
    How to use:
    1. Create instance: detector = AntiSpoofing()
    2. For each frame: result = detector.check_frame(frame)
    3. Result tells if face is real or fake
    """

    def __init__(self):
        self.blink_count = 0
        self.frame_count = 0
        self.consecutive_closed = 0
        self.is_verified = False
        self.status = "Checking... Please blink!"

    def reset(self):
        """Reset detector for new person"""
        self.blink_count = 0
        self.frame_count = 0
        self.consecutive_closed = 0
        self.is_verified = False
        self.status = "Checking... Please blink!"

    def check_frame(self, frame):
        """
        Check if face in frame is real or fake

        Returns:
            dict with:
            - is_real: True/False
            - is_verified: True if enough blinks detected
            - blink_count: how many blinks detected
            - status: message to display
            - ear: current eye aspect ratio
        """
        self.frame_count += 1

        # convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get face landmarks
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

        if len(face_landmarks_list) == 0:
            return {
                "is_real": False,
                "is_verified": False,
                "blink_count": self.blink_count,
                "status": "No face detected!",
                "ear": 0.0
            }

        # get first face landmarks
        face_landmarks = face_landmarks_list[0]

        # get eye points
        left_eye, right_eye = get_eye_points(face_landmarks)

        if len(left_eye) == 0 or len(right_eye) == 0:
            return {
                "is_real": False,
                "is_verified": self.is_verified,
                "blink_count": self.blink_count,
                "status": "Cannot detect eyes!",
                "ear": 0.0
            }

        # calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # check if eyes are closed (blinking)
        if avg_ear < EAR_THRESHOLD:
            self.consecutive_closed += 1
        else:
            # eyes opened after being closed = blink detected!
            if self.consecutive_closed >= BLINK_FRAMES:
                self.blink_count += 1
                print(f"  Blink detected! Total: {self.blink_count}")
            self.consecutive_closed = 0

        # check if enough blinks detected
        if self.blink_count >= BLINKS_NEEDED:
            self.is_verified = True
            self.status = f"REAL FACE ✅ ({self.blink_count} blinks)"
        elif self.frame_count > CHECK_FRAMES and self.blink_count == 0:
            self.status = "FAKE FACE ❌ No blinks detected!"
        else:
            remaining = BLINKS_NEEDED - self.blink_count
            self.status = f"Please blink {remaining} more time(s)!"

        return {
            "is_real": self.is_verified,
            "is_verified": self.is_verified,
            "blink_count": self.blink_count,
            "status": self.status,
            "ear": round(avg_ear, 3)
        }


def run_anti_spoofing_webcam():
    """
    Run anti-spoofing check using IP Webcam
    """
    print("Starting Anti-Spoofing Detection...")
    print("Please blink 2 times to verify you are real!\n")

    cap = cv2.VideoCapture("http://192.168.1.35:8080//videofeed")

    if not cap.isOpened():
        print("ERROR: Cannot connect to camera!")
        return

    detector = AntiSpoofing()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # resize for display
        frame = cv2.resize(frame, (640, 480))

        # check frame
        result = detector.check_frame(frame)

        # choose color based on result
        if result["is_verified"]:
            color = (0, 255, 0)   # green = real
        else:
            color = (0, 0, 255)   # red = fake/checking

        # display status
        cv2.putText(
            frame, result["status"],
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, 2
        )

        # display EAR value
        cv2.putText(
            frame, f"EAR: {result['ear']}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 0), 2
        )

        # display blink count
        cv2.putText(
            frame, f"Blinks: {result['blink_count']}/{BLINKS_NEEDED}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 0), 2
        )

        # show frame
        cv2.imshow("Anti-Spoofing Detection", frame)

        # stop if verified
        if result["is_verified"]:
            print("✅ Real face verified!")
            cv2.waitKey(2000)
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    run_anti_spoofing_webcam()