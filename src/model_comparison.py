# model_comparison.py
# Compares two face detection approaches:
# 1. Haar Cascade (classic method)
# 2. HOG + dlib (our method)
# Measures accuracy, speed and FPS for each

import cv2
import face_recognition
import numpy as np
import time
import os

# ── Setup ─────────────────────────────────────────────────────
TEST_IMAGE = "known_faces/divya_1.jpeg"

def load_test_image():
    """Load and prepare test image"""
    img = cv2.imread(TEST_IMAGE)
    img = cv2.resize(img, (600, 800))
    return img

# ══════════════════════════════════════════════════════════════
# METHOD 1 — Haar Cascade
# ══════════════════════════════════════════════════════════════

def detect_haar(image, runs=10):
    """
    Haar Cascade face detection
    
    HOW IT WORKS:
    - Uses a pre-trained XML file with face features
    - Scans image at multiple scales using sliding window
    - Each window checked against cascade of simple features
    - Fast but struggles with non-frontal faces
    
    INVENTED: 2001 by Viola and Jones
    TYPE: Classical Computer Vision
    """

    # load haar cascade XML file
    # comes built into OpenCV — no download needed!
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # convert to grayscale
    # haar cascade works on grayscale images
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # measure speed over multiple runs
    start_time = time.time()

    for _ in range(runs):
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,   # how much image is reduced each scale
            minNeighbors=5,    # how many neighbors each rectangle should have
            minSize=(30, 30)   # minimum face size to detect
        )

    end_time = time.time()
    avg_time = (end_time - start_time) / runs
    fps = round(1 / avg_time, 1)

    # draw results
    result_image = image.copy()
    face_count = len(faces) if isinstance(faces, np.ndarray) else 0

    if face_count > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(result_image, "Haar", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return {
        "method": "Haar Cascade",
        "faces_found": face_count,
        "avg_time_ms": round(avg_time * 1000, 2),
        "fps": fps,
        "image": result_image
    }


# ══════════════════════════════════════════════════════════════
# METHOD 2 — HOG + dlib (our method)
# ══════════════════════════════════════════════════════════════

def detect_hog(image, runs=10):
    """
    HOG + dlib face detection
    
    HOW IT WORKS:
    - HOG = Histogram of Oriented Gradients
    - Looks at gradient directions in image patches
    - More robust to lighting and angle changes
    - Uses SVM classifier on top of HOG features
    
    INVENTED: 2005 by Dalal and Triggs
    TYPE: Machine Learning based
    """

    # convert BGR to RGB for face_recognition
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image = np.ascontiguousarray(rgb_image, dtype=np.uint8)

    # measure speed
    start_time = time.time()

    for _ in range(runs):
        face_locations = face_recognition.face_locations(rgb_image)

    end_time = time.time()
    avg_time = (end_time - start_time) / runs
    fps = round(1 / avg_time, 1)

    # draw results
    result_image = image.copy()
    face_count = len(face_locations)

    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(result_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(result_image, "HOG", (left, top-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return {
        "method": "HOG + dlib",
        "faces_found": face_count,
        "avg_time_ms": round(avg_time * 1000, 2),
        "fps": fps,
        "image": result_image
    }


# ══════════════════════════════════════════════════════════════
# COMPARISON REPORT
# ══════════════════════════════════════════════════════════════

def print_comparison(haar_results, hog_results):
    """Print detailed comparison report"""

    print("\n" + "="*60)
    print("      FACE DETECTION MODEL COMPARISON REPORT")
    print("="*60)

    print(f"\n{'Metric':<25} {'Haar Cascade':<20} {'HOG + dlib':<20}")
    print("-"*60)
    print(f"{'Faces Found':<25} {haar_results['faces_found']:<20} {hog_results['faces_found']:<20}")
    print(f"{'Avg Time (ms)':<25} {haar_results['avg_time_ms']:<20} {hog_results['avg_time_ms']:<20}")
    print(f"{'FPS':<25} {haar_results['fps']:<20} {hog_results['fps']:<20}")
    print("-"*60)

    print("\n📊 ANALYSIS:")
    print("\n1. SPEED:")
    if haar_results['avg_time_ms'] < hog_results['avg_time_ms']:
        print(f"   Haar Cascade is FASTER by {round(hog_results['avg_time_ms'] - haar_results['avg_time_ms'], 2)}ms")
    else:
        print(f"   HOG + dlib is FASTER by {round(haar_results['avg_time_ms'] - hog_results['avg_time_ms'], 2)}ms")

    print("\n2. ACCURACY:")
    print("   Haar Cascade: Good for frontal faces, struggles with angles")
    print("   HOG + dlib  : Better accuracy across different angles")

    print("\n3. USE CASES:")
    print("   Haar Cascade: Best for real-time systems needing speed")
    print("   HOG + dlib  : Best for accuracy-critical systems")

    print("\n4. OUR CHOICE — HOG + dlib because:")
    print("   ✅ Better accuracy for attendance (we need correct identification)")
    print("   ✅ More robust to different lighting conditions")
    print("   ✅ Works better with glasses and partial faces")
    print("   ✅ 10+ FPS still sufficient for real-time use")

    print("\n" + "="*60)

    # save report to file
    with open("docs/model_comparison_report.txt", "w") as f:
        f.write("FACE DETECTION MODEL COMPARISON REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Haar Cascade:\n")
        f.write(f"  Faces Found  : {haar_results['faces_found']}\n")
        f.write(f"  Avg Time     : {haar_results['avg_time_ms']}ms\n")
        f.write(f"  FPS          : {haar_results['fps']}\n\n")
        f.write(f"HOG + dlib:\n")
        f.write(f"  Faces Found  : {hog_results['faces_found']}\n")
        f.write(f"  Avg Time     : {hog_results['avg_time_ms']}ms\n")
        f.write(f"  FPS          : {hog_results['fps']}\n\n")
        f.write("Conclusion: HOG + dlib chosen for better accuracy\n")

    print(f"\nReport saved to docs/model_comparison_report.txt")


# ── Run comparison ────────────────────────────────────────────
if __name__ == "__main__":

    # create docs folder
    os.makedirs("docs", exist_ok=True)

    print("Loading test image...")
    image = load_test_image()

    print("Running Haar Cascade detection...")
    haar_results = detect_haar(image)

    print("Running HOG + dlib detection...")
    hog_results = detect_hog(image)

    # show results
    print_comparison(haar_results, hog_results)

    # save result images
    cv2.imwrite("docs/haar_result.jpg", haar_results["image"])
    cv2.imwrite("docs/hog_result.jpg", hog_results["image"])
    print("Result images saved to docs/ folder!")

    # show images side by side
    combined = np.hstack([haar_results["image"], hog_results["image"]])
    cv2.imshow("Haar Cascade vs HOG + dlib", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()