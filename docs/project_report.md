# Real-Time Face Recognition Attendance System
## Project Report

**Author:** Divya Maria Manuel
**Project Type:** Computer Vision Internship Task

---

## 1. Introduction

### 1.1 Problem Statement
Traditional attendance systems rely on manual marking or ID cards, which are:
- Time consuming
- Prone to proxy attendance
- Inefficient for large groups

### 1.2 Proposed Solution
A Real-Time Face Recognition Attendance System that:
- Automatically detects and recognizes faces
- Marks attendance without human intervention
- Prevents proxy attendance
- Stores records in database

### 1.3 Objectives
1. Build a real-time face detection system
2. Implement face recognition using embeddings
3. Automate attendance logging
4. Handle unknown faces
5. Provide a dashboard for attendance visualization

---

## 2. System Architecture
Camera Input (IP Webcam)
↓
Image Preprocessing
(Resize + BGR→RGB + Normalize)
↓
Face Detection
(HOG + dlib)
↓
Face Recognition
(128-d Embeddings + Euclidean Distance)
↓
Anti-Spoofing Check
(Blink Detection)
↓
Attendance Logging
(CSV + SQLite)
↓
Streamlit Dashboard

---

## 3. Technical Implementation

### 3.1 Face Detection
**Method:** HOG (Histogram of Oriented Gradients) + dlib

HOG works by:
1. Dividing image into small cells
2. Computing gradient directions in each cell
3. Creating histogram of gradient orientations
4. Using SVM classifier to detect faces

**Why HOG over Haar Cascade:**
- Lower false positive rate
- More robust to lighting changes
- Better with glasses and accessories

### 3.2 Face Recognition
**Method:** 128-dimensional face embeddings

The process:
1. Detect face location in image
2. Extract 68 facial landmarks
3. Pass through dlib's deep neural network
4. Get 128-dimensional embedding vector
5. Compare with known embeddings using Euclidean distance
Known face embedding:  [0.12, -0.45, 0.87, ...]
New face embedding:    [0.11, -0.43, 0.85, ...]
Distance: 0.17 → Same person! (< 0.5 threshold)

### 3.3 Preprocessing Pipeline

| Step | Method | Reason |
|------|--------|--------|
| Resize | fx=0.25 | 16x speed improvement |
| BGR→RGB | cvtColor | face_recognition needs RGB |
| Normalize | /255.0 | consistent 0-1 range |
| Contiguous | ascontiguousarray | dlib C++ requirement |

### 3.4 Attendance Logging
Two storage methods implemented:

**CSV Storage:**
Name,Date,Time
divya,2026-06-01,10:30:45
jeswin,2026-06-01,10:31:22

**SQLite Storage:**
```sql
CREATE TABLE attendance (
    id INTEGER PRIMARY KEY,
    name TEXT,
    date TEXT,
    time TEXT,
    confidence REAL
)
```

Duplicate prevention: same person marked only once per day.

### 3.5 Anti-Spoofing
**Method:** Eye Aspect Ratio (EAR) blink detection
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
Eye OPEN  → EAR ≈ 0.30
Eye CLOSED → EAR ≈ 0.10

Real face blinks → attendance marked ✅
Photo/screen no blinks → rejected ❌

### 3.6 Unknown Face Handling
When face distance > 0.5 threshold:
- Marked as "Unknown" with red bounding box
- Snapshot saved to `unknown_faces/` folder
- Not marked in attendance

---

## 4. Model Comparison

### 4.1 Haar Cascade vs HOG + dlib

| Metric | Haar Cascade | HOG + dlib |
|--------|-------------|------------|
| Speed | 6.6 FPS | 3.0 FPS |
| Faces Found | 2 (false positive!) | 1 (correct) |
| Avg Time | 151.83ms | 328.62ms |
| False Positives | High | Low |
| Our Choice | ❌ | ✅ |

### 4.2 Conclusion
Haar Cascade is faster but produced a false positive — detected 2 faces when only 1 was present. For an attendance system, accuracy is more critical than speed. HOG + dlib was chosen for its superior accuracy.

---

## 5. Performance Evaluation

| Metric | Value |
|--------|-------|
| Recognition Confidence | 83.1% |
| FPS (Real-time) | 10+ FPS |
| False Positives | Low (tolerance=0.5) |
| False Negatives | Low (2 photos per person) |
| Duplicate Prevention | ✅ |
| Multi-face Support | ✅ |

### 5.1 Factors Affecting Accuracy
| Factor | Impact |
|--------|--------|
| Number of training photos | More photos = better accuracy |
| Lighting conditions | Good lighting improves accuracy |
| Face angle | Frontal face works best |
| Glasses/accessories | Slight reduction in accuracy |
| Image resolution | Higher resolution = better |

---

## 6. Challenges & Solutions

### Challenge 1: Windows Camera Bug
- **Problem:** HP Pavilion x360 gave PhotoCaptureStartTimeout error
- **Error Code:** 0xA00F4292
- **Solution:** Used IP Webcam Android app as alternative
- **Learning:** Always have backup camera solution

### Challenge 2: dlib Installation
- **Problem:** `pip install dlib` failed on Windows — needs C++ compiler
- **Solution:** `conda install -c conda-forge dlib`
- **Learning:** C++ libraries need conda on Windows

### Challenge 3: numpy Compatibility
- **Problem:** numpy 2.0 incompatible with dlib
- **Error:** RuntimeError: Unsupported image type
- **Solution:** Downgraded to numpy==1.24.3
- **Learning:** Check library compatibility matrix

### Challenge 4: Mask Detection
- **Problem:** face_recognition couldn't detect masked faces
- **Reason:** dlib model trained on unmasked faces
- **Workaround:** Skin color analysis for lower face region
- **Future Fix:** Dedicated CNN mask detection model

### Challenge 5: DroidCam Integration
- **Problem:** DroidCam OBS version didn't create virtual webcam
- **Solution:** Switched to IP Webcam app with direct HTTP stream
- **Learning:** Direct HTTP streaming more reliable than virtual drivers

---

## 7. Features Implemented

### Core Features ✅
- Real-time face detection and recognition
- Face detection from webcam, images, video files
- 128-dimensional face embeddings
- Euclidean distance similarity matching
- Confidence score display
- Attendance logging — Name, Date, Time
- CSV and SQLite storage
- Unknown face handling + snapshot
- Bounding boxes with labels

### Bonus Features ✅
- Multi-face recognition
- Streamlit dashboard with charts
- Docker deployment
- Anti-spoofing (blink detection)
- Mask detection (skin color)
- Model comparison report

---

## 8. Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.9 | Core language |
| OpenCV | 4.8.0 | Image processing |
| face_recognition | 1.3.0 | Face encoding |
| dlib | 19.24.2 | Deep learning model |
| numpy | 1.24.3 | Array operations |
| pandas | 2.3.3 | Data handling |
| SQLite | Built-in | Database |
| Streamlit | 1.50.0 | Dashboard |
| Plotly | Latest | Charts |
| scipy | Latest | Anti-spoofing |
| Docker | Latest | Deployment |

---

## 9. Future Improvements

1. **Cloud Database** — Supabase PostgreSQL for remote access
2. **GPU Acceleration** — CUDA for faster processing
3. **Better Mask Detection** — Dedicated CNN model
4. **Mobile App** — React Native frontend
5. **PostgreSQL/MySQL** — Enterprise database support
6. **Face Liveness Detection** — 3D depth sensing
7. **Attendance Reports** — PDF export feature

---

## 10. Conclusion

This project successfully implements a Real-Time Face Recognition Attendance System using Computer Vision techniques. The system:

- ✅ Detects faces in real-time at 10+ FPS
- ✅ Recognizes registered persons with 83.1% confidence
- ✅ Automatically marks attendance in CSV and SQLite
- ✅ Handles unknown faces with snapshot saving
- ✅ Includes anti-spoofing to prevent fake attendance
- ✅ Provides a beautiful Streamlit dashboard
- ✅ Supports Docker deployment

The system demonstrates practical application of Computer Vision concepts including face embeddings, similarity matching, and real-time processing. Despite hardware challenges (Windows camera bug), the system was successfully deployed using IP Webcam as an alternative camera source.

---

## References

1. Viola, P., Jones, M. (2001). Rapid Object Detection using a Boosted Cascade of Simple Features. CVPR.
2. Dalal, N., Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. CVPR.
3. King, D.E. (2009). Dlib-ml: A Machine Learning Toolkit. JMLR.
4. Soukupová, T., Čech, J. (2016). Real-Time Eye Blink Detection using Facial Landmarks. CVWW.
5. OpenCV Documentation: https://docs.opencv.org
6. face_recognition Library: https://github.com/ageitgey/face_recognition