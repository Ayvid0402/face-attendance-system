# 👤 Real-Time Face Recognition Attendance System

A Computer Vision system that automatically detects and recognizes faces from webcam input and marks attendance in real-time.

Built as part of Computer Vision Internship Task.

---

## 🗺️ System Architecture
Camera Input (Webcam / IP Webcam)
↓
Preprocessing (Resize + RGB + Normalize)
↓
Face Detection (HOG + dlib)
↓
Face Recognition (128-d Embeddings)
↓
Anti-Spoofing Check (Blink Detection)
↓
Attendance Logging (CSV + SQLite)
↓
Streamlit Dashboard

---

## 📁 Project Structure
face_attendance_system/
│
├── known_faces/             ← photos of registered people
├── unknown_faces/           ← snapshots of unrecognized faces
│
├── attendance/
│   ├── attendance.csv       ← CSV attendance records
│   └── attendance.db        ← SQLite database
│
├── docs/
│   ├── model_comparison_report.md  ← detailed report
│   ├── model_comparison_report.txt ← simple report
│   ├── haar_result.jpg             ← haar detection result
│   └── hog_result.jpg              ← hog detection result
│
├── src/
│   ├── encode_faces.py          ← encode known faces
│   ├── preprocessor.py          ← image preprocessing
│   ├── face_recognizer.py       ← recognize faces
│   ├── attendance_logger.py     ← log to CSV
│   ├── database.py              ← log to SQLite
│   ├── webcam_attendance.py     ← real time webcam system
│   ├── video_attendance.py      ← video file support
│   ├── anti_spoofing.py         ← blink detection
│   ├── mask_detection.py        ← mask detection
│   ├── model_comparison.py      ← model comparison
│   ├── dashboard.py             ← streamlit dashboard
│   └── encodings.pkl            ← saved face encodings
│
├── Dockerfile                   ← Docker deployment
├── docker-compose.yml           ← Docker compose
├── requirements.txt             ← Python dependencies
└── README.md                    ← Documentation

---

## 🔧 Setup & Installation

### 1. Create virtual environment
```bash
conda create -n face_attendance python=3.9
conda activate face_attendance
```

### 2. Install dlib (Windows)
```bash
conda install -c conda-forge dlib
```

### 3. Install Python libraries
```bash
pip install face_recognition opencv-python==4.8.0.76 numpy==1.24.3
pip install pandas pillow streamlit plotly scipy
```

### 4. Add known faces
- Add photos to `known_faces/` folder
- Name them: `personname_1.jpg`, `personname_2.jpg`

### 5. Encode faces
```bash
python src/encode_faces.py
```

### 6. Run real-time attendance
```bash
python src/webcam_attendance.py
```

### 7. Run dashboard
```bash
streamlit run src/dashboard.py
```

---

## 🧠 Technical Concepts

### What are Face Embeddings?
A face embedding is a **128-dimensional vector** that represents a face mathematically. Every person has a unique embedding — like a digital fingerprint.
Face Image → dlib model → [0.12, -0.45, 0.87, ...] (128 numbers)

Two photos of the same person will have embeddings that are **close together** (small distance), while different people will have embeddings **far apart**.

### Detection vs Recognition

| Detection | Recognition |
|-----------|------------|
| WHERE is the face? | WHO is the face? |
| Returns bounding box | Returns person name |
| HOG + dlib | 128-d embeddings |
| Fast | Accurate |

### Similarity Matching
Embeddings are compared using **Euclidean distance**:
Distance < 0.5  → Same person ✅
Distance > 0.5  → Unknown person ❌

### Confidence Score
Confidence % = (1 - distance) × 100
Distance 0.17 → Confidence 83% ✅

### Confidence Thresholds
- Tolerance = 0.5 (default)
- Lower tolerance = stricter matching (fewer false positives)
- Higher tolerance = lenient matching (more false positives)
- We use 0.5 as optimal balance

---

## 📊 Preprocessing Pipeline

| Step | Method | Why Necessary |
|------|--------|--------------|
| Resize | cv2.resize(fx=0.25) | 16x faster processing |
| BGR→RGB | cv2.COLOR_BGR2RGB | face_recognition needs RGB |
| Normalize | pixels/255.0 | consistent 0-1 range |
| Contiguous array | np.ascontiguousarray | dlib C++ memory requirement |

### Why Preprocessing is Necessary:
1. **Resizing** — Full HD image has 2M pixels. Resizing to 25% gives 16x speedup for real-time performance
2. **BGR→RGB** — OpenCV loads BGR but face_recognition needs RGB. Without this, recognition fails!
3. **Normalization** — Scales pixels 0-255 to 0.0-1.0. Makes model consistent across different lighting
4. **Contiguous array** — dlib's C++ backend requires contiguous memory layout

---

## 🔬 Model Comparison

| Metric | Haar Cascade | HOG + dlib |
|--------|-------------|------------|
| Speed | 6.6 FPS ⚡ | 3.0 FPS |
| Accuracy | Lower | Higher ✅ |
| False Positives | High ❌ | Low ✅ |
| Our choice | ❌ | ✅ |

**Conclusion:** HOG + dlib chosen for better accuracy despite being slower.
See full report in `docs/model_comparison_report.md`

---

## 📈 Performance Evaluation

| Metric | Value |
|--------|-------|
| Recognition Confidence | 83.1% |
| FPS | 10+ FPS |
| False Positives | Low (tolerance=0.5) |
| False Negatives | Low (2 encodings per person) |
| Duplicate Prevention | ✅ One entry per day |
| Multi-face Support | ✅ Yes |

---

## ✅ Features Implemented

### Core Features:
- ✅ Real-time face detection and recognition
- ✅ Face detection from webcam feed
- ✅ Face detection from images
- ✅ Face detection from video files
- ✅ 128-d face embeddings
- ✅ Similarity matching with confidence score
- ✅ Attendance logging — Name, Date, Time
- ✅ CSV storage
- ✅ SQLite database storage
- ✅ Unknown face handling + snapshot saved
- ✅ Bounding boxes with name and confidence display

### Bonus Features:
- ✅ Multi-face recognition
- ✅ Streamlit dashboard with charts
- ✅ Docker deployment
- ✅ Anti-spoofing (blink detection)
- ✅ Mask detection (skin color analysis)
- ✅ Model comparison report

---

## 🗄️ Database Notes

### Why SQLite instead of PostgreSQL/MySQL:

| Database | Status | Reason |
|----------|--------|--------|
| CSV | ✅ Implemented | Simple, portable |
| SQLite | ✅ Implemented | No server needed, built-in Python |
| PostgreSQL | ⚠️ Planned | Requires server setup |
| MySQL | ⚠️ Planned | Requires server setup |

### Why we chose SQLite:
1. **No server required** — PostgreSQL and MySQL need a running server
2. **Built into Python** — no extra installation needed
3. **Perfect for single machine** — ideal for attendance system
4. **Easy migration** — can switch to PostgreSQL/MySQL by changing connection string only

### Cloud Database Attempt:
Supabase (PostgreSQL cloud) was explored but requires additional organization permissions. Can be integrated in future by updating `database.py` connection string:

```python
# Current SQLite
engine = create_engine('sqlite:///attendance.db')

# Future PostgreSQL
engine = create_engine('postgresql://user:password@host/dbname')

# Future MySQL
engine = create_engine('mysql://user:password@host/dbname')
```

---

## ⚠️ Known Issues & Solutions

### 1. Windows Camera Bug (0xA00F4292)
- **Issue:** HP Pavilion x360 camera gives PhotoCaptureStartTimeout error
- **Solution:** Used IP Webcam Android app as alternative camera source
- **Learning:** Always have a backup camera solution for Windows systems

### 2. dlib Installation on Windows
- **Issue:** pip install dlib fails — requires C++ compiler
- **Solution:** `conda install -c conda-forge dlib` installs pre-compiled version
- **Learning:** C++ libraries need conda on Windows

### 3. numpy Compatibility
- **Issue:** numpy 2.0 incompatible with dlib
- **Solution:** Downgraded to `numpy==1.24.3`
- **Learning:** Always check library compatibility before installing

### 4. Mask Detection Limitation
- **Issue:** face_recognition struggles to detect masked faces
- **Reason:** dlib model trained on unmasked faces
- **Future Fix:** Use dedicated mask detection CNN model

---

## 🚀 Future Improvements
- Cloud database (Supabase PostgreSQL)
- GPU acceleration with CUDA
- Better mask detection with dedicated CNN model
- Mobile app integration
- PostgreSQL/MySQL integration

---

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| opencv-python | Video capture, image processing |
| face_recognition | Face encoding and matching |
| dlib | Deep learning face model |
| numpy | Array operations |
| pandas | CSV handling |
| sqlite3 | Local database (built-in Python) |
| streamlit | Web dashboard |
| plotly | Interactive charts |
| scipy | Eye aspect ratio for anti-spoofing |

---

## 🐳 Docker Deployment

```bash
docker-compose up
```

Dashboard available at: http://localhost:8501

---

## 👩‍💻 Author
Divya Maria Manuel— Computer Vision Internship Project 2026