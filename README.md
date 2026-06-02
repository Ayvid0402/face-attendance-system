# Real-Time Face Recognition Attendance System

A Computer Vision system that detects and recognizes faces from webcam input and marks attendance automatically.

## 🗺️ System Architecture
Camera Input (Webcam / IP Webcam)
↓
Face Detection (OpenCV + dlib)
↓
Face Recognition (face_recognition library)
↓
Attendance Logging (CSV file)
↓
Dashboard (Streamlit)

## 📁 Project Structure
face_attendance_system/
│
├── known_faces/          ← photos of registered people
├── unknown_faces/        ← snapshots of unrecognized faces
├── attendance/
│   └── attendance.csv   ← attendance records
│
├── src/
│   ├── encode_faces.py      ← Step 1: encode known faces
│   ├── face_recognizer.py   ← Step 2: recognize faces
│   ├── attendance_logger.py ← Step 3: log attendance
│   ├── webcam_attendance.py ← Step 4: real time system
│   ├── dashboard.py         ← Step 5: streamlit dashboard
│   └── encodings.pkl        ← saved face encodings
│
└── README.md

## 🔧 Setup

### 1. Create virtual environment
```bash
conda create -n face_attendance python=3.9
conda activate face_attendance
```

### 2. Install libraries
```bash
conda install -c conda-forge dlib
pip install face_recognition opencv-python numpy pandas pillow streamlit plotly
```

### 3. Add known faces
- Add photos to `known_faces/` folder
- Name them: `personname_1.jpg`, `personname_2.jpg`

### 4. Encode faces
```bash
python src/encode_faces.py
```

### 5. Run attendance system
```bash
python src/webcam_attendance.py
```

### 6. Run dashboard
```bash
streamlit run src/dashboard.py
```

## 🧠 How It Works

### Face Detection
OpenCV reads frames from webcam. Each frame is resized to 25% for speed, then face locations are detected using dlib's HOG-based detector.

### Face Recognition
Each detected face is converted to a **128-dimensional embedding vector** using dlib's deep learning model. This vector is compared against all known face embeddings using Euclidean distance.

### Similarity Matching
Distance < 0.5  → Same person (known) ✅
Distance > 0.5  → Different person (unknown) ❌

### Attendance Logging
When a known face is recognized:
- Name, date, time saved to CSV
- Duplicate prevention — same person marked only once per day

### Unknown Face Handling
If face distance > threshold:
- Marked as "Unknown"
- Face snapshot saved to `unknown_faces/` folder

## 📊 Preprocessing

| Step | Why |
|------|-----|
| Resize to 25% | Faster processing — face_recognition is slow on full frames |
| BGR to RGB | OpenCV uses BGR, face_recognition needs RGB |
| np.ascontiguousarray | Ensures correct memory layout for dlib |

## 🔬 Model Comparison

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| Haar Cascade | Very Fast | Low | Simple detection |
| HOG + dlib | Fast | Medium | Our system ✅ |
| CNN (dlib) | Slow | High | GPU systems |
| MTCNN | Medium | High | Production |

## 📈 Performance

| Metric | Value |
|--------|-------|
| Recognition Confidence | 83.1% |
| FPS | 10+ FPS |
| False Positives | Low (tolerance=0.5) |

## 🎯 Features

- ✅ Real-time face detection and recognition
- ✅ Automatic attendance marking
- ✅ Duplicate prevention
- ✅ Unknown face handling + snapshot
- ✅ Multi-face recognition
- ✅ Streamlit dashboard with charts
- ✅ CSV attendance export

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| opencv-python | Video capture, image processing |
| face_recognition | Face encoding and matching |
| dlib | Deep learning face model |
| numpy | Array operations |
| pandas | CSV handling |
| streamlit | Web dashboard |
| plotly | Interactive charts |# Real-Time Face Recognition Attendance System

A Computer Vision system that detects and recognizes faces from webcam input and marks attendance automatically.

## 🗺️ System Architecture
Camera Input (Webcam / IP Webcam)
↓
Face Detection (OpenCV + dlib)
↓
Face Recognition (face_recognition library)
↓
Attendance Logging (CSV file)
↓
Dashboard (Streamlit)

## 📁 Project Structure
face_attendance_system/
│
├── known_faces/          ← photos of registered people
├── unknown_faces/        ← snapshots of unrecognized faces
├── attendance/
│   └── attendance.csv   ← attendance records
│
├── src/
│   ├── encode_faces.py      ← Step 1: encode known faces
│   ├── face_recognizer.py   ← Step 2: recognize faces
│   ├── attendance_logger.py ← Step 3: log attendance
│   ├── webcam_attendance.py ← Step 4: real time system
│   ├── dashboard.py         ← Step 5: streamlit dashboard
│   └── encodings.pkl        ← saved face encodings
│
└── README.md

## 🔧 Setup

### 1. Create virtual environment
```bash
conda create -n face_attendance python=3.9
conda activate face_attendance
```

### 2. Install libraries
```bash
conda install -c conda-forge dlib
pip install face_recognition opencv-python numpy pandas pillow streamlit plotly
```

### 3. Add known faces
- Add photos to `known_faces/` folder
- Name them: `personname_1.jpg`, `personname_2.jpg`

### 4. Encode faces
```bash
python src/encode_faces.py
```

### 5. Run attendance system
```bash
python src/webcam_attendance.py
```

### 6. Run dashboard
```bash
streamlit run src/dashboard.py
```

## 🧠 How It Works

### Face Detection
OpenCV reads frames from webcam. Each frame is resized to 25% for speed, then face locations are detected using dlib's HOG-based detector.

### Face Recognition
Each detected face is converted to a **128-dimensional embedding vector** using dlib's deep learning model. This vector is compared against all known face embeddings using Euclidean distance.

### Similarity Matching
Distance < 0.5  → Same person (known) ✅
Distance > 0.5  → Different person (unknown) ❌

### Attendance Logging
When a known face is recognized:
- Name, date, time saved to CSV
- Duplicate prevention — same person marked only once per day

### Unknown Face Handling
If face distance > threshold:
- Marked as "Unknown"
- Face snapshot saved to `unknown_faces/` folder

## 📊 Preprocessing

| Step | Why |
|------|-----|
| Resize to 25% | Faster processing — face_recognition is slow on full frames |
| BGR to RGB | OpenCV uses BGR, face_recognition needs RGB |
| np.ascontiguousarray | Ensures correct memory layout for dlib |

## 🔬 Model Comparison

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| Haar Cascade | Very Fast | Low | Simple detection |
| HOG + dlib | Fast | Medium | Our system ✅ |
| CNN (dlib) | Slow | High | GPU systems |
| MTCNN | Medium | High | Production |

## 📈 Performance

| Metric | Value |
|--------|-------|
| Recognition Confidence | 83.1% |
| FPS | 10+ FPS |
| False Positives | Low (tolerance=0.5) |

## 🎯 Features

- ✅ Real-time face detection and recognition
- ✅ Automatic attendance marking
- ✅ Duplicate prevention
- ✅ Unknown face handling + snapshot
- ✅ Multi-face recognition
- ✅ Streamlit dashboard with charts
- ✅ CSV attendance export

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| opencv-python | Video capture, image processing |
| face_recognition | Face encoding and matching |
| dlib | Deep learning face model |
| numpy | Array operations |
| pandas | CSV handling |
| streamlit | Web dashboard |
| plotly | Interactive charts |# Real-Time Face Recognition Attendance System

A Computer Vision system that detects and recognizes faces from webcam input and marks attendance automatically.

## 🗺️ System Architecture
Camera Input (Webcam / IP Webcam)
↓
Face Detection (OpenCV + dlib)
↓
Face Recognition (face_recognition library)
↓
Attendance Logging (CSV file)
↓
Dashboard (Streamlit)

## 📁 Project Structure
face_attendance_system/
│
├── known_faces/          ← photos of registered people
├── unknown_faces/        ← snapshots of unrecognized faces
├── attendance/
│   └── attendance.csv   ← attendance records
│
├── src/
│   ├── encode_faces.py      ← Step 1: encode known faces
│   ├── face_recognizer.py   ← Step 2: recognize faces
│   ├── attendance_logger.py ← Step 3: log attendance
│   ├── webcam_attendance.py ← Step 4: real time system
│   ├── dashboard.py         ← Step 5: streamlit dashboard
│   └── encodings.pkl        ← saved face encodings
│
└── README.md

## 🔧 Setup

### 1. Create virtual environment
```bash
conda create -n face_attendance python=3.9
conda activate face_attendance
```

### 2. Install libraries
```bash
conda install -c conda-forge dlib
pip install face_recognition opencv-python numpy pandas pillow streamlit plotly
```

### 3. Add known faces
- Add photos to `known_faces/` folder
- Name them: `personname_1.jpg`, `personname_2.jpg`

### 4. Encode faces
```bash
python src/encode_faces.py
```

### 5. Run attendance system
```bash
python src/webcam_attendance.py
```

### 6. Run dashboard
```bash
streamlit run src/dashboard.py
```

## 🧠 How It Works

### Face Detection
OpenCV reads frames from webcam. Each frame is resized to 25% for speed, then face locations are detected using dlib's HOG-based detector.

### Face Recognition
Each detected face is converted to a **128-dimensional embedding vector** using dlib's deep learning model. This vector is compared against all known face embeddings using Euclidean distance.

### Similarity Matching
Distance < 0.5  → Same person (known) ✅
Distance > 0.5  → Different person (unknown) ❌

### Attendance Logging
When a known face is recognized:
- Name, date, time saved to CSV
- Duplicate prevention — same person marked only once per day

### Unknown Face Handling
If face distance > threshold:
- Marked as "Unknown"
- Face snapshot saved to `unknown_faces/` folder

## 📊 Preprocessing

| Step | Why |
|------|-----|
| Resize to 25% | Faster processing — face_recognition is slow on full frames |
| BGR to RGB | OpenCV uses BGR, face_recognition needs RGB |
| np.ascontiguousarray | Ensures correct memory layout for dlib |

## 🔬 Model Comparison

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| Haar Cascade | Very Fast | Low | Simple detection |
| HOG + dlib | Fast | Medium | Our system ✅ |
| CNN (dlib) | Slow | High | GPU systems |
| MTCNN | Medium | High | Production |

## 📈 Performance

| Metric | Value |
|--------|-------|
| Recognition Confidence | 83.1% |
| FPS | 10+ FPS |
| False Positives | Low (tolerance=0.5) |

## 🎯 Features

- ✅ Real-time face detection and recognition
- ✅ Automatic attendance marking
- ✅ Duplicate prevention
- ✅ Unknown face handling + snapshot
- ✅ Multi-face recognition
- ✅ Streamlit dashboard with charts
- ✅ CSV attendance export

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| opencv-python | Video capture, image processing |
| face_recognition | Face encoding and matching |
| dlib | Deep learning face model |
| numpy | Array operations |
| pandas | CSV handling |
| streamlit | Web dashboard |
| plotly | Interactive charts |# Real-Time Face Recognition Attendance System

A Computer Vision system that detects and recognizes faces from webcam input and marks attendance automatically.

## 🗺️ System Architecture
Camera Input (Webcam / IP Webcam)
↓
Face Detection (OpenCV + dlib)
↓
Face Recognition (face_recognition library)
↓
Attendance Logging (CSV file)
↓
Dashboard (Streamlit)

## 📁 Project Structure
face_attendance_system/
│
├── known_faces/          ← photos of registered people
├── unknown_faces/        ← snapshots of unrecognized faces
├── attendance/
│   └── attendance.csv   ← attendance records
│
├── src/
│   ├── encode_faces.py      ← Step 1: encode known faces
│   ├── face_recognizer.py   ← Step 2: recognize faces
│   ├── attendance_logger.py ← Step 3: log attendance
│   ├── webcam_attendance.py ← Step 4: real time system
│   ├── dashboard.py         ← Step 5: streamlit dashboard
│   └── encodings.pkl        ← saved face encodings
│
└── README.md

## 🔧 Setup

### 1. Create virtual environment
```bash
conda create -n face_attendance python=3.9
conda activate face_attendance
```

### 2. Install libraries
```bash
conda install -c conda-forge dlib
pip install face_recognition opencv-python numpy pandas pillow streamlit plotly
```

### 3. Add known faces
- Add photos to `known_faces/` folder
- Name them: `personname_1.jpg`, `personname_2.jpg`

### 4. Encode faces
```bash
python src/encode_faces.py
```

### 5. Run attendance system
```bash
python src/webcam_attendance.py
```

### 6. Run dashboard
```bash
streamlit run src/dashboard.py
```

## 🧠 How It Works

### Face Detection
OpenCV reads frames from webcam. Each frame is resized to 25% for speed, then face locations are detected using dlib's HOG-based detector.

### Face Recognition
Each detected face is converted to a **128-dimensional embedding vector** using dlib's deep learning model. This vector is compared against all known face embeddings using Euclidean distance.

### Similarity Matching
Distance < 0.5  → Same person (known) ✅
Distance > 0.5  → Different person (unknown) ❌

### Attendance Logging
When a known face is recognized:
- Name, date, time saved to CSV
- Duplicate prevention — same person marked only once per day

### Unknown Face Handling
If face distance > threshold:
- Marked as "Unknown"
- Face snapshot saved to `unknown_faces/` folder

## 📊 Preprocessing

| Step | Why |
|------|-----|
| Resize to 25% | Faster processing — face_recognition is slow on full frames |
| BGR to RGB | OpenCV uses BGR, face_recognition needs RGB |
| np.ascontiguousarray | Ensures correct memory layout for dlib |

## 🔬 Model Comparison

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| Haar Cascade | Very Fast | Low | Simple detection |
| HOG + dlib | Fast | Medium | Our system ✅ |
| CNN (dlib) | Slow | High | GPU systems |
| MTCNN | Medium | High | Production |

## 📈 Performance

| Metric | Value |
|--------|-------|
| Recognition Confidence | 83.1% |
| FPS | 10+ FPS |
| False Positives | Low (tolerance=0.5) |

## 🎯 Features

- ✅ Real-time face detection and recognition
- ✅ Automatic attendance marking
- ✅ Duplicate prevention
- ✅ Unknown face handling + snapshot
- ✅ Multi-face recognition
- ✅ Streamlit dashboard with charts
- ✅ CSV attendance export

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| opencv-python | Video capture, image processing |
| face_recognition | Face encoding and matching |
| dlib | Deep learning face model |
| numpy | Array operations |
| pandas | CSV handling |
| streamlit | Web dashboard |
| plotly | Interactive charts |# Real-Time Face Recognition Attendance System

A Computer Vision system that detects and recognizes faces from webcam input and marks attendance automatically.

## 🗺️ System Architecture
Camera Input (Webcam / IP Webcam)
↓
Face Detection (OpenCV + dlib)
↓
Face Recognition (face_recognition library)
↓
Attendance Logging (CSV file)
↓
Dashboard (Streamlit)

## 📁 Project Structure
face_attendance_system/
│
├── known_faces/          ← photos of registered people
├── unknown_faces/        ← snapshots of unrecognized faces
├── attendance/
│   └── attendance.csv   ← attendance records
│
├── src/
│   ├── encode_faces.py      ← Step 1: encode known faces
│   ├── face_recognizer.py   ← Step 2: recognize faces
│   ├── attendance_logger.py ← Step 3: log attendance
│   ├── webcam_attendance.py ← Step 4: real time system
│   ├── dashboard.py         ← Step 5: streamlit dashboard
│   └── encodings.pkl        ← saved face encodings
│
└── README.md

## 🔧 Setup

### 1. Create virtual environment
```bash
conda create -n face_attendance python=3.9
conda activate face_attendance
```

### 2. Install libraries
```bash
conda install -c conda-forge dlib
pip install face_recognition opencv-python numpy pandas pillow streamlit plotly
```

### 3. Add known faces
- Add photos to `known_faces/` folder
- Name them: `personname_1.jpg`, `personname_2.jpg`

### 4. Encode faces
```bash
python src/encode_faces.py
```

### 5. Run attendance system
```bash
python src/webcam_attendance.py
```

### 6. Run dashboard
```bash
streamlit run src/dashboard.py
```

## 🧠 How It Works

### Face Detection
OpenCV reads frames from webcam. Each frame is resized to 25% for speed, then face locations are detected using dlib's HOG-based detector.

### Face Recognition
Each detected face is converted to a **128-dimensional embedding vector** using dlib's deep learning model. This vector is compared against all known face embeddings using Euclidean distance.

### Similarity Matching
Distance < 0.5  → Same person (known) ✅
Distance > 0.5  → Different person (unknown) ❌

### Attendance Logging
When a known face is recognized:
- Name, date, time saved to CSV
- Duplicate prevention — same person marked only once per day

### Unknown Face Handling
If face distance > threshold:
- Marked as "Unknown"
- Face snapshot saved to `unknown_faces/` folder

## 📊 Preprocessing

| Step | Why |
|------|-----|
| Resize to 25% | Faster processing — face_recognition is slow on full frames |
| BGR to RGB | OpenCV uses BGR, face_recognition needs RGB |
| np.ascontiguousarray | Ensures correct memory layout for dlib |

## 🔬 Model Comparison

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| Haar Cascade | Very Fast | Low | Simple detection |
| HOG + dlib | Fast | Medium | Our system ✅ |
| CNN (dlib) | Slow | High | GPU systems |
| MTCNN | Medium | High | Production |

## 📈 Performance

| Metric | Value |
|--------|-------|
| Recognition Confidence | 83.1% |
| FPS | 10+ FPS |
| False Positives | Low (tolerance=0.5) |

## 🎯 Features

- ✅ Real-time face detection and recognition
- ✅ Automatic attendance marking
- ✅ Duplicate prevention
- ✅ Unknown face handling + snapshot
- ✅ Multi-face recognition
- ✅ Streamlit dashboard with charts
- ✅ CSV attendance export

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| opencv-python | Video capture, image processing |
| face_recognition | Face encoding and matching |
| dlib | Deep learning face model |
| numpy | Array operations |
| pandas | CSV handling |
| streamlit | Web dashboard |
| plotly | Interactive charts |