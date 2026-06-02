# Dockerfile
# Instructions to build our face attendance system container
# Think of this like a recipe — Docker follows these steps
# to create an environment where our app runs perfectly!

# ── Base image ────────────────────────────────────────────────
# Start with Python 3.9 on Ubuntu
# Like starting with a fresh computer with Python installed
FROM python:3.9-slim

# ── Set working directory ─────────────────────────────────────
# All commands run from this folder inside container
WORKDIR /app

# ── Install system dependencies ───────────────────────────────
# dlib needs cmake and build tools to compile
# These are like installing Visual Studio C++ on Windows
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Copy requirements first ───────────────────────────────────
# Copy requirements before code
# This way Docker caches this layer — faster rebuilds!
COPY requirements.txt .

# ── Install Python libraries ──────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────
# Copy everything into the container
COPY . .

# ── Create necessary folders ──────────────────────────────────
RUN mkdir -p known_faces unknown_faces attendance src docs

# ── Expose port for Streamlit ─────────────────────────────────
# Streamlit runs on port 8501
EXPOSE 8501

# ── Run command ───────────────────────────────────────────────
# Default command when container starts
CMD ["streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]