# User Guide: YOLO11 Talking Objects + MediaPipe Hands

## 1. Targeted Audience

This guide is intended for a **TA, instructor, or first-time users** who need to:
- Set up the project
- Run the demo successfully
- Verify that the object detection, overlays, and spoken narration are working properly

No prior familiarity with the codebase is required for this.

---

## 2. System Overview
This project is a real-time computer vision application that processes live webcam input (or a video file) and produces visual and audio feedback:

The system performs the following tasks:
- Detects objects using **Ultralytics YOLO11**
- Bounding Box Rendering in an **OpenCV** display window
- Hand Landmark Tracking using **MediaPipe**
- Scene Narration using the macOS `say` command on a background thread

### What You Should See When It Works

When running properly:
- A video window opens
- Objects are detected with labeled bounding boxes
- Hand Landmarks appear when hands are visible
- The system speaks short scene descriptions (e.g. cardinal direction + relative distance)

---

## 3. Installation and Setup

### Step 1) Prerequisites
- macOS (uses the built-in `say` command for speech)
- Python **3.9+** recommended
- A webcam (optional if you run on a video file)

To verify Python version:
```bash
python3 --version
```

### Step 2) Create and Activate a Virtual Environment (Recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
```
You should now see ```(.venv)``` in your terminal prompt.

### Step 3) Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4) Run
```bash
python sonr.py --source 0
```
Press `v` to toggle voice, `b` to toggle bounding boxes, `q` or `ESC` to quit. See `docs/api.md` for the full CLI reference.

---
