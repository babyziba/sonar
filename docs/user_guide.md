# User Guide YOLO11 Talking Objects + MediaPipe Hands

## Who this guide is for
This guide is for a **TA, instructor, or any new user** who wants to run the project for the first time and verify it works (camera/video → detections → overlays → spoken narration).

---

## System Overview
This project is a real-time computer vision demo that:

- Detects objects using **Ultralytics YOLO11**
- Displays results in an **OpenCV window** (bounding boxes + labels)
- Draws **MediaPipe hand landmarks** on top of the same video feed
- Speaks a short scene description (e.g., direction + approximate distance) using offline TTS
- (Optional) Uses a **local Ollama LLM** to generate a nicer summary and display/speak it

---

## Installation and Setup

### 1) Prerequisites
- Python **3.9+** recommended
- A webcam (optional if you run on a video file)
- macOS / Windows / Linux supported

### 2) Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell
