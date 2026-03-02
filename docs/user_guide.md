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
- Scene Narration using offline **Text-to-Speech**
- *(Optional)* LLM-Enhanced Summaries using a local Ollama model

### What You Should See When It Works

When running properly:
- A video window opens
- Objects are detected with labeled bounding boxes
- Hand Landmarks appear when hands are visible
- The system speaks short scene descriptions (e.g, cardinal direction + relative distance)
- If  enabled, an LLM-generated summary

---

## 3. Installation and Setup

### Step 1) Prerequisites
- Python **3.9+** recommended
- A webcam (optional if you run on a video file)
- macOS, Windows, and Linux are supported
- (Optional) Ollama installed locally if testing LLM features

To verify Python versions:
```bash
python3 --version     # macOS/Linux   
python --version      # Windows 
```

### Step 2) Create and Activate a Virtual Environment (Recommended)
```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell
```
You should now see ```(.venv)``` in your terminal prompt

---
