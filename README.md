# Sonr

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running](#running)
- [Python version mismatch (3.13+)](#python-version-mismatch-313)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)

## Overview

Sonr is an accessibility-focused app designed to empower visually impaired users to navigate the real world with greater confidence. Using advanced image recognition technology, Sonr interprets the user's surroundings in real time and translates visual information into clear, spoken feedback delivered through discreet speakers. Whether identifying objects, reading signs, or describing the environment, Sonr acts as an intuitive guide, helping users stay informed and independent as they move through daily life.

## Features

- Real-time object detection (Ultralytics YOLO11) with persistent IDs (ByteTrack), so the same person doesn't get re-announced every time they shift.
- Spoken scene narration: direction (left / ahead / right) + distance, composed into natural English sentences.
- Metric distance estimation via Depth Anything v2, running on a background thread so it doesn't slow the camera preview.
- MediaPipe hand-landmark overlay with correct left/right handedness (mirror-flipped frame).
- On-demand text reading (press `r`): runs EasyOCR on the current frame and speaks any text found.

## Requirements

- macOS (TTS uses the built-in `say` command).
- **Python 3.12** specifically — `mediapipe` is pinned to `0.10.14`, which only ships wheels for Python 3.9-3.12. See [Python version mismatch](#python-version-mismatch-313) below if your default `python3` is 3.13+.
- A webcam (built-in or USB).
- ~500 MB of disk for model weights downloaded on first run.

## Installation

```bash
git clone <repo-url>
cd sonar
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

That's it. The first time you run `python sonr.py`, it'll download model weights one-time:

| Model | Size | Source |
|-------|------|--------|
| YOLO11n | ~5 MB | Ultralytics auto-download |
| Depth Anything v2 Small Metric-Indoor | ~100 MB | HuggingFace |
| EasyOCR detector + recognition | ~64 MB | EasyOCR auto-download |

These are cached after the first run.

## Running

```bash
python sonr.py --source 0
```

Use `--source 1` (or higher) for a non-default webcam. Use `--source path/to/video.mp4` to run on a video file (add `--no-mirror` for video files since they're not selfie-view).

Keys during the run:

| Key | Action |
|-----|--------|
| `v` | Toggle voice on/off |
| `b` | Toggle bounding-box drawing |
| `h` | Toggle hand-landmark drawing/processing |
| `r` | Read text in the current frame (OCR) — speaks "No text found" if nothing within 5 s |
| `q` or `ESC` | Quit |

See [`docs/api.md`](docs/api.md) for the full CLI flag reference, distance-bucket thresholds, and tuning knobs.

## Python version mismatch (3.13+)

If `python3 --version` reports 3.13 or higher, `pip install -r requirements.txt` will fail when it tries to install `mediapipe==0.10.14` — there are no wheels for Python 3.13+ at that version, and we can't unpin until we migrate `hands.py` off the legacy `mp.solutions.*` API (planned, not done).

The fix is to install Python 3.12 alongside your default and pin this project to use it. The cleanest way is [pyenv](https://github.com/pyenv/pyenv):

```bash
# 1. Install pyenv (one-time, system-wide)
brew install pyenv

# Add pyenv to your shell — see https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv
# (typically: add the eval line to ~/.zshrc, then restart your terminal)

# 2. Install Python 3.12
pyenv install 3.12.7

# 3. Pin THIS project (only this directory) to 3.12.7
cd sonar
pyenv local 3.12.7

# 4. Verify the pin worked
python3 --version    # should print: Python 3.12.7

# 5. Now proceed with the standard installation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`pyenv local` writes a `.python-version` file in this directory; pyenv reads it whenever you `cd` here and picks the right interpreter automatically. Outside this directory, your default Python is unaffected. The `.python-version` file is gitignored, so each contributor pins their own.

If you'd rather not install pyenv, `brew install python@3.12` works too — then use `python3.12 -m venv .venv` instead of `python3 -m venv .venv` in step 5.

## Troubleshooting

**"Could not open webcam" on first run.** macOS gates camera access per-app. Open *System Settings → Privacy & Security → Camera* and enable access for your terminal (Terminal.app or iTerm2.app).

**Camera lag / preview feels choppy.** Depth estimation is the heaviest model (~125 ms/frame on M-series MPS, runs in the background). To disable: `python sonr.py --source 0 --no-depth`. Distances will fall back to a bbox-height heuristic (less accurate but free).

**OCR (`r`) returns mirrored or empty text.** The `r` key submits the unflipped raw frame, so this should already work even with `--mirror` on. If text still reads empty, the most common causes are: text too small in the frame, glare from a phone screen, or stylized fonts EasyOCR struggles with.

**A small object that's clearly visible doesn't get announced.** YOLO confidence threshold defaults to `0.75`. Hands holding objects often drop confidence below that. Try `--conf 0.5`.

**The same object keeps getting re-announced.** Per-`track_id` cooldown defaults to 8 s (`--per-item-cooldown 8.0`). Increase if too chatty (e.g. `--per-item-cooldown 15`).

## Architecture

The codebase is modular. Each module is small and single-purpose:

```
sonr.py        - entry point, frame loop, key handling
detection.py   - YOLO11 + ByteTrack wrapper
geometry.py    - left/center/right zone helpers + hysteresis
distance.py    - distance bucketing (bbox-height fallback + metric depth)
depth.py       - Depth Anything v2 + threaded worker
tracking.py    - StableTracker for frame-to-frame deduplication
narration.py   - phrase composition + pluralization
speech.py      - macOS `say` worker thread
hands.py       - MediaPipe hand-landmark overlay
ocr.py         - EasyOCR text reader (threaded worker)
```

See [`docs/api.md`](docs/api.md) for what each module exposes and the threading model.
