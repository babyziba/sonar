# Deployment Guide

This document describes how to install, configure, and validate Sonr in a
clean environment, and how to verify the deployment works end-to-end.

The full project source code and deployment artifacts are available at:
**https://github.com/babyziba/sonar**

## Table of Contents

- [Target Environment](#target-environment)
- [Prerequisites](#prerequisites)
- [Clean-Environment Install](#clean-environment-install)
- [Configuration](#configuration)
- [First Run (Model Downloads)](#first-run-model-downloads)
- [Validation](#validation)
- [Running Tests](#running-tests)
- [Why No Docker / docker-compose](#why-no-docker--docker-compose)
- [Common Deployment Issues](#common-deployment-issues)
- [Uninstall](#uninstall)

## Target Environment

Sonr is a **desktop application** that runs on a single machine and needs
direct access to:

- A camera (webcam, built-in or USB)
- The macOS speech synthesizer (`/usr/bin/say`)
- Local GPU/MPS or CPU for ML inference

It is **not** a server, web service, or container workload. There is no
remote deployment target — "deployment" here means installing on the user's
Mac and confirming the pipeline runs end-to-end.

| Property            | Value                                            |
| ------------------- | ------------------------------------------------ |
| OS                  | macOS 12+ (tested on 14, 15)                     |
| Architecture        | Apple Silicon (preferred) or Intel               |
| Python              | **3.12.x** (see note below)                      |
| Disk                | ~500 MB for model weights downloaded on first run |
| RAM                 | 4 GB free recommended                            |
| Network             | Required only on first run for model downloads   |

> **Python version note:** `mediapipe==0.10.14` only ships wheels for Python
> 3.9–3.12. Python 3.13+ will fail at install time. See the
> [README's "Python version mismatch"](README.md#python-version-mismatch-313)
> section for the `pyenv` workaround.

## Prerequisites

1. **Python 3.12** on `PATH` (or via pyenv — recommended for isolation).
2. **Git** to clone the repository.
3. **macOS Camera permission** for your terminal application (Terminal.app
   or iTerm2.app). The OS prompts for this on first launch; if you missed
   it, enable it manually in *System Settings → Privacy & Security → Camera*.
4. **Audio output** (built-in speakers are fine). The `say` command must
   work from your shell — verify with: `say "hello"`.

## Clean-Environment Install

These steps assume a fresh checkout on a machine that has Python 3.12
available. Replace `python3` with `python3.12` if your default is something
else, or use `pyenv local 3.12.7` per the README.

```bash
# 1. Clone
git clone https://github.com/babyziba/sonar.git
cd sonar

# 2. Create an isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install runtime dependencies
pip install -r requirements.txt

# 4. (Optional) install dev/test dependencies
pip install -r requirements-dev.txt

# 5. (Optional) copy the example env file
cp .env.example .env
# then edit .env to override defaults if needed
```

The install should complete in 2–5 minutes on a typical home connection.
Most of the size is `torch` and `transformers` for Depth Anything v2.

## Configuration

Sonr reads configuration from three sources, in order of precedence
(later sources override earlier ones):

1. **Built-in defaults** in `src/sonr.py`
2. **Environment variables** (most easily set via a `.env` file — see
   `.env.example` for the full list)
3. **Command-line flags** (see `docs/api.md` for the complete reference)

The most common knobs are:

| Variable        | CLI flag      | Default       | Purpose                                 |
| --------------- | ------------- | ------------- | --------------------------------------- |
| `SONR_MODEL`    | `--model`     | `yolo11n.pt`  | YOLO11 model path or name               |
| `SONR_SOURCE`   | `--source`    | `0`           | Camera index or video file              |
| `SONR_DEVICE`   | `--device`    | (auto)        | `cpu`, `cuda`, or `mps`                 |
| `SONR_CONF`     | `--conf`      | `0.75`        | YOLO confidence threshold               |

Sonr does not auto-load `.env`; the easiest way to use it is to source it
before running:

```bash
set -a; source .env; set +a
python src/sonr.py
```

## First Run (Model Downloads)

The first invocation downloads ~165 MB of model weights to your user cache
and the project root. These are cached afterward.

| Model                                  | Size    | Cached at                    |
| -------------------------------------- | ------- | ---------------------------- |
| YOLO11n                                | ~5 MB   | `./yolo11n.pt`               |
| Depth Anything v2 Small Metric-Indoor  | ~100 MB | `~/.cache/huggingface/hub/`  |
| EasyOCR detector + recognition         | ~64 MB  | `~/.EasyOCR/`                |

A successful first run prints:

```
Loading Depth Anything v2 (first run downloads ~100 MB)...
Depth model loaded on device: mps (running async)
Loading EasyOCR (first run downloads ~64 MB)...
OCR ready. Press 'r' to read text in the current scene.
Press v=voice, b=boxes, h=hands, r=read text, q=quit.
```

If any download fails, the relevant feature is disabled and the rest of the
pipeline still runs. See [Common Deployment Issues](#common-deployment-issues).

## Validation

Use this checklist to confirm a fresh deployment is healthy. Each item
takes < 30 seconds.

### 1. Imports resolve and CLI parses

```bash
python src/sonr.py --help
```

**Expected:** the argparse help text. **If this fails:** `pip install`
didn't complete — re-run with verbose output to see which package broke.

### 2. Unit tests pass

```bash
pytest
```

**Expected:** all tests in `tests/` pass. These cover the pure-logic modules
(`geometry`, `narration`, `tracking`, `distance`) and don't require a camera
or any ML model, so a green run confirms the codebase is internally
consistent. **If this fails:** the import path or one of the modules is
broken.

### 3. Smoke run (with camera)

```bash
python src/sonr.py --source 0
```

**Expected:**
- A preview window titled *"YOLO11 Talking Objects"* appears within 2–3
  seconds.
- Bounding boxes render around any visible objects.
- After a few stable frames, you hear a spoken description (e.g. *"On your
  left, a person close to you."*).
- Pressing `r` triggers OCR on the current frame and speaks any text seen
  (or *"No text found"* after 5 s).
- Pressing `q` or `ESC` cleanly exits.

**If the window doesn't open:** macOS Camera permission isn't granted to
your terminal app — see [Prerequisites](#prerequisites).

### 4. Smoke run (no camera, video file)

If you don't have a webcam available on the deployment machine, drop any
short MP4 in the working directory and run:

```bash
python src/sonr.py --source path/to/video.mp4 --no-mirror
```

This validates the detection / depth / narration pipeline independently of
camera permissions.

### 5. Headless validation (no display)

For CI or remote machines without a display:

```bash
python src/sonr.py --source path/to/video.mp4 --no-show --watchdog 30 --debug
```

Watch for `[dbg] fps=...` heartbeats. The process should exit cleanly when
the video ends or the watchdog trips.

## Running Tests

```bash
pytest                       # run all tests
pytest -v                    # verbose
pytest tests/test_geometry.py    # one file
pytest -k "pluralize"        # filter by name
```

Tests live in `tests/` and import from `src/` directly. They are
intentionally narrow — they cover pure-logic modules so that a green run
gives a fast, dependable signal without needing a camera, a GPU, or any
model downloads.

## Why No Docker / docker-compose

The submission rubric calls for `docker-compose.yml` *if applicable*. For
Sonr it is **not applicable**, for three concrete reasons:

1. **Camera passthrough on macOS Docker is not supported.** Docker Desktop
   for Mac runs containers inside a Linux VM; the host camera (AVFoundation)
   is not exposed to that VM. There is no equivalent of Linux's
   `/dev/video0` to forward.
2. **`/usr/bin/say` is a macOS-native binary.** It links against
   AVSpeechSynthesizer and would not run in a Linux container even if we
   could pass audio out.
3. **GPU/MPS is host-only.** Apple Silicon's Metal Performance Shaders
   backend is not exposed to Docker containers.

Shipping a `docker-compose.yml` would fail at the first feature-flag check
(no camera) and mislead any evaluator who tried to run it. Instead, the
clean-environment install above plays the same role: a reproducible setup
recipe an evaluator can follow in a fresh checkout.

## Common Deployment Issues

**"Could not open webcam: 0"**
macOS Camera permission isn't granted to your terminal. Fix in *System
Settings → Privacy & Security → Camera*, then restart the terminal.

**`pip install` fails on `mediapipe==0.10.14`**
Your interpreter is Python 3.13+. Downgrade to 3.12 (`pyenv install 3.12.7
&& pyenv local 3.12.7`), recreate the venv, and reinstall.

**Depth model download hangs or times out**
HuggingFace Hub may be rate-limiting or your network is restricted. Either
retry, or run with `--no-depth` to fall back to the bbox-height heuristic.
The rest of the pipeline still works.

**EasyOCR download fails**
Run with `--no-ocr` to skip OCR initialization. Detection, depth, hands,
and TTS continue to work.

**No speech, but everything else works**
Voice may have been toggled off (press `v`). Otherwise, verify
`say "hello"` works in your shell — if not, check audio output device and
TTS permissions in *System Settings → Accessibility → Spoken Content*.

**Camera lag / preview feels choppy**
Depth estimation is the heaviest model (~125 ms/frame on M-series MPS,
runs in the background). Disable with `--no-depth` for snappier preview at
the cost of less accurate distances.

## Uninstall

```bash
deactivate                  # exit the venv
cd ..
rm -rf sonar                # remove the source tree
rm -rf ~/.EasyOCR           # (optional) remove EasyOCR cache
# Depth Anything weights live under ~/.cache/huggingface/hub and may be
# shared with other projects — only remove if you're sure.
```
