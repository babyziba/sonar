# docs/api.md — API Documentation

## Component: `app2_vlm_demo.py` Command-Line Interface (CLI)

### Name
**YOLO11 “Talking Objects” Demo + MediaPipe Hands** :contentReference[oaicite:1]{index=1}

### Short Description
Runs a real-time object detection loop using **Ultralytics YOLO11** on a webcam or video file, overlays bounding boxes, draws **MediaPipe hand landmarks**, and speaks short scene descriptions (direction + approximate distance) using offline TTS (pyttsx3 or macOS `say`). Optionally, it can request a short rephrased summary from a **local Ollama LLM** without blocking the main CV loop. :contentReference[oaicite:2]{index=2}

### Signature or endpoint
This is a CLI script, so the “signature” is its command usage:
python app2_vlm_demo.py [--model MODEL] [--source SOURCE] [--conf CONF] [--iou IOU] [--device DEVICE]
                        [--speak-interval SECONDS] [--speak-on-change] [--tts-backend {auto,pyttsx3,say}]
                        [--min-speak-hnorm HNORM] [--new-only] [--repeat-after SECONDS]
                        [--min-stable N] [--hold-seconds SECONDS] [--zone-deadband X] [--dist-deadband X]
                        [--debug] [--log-every SECONDS] [--quiet-cooldowns] [--no-tty-quit] [--watchdog SECONDS]
                        [--llm {none,ollama}] [--llm-model NAME] [--llm-url URL] [--llm-timeout SECONDS]
                        [--llm-image {person,full,none}] [--llm-jpeg-quality Q] [--llm-overlay-seconds SECONDS]
                        [--llm-auto-speak]
                        
### Parameters (name, type, and description of each)
Input / detection

--model (str): YOLO11 model path/name (e.g., yolo11n.pt)

--source (str): "0" for webcam or a video file path

--conf (float): detection confidence threshold

--iou (float): NMS IoU threshold

--device (str | None): device hint (cpu, cuda, mps)

Narration / TTS

--speak-interval (float): minimum seconds between spoken announcements

--speak-on-change (bool flag): speak immediately on phrase changes (still rate-limited)

--tts-backend (str): choose TTS backend (auto, pyttsx3, say)

--min-speak-hnorm (float): ignore tiny detections below this normalized bbox height

--new-only (bool flag): announce only newly stable objects

--repeat-after (float): allow repeats after N seconds; 0 disables repeats

Stability / anti-flicker

--min-stable (int): frames required before an object becomes “stable”

--hold-seconds (float): keep recently seen objects alive briefly

--zone-deadband (float): hysteresis near left/center/right boundaries

--dist-deadband (float): hysteresis near distance bucket boundaries

Debug / runtime controls

--debug (bool flag): verbose logging

--log-every (float): heartbeat log period (seconds)

--quiet-cooldowns (bool flag): suppress cooldown spam in logs

--no-tty-quit (bool flag): ignore q key (ESC still quits)

--watchdog (float): exit if no frames arrive for this many seconds (0 disables)

Optional local LLM (Ollama)

--llm (str): enable local LLM mode (none or ollama)

--llm-model (str): Ollama model name

--llm-url (str): Ollama base URL (default http://localhost:11434)

--llm-timeout (float): request timeout (seconds)

--llm-image (str): send person crop, full frame, or none

--llm-jpeg-quality (int): JPEG quality when sending images

--llm-overlay-seconds (float): how long overlay text stays visible

--llm-auto-speak (bool flag): auto-speak LLM summary

### Return values (type and structure):
None

### Errors or exceptions:
Dependency missing (e.g., Ultralytics not installed) → import error / friendly message and exit.
Cannot open video source (bad path, webcam in use, permissions blocked) → program exits early with an error.
No frames received (camera freeze / bad stream) → may retry; may exit if --watchdog is set.
TTS backend failure → narration may fail; fallback may be attempted (depending on OS/backend).
Ollama unreachable / timeout → LLM overlay shows an error message; core detection continues.

### One example showing how to call it:
Webcam demo with narration:
python app2_vlm_demo.py --source 0 --model yolo11n.pt --conf 0.35

### Other important info:
“Distance” is approximate and inferred from bounding box height (not true depth).
Performance depends on hardware + model size; yolo11n.pt is faster than larger models.
MediaPipe hand landmarks require good lighting and visible hands for stable tracking.
Ollama features require a local Ollama server; they are optional and not needed for the core demo.
