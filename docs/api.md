# docs/api.md API Documentation

## Component: `sonr.py` Command-Line Interface (CLI)

### Name
**Sonr — YOLO11 Talking Objects + MediaPipe Hands + Depth Anything v2 + OCR**

### Short Description
Real-time accessibility vision pipeline. Runs **Ultralytics YOLO11** with **ByteTrack** for object detection + persistent IDs, overlays **MediaPipe** hand landmarks, estimates monocular metric depth via **Depth Anything v2** on a background thread, and speaks short scene descriptions via the macOS `say` command. On demand (key `r`), reads arbitrary text in the frame via EasyOCR.

### Signature

```
python sonr.py [--model MODEL] [--source SOURCE] [--conf CONF] [--iou IOU] [--device DEVICE]
               [--mirror | --no-mirror]
               [--depth | --no-depth] [--metric-deadband M]
               [--ocr | --no-ocr]
               [--speak-interval SECONDS] [--per-item-cooldown SECONDS]
               [--min-speak-hnorm HNORM]
               [--min-stable N] [--hold-seconds SECONDS]
               [--zone-deadband X] [--dist-deadband X]
               [--no-show] [--no-tty-quit] [--watchdog SECONDS]
               [--debug] [--log-every SECONDS] [--quiet-cooldowns]
```

### Parameters

**Input / detection**

- `--model` (str, default `yolo11n.pt`): YOLO11 model path or name.
- `--source` (str, default `0`): `0` for the default webcam, an integer index for another, or a video file path.
- `--conf` (float, default `0.75`): YOLO confidence threshold.
- `--iou` (float, default `0.5`): NMS IoU threshold.
- `--device` (str | None): device hint (`cpu`, `cuda`, `mps`). If unset, ultralytics chooses; depth estimator auto-selects MPS on Apple Silicon.
- `--mirror` / `--no-mirror` (default on): horizontally flip the frame. On for selfie-view webcams (so hand handedness and left/right narration match the user). Off for video files or forward-facing cameras.

**Distance (Depth Anything v2)**

- `--depth` / `--no-depth` (default on): use Depth Anything v2 metric-indoor for distance estimates. With `--no-depth`, falls back to a bbox-height heuristic.
- `--metric-deadband` (float, default `0.3`): hysteresis around metric distance bucket boundaries (meters). Prevents flapping between e.g. "close to you" and "nearby" when an object sits near a boundary.

Distance buckets in meters:
- `≤ 0.7 m` → "very close to you"
- `≤ 1.5 m` → "close to you"
- `≤ 3.0 m` → "nearby"
- `≤ 6.0 m` → "in the distance"
- `> 6.0 m` → "far away"

**OCR**

- `--ocr` / `--no-ocr` (default on): enable on-demand text reading. Press `r` during the run to OCR the current frame and have the text spoken. Submitted on the unflipped raw frame so mirrored text (when `--mirror` is on) reads correctly.

**Narration / TTS**

- `--speak-interval` (float, default `2.0`): minimum seconds between spoken utterances (anti-spam floor).
- `--per-item-cooldown` (float, default `8.0`): seconds before the same `track_id` (or `(label, zone)` for untracked detections) can be re-announced. Different items can speak in between, so the system stays responsive without nagging about the same chair every 2 seconds.
- `--min-speak-hnorm` (float, default `0.05`): ignore detections smaller than this normalized bbox height.

**Stability / anti-flicker**

- `--min-stable` (int, default `3`): frames an object must persist before becoming "stable".
- `--hold-seconds` (float, default `0.75`): keep recently-seen objects alive briefly across frame drops.
- `--zone-deadband` (float, default `0.06`): hysteresis around left/center/right zone boundaries.
- `--dist-deadband` (float, default `0.04`): hysteresis around bbox-height distance boundaries (only used when `--no-depth`).

**Display / runtime**

- `--no-show` (flag): disable the OpenCV preview window.
- `--debug` (flag): verbose logs (FPS, detections, TTS events, OCR events).
- `--log-every` (float, default `1.0`): heartbeat log period (seconds).
- `--quiet-cooldowns` (flag): suppress cooldown-related debug spam.
- `--no-tty-quit` (flag): ignore the `q` key (ESC still quits).
- `--watchdog` (float, default `0.0`): exit if no frames arrive for this many seconds. `0` disables.

### Interactive keys

- `v` — toggle voice on/off
- `b` — toggle bounding-box drawing
- `h` — toggle hand-landmark drawing/processing
- `r` — read text in the current frame (OCR). Speaks "No text found" if nothing detected within 5 s.
- `q` or `ESC` — quit

### TTS behavior

A new phrase drains any pending queued phrases and is enqueued. The currently-speaking utterance is allowed to finish naturally (interrupting mid-word felt jarring during testing, and with phrases capped to ~2 seconds the worst-case lag is small enough to live with).

Phrase length is capped (top 3 items by importance, max 2 per zone) to keep individual utterances short. Pluralization is applied (`person → people`, `cup → cups`, etc.).

### Threading model

- **Detection** runs synchronously per frame in the main loop (~50 ms YOLO).
- **Hand landmarks** run synchronously per frame (~10–20 ms MediaPipe).
- **Depth estimation** runs on a background thread (`DepthWorker`). Main loop submits the latest frame and reads whatever depth map is currently ready (~125 ms behind on M-series MPS). Until the first inference completes, the bbox-height heuristic is used as a transient fallback.
- **Text OCR** runs on a background thread (`OcrWorker`). User-triggered via `r`. Receives the unflipped raw frame so mirrored text reads correctly.
- **TTS** runs on a background thread (`TTSWorker`).

Net effect: main capture/detection loop runs at full YOLO speed; all heavier ML inference is decoupled.

### Errors

- Dependency missing → import error and exit with install hint.
- Cannot open video source (bad path, webcam in use, permissions blocked) → early exit with an error.
- No frames received (camera freeze / bad stream) → may retry; may exit if `--watchdog` is set.
- HuggingFace download fails → depth or OCR may not initialize; the rest of the pipeline still runs.

### Examples

Default webcam demo:

```
python sonr.py --source 0
```

Lower confidence to catch held objects:

```
python sonr.py --source 0 --conf 0.5
```

Video file, disable depth (faster), disable mirror (text/plates in scene aren't reversed):

```
python sonr.py --source clip.mp4 --no-mirror --no-depth
```

### Notes

- macOS only: TTS uses the built-in `say` command.
- Performance on M-series Mac: ~15 fps preview with depth + plates enabled.
- The metric-indoor depth model saturates outdoors (~6-10 m); use `--no-depth` if you'll be testing outside.
- MediaPipe hand landmarks are pinned to `mediapipe==0.10.14` because newer versions removed the legacy `solutions` API. See `requirements.txt`.

### Module layout

The CLI script `sonr.py` orchestrates the following modules:

- `detection.py` — YOLO11 wrapper with ByteTrack (`Detector`, `Detection` with `track_id`).
- `geometry.py` — left/center/right zone helpers with hysteresis.
- `distance.py` — distance bucketing (bbox-height fallback + metric depth).
- `depth.py` — Depth Anything v2 wrapper (`DepthEstimator`) and threaded worker (`DepthWorker`).
- `tracking.py` — `StableTracker` for frame-to-frame deduplication.
- `narration.py` — phrase composition with pluralization (`SpeakItem`, `compose_phrase`).
- `speech.py` — background-thread macOS `say` worker (`TTSWorker`).
- `hands.py` — MediaPipe hand-landmark overlay (`HandTracker`).
- `ocr.py` — EasyOCR-based text reader (`OcrWorker`).
