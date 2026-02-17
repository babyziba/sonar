#!/usr/bin/env python3
"""
YOLO11 "Talking Objects" Demo + MediaPipe hands
- Runs YOLO11 on a webcam or video.
- Announces objects and simple spatial relation: left/center/right + near/medium/far.
- Offline TTS via pyttsx3 (or macOS `say` fallback).

Usage:
  python app2.py --source 0           # webcam
  python app2.py --source path/to/video.mp4
  python app2.py --model yolo11n.pt   # or any YOLO11 variant
  python app2.py --conf 0.35 --iou 0.5 --speak-interval 2.0
"""

import argparse
import queue
import threading
import time
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp

# -------------------- MediaPipe Hand Detection (UNCHANGED) -------------------- #

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# Global Hands instance
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# -------------------- YOLO Import -------------------- #

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit("Ultralytics is required. Install with: pip install ultralytics") from e

# -------------------- Class Families (for grouping similar stuff) -------------------- #

_CLASS_FAMILIES = {
    # drinkware / containers
    "bottle": "container",
    "cup": "container",
    "can": "container",
    "wine glass": "container",
    "glass": "container",
    "mug": "container",
    # handheld electronics
    "cell phone": "device",
    "remote": "device",
    "laptop": "device",
    "keyboard": "device",
    "mouse": "device",
    "tv": "device",
}


def _label_family(lbl: str) -> str:
    return _CLASS_FAMILIES.get(lbl.lower(), lbl.lower())


# -------------------- TTS Worker -------------------- #

def _tts_worker(tts_queue: "queue.Queue[str]", voice_enabled_flag: list, backend_choice: str = "auto"):
    """
    Background thread that speaks phrases from a queue.
    Uses pyttsx3 if available, falls back to macOS `say` if needed.

    Fixes:
    - If pyttsx3 fails at runtime, we fall back to `say` (macOS).
    - Keeps a lock so multiple speaks never overlap.
    """
    engine = None
    use_say = False
    speak_lock = threading.Lock()

    def _init_pyttsx3():
        try:
            import pyttsx3  # type: ignore
            eng = pyttsx3.init()
            return eng
        except Exception:
            return None

    # Try pyttsx3 first
    if backend_choice in ("auto", "pyttsx3"):
        engine = _init_pyttsx3()

    # Fallback to `say` on macOS
    if backend_choice == "say" or (engine is None and backend_choice == "auto"):
        use_say = True

    while True:
        phrase = tts_queue.get()
        if phrase is None:
            break  # sentinel to stop thread

        if not voice_enabled_flag[0]:
            continue

        # Prefer pyttsx3 if working
        if engine is not None:
            try:
                with speak_lock:
                    engine.say(phrase)
                    engine.runAndWait()
                continue
            except Exception:
                # pyttsx3 borked mid-run; disable and fall back
                engine = None
                use_say = True

        if use_say:
            try:
                import subprocess
                import sys
                if sys.platform == "darwin":
                    with speak_lock:
                        subprocess.run(["say", phrase], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        # else: no TTS available → drop


# -------------------- Geometry Helpers -------------------- #

def bearing_zone(cx_norm: float) -> str:
    """Left / center / right by horizontal thirds."""
    if cx_norm < 0.33:
        return "to your left"
    elif cx_norm < 0.66:
        return "in front of you"
    else:
        return "to your right"


def distance_bucket(h_norm: float) -> str:
    """
    Crude monocular proxy using bbox height relative to frame height.
    Bigger bbox → closer.

    FIX:
    - Original thresholds made most real webcam detections fall into the far bucket,
      and the main loop *skipped* far items entirely, so it often never spoke.
    - These thresholds are more forgiving for typical webcam FOV.
    """
    if h_norm >= 0.38:
        return "right in front of you"
    elif h_norm >= 0.24:
        return "very close"
    elif h_norm >= 0.14:
        return "nearby"
    elif h_norm >= 0.07:
        return "a little further"
    else:
        return "far away"


def dbg_print(enabled: bool, *args):
    if enabled:
        print(*args)


# -------------------- Stability Tracker -------------------- #

class StableTracker:
    """
    Tracks per-(label, zone, dist) stability across frames.
    An item must appear in >= min_stable consecutive frames to be 'stable'.
    A small time-based hold avoids flicker when it temporarily disappears.
    """

    def __init__(self, min_stable_frames: int = 3, hold_seconds: float = 0.75):
        self.min_stable = min_stable_frames
        self.hold = hold_seconds
        self.counts = {}     # key -> consecutive frames present
        self.last_seen = {}  # key -> last timestamp seen

    def update_frame(self, present_keys, now):
        present_set = set(present_keys)

        # increment present
        for k in present_set:
            self.counts[k] = self.counts.get(k, 0) + 1
            self.last_seen[k] = now

        # maintain absent keys for hold interval; drop if expired
        to_delete = []
        for k, last in list(self.last_seen.items()):
            if k not in present_set and (now - last) > self.hold:
                to_delete.append(k)
        for k in to_delete:
            self.counts.pop(k, None)
            self.last_seen.pop(k, None)

    def stable_now(self):
        return {k for k, c in self.counts.items() if c >= self.min_stable}


# -------------------- Hysteresis Helpers (sticky zone / distance) -------------------- #

def _zone_with_deadband(cx_norm: float, prev_zone: str, deadband: float) -> str:
    """
    Sticky band around 0.33 and 0.66 so zones don't flap when near the boundary.
    If we're within ±deadband of a boundary and have a prev_zone, keep it.
    """
    z_basic = bearing_zone(cx_norm)
    if not prev_zone:
        return z_basic
    b1, b2 = 0.33, 0.66
    near_b1 = abs(cx_norm - b1) <= deadband
    near_b2 = abs(cx_norm - b2) <= deadband
    if z_basic != prev_zone and (near_b1 or near_b2):
        return prev_zone
    return z_basic


def _dist_with_deadband(h_norm: float, prev_dist: str, deadband: float) -> str:
    """
    Sticky band around distance boundaries so distance words don't flap.
    """
    d_basic = distance_bucket(h_norm)
    if not prev_dist:
        return d_basic

    # Boundaries aligned with distance_bucket() thresholds
    boundaries = [0.07, 0.14, 0.24, 0.38]
    near_any = any(abs(h_norm - b) <= deadband for b in boundaries)
    if d_basic != prev_dist and near_any:
        return prev_dist
    return d_basic


# -------------------- Hand Annotation (UNCHANGED) -------------------- #

def annotate_hands(frame):
    """
    Draw MediaPipe hand landmarks + simple bounding boxes onto the frame.
    """
    if frame is None or frame.size == 0:
        return frame

    # Convert BGR -> RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return frame

    h, w = frame.shape[:2]
    hand_landmarks_list = result.multi_hand_landmarks
    handedness_list = result.multi_handedness or []

    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        # Draw landmarks + connections
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_style.get_default_hand_landmarks_style(),
            mp_style.get_default_hand_connections_style(),
        )

        # Build a bounding box around the landmarks
        xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
        ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        # Optional: handedness label (Left / Right)
        label_text = ""
        if idx < len(handedness_list) and handedness_list[idx].classification:
            label_text = handedness_list[idx].classification[0].label  # "Left" / "Right"

        cv2.rectangle(frame, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (0, 255, 255), 2)
        if label_text:
            cv2.putText(
                frame,
                label_text,
                (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return frame


# Wrap cv2.imshow so we can inject hand tracking without touching main()
_original_imshow = cv2.imshow


def _imshow_with_hands(winname, frame):
    # Only annotate the main YOLO window; leave others alone
    if winname == "YOLO11 Talking Objects":
        frame = annotate_hands(frame)
    return _original_imshow(winname, frame)


cv2.imshow = _imshow_with_hands


# -------------------- Main -------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO11 model path or name")
    ap.add_argument("--source", type=str, default="0", help="Video source (0 for webcam)")
    ap.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU")
    ap.add_argument("--device", type=str, default=None, help="cuda, cpu, or mps")
    ap.add_argument("--speak-interval", type=float, default=2.0,
                    help="Seconds between announcements of the same phrase")
    ap.add_argument("--show", action="store_true", help="Show window with boxes", default=True)

    # debug flags
    ap.add_argument("--debug", action="store_true", help="Verbose logs: FPS, detections, TTS events")
    ap.add_argument("--log-every", type=float, default=1.0, help="Seconds between debug log heartbeats")
    ap.add_argument("--min-stable", type=int, default=3,
                    help="Frames an object must persist (per label+zone+dist) before speaking")
    ap.add_argument("--hold-seconds", type=float, default=0.75,
                    help="Post-detect hold to reduce flicker (seconds)")
    ap.add_argument("--no-tty-quit", action="store_true",
                    help="Ignore 'q' key; use ESC only to quit")
    ap.add_argument("--watchdog", type=float, default=0.0,
                    help="Exit if no frames arrive for this many seconds (0 to disable)")
    ap.add_argument(
        "--speak-on-change",
        action="store_true",
        help="Only announce when the scene description changes or after interval."
    )
    ap.add_argument("--quiet-cooldowns", action="store_true",
                    help="Suppress cooldown logs even with --debug")
    ap.add_argument("--label-interval", type=float, default=None,
                    help="(Reserved) Per-label min seconds between announcements.")
    ap.add_argument("--family-interval", type=float, default=None,
                    help="(Reserved) Per-family min seconds between announcements.")
    ap.add_argument("--zone-deadband", type=float, default=0.06,
                    help="Sticky band (0-1 normalized) around left/center/right boundaries.")
    ap.add_argument("--dist-deadband", type=float, default=0.04,
                    help="Sticky band around distance boundaries.")
    ap.add_argument(
        "--tts-backend",
        type=str,
        default="auto",
        choices=["auto", "pyttsx3", "say"],
        help="Force TTS backend (auto tries pyttsx3, then say)",
    )

    # FIX: avoid speaking microscopic detections that are basically noise
    ap.add_argument(
        "--min-speak-hnorm",
        type=float,
        default=0.05,
        help="Minimum bbox height (normalized by frame height) required to speak an object.",
    )

    # NEW: announce only newly detected stable objects (no repeating)
    ap.add_argument(
        "--new-only",
        action="store_true",
        help="Only announce newly detected stable objects (no repeating).",
    )
    ap.add_argument(
        "--repeat-after",
        type=float,
        default=0.0,
        help="If >0, allow re-announcing the SAME object key after N seconds (0 = never).",
    )

    args = ap.parse_args()

    # Right now label_interval / family_interval are not used in gating, but we keep them parsed.
    label_interval = args.label_interval if args.label_interval is not None else args.speak_interval
    family_interval = args.family_interval if args.family_interval is not None else label_interval
    _ = (label_interval, family_interval)  # silence lints

    # Load YOLO model
    model = YOLO(args.model)

    # Open source (UNCHANGED)
    src = 0 if args.source == "0" else args.source
    cap = None
    is_webcam = False
    try:
        cam_index = int(args.source)
        cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
        is_webcam = True
    except ValueError:
        cap = cv2.VideoCapture(src)

    if not cap or not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    # Window
    if args.show:
        cv2.namedWindow("YOLO11 Talking Objects", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO11 Talking Objects", 960, 540)

    # Warm-up: try to pull a frame (UNCHANGED)
    def warmup(capture, tries=40, pause=0.1):
        ok, frame = False, None
        for _ in range(tries):
            ok, frame = capture.read()
            if ok:
                return True, frame
            time.sleep(pause)
        return False, None

    warm_ok, _ = warmup(cap)
    if not warm_ok and is_webcam:
        try:
            alt_idx = 1 if int(args.source) == 0 else 0
            alt = cv2.VideoCapture(alt_idx, cv2.CAP_AVFOUNDATION)
            ok2, _ = warmup(alt, tries=40, pause=0.1)
            if ok2:
                cap.release()
                cap = alt
                print(f"Switched to camera index {alt_idx}")
                warm_ok = True
        except ValueError:
            pass

    # TTS thread
    tts_queue: "queue.Queue[str]" = queue.Queue()
    voice_enabled_flag = [True]
    t_thread = threading.Thread(
        target=_tts_worker,
        args=(tts_queue, voice_enabled_flag, args.tts_backend),
        daemon=True,
    )
    t_thread.start()

    # TTS state
    last_spoken = defaultdict(float)   # phrase -> time
    last_any_spoken = [0.0]            # last time anything was spoken
    last_phrase = [""]                 # last spoken full scene phrase

    # NEW: track which (label, zone, dist) we've already announced
    announced_at = {}  # (label, zone, dist) -> timestamp last announced

    # Hysteresis memory per label
    last_zone_seen = {}                # label -> last zone decided
    last_dist_seen = {}                # label -> last dist decided

    draw_boxes = True
    print("Press v to toggle voice, b to toggle boxes, q to quit.")

    # Tracker + debug state
    tracker = StableTracker(min_stable_frames=args.min_stable, hold_seconds=args.hold_seconds)
    last_log = time.time()
    frame_counter = 0
    fps_last = time.time()
    fps = 0.0
    last_frame_time = time.time()

    MAX_TTS_QUEUE = 3

    try:
        while True:
            now = time.time()

            # Read frame
            ret, frame = cap.read()
            if ret:
                last_frame_time = now
            if not ret:
                if args.debug:
                    print("[dbg] no frame from camera")

                if args.show:
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        placeholder,
                        "No frame received. Retrying…",
                        (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("YOLO11 Talking Objects", placeholder)
                    _ = cv2.waitKey(1) & 0xFF

                # quick re-open attempt for webcams (UNCHANGED)
                if is_webcam:
                    cap.release()
                    try:
                        cap = cv2.VideoCapture(int(args.source), cv2.CAP_AVFOUNDATION)
                    except ValueError:
                        cap = cv2.VideoCapture(src)

                time.sleep(0.1)

                # watchdog: bail if camera is silent too long (disabled by default)
                if args.watchdog > 0 and (now - last_frame_time) > args.watchdog:
                    print(f"[exit] watchdog: no frames for {args.watchdog}s")
                    break
                continue

            H, W = frame.shape[:2]

            # Inference
            infer_start = time.time()
            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            infer_time = time.time() - infer_start

            frame_counter += 1
            now = time.time()
            if now - fps_last >= 0.5:
                fps = frame_counter / (now - fps_last)
                frame_counter = 0
                fps_last = now

            present_keys = []
            summary = {}  # (label, zone, dist) -> {"count": int, "best_score": float, "best_h": float}
            det_total = 0

            # Parse detections
            if len(results):
                r = results[0]
                if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    clss = r.boxes.cls.cpu().numpy().astype(int)
                    names = r.names if hasattr(r, "names") else model.names

                    det_total = len(xyxy)

                    for box, conf, cls in zip(xyxy, confs, clss):
                        x1, y1, x2, y2 = box
                        w = max(1.0, x2 - x1)
                        h = max(1.0, y2 - y1)
                        cx = (x1 + x2) / 2.0
                        _cy = (y1 + y2) / 2.0  # unused but kept for future
                        cx_norm = cx / W
                        h_norm = h / H

                        label = names.get(cls, str(cls)) if isinstance(names, dict) else names[cls]

                        # Sticky zone / distance
                        prev_zone = last_zone_seen.get(label)
                        zone = _zone_with_deadband(cx_norm, prev_zone, args.zone_deadband)
                        last_zone_seen[label] = zone

                        prev_dist = last_dist_seen.get(label)
                        dist = _dist_with_deadband(h_norm, prev_dist, args.dist_deadband)
                        last_dist_seen[label] = dist

                        key = (label, zone, dist)
                        present_keys.append(key)

                        # Importance heuristic: bigger + higher conf first
                        importance = float(h_norm) * float(conf)

                        if key not in summary:
                            summary[key] = {"count": 1, "best_score": importance, "best_h": float(h_norm)}
                        else:
                            summary[key]["count"] += 1
                            if importance > summary[key]["best_score"]:
                                summary[key]["best_score"] = importance
                            if float(h_norm) > summary[key]["best_h"]:
                                summary[key]["best_h"] = float(h_norm)

                        # Draw boxes
                        if args.show and draw_boxes:
                            txt = f"{label} {conf:.2f}"
                            color = (0, 255, 0)
                            cv2.rectangle(
                                frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                color,
                                2,
                            )
                            cv2.putText(
                                frame,
                                txt,
                                (int(x1), int(y1) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2,
                                cv2.LINE_AA,
                            )

            # Update stability and decide what can be spoken
            tracker.update_frame(present_keys, now)
            stable = tracker.stable_now()

            # NEW: If an object key disappears long enough to be dropped from tracker, forget its announcement
            active_keys = set(tracker.last_seen.keys())
            for k in list(announced_at.keys()):
                if k not in active_keys:
                    announced_at.pop(k, None)

            # Debug heartbeat (FIX: don't spam every frame unless debug is on)
            if args.debug:
                if (now - last_log) >= args.log_every:
                    dbg_print(
                        True,
                        f"[dbg] fps={fps:.1f} dets={det_total} stable={len(stable)} infer={infer_time*1000:.1f}ms",
                    )
                    last_log = now

            # === Build list of stable speakable items ===
            speak_items = []
            for (label, zone, dist), info in summary.items():
                if (label, zone, dist) not in stable:
                    if args.debug and not args.quiet_cooldowns:
                        dbg_print(
                            True,
                            f"[tts-skip] not stable yet → {label} {zone}, {dist} (count={info['count']})",
                        )
                    continue

                # FIX: speak detections as long as they're not microscopic
                if info["best_h"] < args.min_speak_hnorm:
                    if args.debug and not args.quiet_cooldowns:
                        dbg_print(True, f"[tts-skip] too small → {label} (h_norm={info['best_h']:.3f})")
                    continue

                # NEW-ONLY gating: only speak when we haven't announced this key yet
                key = (label, zone, dist)
                if args.new_only:
                    last_t = announced_at.get(key)
                    if last_t is not None:
                        if args.repeat_after <= 0:
                            continue
                        if (now - last_t) < args.repeat_after:
                            continue

                count = info["count"]
                score = info["best_score"]
                family = _label_family(label)

                speak_items.append(
                    {
                        "label": label,
                        "zone": zone,
                        "dist": dist,
                        "count": count,
                        "family": family,
                        "score": score,
                    }
                )

            # If we have stable items, build a single sentence that includes everything
            if speak_items:
                # Group by zone
                zones = {}
                for it in speak_items:
                    zones.setdefault(it["zone"], []).append(it)

                # Sort each zone's items by importance
                for z in zones:
                    zones[z].sort(key=lambda it: it["score"], reverse=True)

                # Prefer front, then left/right
                zone_priority = {
                    "in front of you": 0,
                    "to your left": 1,
                    "to your right": 2,
                }
                zone_order = sorted(
                    zones.keys(),
                    key=lambda z: (zone_priority.get(z, 99), -zones[z][0]["score"]),
                )

                max_zones = 3
                selected_zones = zone_order[:max_zones]

                prefix_map = {
                    "in front of you": "Ahead of you",
                    "to your left": "On your left",
                    "to your right": "On your right",
                }

                per_zone_phrases = []

                for z in selected_zones:
                    items = zones[z]
                    prefix = prefix_map.get(z, z)

                    item_strs = []
                    for it in items:
                        label = it["label"]
                        dist = it["dist"]
                        count = it["count"]
                        if count > 1:
                            item_strs.append(f"{count} {label} {dist}")
                        else:
                            article = "an" if label[:1].lower() in "aeiou" else "a"
                            item_strs.append(f"{article} {label} {dist}")

                    if len(item_strs) == 1:
                        items_part = item_strs[0]
                    elif len(item_strs) == 2:
                        items_part = f"{item_strs[0]} and {item_strs[1]}"
                    else:
                        items_part = ", ".join(item_strs[:-1]) + f", and {item_strs[-1]}"

                    per_zone_phrases.append(f"{prefix}: {items_part}")

                final_phrase = " ".join(per_zone_phrases)

                # --------------- TTS Gating ---------------- #
                should_speak = False
                time_since_last = now - last_any_spoken[0]

                if args.speak_on_change:
                    # Speak immediately if the phrase changed, otherwise rate-limit
                    if final_phrase != last_phrase[0]:
                        should_speak = True
                        if args.debug:
                            dbg_print(True, "[tts] phrase changed → speaking immediately")
                    elif time_since_last >= args.speak_interval:
                        should_speak = True
                        if args.debug and not args.quiet_cooldowns:
                            dbg_print(True, "[tts] same phrase but interval elapsed → repeat")
                    else:
                        if args.debug and not args.quiet_cooldowns:
                            dbg_print(True, "[tts-skip] same phrase & still in cooldown")
                else:
                    if time_since_last >= args.speak_interval:
                        should_speak = True
                        if args.debug:
                            dbg_print(True, "[tts] interval elapsed → speaking")
                    else:
                        if args.debug and not args.quiet_cooldowns:
                            dbg_print(True, "[tts-skip] cooldown (interval not elapsed)")

                # NEW: if we're in new-only mode, we should not use phrase-based cooldown repeats.
                # Speak only if there is at least one newly-eligible item.
                if args.new_only:
                    # If speak_items exists, it's already filtered to NEW (or repeat_after passed).
                    # So we can just ignore phrase repetition rules and speak immediately.
                    should_speak = True

                # Manage queue: don't let it grow forever
                if should_speak and final_phrase:
                    qsize = tts_queue.qsize()
                    if qsize >= MAX_TTS_QUEUE:
                        try:
                            _ = tts_queue.get_nowait()  # drop oldest
                            if args.debug:
                                dbg_print(True, "[tts] queue full → dropped oldest phrase")
                        except Exception:
                            if args.debug:
                                dbg_print(True, "[tts-skip] queue full and could not drop oldest")
                            should_speak = False

                if should_speak and final_phrase:
                    last_any_spoken[0] = now
                    last_phrase[0] = final_phrase
                    last_spoken[final_phrase] = now

                    # NEW: mark items as announced so we don't repeat them
                    if args.new_only:
                        for it in speak_items:
                            announced_at[(it["label"], it["zone"], it["dist"])] = now

                    if voice_enabled_flag[0]:
                        if args.debug:
                            dbg_print(True, f"[tts] speak → {final_phrase}")
                        tts_queue.put(final_phrase)
                    else:
                        if args.debug:
                            dbg_print(True, f"[tts-skip] voice off → {final_phrase}")

            # -------------------- Drawing & Key Handling -------------------- #
            if args.show:
                # Overlay vertical lines for left/center/right
                if draw_boxes:
                    cv2.line(frame, (int(W / 3), 0), (int(W / 3), H), (255, 255, 255), 1)
                    cv2.line(
                        frame,
                        (int(2 * W / 3), 0),
                        (int(2 * W / 3), H),
                        (255, 255, 255),
                        1,
                    )

                cv2.imshow("YOLO11 Talking Objects", frame)

                # Window keep-alive + key logging
                try:
                    prop = cv2.getWindowProperty("YOLO11 Talking Objects", cv2.WND_PROP_VISIBLE)
                    if prop < 1:
                        print("[exit] window closed by OS")
                        break
                except cv2.error:
                    print("[exit] window property check failed; assuming closed")
                    break

                key = cv2.waitKey(1) & 0xFF
                if key != 255 and args.debug:
                    print(f"[dbg] keycode={key}")

                # ESC always quits
                if key == 27:  # ESC
                    print("[exit] esc pressed")
                    break
                # Optional: ignore 'q' if layout is noisy
                if not args.no_tty_quit and key == ord("q"):
                    print("[exit] q pressed")
                    break
                elif key == ord("v"):
                    voice_enabled_flag[0] = not voice_enabled_flag[0]
                    print(f"Voice {'ON' if voice_enabled_flag[0] else 'OFF'}")
                elif key == ord("b"):
                    draw_boxes = not draw_boxes
                    print(f"Boxes {'ON' if draw_boxes else 'OFF'}")

    finally:
        # Stop TTS thread
        try:
            tts_queue.put(None)
        except Exception:
            pass
        if cap:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
