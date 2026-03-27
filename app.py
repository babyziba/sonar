#!/usr/bin/env python3
"""
YOLO11 "Talking Objects" Demo
- Runs YOLO11 on a webcam or video.
- Announces objects and simple spatial relation: left/center/right + near/medium/far.
- Offline TTS via pyttsx3 (or macOS `say` fallback).

Usage:
  python app.py --source 0           # webcam
  python app.py --source path/to/video.mp4
  python app.py --model yolo11n.pt   # or any YOLO11 variant
  python app.py --conf 0.35 --iou 0.5 --speak-interval 2.0

Keys:
  q = quit, v = toggle voice, b = toggle drawing boxes
"""
import argparse
import math
import queue
import threading
import time
import signal
from collections import defaultdict
from typing import Tuple

import cv2
import numpy as np
import mediapipe as mp

#MediaPipe Hand detection 
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

#global Hands instance 
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


#YOLO
try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit("Ultralytics is required. Install with: pip install ultralytics") from e

#class families to suppress cross-label chatter 
_CLASS_FAMILIES = {
    #drinkware / containers
    "bottle": "container",
    "cup": "container",
    "can": "container",
    "wine glass": "container",
    "glass": "container",
    "mug": "container",
    #handheld electronics
    "cell phone": "device",
    "remote": "device",
    "laptop": "device",
    "keyboard": "device",
    "mouse": "device",
    "tv": "device",
}
def _label_family(lbl: str) -> str:
    return _CLASS_FAMILIES.get(lbl.lower(), lbl.lower())

#TTS setup (offline)  [FIX: make macOS 'say' blocking to avoid overlapping/stutter; serialize with a lock]
def _tts_worker(tts_queue: "queue.Queue[str]", voice_enabled_flag: list, backend_choice: str = "auto"):
    engine = None
    use_say = False
    speak_lock = threading.Lock()  #ensure only one speaks at a time

    #Choose backend
    if backend_choice in ("auto", "pyttsx3"):
        try:
            import pyttsx3
            engine = pyttsx3.init()
        except Exception:
            engine = None

    if backend_choice == "say" or (engine is None and backend_choice == "auto"):
        use_say = True

    while True:
        phrase = tts_queue.get()
        if phrase is None:
            break
        if not voice_enabled_flag[0]:
            continue

        if engine is not None:
            try:
                with speak_lock:
                    engine.say(phrase)
                    engine.runAndWait()   
            except Exception:
                pass
        elif use_say:
            #macOS say: make it BLOCKING so phrases don't overlap/clip
            try:
                import subprocess, sys
                if sys.platform == "darwin":
                    with speak_lock:
                        subprocess.run(["say", phrase])  
            except Exception:
                pass
        else:
            #last-resort: just drop
            pass

def bearing_zone(cx_norm: float) -> str:
    #left / center / right by thirds
    if cx_norm < 0.33:
        return "to your left"
    elif cx_norm < 0.66:
        return "in front of you"
    else:
        return "to your right"

def distance_bucket(h_norm: float) -> str:
    #crude monocular proxy using bbox height relative to frame height
    if h_norm >= 0.55:
        return "right in front of you"
    elif h_norm >= 0.35:
        return "very close"
    elif h_norm >= 0.2:
        return "nearby"
    else:
        return "a little further"

def describe_detection(name: str, zone: str, dist: str) -> str:
    article = "an" if name[:1].lower() in "aeiou" else "a"
    return f"{article} {name} {zone}, {dist}."

#debug helpers + stability tracker

def dbg_print(enabled: bool, *args):
    if enabled:
        print(*args)

class StableTracker:
    """
    Tracks per-(label, zone) stability across frames.
    An item must appear in >= min_stable consecutive frames to be considered 'stable'.
    A small time-based hold avoids flicker when it temporarily disappears.
    """
    def __init__(self, min_stable_frames: int = 3, hold_seconds: float = 0.75):
        self.min_stable = min_stable_frames
        self.hold = hold_seconds
        self.counts = {}        #key -> consecutive frames present
        self.last_seen = {}     #key -> last timestamp seen

    def update_frame(self, present_keys, now):
        present_set = set(present_keys)
        #increment present
        for k in present_set:
            self.counts[k] = self.counts.get(k, 0) + 1
            self.last_seen[k] = now
        #maintain absent keys for hold interval; drop if expired
        to_delete = []
        for k, last in list(self.last_seen.items()):
            if k not in present_set and (now - last) > self.hold:
                to_delete.append(k)
        for k in to_delete:
            self.counts.pop(k, None)
            self.last_seen.pop(k, None)

    def stable_now(self):
        return {k for k, c in self.counts.items() if c >= self.min_stable}

# Hysteresis helpers (zone & distance) 

def _zone_with_deadband(cx_norm: float, prev_zone: str, deadband: float) -> str:
    """
    Apply a sticky band around 0.33 and 0.66 so zones don't flap when near the boundary.
    If we're within ±deadband of a boundary and have a prev_zone, keep it.
    """
    z_basic = bearing_zone(cx_norm)
    if not prev_zone:
        return z_basic
    #boundaries for thirds
    b1, b2 = 0.33, 0.66
    near_b1 = abs(cx_norm - b1) <= deadband
    near_b2 = abs(cx_norm - b2) <= deadband
    if z_basic != prev_zone and (near_b1 or near_b2):
        return prev_zone
    return z_basic

def _dist_with_deadband(h_norm: float, prev_dist: str, deadband: float) -> str:
    """
    Sticky band around 0.25 and 0.45 so distance words don't flap.
    """
    d_basic = distance_bucket(h_norm)
    if not prev_dist:
        return d_basic
    b1, b2 = 0.25, 0.45
    near_b1 = abs(h_norm - b1) <= deadband
    near_b2 = abs(h_norm - b2) <= deadband
    if d_basic != prev_dist and (near_b1 or near_b2):
        return prev_dist
    return d_basic


def annotate_hands(frame):
    """
    Draw MediaPipe hand landmarks + simple bounding boxes onto the frame.
    Called indirectly via cv2.imshow wrapper so we don't touch main().
    """
    if frame is None or frame.size == 0:
        return frame

    #Convert BGR -> RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return frame

    h, w = frame.shape[:2]
    hand_landmarks_list = result.multi_hand_landmarks
    handedness_list = result.multi_handedness or []

    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        #Draw landmarks + connections
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_style.get_default_hand_landmarks_style(),
            mp_style.get_default_hand_connections_style(),
        )

        #Build a bounding box around the landmarks
        xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
        ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        #Optional: handedness label (Left / Right)
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

#Wrap cv2.imshow so we can inject hand tracking without touching main() ---

_original_imshow = cv2.imshow

def _imshow_with_hands(winname, frame):
    #Only annotate the main YOLO window; leave others alone
    if winname == "YOLO11 Talking Objects":
        frame = annotate_hands(frame)
    return _original_imshow(winname, frame)

cv2.imshow = _imshow_with_hands



#args use in terminal 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO11 model path or name")
    ap.add_argument("--source", type=str, default="0", help="Video source (0 for webcam)")
    ap.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    #intersection over union for merging frames :) 
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU") 
    ap.add_argument("--device", type=str, default=None, help="cuda, cpu, or mps")
    ap.add_argument("--speak-interval", type=float, default=2.0, help="Seconds between repeated announcements of same phrase")
    ap.add_argument("--show", action="store_true", help="Show window with boxes", default=True)
    #debug flags
    ap.add_argument("--debug", action="store_true", help="Verbose logs: FPS, detections, TTS events")
    ap.add_argument("--log-every", type=float, default=1.0, help="Seconds between debug log heartbeats")
    ap.add_argument("--min-stable", type=int, default=3, help="Frames an object must persist (per label+zone) before speaking")
    ap.add_argument("--hold-seconds", type=float, default=0.75, help="Post-detect hold to reduce flicker (seconds)")
    ap.add_argument("--no-tty-quit", action="store_true", help="Ignore 'q' key (for layouts sending stray q/ç); use ESC to quit")
    ap.add_argument("--watchdog", type=float, default=0.0, help="Exit with reason if no frames arrive for this many seconds (0 to disable)")
    #speak only when state changes + quiet cooldown logs 
    ap.add_argument("--speak-on-change", action="store_true",
                    help="Only announce when a stable item's side/distance changes or first appears")
    ap.add_argument("--quiet-cooldowns", action="store_true",
                    help="Suppress cooldown skip logs even with --debug")
    #perlabel gate to avoid rapid repeating!! variants for the same label
    ap.add_argument("--label-interval", type=float, default=None,
                    help="Min seconds between announcements for the same (any wording). "
                         "Defaults to --speak-interval if not set.")
    #family-level gate to avoid bottle/cup/can/etc flips in same zone!! 
    ap.add_argument("--family-interval", type=float, default=None,
                    help="Min seconds between announcements for similar labels (e.g., bottle/cup/can) in the same zone. "
                         "Defaults to --label-interval (or --speak-interval).")
    #deadbands to stop zone/distance boundary jitter (annoying)
    ap.add_argument("--zone-deadband", type=float, default=0.06,
                    help="Sticky band (0-1 normalized) around left/center/right boundaries to reduce zone flicker (default 0.06).")
    ap.add_argument("--dist-deadband", type=float, default=0.04,
                    help="Sticky band (0-1 normalized) around distance boundaries to reduce near/very-close flicker (default 0.04).")
    #TTS backend control
    ap.add_argument("--tts-backend", type=str, default="auto", choices=["auto", "pyttsx3", "say"], help="Force TTS backend (auto tries pyttsx3, then say)")
    args = ap.parse_args()

    #Resolve per-label/family interval defaults
    label_interval = args.label_interval if args.label_interval is not None else args.speak_interval
    family_interval = args.family_interval if args.family_interval is not None else label_interval

    #Load model
    model = YOLO(args.model)

    #Open source
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

    #Window (helps on macOS)
    if args.show:
        cv2.namedWindow("YOLO11 Talking Objects", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO11 Talking Objects", 960, 540)

    #Warm-up: try to pull a frame; if it fails, retry & failover once
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

    #TTS thread
    tts_queue: "queue.Queue[str]" = queue.Queue()
    voice_enabled_flag = [True]
    t_thread = threading.Thread(target=_tts_worker, args=(tts_queue, voice_enabled_flag, args.tts_backend), daemon=True)
    t_thread.start()

    last_spoken = defaultdict(float)                  # phrase -> time
    last_spoken_label = defaultdict(float)            # label -> time
    last_announced_state = {}                         # label -> (zone, dist, t_last_spoken)
    # NEW: family gating (family, zone) -> time/last label to suppress cross-label chatter
    last_spoken_family_zone = defaultdict(float)      # (family, zone) -> time
    last_family_label = {}                            # (family, zone) -> last label actually spoken
    # NEW: hysteresis memory per label
    last_zone_seen = {}                               # label -> last zone decided (string)
    last_dist_seen = {}                               # label -> last distance decided (string)

    draw_boxes = True
    print("Press v to toggle voice, b to toggle boxes, q to quit.")

    last_heartbeat = time.time()

    last_any_spoken = [0.0]     #last_any_spoken[0] = last time anything was announced
    MAX_TTS_QUEUE = 2  

    #tracker + debug state
    tracker = StableTracker(min_stable_frames=args.min_stable, hold_seconds=args.hold_seconds)
    last_log = time.time()
    frame_counter = 0
    fps_last = time.time()
    fps = 0.0
    last_frame_time = time.time()

    #exit reason + signal handlers
    exit_reason = ["unknown"]
    def _set_exit(reason):
        exit_reason[0] = reason
        print(f"[exit] reason → {reason}")

    def _sig_handler(signum, frame):
        _set_exit(f"signal {signum}")
        raise SystemExit(0)

    try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
    except Exception:
        pass

    try:
        while True:
            #Heartbeat every ~1s so you know it's alive
            now = time.time()
            if now - last_heartbeat > 1.0:
                print("[running] press q to quit…")
                last_heartbeat = now

            ret, frame = cap.read()
            if ret:
                last_frame_time = now
            if not ret:
                if args.debug:
                    print("[dbg] no frame from camera")
                #keep app alive; show placeholder if --show
                if args.show:
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "No frame received. Retrying…", (30, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                    cv2.imshow("YOLO11 Talking Objects", placeholder)
                    _ = cv2.waitKey(1) & 0xFF
                #quick re-open attempt for webcams
                if is_webcam:
                    cap.release()
                    try:
                        cap = cv2.VideoCapture(int(args.source), cv2.CAP_AVFOUNDATION)
                    except ValueError:
                        cap = cv2.VideoCapture(src)
                time.sleep(0.1)

                #watchdog: bail if camera is silent too long (disabled by default)
                if args.watchdog > 0 and (now - last_frame_time) > args.watchdog:
                    _set_exit(f"watchdog: no frames for {args.watchdog}s")
                    break
                continue

            H, W = frame.shape[:2]

            #timing + FPS
            infer_start = time.time()
            results = model.predict(source=frame, conf=args.conf, iou=args.iou, device=args.device, verbose=False)
            infer_time = time.time() - infer_start

            frame_counter += 1
            now = time.time()
            if now - fps_last >= 0.5:
                fps = frame_counter / (now - fps_last)
                frame_counter = 0
                fps_last = now

            present_keys = []
            summary = {}  #(label, zone, dist) -> {"count": int, "best_score": float}
            det_total = 0

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
                        cy = (y1 + y2) / 2.0

                        cx_norm = cx / W
                        h_norm = h / H

                        label = names.get(cls, str(cls)) if isinstance(names, dict) else names[cls]

                        prev_zone = last_zone_seen.get(label)
                        zone = _zone_with_deadband(cx_norm, prev_zone, args.zone_deadband)
                        last_zone_seen[label] = zone

                        prev_dist = last_dist_seen.get(label)
                        dist = _dist_with_deadband(h_norm, prev_dist, args.dist_deadband)
                        last_dist_seen[label] = dist

                        key = (label, zone, dist)
                        present_keys.append(key)

                        #importance: bigger + closer + confident
                        importance = float(h_norm) * float(conf)

                        if key not in summary:
                            summary[key] = {"count": 1, "best_score": importance}
                        else:
                            summary[key]["count"] += 1
                            if importance > summary[key]["best_score"]:
                                summary[key]["best_score"] = importance

                        if args.show and draw_boxes:
                            txt = f"{label} {conf:.2f}"
                            color = (0, 255, 0)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(frame, txt, (int(x1), int(y1) - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)




            #Update stability and decide what can be spoken
            tracker.update_frame(present_keys, now)
            stable = tracker.stable_now()

            #debug heartbeat
            if now - last_log >= args.log_every or args.debug:
                dbg_print(args.debug, f"[dbg] fps={fps:.1f} dets={det_total} stable={len(stable)} infer={infer_time*1000:.1f}ms")
                last_log = now

            #here
            #Build candidate list from stable items, sorted by importance (closest first)
            candidates = []
            for (label, zone, dist), info in summary.items():
                if (label, zone, dist) not in stable:
                    dbg_print(args.debug,
                            f"[tts-skip] not stable yet → {label} {zone}, {dist} (count={info['count']})")
                    continue

                #Optional: only speak when the state for this label actually changed
                #optional: only speak when the state for this label actually changed
                if args.speak_on_change:
                    prev = last_announced_state.get(label)
                    if prev is not None:
                        prev_zone, prev_dist, prev_t = prev
                        # Same zone AND same distance word? then skip completely.
                        if prev_zone == zone and prev_dist == dist:
                            dbg_print(args.debug,
                                    f"[tts-skip] unchanged → {label} {zone}, {dist}")
                            continue


                phrase = describe_detection(label, zone, dist)
                count = info["count"]
                if count > 1:
                    phrase = phrase.replace(".", f", {count} of them.")

                family = _label_family(label)
                score = info["best_score"]

                candidates.append((score, label, zone, dist, phrase, family, count))

            #Closest / biggest / most confident first
            candidates.sort(key=lambda c: c[0], reverse=True)

            for score, label, zone, dist, phrase, family, count in candidates:
                #Global cooldown: only one NEW phrase every speak_interval seconds
                if now - last_any_spoken[0] < args.speak_interval:
                    dbg_print(args.debug, "[tts-skip] global cooldown")
                    break

                #Per-label cooldown (gate all variants of the same label)
                if (now - last_spoken_label[label]) < label_interval:
                    dbg_print(args.debug,
                            f"[tts-skip] label-cooldown → {label}")
                    continue

                #Per-family cooldown (bottle/cup/can etc.) in the same zone
                fam_key = (family, zone)
                if (now - last_spoken_family_zone[fam_key]) < family_interval:
                    dbg_print(args.debug,
                            f"[tts-skip] family-cooldown → {family} {zone}")
                    continue

                if not args.speak_on_change:
                    if (now - last_spoken[phrase]) < args.speak_interval:
                        dbg_print(args.debug,
                                f"[tts-skip] phrase-cooldown → {phrase}")
                        continue

                if tts_queue.qsize() >= MAX_TTS_QUEUE:
                    dbg_print(args.debug, "[tts-skip] TTS queue full")
                    break

                #Passed all gates → speak this one
                last_any_spoken[0] = now
                last_spoken[phrase] = now
                last_spoken_label[label] = now
                last_spoken_family_zone[fam_key] = now
                last_announced_state[label] = (zone, dist, now)

                if voice_enabled_flag[0]:
                    dbg_print(args.debug, f"[tts] speak → {phrase}")
                    tts_queue.put(phrase)
                else:
                    dbg_print(args.debug, f"[tts-skip] voice off → {phrase}")

                #Only enqueue one phrase per frame; next frame we'll try the next-most-important
                break


            




            if args.show:
                #overlay hints
                if draw_boxes:
                    cv2.line(frame, (int(W/3), 0), (int(W/3), H), (255, 255, 255), 1)
                    cv2.line(frame, (int(2*W/3), 0), (int(2*W/3), H), (255, 255, 255), 1)
                cv2.imshow("YOLO11 Talking Objects", frame)

                #window keep-alive + key logging
                try:
                    prop = cv2.getWindowProperty("YOLO11 Talking Objects", cv2.WND_PROP_VISIBLE)
                    if prop < 1:
                        _set_exit("window closed by OS")
                        break
                except cv2.error:
                    _set_exit("window property check failed; assuming closed")
                    break

                key = cv2.waitKey(1) & 0xFF
                if key != 255 and args.debug:
                    print(f"[dbg] keycode={key}")
                #ESC always quits
                if key == 27:  # ESC
                    _set_exit("esc pressed")
                    break
                #Optional: ignore 'q' if layout is noisy
                if not args.no_tty_quit and key == ord('q'):
                    _set_exit("q pressed")
                    break
                elif key == ord('v'):
                    voice_enabled_flag[0] = not voice_enabled_flag[0]
                    print(f"Voice {'ON' if voice_enabled_flag[0] else 'OFF'}")
                elif key == ord('b'):
                    draw_boxes = not draw_boxes
                    print(f"Boxes {'ON' if draw_boxes else 'OFF'}")

    except SystemExit:
        pass
    except BaseException as ex:
        #make any silent exit visible, including KeyboardInterrupt
        import traceback
        print("[fatal]", ex)
        traceback.print_exc()
        exit_reason[0] = exit_reason[0] if exit_reason[0] != "unknown" else "exception"
    finally:
        #stop TTS first so goodbye doesn't fight
        try:
            tts_queue.put(None)
        except Exception:
            pass
        #Say goodbye with the reason
        try:
            import subprocess, sys
            msg = f"Shutting down. Reason: {exit_reason[0]}."
            if sys.platform == "darwin":
                subprocess.run(["say", msg])  # blocking so you hear it
            else:
                print(msg)
        except Exception:
            print(f"Shutting down. Reason: {exit_reason[0]}.")
        if cap:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
