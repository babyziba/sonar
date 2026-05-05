#!/usr/bin/env python3
"""
Sonr — assistive vision: YOLO11 detection + spoken scene narration + hand overlay.

Usage:
  python sonr.py --source 0
  python sonr.py --source path/to/video.mp4
  python sonr.py --model yolo11n.pt --conf 0.35 --speak-interval 2.0

Keys: v = toggle voice, b = toggle boxes, h = toggle hand tracking,
      r = read text in scene (OCR), q / ESC = quit.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from detection import Detector
from distance import dist_with_deadband, metric_with_deadband
from geometry import zone_with_deadband
from hands import HandTracker
from narration import SpeakItem, compose_phrase
from speech import TTSWorker
from tracking import StableTracker


WINDOW_NAME = "YOLO11 Talking Objects"


def _env(name: str, default):
    """Return os.environ[name] if set and non-empty, else `default`. Preserves
    `default`'s type so callers can supply `None`, a string, a float, etc."""
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=_env("SONR_MODEL", "yolo11n.pt"),
                    help="YOLO11 model path or name (env: SONR_MODEL)")
    ap.add_argument("--source", type=str, default=_env("SONR_SOURCE", "0"),
                    help="Video source, 0 for webcam (env: SONR_SOURCE)")
    ap.add_argument("--conf", type=float, default=float(_env("SONR_CONF", 0.75)),
                    help="Confidence threshold (env: SONR_CONF)")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU")
    ap.add_argument("--device", type=str, default=_env("SONR_DEVICE", None),
                    help="cuda, cpu, or mps (env: SONR_DEVICE)")
    ap.add_argument("--speak-interval", type=float, default=2.0,
                    help="Seconds between announcements of the same phrase")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="Disable the OpenCV preview window")
    ap.set_defaults(show=True)

    ap.add_argument("--debug", action="store_true",
                    help="Verbose logs: FPS, detections, TTS events")
    ap.add_argument("--log-every", type=float, default=1.0,
                    help="Seconds between debug log heartbeats")
    ap.add_argument("--min-stable", type=int, default=3,
                    help="Frames an object must persist before speaking")
    ap.add_argument("--hold-seconds", type=float, default=0.75,
                    help="Post-detect hold to reduce flicker (seconds)")
    ap.add_argument("--no-tty-quit", action="store_true",
                    help="Ignore 'q' key; use ESC only to quit")
    ap.add_argument("--watchdog", type=float, default=0.0,
                    help="Exit if no frames arrive for this many seconds (0 to disable)")
    ap.add_argument("--quiet-cooldowns", action="store_true",
                    help="Suppress cooldown logs even with --debug")
    ap.add_argument("--zone-deadband", type=float, default=0.06,
                    help="Sticky band around left/center/right boundaries")
    ap.add_argument("--dist-deadband", type=float, default=0.04,
                    help="Sticky band around distance boundaries")
    ap.add_argument("--min-speak-hnorm", type=float, default=0.05,
                    help="Minimum bbox height (normalized) required to speak an object")
    ap.add_argument("--per-item-cooldown", type=float, default=8.0,
                    help="Seconds before the same (label, zone, distance) item can be re-announced")
    ap.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True,
                    help="Mirror the frame horizontally (correct for selfie-view webcams)")
    ap.add_argument("--depth", action=argparse.BooleanOptionalAction, default=True,
                    help="Use Depth Anything v2 for metric distance (else fall back to bbox-height)")
    ap.add_argument("--metric-deadband", type=float, default=0.3,
                    help="Hysteresis around metric distance bucket boundaries (meters)")
    ap.add_argument("--ocr", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable on-demand text OCR (press 'r' to read text in current frame)")
    return ap.parse_args()


def open_capture(source: str):
    """Open webcam (numeric source) or video file. Returns (capture, is_webcam)."""
    try:
        cam_idx = int(source)
        return cv2.VideoCapture(cam_idx, cv2.CAP_AVFOUNDATION), True
    except ValueError:
        return cv2.VideoCapture(source), False


def warmup(cap, tries: int = 40, pause: float = 0.1) -> bool:
    for _ in range(tries):
        ok, _ = cap.read()
        if ok:
            return True
        time.sleep(pause)
    return False


def dbg(enabled: bool, *args) -> None:
    if enabled:
        print(*args)


def main() -> None:
    args = parse_args()

    detector = Detector(args.model, conf=args.conf, iou=args.iou, device=args.device)
    hand_tracker = HandTracker()
    tts = TTSWorker()
    tracker = StableTracker(min_stable_frames=args.min_stable, hold_seconds=args.hold_seconds)

    depth_estimator = None
    depth_worker = None
    if args.depth:
        from depth import DepthEstimator, DepthWorker
        print("Loading Depth Anything v2 (first run downloads ~100 MB)...")
        depth_estimator = DepthEstimator(device=args.device)
        depth_worker = DepthWorker(depth_estimator)
        print(f"Depth model loaded on device: {depth_estimator.device} (running async)")

    ocr_worker = None
    if args.ocr:
        from ocr import OcrWorker
        print("Loading EasyOCR (first run downloads ~64 MB)...")
        ocr_worker = OcrWorker()
        print("OCR ready. Press 'r' to read text in the current scene.")

    cap, is_webcam = open_capture(args.source)
    if not cap or not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    if args.show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 540)

    if not warmup(cap) and is_webcam:
        try:
            alt_idx = 1 if int(args.source) == 0 else 0
            alt = cv2.VideoCapture(alt_idx, cv2.CAP_AVFOUNDATION)
            if warmup(alt):
                cap.release()
                cap = alt
                print(f"Switched to camera index {alt_idx}")
        except ValueError:
            pass

    print("Press v=voice, b=boxes, h=hands, r=read text, q=quit.")

    last_zone_seen: dict[str, str] = {}
    last_dist_seen: dict[str, str] = {}
    announced_at: dict[tuple[str, str, str], float] = {}
    last_any_spoken = 0.0
    user_pending: dict[str, float] = {}  # tag -> deadline for user-triggered OCR/plate jobs

    draw_boxes = True
    draw_hands = True
    last_log = time.time()
    frame_counter = 0
    fps_last = time.time()
    fps = 0.0
    last_frame_time = time.time()

    try:
        while True:
            now = time.time()

            ret, raw_frame = cap.read()
            if ret:
                last_frame_time = now
                # `frame` is what YOLO / MediaPipe / depth / display all use.
                # `raw_frame` stays unflipped for OCR — flipped text is unreadable.
                frame = cv2.flip(raw_frame, 1) if args.mirror else raw_frame
            else:
                dbg(args.debug, "[dbg] no frame from camera")
                if args.show:
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "No frame received. Retrying...",
                                (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(WINDOW_NAME, placeholder)
                    cv2.waitKey(1)
                if is_webcam:
                    cap.release()
                    cap, _ = open_capture(args.source)
                time.sleep(0.1)
                if args.watchdog > 0 and (now - last_frame_time) > args.watchdog:
                    print(f"[exit] watchdog: no frames for {args.watchdog}s")
                    break
                continue

            H, W = frame.shape[:2]

            if depth_worker is not None:
                depth_worker.submit(frame)
            depth_map = depth_worker.get_depth() if depth_worker is not None else None

            infer_start = time.time()
            detections = detector.detect(frame)
            infer_time = time.time() - infer_start

            frame_counter += 1
            if now - fps_last >= 0.5:
                fps = frame_counter / (now - fps_last)
                frame_counter = 0
                fps_last = now

            present_keys: list[tuple[str, str, str]] = []
            summary: dict[tuple[str, str, str], dict] = {}

            for det in detections:
                # Hysteresis state per-track when available, else per-label.
                hyst_key = det.track_id if det.track_id is not None else det.label
                prev_zone = last_zone_seen.get(hyst_key)
                zone = zone_with_deadband(det.cx_norm, prev_zone, args.zone_deadband)
                last_zone_seen[hyst_key] = zone

                prev_dist = last_dist_seen.get(hyst_key)
                if depth_map is not None:
                    from depth import DepthEstimator
                    meters = DepthEstimator.sample_at_bbox(depth_map, det.bbox)
                    dist = metric_with_deadband(meters, prev_dist, args.metric_deadband)
                else:
                    dist = dist_with_deadband(det.h_norm, prev_dist, args.dist_deadband)
                last_dist_seen[hyst_key] = dist

                key = (det.label, zone, dist)
                present_keys.append(key)

                importance = det.h_norm * det.confidence
                if key not in summary:
                    summary[key] = {
                        "count": 1, "best_score": importance, "best_h": det.h_norm,
                        "track_ids": [det.track_id],
                    }
                else:
                    summary[key]["count"] += 1
                    if importance > summary[key]["best_score"]:
                        summary[key]["best_score"] = importance
                    if det.h_norm > summary[key]["best_h"]:
                        summary[key]["best_h"] = det.h_norm
                    summary[key]["track_ids"].append(det.track_id)

                if args.show and draw_boxes:
                    x1, y1, x2, y2 = det.bbox
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    tid_suffix = f"#{det.track_id}" if det.track_id is not None else ""
                    cv2.putText(frame, f"{det.label}{tid_suffix} {det.confidence:.2f}",
                                (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2, cv2.LINE_AA)


            tracker.update_frame(present_keys, now)
            stable = tracker.stable_now()

            if ocr_worker is not None:
                for tag, text in ocr_worker.get_results():
                    if tag == "user":
                        phrase = f"It says: {text}"
                        dbg(args.debug, f"[ocr] {phrase}")
                        if tts.enabled:
                            tts.say(phrase)
                        user_pending.pop(tag, None)

            # Surface "no text found" for user-triggered OCR after a deadline.
            for tag in list(user_pending.keys()):
                if now > user_pending[tag]:
                    if tts.enabled:
                        tts.say("No text found")
                    user_pending.pop(tag, None)

            # Age-based GC for announced_at — drop entries older than 4× cooldown.
            gc_age = max(args.per_item_cooldown * 4, 30.0)
            for k in list(announced_at.keys()):
                if (now - announced_at[k]) > gc_age:
                    announced_at.pop(k, None)

            if args.debug and (now - last_log) >= args.log_every:
                dbg(True, f"[dbg] fps={fps:.1f} dets={len(detections)} "
                          f"stable={len(stable)} infer={infer_time*1000:.1f}ms")
                last_log = now

            def _cooldown_key(tid, label, zone):
                # Per-track cooldown when ByteTrack assigned an id; otherwise
                # fall back to (label, zone) so untracked detections still cool down.
                return tid if tid is not None else ("__notrack__", label, zone)

            speak_items: list[SpeakItem] = []
            for key, info in summary.items():
                label, zone, dist = key
                if key not in stable:
                    if args.debug and not args.quiet_cooldowns:
                        dbg(True, f"[tts-skip] not stable yet -> {label} {zone}, {dist} (count={info['count']})")
                    continue
                if info["best_h"] < args.min_speak_hnorm:
                    if args.debug and not args.quiet_cooldowns:
                        dbg(True, f"[tts-skip] too small -> {label} (h_norm={info['best_h']:.3f})")
                    continue

                eligible_tids = []
                for tid in info["track_ids"]:
                    ck = _cooldown_key(tid, label, zone)
                    last_t = announced_at.get(ck)
                    if last_t is None or (now - last_t) >= args.per_item_cooldown:
                        eligible_tids.append(tid)

                if not eligible_tids:
                    if args.debug and not args.quiet_cooldowns:
                        dbg(True, f"[tts-skip] all tracks cooled -> {label} {zone}, {dist}")
                    continue

                speak_items.append(SpeakItem(
                    label=label, zone=zone, distance=dist,
                    count=len(eligible_tids), score=info["best_score"],
                    track_ids=eligible_tids,
                ))

            if speak_items:
                time_since = now - last_any_spoken
                if time_since < args.speak_interval:
                    if args.debug and not args.quiet_cooldowns:
                        dbg(True, f"[tts-skip] interval gate ({time_since:.1f}s < {args.speak_interval}s)")
                else:
                    final_phrase = compose_phrase(speak_items)
                    if final_phrase:
                        last_any_spoken = now
                        for it in speak_items:
                            for tid in it.track_ids:
                                announced_at[_cooldown_key(tid, it.label, it.zone)] = now
                        if tts.enabled:
                            dbg(args.debug, f"[tts] speak -> {final_phrase}")
                            tts.say(final_phrase)
                    else:
                        dbg(args.debug, f"[tts-skip] voice off -> {final_phrase}")

            if args.show:
                if draw_boxes:
                    cv2.line(frame, (int(W / 3), 0), (int(W / 3), H), (255, 255, 255), 1)
                    cv2.line(frame, (int(2 * W / 3), 0), (int(2 * W / 3), H), (255, 255, 255), 1)

                if draw_hands:
                    hand_tracker.annotate(frame)
                cv2.imshow(WINDOW_NAME, frame)

                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        print("[exit] window closed by OS")
                        break
                except cv2.error:
                    print("[exit] window property check failed; assuming closed")
                    break

                key = cv2.waitKey(1) & 0xFF
                if key != 255 and args.debug:
                    print(f"[dbg] keycode={key}")
                if key == 27:
                    print("[exit] esc pressed")
                    break
                if not args.no_tty_quit and key == ord("q"):
                    print("[exit] q pressed")
                    break
                if key == ord("v"):
                    tts.set_enabled(not tts.enabled)
                    print(f"Voice {'ON' if tts.enabled else 'OFF'}")
                elif key == ord("b"):
                    draw_boxes = not draw_boxes
                    print(f"Boxes {'ON' if draw_boxes else 'OFF'}")
                elif key == ord("h"):
                    draw_hands = not draw_hands
                    print(f"Hands {'ON' if draw_hands else 'OFF'}")
                elif key == ord("r") and ocr_worker is not None:
                    ocr_worker.submit("user", raw_frame)
                    user_pending["user"] = now + 5.0
                    print("[ocr] reading scene...")
    finally:
        tts.stop()
        if depth_worker is not None:
            depth_worker.stop()
        if ocr_worker is not None:
            ocr_worker.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
