"""OCR for arbitrary text in the scene, via EasyOCR.

`OcrWorker` runs EasyOCR on a background thread so the main loop is never
blocked. Producers (main loop key handler, plate pipeline, ...) call
`submit(tag, image)`; consumers poll `get_results()` each frame to collect
completed `(tag, text)` pairs.

EasyOCR is slow (~500 ms-2 s per call on CPU; faster on GPU when available)
and downloads ~64 MB of detector + recognizer weights on first construction.
"""

from __future__ import annotations

import queue
import threading

import numpy as np

try:
    import easyocr
except ImportError as e:
    raise SystemExit(
        "easyocr is required. Install with: pip install easyocr"
    ) from e


class OcrWorker:
    DEFAULT_LANGS = ["en"]
    MAX_QUEUE = 3
    MIN_CONFIDENCE = 0.4

    def __init__(self, langs: list[str] | None = None, gpu: bool = True) -> None:
        self._reader = easyocr.Reader(langs or self.DEFAULT_LANGS, gpu=gpu)
        self._submit_queue: "queue.Queue[tuple[str, np.ndarray] | None]" = queue.Queue()
        self._results: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, tag: str, image: np.ndarray) -> bool:
        """Submit an image for OCR. Drops oldest if queue is full."""
        if self._submit_queue.qsize() >= self.MAX_QUEUE:
            try:
                self._submit_queue.get_nowait()
            except queue.Empty:
                return False
        self._submit_queue.put((tag, image.copy()))
        return True

    def get_results(self) -> list[tuple[str, str]]:
        """Drain and return all completed (tag, text) results."""
        out = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out

    def stop(self) -> None:
        self._stop.set()
        self._submit_queue.put(None)

    def _run(self) -> None:
        while True:
            item = self._submit_queue.get()
            if item is None or self._stop.is_set():
                return
            tag, image = item
            try:
                detections = self._reader.readtext(image)
                # EasyOCR returns list of (bbox, text, confidence).
                texts = [
                    t.strip()
                    for (_, t, conf) in detections
                    if conf >= self.MIN_CONFIDENCE and t.strip()
                ]
                text = " ".join(texts)
                if text:
                    self._results.put((tag, text))
            except Exception:
                pass
