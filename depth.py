"""Monocular depth estimation via Depth Anything v2.

The metric-indoor variant outputs approximately metric depth (meters) for
typical indoor scenes up to ~10 m. For outdoor or unusual setups the
estimates degrade — but the relative ordering still beats the bbox-height
heuristic.

`DepthWorker` runs the model on a background thread so the main capture/
detection loop is never blocked by depth latency. The main loop submits the
latest frame and reads whatever the most recent depth map happens to be —
it will lag the live frame by a few hundred ms but that's invisible for
"is this close or nearby" bucketing.

The model is downloaded from HuggingFace on first construction (~100 MB)
and cached locally; subsequent runs use the cache.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
from PIL import Image

try:
    from transformers import pipeline
except ImportError as e:
    raise SystemExit(
        "transformers is required for depth estimation. "
        "Install with: pip install transformers accelerate"
    ) from e


DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


def _autodetect_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class DepthEstimator:
    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None) -> None:
        device = device or _autodetect_device()
        self._pipe = pipeline("depth-estimation", model=model, device=device)
        self.device = device

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a depth map (HxW float32, meters) at the input frame size."""
        rgb = frame_bgr[:, :, ::-1]
        pil = Image.fromarray(rgb)
        result = self._pipe(pil)
        depth_tensor = result["predicted_depth"]
        depth = depth_tensor.cpu().numpy().astype(np.float32)
        if depth.ndim == 3:
            depth = depth[0]
        if depth.shape != frame_bgr.shape[:2]:
            depth = cv2.resize(
                depth,
                (frame_bgr.shape[1], frame_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return depth

    @staticmethod
    def sample_at_bbox(depth: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
        """Median depth across the central 30% of a bbox.

        Center sampling avoids edge pixels that often pick up the background
        behind the object, which would otherwise bias the distance estimate.
        """
        x1, y1, x2, y2 = bbox
        H, W = depth.shape
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        sx = max(0, int(cx - bw * 0.15))
        ex = min(W, int(cx + bw * 0.15) + 1)
        sy = max(0, int(cy - bh * 0.15))
        ey = min(H, int(cy + bh * 0.15) + 1)
        region = depth[sy:ey, sx:ex]
        if region.size == 0:
            return float("nan")
        return float(np.median(region))


class DepthWorker:
    """Runs `DepthEstimator.estimate` on a background thread.

    Latest-wins: `submit` overwrites any pending frame, so the worker always
    processes the freshest available frame and intermediate frames are
    silently dropped. `get_depth` returns the most recent completed map (or
    None until the first inference finishes).
    """

    def __init__(self, estimator: DepthEstimator) -> None:
        self._estimator = estimator
        self._latest_frame: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """Hand a frame to the worker. Overwrites any pending frame."""
        with self._lock:
            self._latest_frame = frame.copy()
        self._wake.set()

    def get_depth(self) -> np.ndarray | None:
        """Return the most recent completed depth map, or None."""
        with self._lock:
            return self._latest_depth

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            if self._stop.is_set():
                return
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is None:
                continue
            try:
                depth = self._estimator.estimate(frame)
                with self._lock:
                    self._latest_depth = depth
            except Exception:
                pass
