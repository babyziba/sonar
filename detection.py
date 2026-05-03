"""YOLO11 wrapper. Loads a model and produces normalized Detection records."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit("Ultralytics is required. Install with: pip install ultralytics") from e


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    cx_norm: float                            # bbox center x / frame width
    h_norm: float                             # bbox height / frame height
    track_id: int | None = None               # ByteTrack id, persistent across frames


class Detector:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf: float = 0.35,
        iou: float = 0.5,
        device: str | None = None,
    ) -> None:
        self._model = YOLO(model_path)
        self._conf = conf
        self._iou = iou
        self._device = device

    @property
    def names(self):
        return self._model.names

    def detect(self, frame) -> list[Detection]:
        H, W = frame.shape[:2]
        results = self._model.track(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            persist=True,
            verbose=False,
        )
        if not results:
            return []

        r = results[0]
        if not (hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0):
            return []

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        if r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
        else:
            ids = [None] * len(xyxy)
        names = r.names if hasattr(r, "names") else self._model.names

        out: list[Detection] = []
        for i, (box, conf, cls) in enumerate(zip(xyxy, confs, clss)):
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            h = max(1.0, y2 - y1)
            cx = (x1 + x2) / 2.0
            label = names.get(cls, str(cls)) if isinstance(names, dict) else names[cls]
            tid = ids[i]
            out.append(Detection(
                label=label,
                confidence=float(conf),
                bbox=(x1, y1, x2, y2),
                cx_norm=cx / W,
                h_norm=h / H,
                track_id=int(tid) if tid is not None else None,
            ))
        return out
