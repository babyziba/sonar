"""MediaPipe hand-landmark overlay."""

from __future__ import annotations

import cv2
import mediapipe as mp


class HandTracker:
    """Detect and draw hand landmarks + bounding box on a BGR frame."""

    def __init__(
        self,
        max_num_hands: int = 2,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_style = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def annotate(self, frame):
        """Draw landmarks + per-hand bounding box on `frame` (modified in place)."""
        if frame is None or frame.size == 0:
            return frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return frame

        h, w = frame.shape[:2]
        handedness = result.multi_handedness or []

        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            self._mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_style.get_default_hand_landmarks_style(),
                self._mp_style.get_default_hand_connections_style(),
            )

            xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
            ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            label_text = ""
            if idx < len(handedness) and handedness[idx].classification:
                label_text = handedness[idx].classification[0].label

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
