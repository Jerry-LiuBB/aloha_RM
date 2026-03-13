from __future__ import annotations

import importlib

import numpy as np


class OpenCVCamera:
    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480) -> None:
        cv2 = importlib.import_module("cv2")
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def capture_rgb(self) -> np.ndarray:
        ok, frame_bgr = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from camera")
        return self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()
