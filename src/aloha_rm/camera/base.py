from __future__ import annotations

from typing import Protocol

import numpy as np


class Camera(Protocol):
    def capture_rgb(self) -> np.ndarray:
        """Capture one RGB frame with shape [H, W, 3], dtype=uint8."""
