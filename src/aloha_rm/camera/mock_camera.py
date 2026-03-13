from __future__ import annotations

import numpy as np


class MockCamera:
    """Deterministic mock camera for pipeline/debug testing."""

    def __init__(self, width: int = 160, height: int = 120, seed: int = 42) -> None:
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)

    def capture_rgb(self) -> np.ndarray:
        frame = self.rng.integers(0, 256, size=(self.height, self.width, 3), dtype=np.uint8)
        return frame
