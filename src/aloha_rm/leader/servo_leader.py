from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class LeaderSample:
    timestamp: float
    joints_rad: np.ndarray


class ServoLeaderArm:
    """Leader arm adapter.

    Replace `read_joint_degrees` with actual serial/CAN read implementation.
    """

    def __init__(self, joint_count: int, scale: float = 1.0, offset: float = 0.0) -> None:
        self.joint_count = joint_count
        self.scale = scale
        self.offset = offset

    def read_joint_degrees(self) -> np.ndarray:
        # Placeholder demo signal to keep pipeline runnable before hardware integration.
        t = time.time()
        return np.array([10 * math.sin(t + i) for i in range(self.joint_count)], dtype=np.float32)

    def sample(self) -> LeaderSample:
        deg = self.read_joint_degrees() * self.scale + self.offset
        rad = np.deg2rad(deg).astype(np.float32)
        return LeaderSample(timestamp=time.time(), joints_rad=rad)
