from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import pyrealsense2 as rs
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    rs = None


@dataclass(slots=True)
class CameraFrame:
    image_rgb: np.ndarray
    timestamp: float


class RealSenseD435Camera:
    """RealSense D435 RGB camera wrapper.

    Supports optional serial number binding and hardware timestamps.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial_no: str | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.serial_no = serial_no
        self.pipeline = None

    def start(self) -> None:
        if rs is None:
            raise ModuleNotFoundError("pyrealsense2 is required to use RealSenseD435Camera")

        config = rs.config()
        if self.serial_no:
            config.enable_device(self.serial_no)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        self.pipeline = rs.pipeline()
        self.pipeline.start(config)

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

    def capture(self) -> CameraFrame:
        if self.pipeline is None:
            raise RuntimeError("Camera has not been started")

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError("Failed to get RealSense color frame")

        image = np.asarray(color_frame.get_data(), dtype=np.uint8)
        # RealSense timestamp unit is milliseconds.
        hw_ts_s = float(color_frame.get_timestamp()) / 1000.0
        return CameraFrame(image_rgb=image, timestamp=hw_ts_s)
