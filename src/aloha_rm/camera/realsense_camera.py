from __future__ import annotations

import importlib

import numpy as np


class RealSenseCamera:
    """Intel RealSense D435 RGB camera backend (pyrealsense2)."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial_no: str | None = None,
    ) -> None:
        rs = importlib.import_module("pyrealsense2")
        self.rs = rs

        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial_no:
            config.enable_device(serial_no)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.align = rs.align(rs.stream.color)
        self.pipeline.start(config)

    def capture_rgb(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError("Failed to capture color frame from RealSense")

        frame_bgr = np.asanyarray(color_frame.get_data())
        return frame_bgr[..., ::-1].copy()

    def close(self) -> None:
        self.pipeline.stop()
