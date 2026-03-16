from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    h5py = None

from aloha_rm.follower.realman_client import RealmanClient
from aloha_rm.leader.servo_leader import ServoLeaderArm


class EpisodeCollector:
    def __init__(
        self,
        leader: ServoLeaderArm,
        follower: RealmanClient,
        hz: int,
        max_steps: int,
        camera: Any | None = None,
        secondary_camera: Any | None = None,
        base_camera_name: str = "wrist",
    ) -> None:
        self.leader = leader
        self.follower = follower
        self.hz = hz
        self.max_steps = max_steps
        self.camera = camera
        self.secondary_camera = secondary_camera
        self.base_camera_name = base_camera_name

    @staticmethod
    def _start_camera(camera: Any | None) -> None:
        if camera is not None and hasattr(camera, "start"):
            camera.start()

    @staticmethod
    def _stop_camera(camera: Any | None) -> None:
        if camera is not None and hasattr(camera, "stop"):
            camera.stop()

    @staticmethod
    def _capture_camera(camera: Any) -> tuple[np.ndarray, float]:
        if hasattr(camera, "capture"):
            frame = camera.capture()
            return frame.image_rgb, float(frame.timestamp)
        if hasattr(camera, "capture_rgb"):
            return camera.capture_rgb(), time.time()
        raise TypeError("Camera must implement capture() or capture_rgb()")

    def collect(
        self,
        episode_name: str,
        output_dir: str,
        command_speed: float = 20.0,
        command_acc: float = 20.0,
        dataset_format: str = "npz",
    ) -> Path:
        dt = 1.0 / self.hz
        obs, act, ts, cmd_ok = [], [], [], []
        images_primary, image_ts_primary = [], []
        images_secondary, image_ts_secondary = [], []
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self._start_camera(self.camera)
        self._start_camera(self.secondary_camera)

        try:
            for _ in range(self.max_steps):
                start = time.time()
                leader_sample = self.leader.sample()
                cmd = leader_sample.joints_rad
                result = self.follower.movej(cmd, speed=command_speed, acc=command_acc)
                follower_state = self.follower.get_joint_state()

                obs.append(follower_state)
                act.append(cmd)
                cmd_ok.append(result.success)

                if self.camera is not None:
                    img, img_ts = self._capture_camera(self.camera)
                    images_primary.append(img)
                    image_ts_primary.append(img_ts)
                    ts.append(img_ts)
                else:
                    ts.append(leader_sample.timestamp)

                if self.secondary_camera is not None:
                    img2, img2_ts = self._capture_camera(self.secondary_camera)
                    images_secondary.append(img2)
                    image_ts_secondary.append(img2_ts)

                elapsed = time.time() - start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        finally:
            self._stop_camera(self.secondary_camera)
            self._stop_camera(self.camera)

        obs_arr = np.asarray(obs, dtype=np.float32)
        act_arr = np.asarray(act, dtype=np.float32)
        ts_arr = np.asarray(ts, dtype=np.float64)
        cmd_ok_arr = np.asarray(cmd_ok, dtype=np.bool_)

        primary_images_arr = np.asarray(images_primary, dtype=np.uint8) if images_primary else None
        primary_ts_arr = np.asarray(image_ts_primary, dtype=np.float64) if image_ts_primary else None
        secondary_images_arr = np.asarray(images_secondary, dtype=np.uint8) if images_secondary else None
        secondary_ts_arr = np.asarray(image_ts_secondary, dtype=np.float64) if image_ts_secondary else None

        if dataset_format not in {"npz", "hdf5"}:
            raise ValueError(f"Unsupported dataset_format={dataset_format}, expected 'npz' or 'hdf5'")

        if dataset_format == "hdf5":
            if h5py is None:
                raise ModuleNotFoundError("h5py is required when dataset_format='hdf5'")
            episode_path = out_dir / f"{episode_name}.hdf5"
            with h5py.File(episode_path, "w") as f:
                f.create_dataset("observations", data=obs_arr)
                f.create_dataset("actions", data=act_arr)
                f.create_dataset("timestamps", data=ts_arr)
                f.create_dataset("command_ok", data=cmd_ok_arr)
                if primary_images_arr is not None and primary_ts_arr is not None:
                    g = f.require_group("observations/images")
                    g.create_dataset(self.base_camera_name, data=primary_images_arr)
                    f.create_dataset("image_timestamps", data=primary_ts_arr)
                if secondary_images_arr is not None and secondary_ts_arr is not None:
                    g = f.require_group("observations/images")
                    other_name = "external" if self.base_camera_name != "external" else "secondary"
                    g.create_dataset(other_name, data=secondary_images_arr)
                    f.create_dataset("secondary_image_timestamps", data=secondary_ts_arr)
        else:
            episode_path = out_dir / f"{episode_name}.npz"
            payload = {
                "observations": obs_arr,
                "actions": act_arr,
                "timestamps": ts_arr,
                "command_ok": cmd_ok_arr,
            }
            if primary_images_arr is not None and primary_ts_arr is not None:
                payload["images"] = primary_images_arr
                payload["image_timestamps"] = primary_ts_arr
            if secondary_images_arr is not None and secondary_ts_arr is not None:
                payload["images_secondary"] = secondary_images_arr
                payload["image_timestamps_secondary"] = secondary_ts_arr
            np.savez_compressed(episode_path, **payload)

        image_shape: list[int] = list(primary_images_arr.shape) if primary_images_arr is not None else []
        secondary_image_shape: list[int] = list(secondary_images_arr.shape) if secondary_images_arr is not None else []
        image_sync_mean_ms = 0.0
        if primary_ts_arr is not None and primary_ts_arr.size > 0:
            image_sync_mean_ms = float(np.mean(np.abs(primary_ts_arr - ts_arr)) * 1000.0)

        meta = {
            "hz": self.hz,
            "max_steps": self.max_steps,
            "episode": episode_name,
            "shape_observations": list(obs_arr.shape),
            "shape_actions": list(act_arr.shape),
            "shape_images": image_shape,
            "shape_images_secondary": secondary_image_shape,
            "command_success_rate": float(cmd_ok_arr.mean()) if cmd_ok_arr.size else 0.0,
            "dataset_format": dataset_format,
            "time_base": "camera" if self.camera is not None else "leader",
            "image_sync_mean_abs_error_ms": image_sync_mean_ms,
        }
        with (out_dir / f"{episode_name}.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return episode_path
