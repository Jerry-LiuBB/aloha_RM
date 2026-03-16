from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    h5py = None

from aloha_rm.follower.realman_client import RealmanClient
from aloha_rm.leader.servo_leader import ServoLeaderArm
from aloha_rm.sensors.realsense_camera import RealSenseD435Camera


class EpisodeCollector:
    def __init__(
        self,
        leader: ServoLeaderArm,
        follower: RealmanClient,
        hz: int,
        max_steps: int,
        camera: RealSenseD435Camera | None = None,
    ) -> None:
        self.leader = leader
        self.follower = follower
        self.hz = hz
        self.max_steps = max_steps
        self.camera = camera

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
        images, image_ts = [], []
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.camera is not None:
            self.camera.start()

        try:
            for _ in range(self.max_steps):
                start = time.time()
                leader_sample = self.leader.sample()
                cmd = leader_sample.joints_rad
                result = self.follower.movej(cmd, speed=command_speed, acc=command_acc)
                follower_state = self.follower.get_joint_state()

                obs.append(follower_state)
                act.append(cmd)
                ts.append(leader_sample.timestamp)
                cmd_ok.append(result.success)

                if self.camera is not None:
                    frame = self.camera.capture()
                    images.append(frame.image_rgb)
                    image_ts.append(frame.timestamp)

                elapsed = time.time() - start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        finally:
            if self.camera is not None:
                self.camera.stop()

        obs_arr = np.asarray(obs, dtype=np.float32)
        act_arr = np.asarray(act, dtype=np.float32)
        ts_arr = np.asarray(ts, dtype=np.float64)
        cmd_ok_arr = np.asarray(cmd_ok, dtype=np.bool_)

        images_arr = np.asarray(images, dtype=np.uint8) if images else None
        image_ts_arr = np.asarray(image_ts, dtype=np.float64) if image_ts else None

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
                if images_arr is not None and image_ts_arr is not None:
                    f.create_dataset("images", data=images_arr)
                    f.create_dataset("image_timestamps", data=image_ts_arr)
        else:
            episode_path = out_dir / f"{episode_name}.npz"
            payload = {
                "observations": obs_arr,
                "actions": act_arr,
                "timestamps": ts_arr,
                "command_ok": cmd_ok_arr,
            }
            if images_arr is not None and image_ts_arr is not None:
                payload["images"] = images_arr
                payload["image_timestamps"] = image_ts_arr
            np.savez_compressed(episode_path, **payload)

        image_sync_mean_ms = 0.0
        image_shape: list[int] = []
        if image_ts_arr is not None and image_ts_arr.size > 0:
            image_sync_mean_ms = float(np.mean(np.abs(image_ts_arr - ts_arr)) * 1000.0)
            image_shape = list(images_arr.shape)

        meta = {
            "hz": self.hz,
            "max_steps": self.max_steps,
            "episode": episode_name,
            "shape_observations": list(obs_arr.shape),
            "shape_actions": list(act_arr.shape),
            "shape_images": image_shape,
            "command_success_rate": float(cmd_ok_arr.mean()) if cmd_ok_arr.size else 0.0,
            "dataset_format": dataset_format,
            "image_sync_mean_abs_error_ms": image_sync_mean_ms,
        }
        with (out_dir / f"{episode_name}.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return episode_path
