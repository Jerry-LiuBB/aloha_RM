from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from aloha_rm.camera.base import Camera
from aloha_rm.follower.realman_client import RealmanClient
from aloha_rm.leader.servo_leader import ServoLeaderArm


class EpisodeCollector:
    def __init__(
        self,
        leader: ServoLeaderArm,
        follower: RealmanClient,
        hz: int,
        max_steps: int,
        camera: Camera | None = None,
    ) -> None:
        self.leader = leader
        self.follower = follower
        self.hz = hz
        self.max_steps = max_steps
        self.camera = camera

    def collect(self, episode_name: str, output_dir: str, command_speed: float = 20.0, command_acc: float = 20.0) -> Path:
        dt = 1.0 / self.hz
        obs, act, ts, cmd_ok, images = [], [], [], [], []
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

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
                images.append(self.camera.capture_rgb())

            elapsed = time.time() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)

        obs_arr = np.asarray(obs, dtype=np.float32)
        act_arr = np.asarray(act, dtype=np.float32)
        ts_arr = np.asarray(ts, dtype=np.float64)
        cmd_ok_arr = np.asarray(cmd_ok, dtype=np.bool_)

        episode_path = out_dir / f"{episode_name}.npz"
        payload: dict[str, np.ndarray] = {
            "observations": obs_arr,
            "actions": act_arr,
            "timestamps": ts_arr,
            "command_ok": cmd_ok_arr,
        }
        if images:
            payload["images"] = np.asarray(images, dtype=np.uint8)

        np.savez_compressed(episode_path, **payload)

        meta = {
            "hz": self.hz,
            "max_steps": self.max_steps,
            "episode": episode_name,
            "shape_observations": list(obs_arr.shape),
            "shape_actions": list(act_arr.shape),
            "shape_images": list(payload["images"].shape) if "images" in payload else None,
            "command_success_rate": float(cmd_ok_arr.mean()) if cmd_ok_arr.size else 0.0,
        }
        with (out_dir / f"{episode_name}.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return episode_path
