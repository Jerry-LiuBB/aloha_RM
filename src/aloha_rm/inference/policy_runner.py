from __future__ import annotations

import time

import numpy as np
import torch

from aloha_rm.follower.realman_client import RealmanClient
from aloha_rm.sensors.realsense_camera import RealSenseD435Camera
from aloha_rm.training.model import BCMLP


class PolicyRunner:
    def __init__(
        self,
        follower: RealmanClient,
        model_path: str,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        command_speed: float = 20.0,
        command_acc: float = 20.0,
        camera: RealSenseD435Camera | None = None,
    ) -> None:
        self.follower = follower
        self.command_speed = command_speed
        self.command_acc = command_acc
        self.camera = camera

        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        obs_dim = int(checkpoint.get("obs_dim", obs_dim)) if isinstance(checkpoint, dict) else obs_dim
        act_dim = int(checkpoint.get("act_dim", act_dim)) if isinstance(checkpoint, dict) else act_dim
        hidden_dim = int(checkpoint.get("hidden_dim", hidden_dim)) if isinstance(checkpoint, dict) else hidden_dim

        self.model = BCMLP(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def run(self, hz: int, steps: int) -> None:
        dt = 1.0 / hz

        if self.camera is not None:
            self.camera.start()

        try:
            for _ in range(steps):
                start = time.time()
                obs = self.follower.get_joint_state().astype(np.float32)
                if self.camera is not None:
                    frame = self.camera.capture()
                    image_flat = frame.image_rgb.astype(np.float32).reshape(-1) / 255.0
                    obs = np.concatenate([obs, image_flat], axis=0)

                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                action = self.model(obs_t).squeeze(0).cpu().numpy()
                self.follower.movej(action, speed=self.command_speed, acc=self.command_acc)
                elapsed = time.time() - start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        finally:
            if self.camera is not None:
                self.camera.stop()
