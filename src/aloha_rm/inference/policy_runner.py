from __future__ import annotations

import time

import numpy as np
import torch

from aloha_rm.follower.realman_client import RealmanClient
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
    ) -> None:
        self.follower = follower
        self.command_speed = command_speed
        self.command_acc = command_acc
        self.model = BCMLP(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def run(self, hz: int, steps: int) -> None:
        dt = 1.0 / hz
        for _ in range(steps):
            start = time.time()
            obs = self.follower.get_joint_state()
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            action = self.model(obs_t).squeeze(0).cpu().numpy()
            self.follower.movej(action, speed=self.command_speed, acc=self.command_acc)
            elapsed = time.time() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)
