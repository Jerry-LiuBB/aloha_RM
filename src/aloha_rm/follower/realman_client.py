from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import requests


@dataclass(slots=True)
class RealmanCommandResult:
    success: bool
    raw: dict[str, Any]


class RealmanClient:
    """Realman JSON API client with configurable response keys/auth."""

    def __init__(
        self,
        host: str,
        port: int,
        movej_api: str,
        state_api: str,
        timeout_s: float = 1.0,
        success_code: int = 0,
        joint_state_key: str = "joint",
        token: str | None = None,
    ) -> None:
        self.base_url = f"http://{host}:{port}"
        self.movej_api = movej_api
        self.state_api = state_api
        self.timeout_s = timeout_s
        self.success_code = success_code
        self.joint_state_key = joint_state_key
        self.session = requests.Session()
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

    def movej(self, joint_rad: np.ndarray, speed: float = 20.0, acc: float = 20.0) -> RealmanCommandResult:
        payload = {
            "command": "movej",
            "joint": np.asarray(joint_rad, dtype=np.float32).tolist(),
            "speed": float(speed),
            "acc": float(acc),
            "v": float(speed),
            "a": float(acc),
        }
        response = self.session.post(
            f"{self.base_url}{self.movej_api}",
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        raw = response.json()
        return RealmanCommandResult(success=raw.get("code", -1) == self.success_code, raw=raw)

    def get_joint_state(self) -> np.ndarray:
        response = self.session.get(f"{self.base_url}{self.state_api}", timeout=self.timeout_s)
        response.raise_for_status()
        raw = response.json()
        joints = raw.get(self.joint_state_key, raw.get("joint", []))
        return np.asarray(joints, dtype=np.float32)
