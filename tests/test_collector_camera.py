from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

if importlib.util.find_spec("numpy") is None:
    pytest.skip("numpy is not installed in this environment", allow_module_level=True)

import numpy as np

from aloha_rm.camera.mock_camera import MockCamera
from aloha_rm.teleop.collector import EpisodeCollector


class DummyLeader:
    def sample(self):
        class Sample:
            timestamp = 1.23
            joints_rad = np.zeros((6,), dtype=np.float32)

        return Sample()


class DummyFollower:
    def movej(self, joint_rad: np.ndarray, speed: float, acc: float):
        class Result:
            success = True

        return Result()

    def get_joint_state(self) -> np.ndarray:
        return np.ones((6,), dtype=np.float32)


def test_collector_saves_images(tmp_path: Path) -> None:
    collector = EpisodeCollector(
        leader=DummyLeader(),
        follower=DummyFollower(),
        hz=100,
        max_steps=3,
        camera=MockCamera(width=32, height=24, seed=1),
    )
    ep_path = collector.collect("ep_cam", str(tmp_path), command_speed=10.0, command_acc=10.0)
    assert ep_path.exists()

    data = np.load(ep_path)
    assert data["images"].shape == (3, 24, 32, 3)
    assert data["command_ok"].dtype == np.bool_
