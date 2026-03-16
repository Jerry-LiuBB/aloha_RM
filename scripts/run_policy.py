from __future__ import annotations

import argparse
import numpy as np

from aloha_rm.config import load_config
from aloha_rm.follower.realman_client import RealmanClient
from aloha_rm.inference.policy_runner import PolicyRunner
from aloha_rm.sensors.realsense_camera import RealSenseD435Camera


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--model", default="artifacts/models/bc_mlp.pt")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    cfg = load_config(args.config)
    follower = RealmanClient(
        host=cfg.realman.host,
        port=cfg.realman.port,
        movej_api=cfg.realman.movej_api,
        state_api=cfg.realman.state_api,
        timeout_s=cfg.realman.timeout_s,
        success_code=cfg.realman.success_code,
        joint_state_key=cfg.realman.joint_state_key,
        token=cfg.realman.token,
    )

    camera = None
    if cfg.camera.enabled:
        if cfg.camera.model != "realsense_d435":
            raise ValueError(f"Unsupported camera model={cfg.camera.model}")
        primary_serial = cfg.camera.serial_no or cfg.camera.wrist_serial_no
        camera = RealSenseD435Camera(width=cfg.camera.width, height=cfg.camera.height, fps=cfg.camera.fps, serial_no=primary_serial)

    obs_dim = np.asarray(follower.get_joint_state()).size
    if cfg.camera.enabled:
        obs_dim += cfg.camera.width * cfg.camera.height * 3

    act_dim = cfg.leader.joint_count
    runner = PolicyRunner(
        follower=follower,
        model_path=args.model,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=cfg.training.hidden_dim,
        command_speed=cfg.collection.command_speed,
        command_acc=cfg.collection.command_acc,
        camera=camera,
    )
    runner.run(hz=cfg.inference.hz, steps=args.steps)


if __name__ == "__main__":
    main()
