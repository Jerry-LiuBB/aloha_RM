from __future__ import annotations

import argparse

from aloha_rm.config import load_config
from aloha_rm.follower.realman_client import RealmanClient
from aloha_rm.leader.servo_leader import ServoLeaderArm
from aloha_rm.sensors.realsense_camera import RealSenseD435Camera
from aloha_rm.teleop.collector import EpisodeCollector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--episode", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    leader = ServoLeaderArm(
        joint_count=cfg.leader.joint_count,
        scale=cfg.leader.scale,
        offset=cfg.leader.offset,
    )
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
    camera_external = None
    if cfg.camera.enabled:
        if cfg.camera.model != "realsense_d435":
            raise ValueError(f"Unsupported camera model={cfg.camera.model}")
        primary_serial = cfg.camera.serial_no or cfg.camera.wrist_serial_no
        if primary_serial is not None:
            camera = RealSenseD435Camera(width=cfg.camera.width, height=cfg.camera.height, fps=cfg.camera.fps, serial_no=primary_serial)
        else:
            camera = RealSenseD435Camera(width=cfg.camera.width, height=cfg.camera.height, fps=cfg.camera.fps)
        if cfg.camera.external_serial_no:
            camera_external = RealSenseD435Camera(width=cfg.camera.width, height=cfg.camera.height, fps=cfg.camera.fps, serial_no=cfg.camera.external_serial_no)

    collector = EpisodeCollector(
        leader,
        follower,
        hz=cfg.collection.hz,
        max_steps=cfg.collection.max_steps,
        camera=camera,
        secondary_camera=camera_external,
        base_camera_name=cfg.camera.base_camera_name,
    )
    path = collector.collect(
        args.episode,
        cfg.collection.output_dir,
        command_speed=cfg.collection.command_speed,
        command_acc=cfg.collection.command_acc,
        dataset_format=cfg.collection.dataset_format,
    )
    print(f"saved episode -> {path}")


if __name__ == "__main__":
    main()
