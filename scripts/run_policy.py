from __future__ import annotations

import argparse
import numpy as np

from aloha_rm.config import load_config
from aloha_rm.follower.realman_client import RealmanClient
from aloha_rm.inference.policy_runner import PolicyRunner


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

    obs_dim = np.asarray(follower.get_joint_state()).size
    act_dim = cfg.leader.joint_count
    runner = PolicyRunner(
        follower=follower,
        model_path=args.model,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=cfg.training.hidden_dim,
        command_speed=cfg.collection.command_speed,
        command_acc=cfg.collection.command_acc,
    )
    runner.run(hz=cfg.inference.hz, steps=args.steps)


if __name__ == "__main__":
    main()
