from __future__ import annotations

import argparse

from aloha_rm.config import load_config
from aloha_rm.training.train import train_bc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    out = train_bc(
        dataset_dir=cfg.training.dataset_dir,
        model_dir=cfg.training.model_dir,
        hidden_dim=cfg.training.hidden_dim,
        batch_size=cfg.training.batch_size,
        epochs=cfg.training.epochs,
        learning_rate=cfg.training.learning_rate,
        val_split=cfg.training.val_split,
        seed=cfg.training.seed,
    )
    print(f"saved model -> {out}")


if __name__ == "__main__":
    main()
