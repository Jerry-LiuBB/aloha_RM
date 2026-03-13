from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RealmanConfig:
    host: str
    port: int
    movej_api: str = "/api/movej"
    state_api: str = "/api/joint_state"
    timeout_s: float = 1.0
    success_code: int = 0
    joint_state_key: str = "joint"
    token: str | None = None


@dataclass(slots=True)
class LeaderConfig:
    joint_count: int
    scale: float = 1.0
    offset: float = 0.0


@dataclass(slots=True)
class CollectionConfig:
    hz: int
    max_steps: int
    output_dir: str
    command_speed: float = 20.0
    command_acc: float = 20.0


@dataclass(slots=True)
class TrainingConfig:
    dataset_dir: str
    model_dir: str
    hidden_dim: int
    batch_size: int
    epochs: int
    learning_rate: float
    val_split: float = 0.1
    seed: int = 42


@dataclass(slots=True)
class InferenceConfig:
    hz: int = 30


@dataclass(slots=True)
class PipelineConfig:
    realman: RealmanConfig
    leader: LeaderConfig
    collection: CollectionConfig
    training: TrainingConfig
    inference: InferenceConfig


def _section(raw: dict[str, Any], key: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    value = raw.get(key)
    if value is None:
        return default or {}
    return value


def load_config(path: str | Path) -> PipelineConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return PipelineConfig(
        realman=RealmanConfig(**_section(raw, "realman")),
        leader=LeaderConfig(**_section(raw, "leader")),
        collection=CollectionConfig(**_section(raw, "collection")),
        training=TrainingConfig(**_section(raw, "training")),
        inference=InferenceConfig(**_section(raw, "inference")),
    )
