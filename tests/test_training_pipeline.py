from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

if importlib.util.find_spec("numpy") is None:
    pytest.skip("numpy is not installed in this environment", allow_module_level=True)
if importlib.util.find_spec("torch") is None:
    pytest.skip("torch is not installed in this environment", allow_module_level=True)

import numpy as np

from aloha_rm.training.dataset import EpisodeDataset
from aloha_rm.training.train import train_bc


def test_dataset_and_training(tmp_path: Path) -> None:
    data_dir = tmp_path / "datasets"
    model_dir = tmp_path / "models"
    data_dir.mkdir()

    obs = np.random.randn(32, 6).astype(np.float32)
    act = (obs * 0.7).astype(np.float32)
    ts = np.linspace(0, 1, 32)
    np.savez_compressed(data_dir / "ep1.npz", observations=obs, actions=act, timestamps=ts)

    ds = EpisodeDataset(str(data_dir))
    assert len(ds) == 32

    model_path = train_bc(
        dataset_dir=str(data_dir),
        model_dir=str(model_dir),
        hidden_dim=32,
        batch_size=8,
        epochs=2,
        learning_rate=1e-3,
        val_split=0.2,
        seed=123,
    )
    assert model_path.exists()
    assert (model_dir / "metrics.json").exists()
