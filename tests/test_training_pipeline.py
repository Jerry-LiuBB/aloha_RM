from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

if importlib.util.find_spec("numpy") is None:
    pytest.skip("numpy is not installed in this environment", allow_module_level=True)
if importlib.util.find_spec("torch") is None:
    pytest.skip("torch is not installed in this environment", allow_module_level=True)

import numpy as np

H5PY_AVAILABLE = importlib.util.find_spec("h5py") is not None
if H5PY_AVAILABLE:
    import h5py

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


def test_npz_dataset_with_images(tmp_path: Path) -> None:
    data_dir = tmp_path / "datasets_img"
    data_dir.mkdir()

    obs = np.random.randn(8, 6).astype(np.float32)
    act = np.random.randn(8, 6).astype(np.float32)
    images = np.random.randint(0, 255, size=(8, 16, 16, 3), dtype=np.uint8)
    ts = np.linspace(0, 1, 8)
    img_ts = np.linspace(0, 1, 8)

    np.savez_compressed(
        data_dir / "ep_img.npz",
        observations=obs,
        actions=act,
        timestamps=ts,
        images=images,
        image_timestamps=img_ts,
    )

    ds = EpisodeDataset(str(data_dir), include_images=True)
    sample_obs, sample_act = ds[0]
    assert sample_act.numel() == 6
    assert sample_obs.numel() == 6 + 16 * 16 * 3


def test_hdf5_dataset_loading(tmp_path: Path) -> None:
    if not H5PY_AVAILABLE:
        pytest.skip("h5py is not installed in this environment")

    data_dir = tmp_path / "datasets_h5"
    data_dir.mkdir()

    obs = np.random.randn(16, 6).astype(np.float32)
    act = (obs * 0.5).astype(np.float32)
    ts = np.linspace(0, 1, 16)
    images = np.random.randint(0, 255, size=(16, 8, 8, 3), dtype=np.uint8)
    with h5py.File(data_dir / "ep1.hdf5", "w") as f:
        f.create_dataset("observations", data=obs)
        f.create_dataset("actions", data=act)
        f.create_dataset("timestamps", data=ts)
        f.create_dataset("images", data=images)

    ds = EpisodeDataset(str(data_dir))
    assert len(ds) == 16
    sample_obs, _ = ds[0]
    assert sample_obs.numel() == 6 + 8 * 8 * 3


def test_hdf5_dataset_loading_mobile_aloha_images(tmp_path: Path) -> None:
    if not H5PY_AVAILABLE:
        pytest.skip("h5py is not installed in this environment")

    data_dir = tmp_path / "datasets_h5_mobile"
    data_dir.mkdir()

    obs = np.random.randn(10, 6).astype(np.float32)
    act = (obs * 0.5).astype(np.float32)
    ts = np.linspace(0, 1, 10)
    images = np.random.randint(0, 255, size=(10, 8, 8, 3), dtype=np.uint8)
    with h5py.File(data_dir / "ep1.hdf5", "w") as f:
        f.create_dataset("observations", data=obs)
        f.create_dataset("actions", data=act)
        f.create_dataset("timestamps", data=ts)
        g = f.require_group("observations/images")
        g.create_dataset("wrist", data=images)

    ds = EpisodeDataset(str(data_dir))
    sample_obs, _ = ds[0]
    assert sample_obs.numel() == 6 + 8 * 8 * 3
