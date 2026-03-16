from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    h5py = None


class EpisodeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, dataset_dir: str, include_images: bool = True) -> None:
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.include_images = include_images

        for path in sorted(Path(dataset_dir).glob("*.npz")):
            data = np.load(path)
            obs = data["observations"]
            act = data["actions"]
            images = data["images"] if include_images and "images" in data else None
            for idx, (o, a) in enumerate(zip(obs, act)):
                if images is not None:
                    image_flat = images[idx].astype(np.float32).reshape(-1) / 255.0
                    o = np.concatenate([o.astype(np.float32), image_flat], axis=0)
                self.samples.append((o.astype(np.float32), a.astype(np.float32)))

        if h5py is not None:
            for path in sorted(Path(dataset_dir).glob("*.hdf5")):
                with h5py.File(path, "r") as data:
                    obs = np.asarray(data["observations"])
                    act = np.asarray(data["actions"])
                    images = np.asarray(data["images"]) if include_images and "images" in data else None
                    for idx, (o, a) in enumerate(zip(obs, act)):
                        if images is not None:
                            image_flat = images[idx].astype(np.float32).reshape(-1) / 255.0
                            o = np.concatenate([o.astype(np.float32), image_flat], axis=0)
                        self.samples.append((o.astype(np.float32), a.astype(np.float32)))

        if not self.samples:
            raise ValueError(f"No .npz or .hdf5 episodes found in {dataset_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        o, a = self.samples[idx]
        return torch.from_numpy(o), torch.from_numpy(a)
