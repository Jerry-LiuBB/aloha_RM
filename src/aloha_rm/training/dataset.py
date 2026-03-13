from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EpisodeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, dataset_dir: str) -> None:
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        for path in sorted(Path(dataset_dir).glob("*.npz")):
            data = np.load(path)
            obs = data["observations"]
            act = data["actions"]
            for o, a in zip(obs, act):
                self.samples.append((o.astype(np.float32), a.astype(np.float32)))

        if not self.samples:
            raise ValueError(f"No .npz episodes found in {dataset_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        o, a = self.samples[idx]
        return torch.from_numpy(o), torch.from_numpy(a)
