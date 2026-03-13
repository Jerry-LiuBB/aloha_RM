from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from aloha_rm.training.dataset import EpisodeDataset
from aloha_rm.training.model import BCMLP


def train_bc(
    dataset_dir: str,
    model_dir: str,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    val_split: float = 0.1,
    seed: int = 42,
) -> Path:
    random.seed(seed)
    torch.manual_seed(seed)

    dataset = EpisodeDataset(dataset_dir)
    val_count = int(len(dataset) * val_split)
    train_count = len(dataset) - val_count
    train_set, val_set = random_split(dataset, [train_count, val_count])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False) if val_count > 0 else None

    obs0, act0 = dataset[0]
    model = BCMLP(obs_dim=obs0.numel(), act_dim=act0.numel(), hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    history: list[dict[str, float]] = []
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for obs, act in train_loader:
            pred = model(obs)
            loss = loss_fn(pred, act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * obs.shape[0]

        train_loss = loss_sum / max(len(train_set), 1)

        val_loss = 0.0
        if val_loader:
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for obs, act in val_loader:
                    pred = model(obs)
                    val_loss_sum += loss_fn(pred, act).item() * obs.shape[0]
            val_loss = val_loss_sum / len(val_set)
            best_val = min(best_val, val_loss)
            print(f"epoch={epoch + 1}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        else:
            best_val = min(best_val, train_loss)
            print(f"epoch={epoch + 1}/{epochs} train_loss={train_loss:.6f}")

        history.append({"epoch": float(epoch + 1), "train_loss": train_loss, "val_loss": val_loss})

    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bc_mlp.pt"
    torch.save(model.state_dict(), out_path)

    metrics = {
        "samples": len(dataset),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "best_val_loss": best_val,
        "history": history,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return out_path
