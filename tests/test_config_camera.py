from __future__ import annotations

from pathlib import Path

from aloha_rm.config import load_config


def test_load_camera_realsense_config(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pipeline.yaml"
    cfg_file.write_text(
        """
realman:
  host: "127.0.0.1"
  port: 8080
leader:
  joint_count: 6
collection:
  hz: 30
  max_steps: 10
  output_dir: "artifacts/datasets"
training:
  dataset_dir: "artifacts/datasets"
  model_dir: "artifacts/models"
  hidden_dim: 128
  batch_size: 16
  epochs: 2
  learning_rate: 0.001
camera:
  enabled: true
  backend: "realsense"
  width: 640
  height: 480
  fps: 30
  serial_no: "ABC123"
inference:
  hz: 20
""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_file)
    assert cfg.camera.enabled is True
    assert cfg.camera.backend == "realsense"
    assert cfg.camera.width == 640
    assert cfg.camera.height == 480
    assert cfg.camera.fps == 30
    assert cfg.camera.serial_no == "ABC123"
