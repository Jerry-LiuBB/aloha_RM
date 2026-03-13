# aloha_RM

这个仓库用于复现你要的闭环流程：**舵机主动臂控制 Realman 从动臂采集数据 → 训练策略 → 模型驱动执行任务**。

## 功能总览

- 实时遥操作采集：主动臂读关节角、下发 Realman `movej`、回读关节状态。
- 支持相机图像采集：每一步可同步保存 RGB 图像到 episode 数据。
- 数据集产出：`npz`（obs/action/timestamp/command_ok/images）+ `json` 元数据。
- 训练：行为克隆 MLP，带 train/val 划分和 `metrics.json` 指标导出。
- 部署：加载模型后按固定频率闭环推理并下发 Realman。

## 目录

- `scripts/collect_data.py`：采集脚本（含相机采集开关）。
- `scripts/train_policy.py`：训练脚本。
- `scripts/run_policy.py`：策略部署脚本。
- `src/aloha_rm/follower/realman_client.py`：Realman JSON API 客户端。
- `src/aloha_rm/leader/servo_leader.py`：主动臂舵机读数接口（你要接入真实硬件）。
- `src/aloha_rm/camera/`：相机接口（`MockCamera` / `OpenCVCamera`）。
- `src/aloha_rm/teleop/collector.py`：遥操作采集器。
- `src/aloha_rm/training/`：数据集、模型、训练。
- `src/aloha_rm/inference/policy_runner.py`：在线策略运行。
- `configs/pipeline.yaml`：全局配置。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 配置

编辑 `configs/pipeline.yaml`：

- `realman.host/port`：从动臂控制器地址。
- `realman.movej_api/state_api`：接口路径（按你的固件/网关修改）。
- `realman.success_code/joint_state_key`：适配不同 JSON 返回格式。
- `realman.token`：如有鉴权可填。
- `leader.joint_count`：主动臂关节数。
- `collection.command_speed/command_acc`：下发运动参数。
- `camera.enabled`：是否记录图像。
- `camera.backend`：`mock` 或 `opencv`。

## 1) 采集数据（含图像）

```bash
python scripts/collect_data.py --episode pick_place_001
```

输出：

- `artifacts/datasets/pick_place_001.npz`
  - `observations`: 关节状态
  - `actions`: 控制动作
  - `timestamps`: 时间戳
  - `command_ok`: 每步命令是否成功
  - `images`: RGB 图像（当 `camera.enabled=true`）
- `artifacts/datasets/pick_place_001.json`

## 2) 训练模型

```bash
python scripts/train_policy.py
```

输出：

- `artifacts/models/bc_mlp.pt`
- `artifacts/models/metrics.json`

## 3) 模型驱动执行

```bash
python scripts/run_policy.py --model artifacts/models/bc_mlp.pt --steps 300
```

## 需要你替换的硬件代码

### 主动臂（舵机）

在 `ServoLeaderArm.read_joint_degrees()` 里接入你的串口/CAN 总线读数。

### 相机

- `camera.backend=opencv` 时使用 USB 相机。
- 如果你有工业相机 SDK，建议新增 `camera/<your_camera>.py` 并实现 `capture_rgb()`。

### Realman 从动臂

`RealmanClient` 已支持常见 JSON 字段配置。你可按 Realman 文档微调 payload 与返回字段映射。

## 上传到你的 GitHub 仓库

先配置远端，再推送：

```bash
git remote add origin <你的仓库URL>
git push -u origin work
```

如果你希望推到 `main`：

```bash
git branch -M main
git push -u origin main
```
