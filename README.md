# aloha_RM

这个仓库用于复现你要的闭环流程：**舵机主动臂控制 Realman 从动臂采集数据 → 训练策略 → 模型驱动执行任务**。

## 功能总览

- 实时遥操作采集：主动臂读关节角、下发 Realman `movej`、回读关节状态。
- 双相机采集：支持两路 RealSense D435（腕部视角 + 外部全局视角），以相机时间戳作为数据主时间基准。
- 数据集产出：`hdf5`（默认）或 `npz`，兼容 mobile-aloha/act-plus-plus 常见字段。
- 训练：行为克隆 MLP，带 train/val 划分和 `metrics.json` 指标导出。
- 部署：加载模型后按固定频率闭环推理并下发 Realman。

## 目录

- `scripts/collect_data.py`：采集脚本。
- `scripts/train_policy.py`：训练脚本。
- `scripts/run_policy.py`：策略部署脚本。
- `src/aloha_rm/follower/realman_client.py`：Realman JSON API 客户端。
- `src/aloha_rm/leader/servo_leader.py`：主动臂舵机读数接口（你要接入真实硬件）。
- `src/aloha_rm/sensors/realsense_camera.py`：RealSense D435 相机封装（支持 serial 绑定与硬件时间戳）。
- `src/aloha_rm/teleop/collector.py`：遥操作采集器（机械臂 + 双相机 + 时间同步）。
- `src/aloha_rm/training/`：数据集、模型、训练。
- `src/aloha_rm/inference/policy_runner.py`：在线策略运行。
- `configs/pipeline.yaml`：全局配置。

## 你的硬件接口映射

### 主臂（Servo）

在 `ServoLeaderArm.read_joint_degrees()` 中对接你的主臂代码（如 `shadow_rm_robot/servo_robotic_arm.py` 的读数逻辑），返回每个关节角（单位：度）。

### 从臂（Realman）

- 关节状态读取：`RealmanClient.get_joint_state()`
- 角度透传控制：`RealmanClient.movej()`（movej-canfd 对应透传 payload）

可根据你控制器固件调整 `movej_api` / `state_api` / `joint_state_key`。

### 相机（2x D435）

在 `camera` 配置中填写：

- `wrist_serial_no`: 机械臂上相机序列号
- `external_serial_no`: 外部全景相机序列号
- `fps: 30`

采集时主时间基准默认使用主相机（`base_camera_name`）时间戳。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

如需 RealSense D435 支持：

```bash
pip install -e .[camera]
```

## 配置

编辑 `configs/pipeline.yaml`：

- `collection.hz`: 50（主从机械臂）
- `camera.fps`: 30（D435）
- `collection.dataset_format`: `hdf5`（默认）
- `camera.wrist_serial_no` / `camera.external_serial_no`: 双相机序列号

## 1) 采集数据（相机时间为基准）

```bash
python scripts/collect_data.py --episode pick_place_001
```

输出：

- `artifacts/datasets/pick_place_001.hdf5`
- `artifacts/datasets/pick_place_001.json`

HDF5 关键字段：

- `observations`
- `actions`
- `timestamps`（主时间轴，默认是主相机时间戳）
- `command_ok`
- `observations/images/wrist`（主相机）
- `observations/images/external`（若启用第二相机）
- `image_timestamps`
- `secondary_image_timestamps`（若启用第二相机）

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

如果训练时包含图像特征，部署时也应开启 `camera.enabled=true` 并保持相机分辨率一致。

## 上传到你的 GitHub 仓库

```bash
git remote add origin <你的仓库URL>
git push -u origin work
```
