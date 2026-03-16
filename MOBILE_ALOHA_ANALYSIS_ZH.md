# mobile-aloha 仓库功能分析（结合当前 `aloha_RM` 实现）

> 说明：当前工作区代码是 `aloha_RM`，它实现的是一个与 ALOHA/mobile-aloha 思路一致的最小闭环：
> **示教采集 -> 行为克隆训练 -> 在线推理控制机械臂**。

## 1. 仓库核心能力

`aloha_RM` 的 README 明确了三段式流程：
1) 实时遥操作采集；
2) 训练行为克隆策略；
3) 部署模型闭环执行任务。  
对应产物包括 `.npz` 数据集、`metrics.json`、模型权重文件。  

## 2. 数据采集（Teleoperation）怎么实现

### 2.1 采集循环
`EpisodeCollector.collect()` 在固定频率下循环执行：
- 从主动臂读取关节（leader sample）；
- 把关节命令发给从动臂 `movej`；
- 回读从动臂关节状态；
- 记录 observation / action / timestamp / command_ok。  

这就是标准的 imitation learning 数据采集结构：
- `obs`: 从动臂状态；
- `act`: 人演示动作（由主动臂产生）；
- 并附带时间戳与指令成功标记。

### 2.2 数据落盘格式
每个 episode 会写出：
- `xxx.npz`: `observations`, `actions`, `timestamps`, `command_ok`；
- `xxx.json`: 采样频率、步数、张量形状、命令成功率等元信息。  

这使得后续训练可以直接按监督学习样本读取。

## 3. 机器人接口层（Follower/Leader）怎么抽象

### 3.1 RealmanClient（从动臂）
`RealmanClient` 是 JSON API 客户端，支持：
- 可配置 host/port 与 API path；
- `movej` 下发关节指令（带 speed/acc）；
- `get_joint_state` 拉取当前关节状态；
- 可配置成功码、关节字段键名、鉴权 token。  

这类可配置适配是把同一训练/部署逻辑复用于不同控制器固件版本的关键。

### 3.2 ServoLeaderArm（主动臂）
`ServoLeaderArm` 当前实现中 `read_joint_degrees()` 是占位信号（正弦波），
设计上要求你替换为真实串口/CAN 读取，再由 `sample()` 统一转弧度输出。  

这说明代码把“硬件接入点”清晰隔离在 leader adapter 中。

## 4. 训练模块（Behavior Cloning）怎么实现

### 4.1 Dataset
`EpisodeDataset` 会遍历 `dataset_dir` 下所有 `.npz`，按时间步展开为 `(obs, act)` 样本对。  

### 4.2 模型
`BCMLP` 是三层 MLP（Linear-ReLU-Linear-ReLU-Linear），
输入维度=`obs_dim`，输出维度=`act_dim`。  

### 4.3 训练逻辑
`train_bc()` 主要步骤：
- 按 `val_split` 划分训练/验证；
- MSELoss + Adam 优化；
- 记录每个 epoch 的 train/val loss；
- 导出 `bc_mlp.pt` 与 `metrics.json`。  

这是典型的离线 imitation learning baseline pipeline。

## 5. 推理部署（闭环执行）怎么实现

`PolicyRunner.run()` 在固定频率循环：
- 拉取当前关节状态 `obs`；
- 模型前向得到动作 `action`；
- 发送 `movej(action)`；
- 通过 `dt` 控制频率。  

脚本入口 `scripts/run_policy.py` 会先从机器人读一次状态确定 `obs_dim`，
并使用配置中的 `leader.joint_count` 作为 `act_dim` 构建网络。

## 6. 配置与脚本组织

- 统一配置文件：`configs/pipeline.yaml`；
- 三个入口脚本：
  - `collect_data.py`
  - `train_policy.py`
  - `run_policy.py`  

这种组织非常接近“可复现实验流水线”：采集、训练、部署可独立运行。

## 7. 总结：这个仓库“实现了什么”

如果从功能视角归纳，当前实现完成了一个 **ALOHA 风格最小可用闭环系统**：

1. **人机示教采集**：主动臂驱动从动臂并记录监督数据；
2. **监督学习训练**：用行为克隆把状态映射到动作；
3. **在线闭环执行**：模型实时读状态并输出控制命令；
4. **硬件适配可插拔**：通过 Leader/Follower adapter 隔离真实设备细节。  

它的定位更像“工程骨架/教学版基线”，便于你后续替换真实硬件接口、
扩展视觉输入、多模态策略（如 diffusion policy）或更复杂控制器。
