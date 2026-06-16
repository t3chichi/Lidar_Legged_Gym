# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Extended Isaac Gym Environments for Legged Robots 是基于 NVIDIA Isaac Gym 的腿式机器人强化学习训练框架，扩展自 [legged_gym](https://github.com/leggedrobotics/legged_gym)，并作为 [PegasusFlow](https://github.com/MasterYip/PegasusFlow) 的子模块使用。

主要特性：
- rsl_rl 3.3.0 支持（从 1.0.2 升级）
- NVIDIA Warp SDF 和光线投射集成
- 主环境-滚轮环境（Main-Rollout）架构，支持基于采样的方法
- 受限地形生成和 OBJ 地形支持
- 激光雷达传感器模块（LidarSensor）和 PD-RiskNet 感知策略
- 师生蒸馏训练框架

## 安装和设置

```bash
# 前置条件：安装 NVIDIA Isaac Gym 并设置环境变量
export ISAACGYM_PATH=/path/to/isaacgym
export PYTHONPATH=$ISAACGYM_PATH/python:$PYTHONPATH

# 激活 conda 环境
conda activate li_leggym

# 安装项目组件
pip install -e legged_gym/
pip install -e rsl_rl/
pip install -e LidarSensor/   # 激光雷达功能
```

核心依赖定义在 `legged_gym/setup.py`（isaacgym, torch, warp-lang==1.7.0 等）。

## 常用命令

### 训练
```bash
python legged_gym/legged_gym/scripts/train.py --task=<task_name> [options]

# 常用选项：
#   --num_envs=<N>      环境并行数（默认由配置决定，4096 训练，50 测试）
#   --resume            从检查点恢复训练
#   --headless          无 GUI 模式
#   --rl_device=cuda:0  RL 算法设备
#   --sim_device=cuda:0 物理仿真设备
#   --seed=<N>          随机种子
#   --max_iterations=<N> 最大训练迭代数
#   --experiment_name=<name> 实验名称
#   --run_name=<name>   运行名称
#   --load_run=<name>   加载指定运行
#   --checkpoint=<N>    加载指定 checkpoint 编号（-1 为最新）
```

示例：
```bash
python legged_gym/legged_gym/scripts/train.py --task=go2_lidar_pd_risknet --num_envs=4096 --headless --resume
```

### 测试/演示
```bash
python legged_gym/legged_gym/scripts/play.py --task=<task_name> --num_envs=50 --checkpoint=-1
```

`play.py` 会自动导出策略为 Torch JIT（`logs/<experiment_name>/exported/policies/policy_1.pt`），用于 C++ 部署。

### 运行测试
```bash
python -m pytest legged_gym/legged_gym/tests/test_env.py -v
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
```

测试文件位于 `legged_gym/legged_gym/tests/`：
- `test_env.py` — 基础环境测试
- `test_batch_rollout_env.py` — 批量滚轮环境测试
- `test_confined_terrain.py` — 受限地形测试
- `test_go2_lidar_pd_risknet_math.py` — PD-RiskNet 数学函数测试
- `test_franka.py`, `test_gymapi.py`, `test_env_simstep_time.py` — 其他专项测试

### 代码质量
```bash
pip install isort
isort --check-only --diff rsl_rl/  # 检查导入排序
isort rsl_rl/                       # 自动修复
```

## 项目架构

### 顶层目录结构
```
Lidar_legged_gym/
├── legged_gym/          # Isaac Gym 环境实现（核心）
│   └── legged_gym/
│       ├── envs/        # 机器人环境（anymal_c, go2, elspider_air 等）
│       │   ├── base/    # 基类：LeggedRobot, LeggedRobotRaycast, LeggedRobotDepthCam
│       │   ├── batch_rollout/  # 主-滚轮环境架构基类
│       │   └── <robot>/        # 各机器人专用环境
│       ├── scripts/     # train.py, play.py
│       ├── utils/       # task_registry, terrain, helpers 等工具
│       └── tests/       # 测试文件
├── rsl_rl/              # 强化学习算法
│   └── rsl_rl/
│       ├── algorithms/  # PPO（支持辅助监督损失）
│       ├── modules/     # Actor-Critic, PDRiskNet, StudentTeacher 等网络
│       ├── runners/     # OnPolicyRunner
│       ├── storage/     # RolloutStorage
│       └── utils/       # 工具函数
├── LidarSensor/         # 激光雷达传感器模块（Warp+Taichi 后端）
│   └── LidarSensor/
│       ├── lidar_sensor.py
│       ├── base_sensor.py
│       ├── sensor_config/      # 传感器配置（旋转式/固态 LiDAR, 深度相机）
│       ├── sensor_kernels/     # Taichi kernels
│       └── sensor_pattern/     # 扫描模式
└── doc/                 # 文档资源（图片, 视频）
```

### 任务注册系统

任务在 `legged_gym/envs/__init__.py` 中注册。每个任务包含三个部分：
1. 环境类（继承 `LeggedRobot` 或变体）
2. 环境配置（继承 `LeggedRobotCfg`）
3. 训练配置（继承 `LeggedRobotCfgPPO`）

注册方式：
```python
task_registry.register("task_name", EnvClass, EnvCfg(), TrainCfgPPO())
```

**注册的任务列表**（完整列表见 `__init__.py`）：
- `anymal_c_rough`, `anymal_c_flat` — ANYmal-C 基础地形
- `anymal_c_nav`, `anymal_c_barrier_nav`, `anymal_c_timberpile_nav` — ANYmal-C 导航
- `anymal_c_batch_rollout` — ANYmal-C 滚轮架构
- `anymal_c_rough_teacher/student` — 师生蒸馏
- `go2_flat`, `go2_rough` — Go2 基础
- `go2_lidar_pd_risknet`, `go2_pd_pretrain`, `go2_lidar_pd_risknet_4090` — Go2 激光雷达感知
- `elspider_air_rough`, `elspider_air_flat`, `elspider_air_nav` — ElSpider Air
- `cassie`, `a1`, `franka` — 其他机器人
- `cyber2_stand`, `cyber2_walk` — Cyberdog2（可选依赖）
- `go2_traj_grad_sampling`, `elspider_air_plan_grad_sampling` — 轨迹梯度采样（可选依赖）

### 基础环境变体

`legged_gym/envs/base/` 提供多个基础类：
- **`legged_robot.py`** — 标准腿式机器人基类（核心）
- **`legged_robot_raycast.py`** — 集成 Warp 光线投射的版本
- **`legged_robot_depthcam.py`** — 集成深度相机的版本
- **`base_pose_adapt.py`** — 位姿自适应控制基类

### 滚轮环境架构（Batch Rollout）

`legged_gym/envs/batch_rollout/` 实现了主环境-滚轮环境架构，用于基于采样的运动规划方法（如 MPPI, DialMPC）：
- `robot_batch_rollout.py` — 滚轮环境基类
- `robot_traj_grad_sampling.py` — 轨迹梯度采样
- `robot_plan_grad_sampling.py` — 规划梯度采样
- `robot_batch_rollout_nav.py` — 导航版滚轮环境

### 地形系统

`legged_gym/utils/` 中的地形工具：
- `terrain.py` — 基础地形生成（阶梯、斜坡、离散障碍等）
- `terrain_confine.py` — 受限地形生成（走廊、房间等）
- `terrain_obj.py` — OBJ 文件地形加载支持

### 网络模块（rsl_rl）

`rsl_rl/rsl_rl/modules/`：
- `actor_critic.py` — 标准 Actor-Critic（MLP）
- `actor_critic_recurrent.py` — 循环版（GRU）
- `pd_risknet_actor_critic.py` — PD-RiskNet（近端-远端激光雷达感知网络）
- `student_teacher.py` — 师生蒸馏网络
- `depth_backbone.py` — 深度图像骨干网络
- `rnd.py` — 随机网络蒸馏（RND）
- `terrain_estimator.py` — 地形估计器

## PD-RiskNet（激光雷达感知策略）

PD-RiskNet 是 Omni-Perception 框架的核心感知模块（详见 `rsl_rl/modules/pd_risknet_actor_critic.py`）：

- **近端路径**：最远点采样（FPS）+ GRU，受特权高度监督
- **远端路径**：平均下采样 + GRU
- 时间序列历史长度：`N_hist = 10` 帧
- Actor 隐藏层：`[1024, 512, 256, 128]`，ELU 激活
- 训练时使用辅助损失（`get_auxiliary_loss()`）监督近端特征

**设计文档：**
- `OmniPeception.md` — Omni-Perception 论文全文（方法、实验、消融、域随机化参数）,应重点关注,是复现论文时的唯一参考.
- `特权监督.md` — 特权监督模块详细设计,不是参考,只是对当前代码的描述,每次修改代码时应同步修改
- `避障函数设计.md` — 对越障情况下的避障速度与距离奖励的初步设计构想（含硬约束和软奖励两版方案）非论文设计,当前实现时不应将其作为参考

配置示例在 `legged_gym/envs/go2/lidar_pd_risknet/`：
- `go2_lidar_pd_risknet_config.py` — 基础配置
- `go2_lidar_pd_risknet_4090_config.py` — 4090 GPU 优化配置
- `go2_pd_pretrain_config.py` — 预训练配置

### 师生蒸馏训练

支持教师-学生蒸馏：
```python
# 配置中设置
policy_class_name = 'StudentTeacher'
algorithm_class_name = 'Distillation'
```
- 教师策略使用特权信息（真值高度、地形属性等）
- 学生策略仅使用可部署传感器
- 相关任务：`anymal_c_rough_teacher`, `anymal_c_rough_student`

## 关键文件索引

| 职责 | 路径 |
|------|------|
| 任务注册 | `legged_gym/envs/__init__.py` |
| 基础环境类 | `legged_gym/envs/base/legged_robot.py` |
| 配置基类 | `legged_gym/envs/base/legged_robot_config.py` |
| 训练入口 | `legged_gym/scripts/train.py` |
| 测试入口 | `legged_gym/scripts/play.py` |
| 参数配置 | `legged_gym/utils/helpers.py` |
| PPO 算法 | `rsl_rl/algorithms/ppo.py` |
| 训练运行器 | `rsl_rl/runners/on_policy_runner.py` |
| PD-RiskNet 网络 | `rsl_rl/modules/pd_risknet_actor_critic.py` |
| Laser 环境 | `legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py` |
| LiDAR 传感器 | `LidarSensor/LidarSensor/lidar_sensor.py` |
| 可视化工具 | `legged_gym/utils/gym_visualizer.py` |
| 奖励函数实现 | `legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py` (`compute_reward` 方法) |

## 开发指南

### 添加新机器人
1. 在 `legged_gym/envs/` 下创建机器人目录
2. 创建机器人类（继承 `LeggedRobot`，或 `LeggedRobotRaycast`/`LeggedRobotDepthCam`）
3. 创建配置类（继承 `LeggedRobotCfg` 和 `LeggedRobotCfgPPO`）
4. 在 `__init__.py` 中导入并注册任务

### 策略导出
训练完成后通过 `play.py` 自动导出 JIT 模型，或手动调用：
```python
from legged_gym.utils.helpers import export_policy_as_jit
export_policy_as_jit(actor_critic, output_path)
```
支持标准 MLP（`policy_1.pt`）和 LSTM（`policy_lstm_1.pt`）导出。

## 注意事项

1. **GPU 内存管理**：`height_samples` 变量可能占用大量 GPU 内存。减少 `num_envs` 可缓解。
2. **刚体状态视图**：启用障碍物时 `sim_num_bodies` 可能超过机器人身体数量，使用 `rigid_body_state.view(self.num_envs, self.sim_num_bodies, 13)` 正确访问。
3. **环境间距**：滚轮环境中机器人聚集可能导致 `PxgDynamicsMemoryConfig::foundLostAggregatePairsCapacity` 警告，需设置不同的初始位置。
4. **可选依赖**：`traj_sampling`（轨迹采样）和 `tqdm`（cyberdog2）为可选包，缺失时自动跳过对应任务注册。
5. **conda 环境**：建议使用 `conda activate li_leggym`。
6. **Isaac Gym**：需正确安装并配置 `ISAACGYM_PATH` / `PYTHONPATH`。
