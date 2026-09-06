# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Extended Isaac Gym Environments for Legged Robots 是基于 NVIDIA Isaac Gym 的腿式机器人强化学习训练框架，扩展自 [legged_gym](https://github.com/leggedrobotics/legged_gym)，并作为 [PegasusFlow](https://github.com/MasterYip/PegasusFlow) 的子模块使用。

主要特性：
- NVIDIA Warp SDF 和光线投射集成
- 主环境-滚轮环境（Main-Rollout）架构，支持基于采样的方法
- 受限地形生成和 OBJ 地形支持
- 包内激光雷达模块（`legged_gym/utils/LidarSensor/`，Warp 后端）与 PD-GRU 感知策略（`LidarPDActorCritic`）
- 师生蒸馏训练框架
- 2026-09 完成与实验室仓库 `el4090_legged_gym` 的同步：base/envs/utils/rsl_rl 采用实验室新版本，任务 `pd_gru_lidar` 移植为本仓库的主激光雷达任务线（见"EL_4090 PD-GRU LiDAR"一节）。

## 安装和设置

```bash
# 前置条件：安装 NVIDIA Isaac Gym 并设置环境变量
export ISAACGYM_PATH=/path/to/isaacgym
export PYTHONPATH=$ISAACGYM_PATH/python:$PYTHONPATH

# 激活 conda 环境
conda activate li_leggym

# 安装项目组件（LidarSensor 已并入包内，无需单独安装）
pip install -e legged_gym/
pip install -e rsl_rl/
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

`train.py` 会把 `env_cfg_snapshot.json` / `train_cfg_snapshot.json` 转储到运行目录。

示例：
```bash
python legged_gym/legged_gym/scripts/train.py --task=el4090_lidar_tripod2_low_avoid --num_envs=4096 --headless --resume
```

### 测试/演示
```bash
python legged_gym/legged_gym/scripts/play.py --task=<task_name> --num_envs=50 --checkpoint=-1
# 笔记本演示（无限 viewer 循环，回合边界复位策略状态）：play_laptop.py
```

`play.py` 会自动导出策略为 Torch JIT（`logs/<experiment_name>/exported/policies/policy_1.pt`），用于 C++ 部署。注意：对 `LidarPDActorCritic` 仅导出 actor MLP，双 GRU 感知结构不支持 JIT 导出。

### 运行测试
```bash
python -m pytest legged_gym/legged_gym/tests/test_pointcloud_geometry.py -v
python -m pytest legged_gym/legged_gym/tests/test_confined_terrain.py -v
```

测试文件位于 `legged_gym/legged_gym/tests/`：
- `test_pointcloud_geometry.py` — 点云球坐标排序/切分测试
- `test_confined_terrain.py` — 受限地形测试
- `test_batch_rollout_env.py`, `test_franka.py`, `test_gymapi.py`, `test_env_simstep_time.py` — 其他专项测试
- `test_env.py` 为脚本型测试（需命令行参数），不能直接用 pytest 运行

## 项目架构

### 顶层目录结构
```
Lidar_legged_gym/
├── legged_gym/          # Isaac Gym 环境实现（核心）
│   ├── legged_gym/
│   │   ├── envs/        # 机器人环境（anymal_c, go2, elspider_air, el_4090 等）
│   │   │   ├── base/    # 基类：LeggedRobot(+RewMixin), LeggedRobotRaycast, LeggedRobotDepthCam
│   │   │   ├── el_4090/pd_gru_lidar/  # EL_4090 双 GRU 激光雷达任务（主任务线）
│   │   │   └── <robot>/        # 各机器人专用环境
│   │   ├── scripts/     # train.py, play.py, play_laptop.py
│   │   ├── utils/       # task_registry, terrain, LidarSensor/ 等工具
│   │   └── tests/       # 测试文件
│   └── resources/       # URDF/mesh（含 resources/robots/el_4090/）
├── rsl_rl/              # 强化学习算法（与实验室仓库同步的版本）
│   └── rsl_rl/
│       ├── algorithms/  # PPO（辅助监督损失、对称增广、AMP）
│       ├── modules/     # ActorCritic, LidarPDActorCritic, StudentTeacher 等
│       ├── runners/     # OnPolicyRunner（内置 LidarWrapper 观测重排）
│       ├── storage/     # RolloutStorage（支持 aux_observations）
│       └── utils/       # lidar_wrapper.py 等工具
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
- `el4090_lidar`, `el4090_lidar_tripod2_low`, `el4090_lidar_tripod2_low_avoid` — EL_4090 PD-GRU 激光雷达（主任务线）
- `anymal_c_rough`, `anymal_c_flat` — ANYmal-C 基础地形
- `anymal_c_nav`, `anymal_c_barrier_nav`, `anymal_c_timberpile_nav` — ANYmal-C 导航
- `anymal_c_batch_rollout` — ANYmal-C 滚轮架构
- `anymal_c_rough_teacher/student` — 师生蒸馏
- `go2_flat`, `go2_rough` — Go2 基础
- `elspider_air_rough`, `elspider_air_flat`, `elspider_air_nav` — ElSpider Air
- `cassie`, `a1`, `franka` — 其他机器人
- `cyber2_stand`, `cyber2_walk` — Cyberdog2（可选依赖）
- `go2_traj_grad_sampling`, `elspider_air_plan_grad_sampling` — 轨迹梯度采样（可选依赖）

历史任务 `el4090_lidar_exp` 与 go2 旧 PD 任务（`go2_lidar_pd_risknet` 等）已于 2026-09 移除（git 历史可找回）。

### 基础环境变体

`legged_gym/envs/base/` 提供多个基础类：
- **`legged_robot.py`** — 标准腿式机器人基类（核心；奖励在 `legged_robot_rew_mixin.py`，支持多阶段奖励课程 `multi_stage_rewards`）
- **`legged_robot_raycast.py`** — 集成 Warp 光线投射的版本
- **`legged_robot_depthcam.py`** — 集成深度相机的版本
- **`base_pose_adapt.py`** — 位姿自适应控制基类

### 滚轮环境架构（Batch Rollout）

`legged_gym/envs/batch_rollout/` 实现了主环境-滚轮环境架构，用于基于采样的运动规划方法（如 MPPI, DialMPC）：
- `robot_batch_rollout.py` — 滚轮环境基类
- `robot_traj_grad_sampling.py` — 轨迹梯度采样
- `robot_plan_grad_sampling.py` — 规划梯度采样

### 地形系统

`legged_gym/utils/` 中的地形工具：
- `terrain.py` — 地形生成（斜坡/阶梯/离散障碍/gap/**柱阵 pillar_field**/**正弦通道 sin_curve_channel**；尾部另保留 `curved_corridor_terrain`/`trapezoid_corridor_terrain` 供直接调用）
- `terrain_confine.py` — 受限地形生成（走廊、房间等）
- `terrain_obj.py` — OBJ 文件地形加载支持
- `pillar_mesh.py` — 激光可视柱阵 mesh（遗留工具，当前无任务引用）

### 网络模块（rsl_rl）

`rsl_rl/rsl_rl/modules/`：
- `actor_critic.py` — 标准 Actor-Critic（MLP）
- `actor_critic_recurrent.py` — 循环版（GRU）
- `lidar_pd_actor_critic.py` — LidarPDActorCritic（近端-远端双 GRU 激光雷达感知网络）
- `student_teacher.py` — 师生蒸馏网络
- `depth_backbone.py` — 深度图像骨干网络
- `rnd.py` — 随机网络蒸馏（RND）
- `terrain_estimator.py` — 地形估计器

## EL_4090 PD-GRU LiDAR（主激光雷达任务线）

由实验室仓库 `el4090_legged_gym` 的 `pd_gru_lidar` 任务移植。核心架构：

- **观测**：`obs_buf` 仅 66 维本体感知；激光点云（40×25=1000 点，base 系）经 `env.lidar_points_base` 交给 runner 侧 `LidarWrapper` 实时重排为 `66 + 256×3 + 10×64×3 = 2754` 维（近端 FPS+角排序，远端 10 帧环形历史）。
- **辅助监督**：env 写 `aux_obs_buf`（17×11 高度栅格），runner 自动发现，经 `compute_auxiliary_loss()`（height_supervisor）进 PPO 损失。
- **网络**：`LidarPDActorCritic` — 近端 GRU(3→187)、远端 GRU(3→64)，actor/critic MLP `[512,256,128]`，对称数据增广 + mirror loss + AMP。
- **对称函数**：字符串引用 `legged_gym.envs.el_4090.pd_gru_lidar.el_4090_lidar_symmetry:get_el4090_lidar_xsym_obs_act`。
- **env↔runner 契约**：`cfg.pd_risknet.enabled` 触发；env 需暴露 `lidar_points_base`、`_sensor_offset_quat`、`_sensor_translation`、（可选）`aux_obs_buf`。
- **雷达**：`legged_gym/utils/LidarSensor/`（warp 内核，`LidarType.SIMPLE_GRID`），域随机化：点遮蔽 0.02 + 距离噪声 0.02。
- **任务变体**：`el4090_lidar`（**基座设定**：固化同一 PD 网络共用的雷达/观测/辅助监督配置，供变体继承复用，本身不用于直接训练）、`_tripod2_low`（低蹲步态、平面、多阶段奖励课程）、`_tripod2_low_avoid`（柱阵 trimesh 避障、`terrain_proportions` 索引 7 = pillar_field）。

**设计文档（历史，描述迁移前的 go2 PDRiskNet 架构）：**
- `OmniPeception.md` — Omni-Perception 论文全文（方法、实验、消融、域随机化参数）
- `特权监督.md`、`避障函数设计.md` — 旧架构设计描述，仅作历史参考

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
| 基础环境类 | `legged_gym/envs/base/legged_robot.py`（+ `legged_robot_rew_mixin.py`） |
| 配置基类 | `legged_gym/envs/base/legged_robot_config.py` |
| 训练入口 | `legged_gym/scripts/train.py` |
| 测试/演示入口 | `legged_gym/scripts/play.py`, `play_laptop.py` |
| 参数配置 | `legged_gym/utils/helpers.py` |
| PPO 算法 | `rsl_rl/algorithms/ppo.py` |
| 训练运行器 | `rsl_rl/runners/on_policy_runner.py` |
| PD-GRU 网络 | `rsl_rl/modules/lidar_pd_actor_critic.py` |
| 观测重排 | `rsl_rl/utils/lidar_wrapper.py` |
| LiDAR 任务环境 | `legged_gym/envs/el_4090/pd_gru_lidar/el_4090_lidar.py` |
| LiDAR 传感器 | `legged_gym/utils/LidarSensor/lidar_sensor.py` |
| 可视化工具 | `legged_gym/utils/gym_visualizer.py` |
| 六足本体环境 | `legged_gym/envs/el_4090/spider_nomal/el_4090.py` |

## 开发指南

### 添加新机器人
1. 在 `legged_gym/envs/` 下创建机器人目录
2. 创建机器人类（继承 `LeggedRobot`，或 `LeggedRobotRaycast`/`LeggedRobotDepthCam`）
3. 创建配置类（继承 `LeggedRobotCfg` 和 `LeggedRobotCfgPPO`）
4. 在 `__init__.py` 中导入并注册任务
5. 若 rewards.scales 使用分阶段列表，必须设置 `multi_stage_rewards = True`，否则训练启动即崩

### 策略导出
训练完成后通过 `play.py` 自动导出 JIT 模型，或手动调用：
```python
from legged_gym.utils.helpers import export_policy_as_jit
export_policy_as_jit(actor_critic, output_path)
```
支持标准 MLP（`policy_1.pt`）与 LSTM（`memory_a` 属性检测）导出。

## 注意事项

1. **GPU 内存管理**：`height_samples` 变量可能占用大量 GPU 内存。减少 `num_envs` 可缓解。
2. **刚体状态视图**：启用障碍物时 `sim_num_bodies` 可能超过机器人身体数量，使用 `rigid_body_state.view(self.num_envs, self.sim_num_bodies, 13)` 正确访问。
3. **环境间距**：滚轮环境中机器人聚集可能导致 `PxgDynamicsMemoryConfig::foundLostAggregatePairsCapacity` 警告，需设置不同的初始位置。
4. **可选依赖**：`traj_sampling`（轨迹采样）和 `tqdm`（cyberdog2）为可选包，缺失时自动跳过对应任务注册。
5. **conda 环境**：建议使用 `conda activate li_leggym`。
6. **Isaac Gym**：需正确安装并配置 `ISAACGYM_PATH` / `PYTHONPATH`。
7. **多阶段奖励**：配置中列表型 reward scale 依赖 `multi_stage_rewards=True` 解析；新建配置继承带列表 scale 的父类时注意该开关。
