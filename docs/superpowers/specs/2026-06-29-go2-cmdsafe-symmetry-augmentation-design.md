# Go2CmdSafe Symmetry Augmentation

**日期**: 2026-06-29
**状态**: 设计完成，待实现

## 动机

在 Go2CmdSafe 训练中引入对称数据增强 (Symmetry Data Augmentation)，利用 Go2 四足机器人的左右对称性：
- **数据增强** — 将有效 batch size 翻倍，不增加采样成本
- **镜像损失** — 约束策略在左右对称状态下输出对称动作

参考实现：ElSpider AIR 的 `get_elair_xsym_obs_act` (extended_legged_gym 项目)。

## 架构概览

```
修改/新建的文件:
├── legged_gym/utils/
│   ├── pointcloud_geometry.py          ← 新建：纯几何工具函数（无状态）
│   └── cmd_safe_history_wrapper.py      ← 修改：调用 pointcloud_geometry
├── legged_gym/envs/go2/cmd_safe/
│   ├── go2_cmd_safe.py                  ← 新增对称函数
│   └── go2_cmd_safe_config.py           ← 修改：添加 symmetry_cfg
├── rsl_rl/rsl_rl/runners/
│   └── on_policy_runner.py              ← 修改：_setup_symmetry 注入传感器参数
└── 测试
    ├── test_pointcloud_geometry.py      ← 新建
    ├── test_go2_xsym.py                 ← 新建
    └── test_cmd_safe_history_wrapper.py ← 修改：回归测试
```

关键设计原则：
- `pointcloud_geometry.py` 是纯函数模块，无类状态、无循环依赖
- Wrapper 和 Symmetry 共享同一套几何函数，`sort_points_by_angular_key` 为唯一排序入口
- 传感器参数通过 `functools.partial` 显式绑定，不通过 env/wrapper 隐式捞取

## pointcloud_geometry.py 接口

所有四元数使用 Isaac Gym 约定：`[x, y, z, w]` (scalar-last)。

```python
# 四元数工具
def quaternion_conjugate(q: Tensor) -> Tensor: ...
def quaternion_apply(q: Tensor, v: Tensor) -> Tensor: ...

# 纯几何（无传感器依赖）
def cartesian_to_spherical(points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """输入: points [..., 3]  输出: (r, azimuth, phi) 各 [..., 1]"""

# 传感器相关（显式传参）
def to_sensor_frame(points_base: Tensor, sensor_quat: Tensor,
                    sensor_trans: Tensor) -> Tensor:
    """基座系 → 传感器系。先平移再旋转。"""

def sort_points_by_angular_key(points_base: Tensor, sensor_quat: Tensor,
                                sensor_trans: Tensor) -> Tensor:
    """输入/输出均在基座系。排序键: phi*2π + azimuth。"""
```

## 观测布局（wrapped 4656-dim）

Go2CmdSafe 的观测经 `CmdSafeHistoryWrapper.wrap_obs()` 处理后为 4656 维：

| 切片 | 维度 | 内容 | 镜像操作 |
|------|------|------|---------|
| [0:3] | 3 | 基座线速度 × scale | [1] = -[1] (vy) |
| [3:6] | 3 | 基座角速度 × scale | [3] = -[3], [5] = -[5] (wx, wz) |
| [6:9] | 3 | 投影重力 | [7] = -[7] (gy) |
| [9:12] | 3 | 指令 × scale | [10] = -[10], [11] = -[11] (cmd_vy, cmd_wz) |
| [12:24] | 12 | (dof_pos - default) × scale | FL(12:15)↔FR(15:18), RL(18:21)↔RR(21:24) |
| [24:36] | 12 | dof_vel × scale | 同上 |
| [36:48] | 12 | 上一帧 actions | 同上 |
| [48:816] | 768 | 近程 FPS sorted (256×3) | reshape → Y 取反 → 重排序 → flatten |
| [816:4656] | 3840 | 远端历史 sorted (128×10×3) | reshape → Y 取反 → 重排序 → flatten |

## Go2 DOF 对称映射

12 关节，URDF 顺序：FL → FR → RL → RR，每条腿 hip/thigh/calf。

| 左腿 | 索引 | ↔ | 右腿 | 索引 |
|------|------|---|------|------|
| FL_hip | 12 | ↔ | FR_hip | 15 |
| FL_thigh | 13 | ↔ | FR_thigh | 16 |
| FL_calf | 14 | ↔ | FR_calf | 17 |
| RL_hip | 18 | ↔ | RR_hip | 21 |
| RL_thigh | 19 | ↔ | RR_thigh | 22 |
| RL_calf | 20 | ↔ | RR_calf | 23 |

注：索引为 proprio 中的绝对位置。dof_vel 偏移 +12，actions 偏移 +24。
动作 (12-dim) 映射：[0:3]↔[3:6], [6:9]↔[9:12]。

## 对称函数签名

```python
@torch.no_grad()
def get_go2_cmd_safe_xsym_obs_act(
    obs: Optional[torch.Tensor],
    actions: Optional[torch.Tensor],
    env,
    obs_type: str,
    *,
    sensor_quat: torch.Tensor,
    sensor_trans: torch.Tensor,
    proprio_dim: int = 48,
    proximal_points: int = 256,
    distal_history_points: int = 1280,
    distal_history_length: int = 10,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
```

尾部关键字参数由 `functools.partial` 在 `_setup_symmetry()` 中绑定。

## LiDAR 镜像顺序

```
原始点云 (基座系, 角排序后) [x, y, z]
  → Y 取反 [x, -y, z]
  → sort_points_by_angular_key()  # Y 翻转改变了 azimuth，旧排序失效
  → 输出已排序镜像点云
```

## Runner 集成

`OnPolicyRunner._setup_symmetry()` 改造：

```python
def _setup_symmetry(self):
    if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
        sc = self.alg_cfg["symmetry_cfg"]
        func = sc["data_augmentation_func"]
        if isinstance(func, str):
            func = string_to_callable(func)  # rsl_rl.utils.utils
        func = partial(func,
            sensor_quat=self.env._sensor_offset_quat,
            sensor_trans=self.env._sensor_translation,
            proprio_dim=48, proximal_points=256,
            distal_history_points=1280, distal_history_length=10,
        )
        sc["data_augmentation_func"] = func
        sc["_env"] = self.env
```

PPO 内部调用使用关键字传参（`obs=..., actions=..., env=..., obs_type=...`），`partial` 只补充未传的尾部参数，无错位风险。

## Config 声明

放入 `Go2CmdSafeCfgPPO.algorithm`：

```python
class symmetry_cfg:
    use_data_augmentation = True
    use_mirror_loss = True
    mirror_loss_coeff = 1.0
    data_augmentation_func = (
        "legged_gym.envs.go2.cmd_safe.go2_cmd_safe:get_go2_cmd_safe_xsym_obs_act"
    )
```

## 测试策略

| 测试文件 | 验证内容 | 依赖 |
|---------|---------|------|
| `test_pointcloud_geometry.py` (新) | 5 个公开函数：共轭、旋转、球坐标、传感器变换、排序稳定性 | 仅 torch |
| `test_go2_xsym.py` (新) | 标量镜像、DOF 交换、Y 取反、双重镜像=恒等、batch 翻倍、边界处理、[2048,4656] 压测 | 需 partial 绑定 |
| `test_cmd_safe_history_wrapper.py` (改) | wrap_obs 回归（输出维度、近程/远端分割、排序一致性） | env |
| 集成冒烟 | 训练 1 epoch 不崩溃 + `assert obs_aug.shape[0] == 2 * obs.shape[0]` | 需 GPU |

## 实现顺序

1. **pointcloud_geometry.py** — 抽取纯几何函数
2. **cmd_safe_history_wrapper.py** — 重构，改调 pointcloud_geometry
3. **test_pointcloud_geometry.py** — 验证几何函数正确性
4. **go2_cmd_safe.py** — 新增 `get_go2_cmd_safe_xsym_obs_act`
5. **go2_cmd_safe_config.py** — 添加 `symmetry_cfg`
6. **on_policy_runner.py** — `_setup_symmetry` 改造
7. **test_go2_xsym.py** — 对称函数单元测试
8. **test_cmd_safe_history_wrapper.py** — 回归验证
9. **集成冒烟** — 训练 1 epoch

## 风险点

- **LiDAR 排序键一致性**: Wrapper 和 Symmetry 必须使用完全相同的 `sort_points_by_angular_key`，任何偏差都会导致增强样本的物理不一致。
- **传感器四元数格式**: 务必声明为 `[x, y, z, w]` (Isaac Gym 约定)。
- **partial 绑定的 tensor 生命周期**: `_sensor_offset_quat` 和 `_sensor_translation` 在 `_init_lidar_sensor` 中创建后不会重建，引用稳定。
