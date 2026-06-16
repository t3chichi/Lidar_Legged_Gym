# Go2 偏航轴修复

## 问题

Go2 体坐标系：轴 0 = 垂直（偏航轴），轴 1 = 前，轴 2 = 左。

三个奖励方法错误使用 `base_ang_vel[:, 2]`（roll 轴）跟踪偏航（yaw）行为，
与 Go2 官方 `_reward_tracking_ang_vel` 的 `[:, 0]` 不一致。

## 修改

| 行 | 方法 | 变更 |
|:--:|------|------|
| 530 | `_reward_ang_vel_yaw_penalty` | `[:, 2]` → `[:, 0]` |
| 540 | `_reward_curvature` | `[:, 2]` → `[:, 0]` |
| 799 | `_reward_rays` | `[:, 2]` → `[:, 0]` |

## 不变

- `base_ang_vel * obs_scales`（3 轴全送入观测，网络自行学习）
- `_compute_rays_omega_target`（`quat_apply_yaw_inverse` 使用世界 Z 轴，与体帧约定无关）
