# Collision Replay Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Go2 LiDAR PD-RiskNet 中实现 SEA-Nav 风格的碰撞回放机制，包括软硬碰撞分类、terminate_buf 信号分离、滚动状态缓冲区和回放触发逻辑。

**Architecture:** 所有修改集中在两个文件 — 配置类 `go2_lidar_pd_risknet_config.py` 和环境类 `go2_lidar_pd_risknet.py`。基类 (`legged_robot.py`)、Go2 父类 (`go2.py`) 和 PPO 算法 (`ppo.py`) 不动。通过覆盖 `check_termination`、`_reward_termination`、`reset_idx` 和新增 3 个回放方法实现。

**Tech Stack:** Python 3.8, PyTorch, Isaac Gym Preview 4

---

### Task 1: Config — 添加回放配置和碰撞分类

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py:27-29`

- [ ] **Step 1: 修改 asset 配置，添加 collision parts 分类**

将第 27-29 行：
```python
class Go2LidarPDRiskNetCfg(Go2RoughCfg):
    class asset(Go2RoughCfg.asset):
        terminate_after_contacts_on = []
```
改为：
```python
class Go2LidarPDRiskNetCfg(Go2RoughCfg):
    class asset(Go2RoughCfg.asset):
        terminate_after_contacts_on = ["base", "Head_upper", "Head_lower"]
        penalize_contacts_on = ["thigh", "calf", "Head_upper", "Head_lower", "base"]
```

- [ ] **Step 2: 添加 replay 配置类**

在 `pd_risknet` class 之后（约第 89 行 `goal_enabled = True` 后面）添加：

```python
    class replay:
        enable_collision_replay = True
        replay_prob = 0.8
        early_reset_prob_range = [0.1, 0.5]
        undo_steps_range = [100, 150]
        max_collision_points = 10
```

- [ ] **Step 3: 修改 termination 惩罚倍率**

当前约第 163 行的 `rewards.scales` 中没有 `termination`（继承自 Go2RoughCfg 的 `termination = -0.0`）。在 `class scales` 声明末尾追加重叠覆盖：

```python
            termination = -10.0
```

在现有 `curvature = -0.0` 或最后一个 scale 后添加即可。

- [ ] **Step 4: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py
git commit -m "feat: add collision replay config and soft/hard collision body classification"
```

---

### Task 2: 环境 — `_init_replay_buffers` 和移除旧的 collision_body_indices

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py:22-45`

- [ ] **Step 1: 删除 `collision_body_indices` 硬编码**

`_init_buffers` 中第 30-39 行，删除整个 `self.collision_body_indices` 初始化块：
```python
        # 删除以下 10 行：
        # Body indices for obstacle collision penalty.
        self.collision_body_indices = [
            self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name)
            for name in (
                "base", "Head_upper",
                "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
                "FL_calf",  "FR_calf",  "RL_calf",  "RR_calf",
            )
        ]
```

- [ ] **Step 2: 添加 `_init_replay_buffers` 调用**

在 `_init_buffers` 方法末尾（第 45 行 `self._spawn_angles = None` 之前）插入：
```python
        self._init_replay_buffers()
```

- [ ] **Step 3: 添加 `_init_replay_buffers` 方法**

在 `_init_buffers` 方法之后（约第 46 行空行处）新增完整方法：

```python
    def _init_replay_buffers(self):
        """滚动状态缓冲区 + 碰撞标志，供碰撞回放机制使用。"""
        self.replay_len = 100
        self.replay_root_states = torch.zeros(
            self.num_envs, self.replay_len, 13, device=self.device)
        self.replay_dof_pos = torch.zeros(
            self.num_envs, self.replay_len, self.num_dof, device=self.device)
        self.replay_dof_vel = torch.zeros(
            self.num_envs, self.replay_len, self.num_dof, device=self.device)

        self.collision_occurred = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool)
        self.last_collision_active = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool)
        self.is_replay = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool)
```

- [ ] **Step 4: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: add _init_replay_buffers, remove hardcoded collision_body_indices"
```

---

### Task 3: 环境 — `_update_replay_buffer` 和调用

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py` (在 `_init_buffers` 和 `_post_physics_step_callback` 附近)

- [ ] **Step 1: 添加 `_update_replay_buffer` 方法**

在 `_init_replay_buffers` 方法之后新增：

```python
    def _update_replay_buffer(self):
        """每步滚动更新回放缓冲区。新 episode 前两步用广播填充避免读到脏数据。"""
        self.replay_root_states = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            self.root_states.unsqueeze(1).expand(-1, self.replay_len, -1),
            torch.cat([self.replay_root_states[:, 1:],
                       self.root_states.unsqueeze(1)], dim=1))
        self.replay_dof_pos = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            self.dof_pos.unsqueeze(1).expand(-1, self.replay_len, -1),
            torch.cat([self.replay_dof_pos[:, 1:],
                       self.dof_pos.unsqueeze(1)], dim=1))
        self.replay_dof_vel = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            self.dof_vel.unsqueeze(1).expand(-1, self.replay_len, -1),
            torch.cat([self.replay_dof_vel[:, 1:],
                       self.dof_vel.unsqueeze(1)], dim=1))
```

- [ ] **Step 2: 在 `_post_physics_step_callback` 首行追加调用**

找到 `_post_physics_step_callback`（约第 488 行），在 `super()._post_physics_step_callback()` 之后、`self._update_lidar_history()` 之前插入：

```python
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_replay_buffer()          # ← 新增
        self._update_lidar_history()
        self._compute_v_avoid()
        self._update_smooth_rays_dir()
```

- [ ] **Step 3: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: add _update_replay_buffer with per-step rolling state storage"
```

---

### Task 4: 环境 — `check_termination` 重写

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py:496-518`

- [ ] **Step 1: 完整替换 `check_termination`**

将当前第 496-518 行替换为：

```python
    def check_termination(self):
        """终止检测 + 碰撞追踪 + early_reset 概率触发。"""
        # ── 初始化标志 ──
        self.initial_ = self.episode_length_buf <= 1
        self.extras["bad_masks"] = self.initial_

        # ── 硬碰撞 + timeout（check_termination 基类逻辑内联）──
        # 使用 :2（水平面力），与 SEA-Nav 一致
        hard_collision = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :2],
                       dim=-1) > 1.0, dim=1)
        hard_collision &= (~self.initial_)
        self.terminate_buf = hard_collision
        self.reset_buf = hard_collision.clone()
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

        # ── 翻转/跌落终止（保持原有逻辑）──
        if getattr(self.cfg.env, "enable_fall_termination", False):
            g_thresh = float(getattr(self.cfg.env, "fall_projected_gravity_z_threshold", -0.1))
            h_thresh = float(getattr(self.cfg.env, "fall_base_height_threshold", 0.12))
            flipped = self.projected_gravity[:, 2] > g_thresh
            low_base = self.base_pos[:, 2] < h_thresh
            self.reset_buf |= (flipped | low_base)
            self.terminate_buf |= (flipped | low_base)

        # ── 通道终点到达检测（保持原有逻辑）──
        pd_cfg = self.cfg.pd_risknet
        if self._goal_offsets_table is not None and getattr(pd_cfg, "goal_enabled", False):
            off = self._goal_offsets_table[self.terrain_levels, self.terrain_types]
            gx = self.env_origins[:, 0] + off[:, 0]
            gy = self.env_origins[:, 1] + off[:, 1]
            gr = self.cfg.terrain.goal_radius
            dist = torch.sqrt(
                (self.base_pos[:, 0] - gx) ** 2 +
                (self.base_pos[:, 1] - gy) ** 2
            )
            reached = dist < gr
            self.reset_buf |= reached

        # ── 碰撞回放：碰撞追踪 + early_reset ──
        enable_replay = getattr(self.cfg.replay, 'enable_collision_replay', False)
        if enable_replay:
            # 检测新碰撞：penalised_contact_indices 中任意部位水平力 > 1.0
            new_collisions = torch.any(
                torch.norm(self.contact_forces[:, self.penalised_contact_indices, :2],
                           dim=-1) > 1.0, dim=1)
            new_collisions &= (~self.initial_)

            # 只取碰撞首帧
            is_new_collision = new_collisions & (~self.last_collision_active)

            # early_reset 概率随地形难度线性增长
            prob_range = getattr(self.cfg.replay, 'early_reset_prob_range', [0.1, 0.5])
            early_prob = prob_range[0] + (prob_range[1] - prob_range[0]) * \
                (self.terrain_levels.float() / max(1, self.max_terrain_level)).clamp(max=1.0)
            trigger_early = is_new_collision & \
                (torch.rand(self.num_envs, device=self.device) < early_prob)

            self.reset_buf |= trigger_early
            self.terminate_buf |= trigger_early

            self.collision_occurred |= new_collisions
            self.last_collision_active = new_collisions
```

- [ ] **Step 2: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: rewrite check_termination with collision tracking and early_reset"
```

---

### Task 5: 环境 — `_reward_termination` 覆盖和 `_reward_collision` 更新

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py:838-849`

- [ ] **Step 1: 覆盖 `_reward_termination`，改用 `terminate_buf`**

在 `check_termination` 方法之后新增：

```python
    def _reward_termination(self):
        """终止惩罚：仅对 terminate_buf==True 的环境施加（硬碰撞、early_reset、跌倒）。"""
        if hasattr(self, 'terminate_buf'):
            return self.terminate_buf.float()
        return self.reset_buf * ~self.time_out_buf
```

- [ ] **Step 2: 更新 `_reward_collision`，改用 `penalised_contact_indices`**

将第 838-849 行中的 `self.collision_body_indices` 替换为 `self.penalised_contact_indices`：

```python
    def _reward_collision(self):
        if getattr(self.cfg.pd_risknet, "collision_3d", False):
            return torch.sum(
                (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1).float(),
                dim=1)
        forces_xy = torch.stack([
            torch.norm(self.contact_forces[:, idx, :2], dim=1)
            for idx in self.penalised_contact_indices
        ], dim=1)
        return torch.sum(torch.square(forces_xy), dim=1)
```

- [ ] **Step 3: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: use terminate_buf for termination reward, switch collision reward to penalised_contact_indices"
```

---

### Task 6: 环境 — `_reset_collision_replay` 新方法

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py`

- [ ] **Step 1: 添加 `_reset_collision_replay` 方法**

在 `reset_idx` 之前新增：

```python
    def _reset_collision_replay(self, env_ids):
        """从滚动缓冲区回退机器人状态，模拟"重新尝试"当前场景。"""
        undo_range = getattr(self.cfg.replay, 'undo_steps_range', [100, 150])
        undo_steps = torch.randint(
            undo_range[0], undo_range[1], (len(env_ids),), device=self.device)

        current_len = self.episode_length_buf[env_ids]
        undo_steps = torch.min(undo_steps.long(), current_len.long())
        undo_steps = torch.clamp(undo_steps, max=self.replay_len - 1)

        valid_replay = undo_steps > 20
        replay_ids = env_ids[valid_replay]
        fallback_ids = env_ids[~valid_replay]

        # 历史不够 → 走完整正常重置链
        if len(fallback_ids) > 0:
            super().reset_idx(fallback_ids)
            self.lidar_points_base[fallback_ids] = 0.0
            self.raycast_distances[fallback_ids] = float(self.cfg.pd_risknet.ray_max_distance)
            self._raw_distances[fallback_ids] = float(self.cfg.pd_risknet.ray_max_distance)
            self.v_avoid[fallback_ids] = 0.0
            self._update_lidar_history()
            target_dir_world = self._compute_rays_target_dir()
            self._smooth_dir_world[fallback_ids] = target_dir_world[fallback_ids]

        if len(replay_ids) == 0:
            return

        self.is_replay[replay_ids] = True
        indices = -undo_steps[valid_replay]

        self.root_states[replay_ids] = self.replay_root_states[replay_ids, indices]
        self.dof_pos[replay_ids] = self.replay_dof_pos[replay_ids, indices]
        self.dof_vel[replay_ids] = self.replay_dof_vel[replay_ids, indices]

        env_ids_int32 = replay_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.episode_length_buf[replay_ids] -= undo_steps[valid_replay]
        self.last_actions[replay_ids] = 0.
        self.last_dof_vel[replay_ids] = 0.
```

- [ ] **Step 2: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: add _reset_collision_replay for state rewind from replay buffer"
```

---

### Task 7: 环境 — `reset_idx` 重写（replay/normal 分流）

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py:665-684`

- [ ] **Step 1: 完整替换 `reset_idx`**

将当前第 665-684 行替换为：

```python
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        enable_replay = getattr(self.cfg.replay, 'enable_collision_replay', False)

        # ── 依赖 time_out_buf（基类 check_termination 中设置）──
        time_out = self.time_out_buf if hasattr(self, 'time_out_buf') \
            else torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        if enable_replay:
            is_collision = self.collision_occurred[env_ids]
            prob = getattr(self.cfg.replay, 'replay_prob', 0.8)
            wants_replay = (
                (torch.rand(len(env_ids), device=self.device) < prob)
                & is_collision
                & (~time_out[env_ids])
            )
            # 到达目标的不回放（通过检查 hasattr）
            # 注意：goal_reached_flag 在当前代码中未定义，跳过此条件
            # 如有需要后续可添加

            replay_ids = env_ids[wants_replay]
            normal_ids = env_ids[~wants_replay]

            if len(replay_ids) > 0:
                self._reset_collision_replay(replay_ids)
            if len(normal_ids) > 0:
                super().reset_idx(normal_ids)
        else:
            super().reset_idx(env_ids)
            normal_ids = env_ids
            replay_ids = torch.tensor([], device=self.device, dtype=torch.long)

        # ── LiDAR 专用重置：仅对非回放 env ──
        non_replay_ids = normal_ids if enable_replay else env_ids
        if len(non_replay_ids) > 0:
            self.lidar_points_base[non_replay_ids] = 0.0
            self.raycast_distances[non_replay_ids] = float(self.cfg.pd_risknet.ray_max_distance)
            self._raw_distances[non_replay_ids] = float(self.cfg.pd_risknet.ray_max_distance)
            self.v_avoid[non_replay_ids] = 0.0
            self.last_dist[non_replay_ids] = torch.norm(
                self.base_pos[non_replay_ids, :2] - self.env_origins[non_replay_ids, :2], dim=1)
            self._last_channel_pos[non_replay_ids] = torch.sum(
                self.base_pos[non_replay_ids, :2] * self._channel_forward[non_replay_ids], dim=1)
            if hasattr(self, 'last_last_actions'):
                self.last_last_actions[non_replay_ids] = 0.
            self._update_lidar_history()
            target_dir_world = self._compute_rays_target_dir()
            self._smooth_dir_world[non_replay_ids] = target_dir_world[non_replay_ids]

        # ── 公共清理：所有 env ──
        self.collision_occurred[env_ids] = False
        self.last_collision_active[env_ids] = False
        self.is_replay[env_ids] = False
```

- [ ] **Step 2: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: rewrite reset_idx with replay/normal split and LiDAR reset gating"
```

---

### Task 8: 验证 — 运行环境测试

**Files:** (no changes)

- [ ] **Step 1: 运行基础环境测试**

```bash
python -m pytest legged_gym/legged_gym/tests/test_env.py -v
```

预期: PASS（验证环境创建和基础 step 不受影响）

- [ ] **Step 2: 运行 PD-RiskNet 数学测试**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
```

预期: PASS（验证奖励函数不受影响）

- [ ] **Step 3: 短训练冒烟测试**

```bash
python legged_gym/legged_gym/scripts/train.py --task=go2_lidar_pd_risknet --num_envs=64 --max_iterations=10 --headless
```

预期: 训练正常启动，无 crash，wandb/tensorboard 正常记录。

---

## Implementation Order Dependency

```
Task 1 (config) → Task 2 (init buffers) → Task 3 (update buffer) → Task 4 (check_termination)
                                                                       ↓
                                              Task 6 (_reset_collision_replay) ← Task 5 (reward overrides)
                                                                       ↓
                                              Task 7 (reset_idx) → Task 8 (verify)
```

Task 2-3 可以合并为一个 commit 操作。Task 4-5 有逻辑依赖（check_termination 设置 terminate_buf，_reward_termination 读取它）。Task 6-7 紧密耦合（_reset_collision_replay 被 reset_idx 调用）。
