# Stuck Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 添加卡住检测：双条件(v_low/d_low) + 累计计时器，3秒不动触发止损终止。

**Architecture:** 仅修改 `go2_lidar_pd_risknet.py`，四处改动：init buffers、post_physics_step、check_termination、reset_idx cleanup。

**Tech Stack:** PyTorch, Isaac Gym

---

### Task 1: 添加 pos_hist + stay_timer + 卡住判定

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py`

**Step 1: `_init_replay_buffers` 末尾追加 pos_hist + stay_timer**

在第 52 行 `self.is_replay = ...` 之后追加：

```python
        self.pos_hist = torch.zeros(
            self.num_envs, 10, 2, device=self.device)
        self.stay_timer = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int)
```

**Step 2: `_post_physics_step_callback` 末尾追加 pos_hist 滚动更新**

在第 521 行 `self._update_smooth_rays_dir()` 之后追加：

```python
        # 每 10 步更新位置历史（~2 秒窗口）
        update_ids = (self.episode_length_buf % 10 == 0).nonzero(as_tuple=False).flatten()
        if len(update_ids) > 0:
            self.pos_hist[update_ids] = torch.cat([
                self.pos_hist[update_ids, 1:],
                self.root_states[update_ids, :2].unsqueeze(1)], dim=1)
```

**Step 3: `check_termination` 中 `collision_occurred` 更新之后追加卡住判定**

在第 586 行 `self.last_collision_active = new_collisions` 之后、`_reward_termination` 之前追加：

```python
        # ── 卡住检测：瞬时静止 or 长期无位移 ──
        v_low = (torch.norm(self.base_lin_vel[:, :2], dim=-1) < 0.1) & \
                (torch.abs(self.base_ang_vel[:, 2]) < 0.1)
        d_low = torch.norm(
            self.root_states[:, :2] - self.pos_hist[:, 0, :2], dim=-1) < 0.2
        not_just_reset = (self.episode_length_buf.float() /
                          self.max_episode_length) > 0.1
        self.static = (v_low | d_low) & not_just_reset
        self.stay_timer += self.static.int()
        stand_still_flag = self.stay_timer >= 150
        self.reset_buf |= stand_still_flag
```

**Step 4: `reset_idx` 公共清理区追加 stay_timer + pos_hist 清零**

在第 858 行 `self.is_replay[env_ids] = False` 之后追加：

```python
        self.stay_timer[env_ids] = 0
        self.pos_hist[env_ids, :, :] = 0.
```

**Step 5: 语法验证 + 提交**

```bash
python3 -c "import ast; ast.parse(open('legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py').read())"
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: add stuck detection with dual-condition (v_low/d_low) and 3s timer"
```
