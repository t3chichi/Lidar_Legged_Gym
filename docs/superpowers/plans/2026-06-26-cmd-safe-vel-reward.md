# Command-Safe Velocity Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement new reward design (cmd_safe_vel + sector_dist_penalty) replacing vel_avoid/rays, with LiDAR at body center and dual z-filtering, in a separate config/environment that leaves existing go2_lidar_pd_risknet training untouched.

**Architecture:** New config class `Go2CmdSafeCfg` extends `Go2LidarPDRiskNetCfg` with sensor at [0,0,0], new reward scales, and safety parameters. New env class `Go2CmdSafe` extends `Go2LidarPDRiskNet`, overrides `_post_physics_step_callback` to compute sector safety instead of v_avoid/rays, and adds two reward functions. Both new files live under `go2/cmd_safe/`.

**Tech Stack:** Python, PyTorch, Isaac Gym, existing LidarSensor/PD-RiskNet infrastructure

---

### Task 1: Create directory and config file

**Files:**
- Create: `legged_gym/legged_gym/envs/go2/cmd_safe/__init__.py`
- Create: `legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe_config.py`

- [ ] **Step 1: Create the cmd_safe package init**

```bash
mkdir -p legged_gym/legged_gym/envs/go2/cmd_safe
```

```python
# legged_gym/legged_gym/envs/go2/cmd_safe/__init__.py
from .go2_cmd_safe import Go2CmdSafe
from .go2_cmd_safe_config import Go2CmdSafeCfg, Go2CmdSafeCfgPPO
```

- [ ] **Step 2: Write the config file**

```python
# legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe_config.py

from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet_config import (
    Go2LidarPDRiskNetCfg,
    Go2LidarPDRiskNetCfgPPO,
    OBS_HISTORY_LENGTH,
    PROX_HISTORY_LENGTH,
    DIST_HISTORY_LENGTH,
    PD_SPHERICAL_AZIMUTH,
    PD_SPHERICAL_ELEVATION,
    PD_NUM_LIDAR_POINTS,
    PD_PROXIMAL_POINTS,
    PD_DISTAL_POINTS,
    PD_PROXIMAL_FEATURE_DIM,
    PD_DISTAL_FEATURE_DIM,
    HEADING_OBS_ENABLED,
    PD_PROPRIO_DIM,
    PD_THETA_DEG,
    MEASURED_GRID_X_COUNT,
    MEASURED_GRID_Y_COUNT,
    MEASURED_GRID_X_RANGE,
    MEASURED_GRID_Y_RANGE,
    PD_PRIV_HEIGHT_DIM,
    PD_PRIV_CRITIC_DIM,
)


class Go2CmdSafeCfg(Go2LidarPDRiskNetCfg):
    """Command-safe velocity reward config.

    Differs from Go2LidarPDRiskNetCfg:
    - LiDAR at body centre [0,0,0] with horizontal orientation
    - vel_avoid and rays rewards set to zero
    - base_height reward set to zero (replaced by dual z-filtering)
    - New cmd_safe_vel and sector_dist_penalty rewards
    - No velocity command curriculum
    """

    class cmd_safe:
        # Body ellipse parameters (from URDF collision box)
        body_semi_length = 0.188   # a: collision box L/2 = 0.3762/2
        body_semi_width  = 0.047   # b: collision box W/2 = 0.0935/2

        # z-filtering thresholds (body frame, LiDAR at [0,0,0])
        z_thresh_high = 0.10       # body top + 3.3 cm clearance
        z_thresh_low  = -0.20      # body bottom + 14 cm leg clearance

        # safety factor parameters
        d_safety   = 0.10          # additional safety gap (m)
        d_safe_max = 1.0           # distance above which safe=1

        # cmd_safe_vel reward
        cmd_safe_sigma = 0.25      # gaussian kernel width

        # sector_dist_penalty reward
        dist_penalty_thresh = 0.5  # penalty activates below this (m)
        dist_penalty_alpha  = 0.5  # penalty scale factor

    class raycaster(Go2LidarPDRiskNetCfg.raycaster):
        offset_pos = [0.0, 0.0, 0.0]          # LiDAR at body centre
        sensor_offset_rpy = [0.0, 0.0, 0.0]   # horizontal, no tilt

    class rewards(Go2LidarPDRiskNetCfg.rewards):
        class scales:
            # ── New rewards (replace vel_avoid + rays) ──
            cmd_safe_vel        = 2.0
            sector_dist_penalty = 0.5

            # ── Turn off old avoidance rewards ──
            vel_avoid = 0.0
            rays      = 0.0

            # ── Turn off base_height (z-filtering handles body regulation) ──
            base_height = 0.0

            # ── Keep auxiliary rewards (same as parent) ──
            lin_vel_z    = -3.0e-4
            feet_stumble = -2.0e-2
            collision    = -2.0e-2
            dof_pos_limits = -0.2
            torques      = -1.0e-6
            dof_vel      = -1.0e-6
            dof_acc      = -2.5e-7
            action_rate  = -5.0e-3
            action_rate2 = -5.0e-3
            termination  = -10.0

            # ── Task-specific rewards ──
            goal            = 20.0
            channel_forward = 10.0
            curvature       = 0.0
            ang_vel_yaw_penalty = 0.0
            stand_still     = 0.0

            # ── Flat reward overrides (set to zero, not used) ──
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            ang_vel_xy       = 0.0
            orientation      = 0.0
            feet_air_time    = 0.0
            gait_2_step      = 0.0

    class commands(Go2LidarPDRiskNetCfg.commands):
        curriculum = False  # disable velocity command curriculum


class Go2CmdSafeCfgPPO(Go2LidarPDRiskNetCfgPPO):
    class runner(Go2LidarPDRiskNetCfgPPO.runner):
        experiment_name = "go2_cmd_safe"
        run_name = ""
        max_iterations = 4000
```

- [ ] **Step 3: Verify the config imports cleanly**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -c "
from legged_gym.envs.go2.cmd_safe.go2_cmd_safe_config import Go2CmdSafeCfg, Go2CmdSafeCfgPPO
cfg = Go2CmdSafeCfg()
print('sensor offset:', cfg.raycaster.offset_pos)
print('z_thresh_high:', cfg.cmd_safe.z_thresh_high)
print('z_thresh_low:', cfg.cmd_safe.z_thresh_low)
print('cmd_safe_vel scale:', cfg.rewards.scales.cmd_safe_vel)
print('sector_dist_penalty scale:', cfg.rewards.scales.sector_dist_penalty)
print('vel_avoid scale:', cfg.rewards.scales.vel_avoid)
print('rays scale:', cfg.rewards.scales.rays)
print('Config OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/cmd_safe/
git commit -m "feat: add Go2CmdSafe config with centre-mounted LiDAR and new reward scales"
```

---

### Task 2: Create environment class skeleton

**Files:**
- Create: `legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe.py`

- [ ] **Step 1: Write the skeleton with buffer init and sector safety computation**

```python
# legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe.py

import math
import torch

from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet import Go2LidarPDRiskNet


class Go2CmdSafe(Go2LidarPDRiskNet):
    """Go2 environment with command-safe velocity reward.

    Replaces vel_avoid / rays with:
    - cmd_safe_vel: tracks a 2D safe velocity (cmd projected through
      per-sector safety factors, each direction scaled independently)
    - sector_dist_penalty: omnidirectional background distance penalty

    Fully compatible parent: PD-RiskNet, collision replay, stuck detection,
    terrain curriculum all inherited unchanged.
    """

    def _init_pd_risknet_buffers(self):
        super()._init_pd_risknet_buffers()
        self._init_cmd_safe_buffers()

    def _init_cmd_safe_buffers(self):
        cd_cfg = self.cfg.cmd_safe
        n_sec = int(self.cfg.pd_risknet.n_sectors)  # 36

        # ── Precompute body radius per sector (ellipse) ──
        a = float(cd_cfg.body_semi_length)  # 0.188
        b = float(cd_cfg.body_semi_width)   # 0.047
        sec_size = 2.0 * math.pi / n_sec
        angles = torch.linspace(
            -math.pi + 0.5 * sec_size,
            math.pi - 0.5 * sec_size,
            n_sec, device=self.device,
        )
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        self._body_radius = a * b / torch.sqrt(
            b * b * cos_a * cos_a + a * a * sin_a * sin_a
        )  # (36,)

        self._sector_centers = torch.stack(
            (torch.cos(angles), torch.sin(angles)), dim=1
        )  # (36, 2)

        # ── Runtime buffers ──
        self._sector_dists = torch.zeros(
            self.num_envs, n_sec, device=self.device, dtype=torch.float,
        )
        self._sector_safe = torch.zeros(
            self.num_envs, n_sec, device=self.device, dtype=torch.float,
        )
        self._safe_distances = torch.full(
            (self.num_envs, int(self.cfg.pd_risknet.num_lidar_points)),
            float(self.cfg.pd_risknet.ray_max_distance),
            device=self.device, dtype=torch.float,
        )

    def _compute_sector_safety(self):
        """Compute per-sector z-filtered effective distances and safety factors.

        Replaces _compute_v_avoid() + _update_smooth_rays_dir().
        Called every step from _post_physics_step_callback.
        """
        cd_cfg = self.cfg.cmd_safe
        n_sec = int(self.cfg.pd_risknet.n_sectors)
        sec_size = 2.0 * math.pi / n_sec
        d_max = float(self.cfg.pd_risknet.ray_max_distance)

        dist = self._raw_distances.clone()
        pts = self.lidar_points_base

        # ── Step 1: z-filtering ──
        z = pts[..., 2]
        z_mask = (z > cd_cfg.z_thresh_high) | (z < cd_cfg.z_thresh_low)
        dist = torch.where(z_mask, torch.full_like(dist, d_max), dist)
        self._safe_distances.copy_(dist)

        # ── Step 2: per-sector min distance ──
        angles = torch.atan2(pts[..., 1], pts[..., 0])
        sec_ids = torch.floor(
            (angles + math.pi) / sec_size
        ).long().clamp(min=0, max=n_sec - 1)

        min_dist = torch.full(
            (dist.shape[0], n_sec), 1e9, device=dist.device, dtype=dist.dtype,
        )
        min_dist.scatter_reduce_(
            1, sec_ids, dist, reduce='amin', include_self=False,
        )

        # ── Step 3: body radius compensation → effective distance ──
        d_eff = torch.clamp(min_dist - self._body_radius.unsqueeze(0), min=0.0)
        self._sector_dists.copy_(d_eff)

        # ── Step 4: safety factor ──
        d_safety = float(cd_cfg.d_safety)
        d_safe_max = float(cd_cfg.d_safe_max)
        safe = torch.clamp(
            (d_eff - d_safety) / (d_safe_max - d_safety), 0.0, 1.0,
        )
        self._sector_safe.copy_(safe)
```

- [ ] **Step 2: Verify the class imports**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -c "
from legged_gym.envs.go2.cmd_safe.go2_cmd_safe import Go2CmdSafe
print('Go2CmdSafe imported OK')
print('MRO:', [c.__name__ for c in Go2CmdSafe.__mro__])
"
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe.py
git commit -m "feat: add Go2CmdSafe env skeleton with sector safety computation"
```

---

### Task 3: Add reward functions

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe.py` — append reward methods

- [ ] **Step 1: Add _reward_cmd_safe_vel and _reward_sector_dist_penalty methods**

Append these methods to the `Go2CmdSafe` class:

```python
    def _reward_cmd_safe_vel(self):
        """Sector-constrained safe velocity tracking reward.

        v_safe_2d = sum_j safe_j * align(cmd, sector_j) * sector_j
        r = exp(-||v_actual - v_safe_2d||^2 / sigma^2)
        """
        cd_cfg = self.cfg.cmd_safe
        cmd_2d = self.commands[:, :2]                    # (N, 2), body frame
        cmd_norm = torch.norm(cmd_2d, dim=1, keepdim=True).clamp(min=1e-8)

        safe = self._sector_safe                         # (N, 36)
        centers = self._sector_centers                   # (36, 2)

        # Per-sector alignment: cmd projection onto each sector direction
        align = torch.matmul(cmd_2d, centers.T)          # (N, 36)
        align = torch.clamp(align, min=0.0)              # only forward sectors

        # v_safe_2d = sum_j safe_j * align_j * sector_j
        weighted = safe * align                          # (N, 36)
        v_safe = torch.matmul(weighted, centers)         # (N, 2)

        # Clamp norm to ||cmd|| so obstacles never increase target speed
        v_safe_norm = torch.norm(v_safe, dim=1, keepdim=True).clamp(min=1e-8)
        scale = torch.clamp(cmd_norm / v_safe_norm, max=1.0)
        v_safe = v_safe * scale

        v_actual = self.base_lin_vel[:, :2]
        vel_err = torch.sum(torch.square(v_actual - v_safe), dim=1)
        sigma = float(cd_cfg.cmd_safe_sigma)
        return torch.exp(-vel_err / sigma)

    def _reward_sector_dist_penalty(self):
        """Omnidirectional background distance penalty.

        r = -alpha * mean_j(relu(d_thresh - d_eff_j)^2)
        Active for all 36 sectors, providing early warning of proximity.
        """
        cd_cfg = self.cfg.cmd_safe
        d_thresh = float(cd_cfg.dist_penalty_thresh)
        alpha = float(cd_cfg.dist_penalty_alpha)
        penalty = torch.relu(d_thresh - self._sector_dists).square()
        return -alpha * penalty.mean(dim=1)
```

- [ ] **Step 2: Verify both reward methods are callable**

Check that `_prepare_reward_function` in the base class will find them (it looks for `_reward_<name>` methods matching non-zero scales in the config):

```bash
cd /home/t3chichi/Lidar_legged_gym && python -c "
from legged_gym.envs.go2.cmd_safe.go2_cmd_safe_config import Go2CmdSafeCfg
cfg = Go2CmdSafeCfg()
for name, scale in cfg.rewards.scales.__dict__.items():
    if not name.startswith('_') and scale != 0:
        expected = '_reward_' + name
        print(f'  {name}: scale={scale}, expects {expected}')
"
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe.py
git commit -m "feat: add cmd_safe_vel and sector_dist_penalty reward functions"
```

---

### Task 4: Override _post_physics_step_callback

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe.py` — add callback override

- [ ] **Step 1: Override _post_physics_step_callback**

Append to the `Go2CmdSafe` class. This replaces `_compute_v_avoid()` + `_update_smooth_rays_dir()` with `_compute_sector_safety()`, but preserves all other logic (replay buffer, LiDAR history, pos history):

```python
    def _post_physics_step_callback(self):
        """Override: replace v_avoid + rays with sector safety computation."""
        # Call Go2._post_physics_step_callback() (skip Go2LidarPDRiskNet's)
        super(Go2LidarPDRiskNet, self)._post_physics_step_callback()

        self._update_replay_buffer()
        self._update_lidar_history()

        # New: sector safety replaces _compute_v_avoid + _update_smooth_rays_dir
        self._compute_sector_safety()

        # Keep pos_hist update from parent (used by stuck detection)
        update_ids = (self.episode_length_buf % 10 == 0).nonzero(as_tuple=False).flatten()
        if len(update_ids) > 0:
            self.pos_hist[update_ids] = torch.cat([
                self.pos_hist[update_ids, 1:],
                self.root_states[update_ids, :2].unsqueeze(1),
            ], dim=1)
```

- [ ] **Step 2: Verify the callback chain executes without error**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -c "
from legged_gym.envs.go2.cmd_safe.go2_cmd_safe import Go2CmdSafe
# Check that _post_physics_step_callback is defined on the class
assert hasattr(Go2CmdSafe, '_post_physics_step_callback'), 'callback missing'
print('Callback override OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe.py
git commit -m "feat: override _post_physics_step_callback with sector safety"
```

---

### Task 5: Register new task

**Files:**
- Modify: `legged_gym/legged_gym/envs/__init__.py` — add import and registration

- [ ] **Step 1: Read the existing registration pattern**

```bash
grep -n "go2_lidar_pd_risknet" /home/t3chichi/Lidar_legged_gym/legged_gym/legged_gym/envs/__init__.py
```

- [ ] **Step 2: Add import and registration**

Insert after the `go2_lidar_pd_risknet` registration block. The exact insertion point depends on the file structure; add:

```python
# ── Command-Safe Velocity ──
from legged_gym.envs.go2.cmd_safe import Go2CmdSafe
from legged_gym.envs.go2.cmd_safe.go2_cmd_safe_config import Go2CmdSafeCfg, Go2CmdSafeCfgPPO
task_registry.register("go2_cmd_safe", Go2CmdSafe, Go2CmdSafeCfg(), Go2CmdSafeCfgPPO())
```

- [ ] **Step 3: Verify the task is registered**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -c "
from legged_gym.envs import task_registry
print('go2_cmd_safe' in task_registry.task_classes)
print('Registered tasks containing cmd_safe:')
for name in sorted(task_registry.task_classes.keys()):
    if 'cmd_safe' in name:
        print(f'  {name}')
"
```

- [ ] **Step 4: Commit**

```bash
git add legged_gym/legged_gym/envs/__init__.py
git commit -m "feat: register go2_cmd_safe task"
```

---

### Task 6: Smoke test — environment creation and reward computation

**Files:**
- Verify: all new and modified files

- [ ] **Step 1: Create environment with minimal config**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -c "
import torch
from legged_gym.envs.go2.cmd_safe.go2_cmd_safe_config import Go2CmdSafeCfg, Go2CmdSafeCfgPPO

# Instantiate configs
env_cfg = Go2CmdSafeCfg()
train_cfg = Go2CmdSafeCfgPPO()

# Set minimal env count for testing
env_cfg.env.num_envs = 4
env_cfg.terrain.num_rows = 1
env_cfg.terrain.num_cols = 1
env_cfg.terrain.mesh_type = 'plane'
env_cfg.terrain.curriculum = False
env_cfg.replay.enable_collision_replay = False
env_cfg.commands.curriculum = False
env_cfg.commands.resampling_time = 100.0  # don't resample during smoke test

print('Configs created OK')
print(f'  num_obs: {env_cfg.env.num_observations}')
print(f'  sensor pos: {env_cfg.raycaster.offset_pos}')
print(f'  sensor rpy: {env_cfg.raycaster.sensor_offset_rpy}')
print(f'  cmd_safe_vel scale: {env_cfg.rewards.scales.cmd_safe_vel}')
print(f'  sector_dist_penalty scale: {env_cfg.rewards.scales.sector_dist_penalty}')
print(f'  vel_avoid scale: {env_cfg.rewards.scales.vel_avoid}')
print(f'  rays scale: {env_cfg.rewards.scales.rays}')
"
```

- [ ] **Step 2: Create full Isaac Gym environment (headless) and step once**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -c "
import torch
import isaacgym
from legged_gym.envs import task_registry
from legged_gym.utils.helpers import get_args, parse_sim_params

# Override args for minimal smoke test
import sys
sys.argv = ['smoke_test', '--task=go2_cmd_safe', '--num_envs=4', '--headless']

args = get_args()
env_cfg, train_cfg = task_registry.get_cfgs(args.task)

# Minimal config
env_cfg.env.num_envs = 4
env_cfg.terrain.num_rows = 1
env_cfg.terrain.num_cols = 1
env_cfg.terrain.mesh_type = 'plane'
env_cfg.terrain.curriculum = False
env_cfg.replay.enable_collision_replay = False
env_cfg.domain_rand.lidar_point_mask_ratio = 0.0
env_cfg.domain_rand.lidar_distance_noise_ratio = 0.0

sim_params = {'sim': dict(env_cfg.sim), 'physics_engine': 'physx'}
env, _ = task_registry.make_env(args.task, env_cfg, args)

print(f'Environment created: {env.num_envs} envs')
print(f'Observation shape: {env.obs_buf.shape}')

# Step once
obs, critic_obs = env.reset()
actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
obs, rewards, done_buf, extras = env.step(actions.detach())

print(f'Step completed')
print(f'  reward keys: {sorted(extras[\"episode\"].keys()) if hasattr(extras, \"__getitem__\") else extras.keys()}')
print(f'  reward sum per env: {env.rew_buf.tolist()}')
"
```

- [ ] **Step 3: Check specific reward values are computed**

After the step, verify that `cmd_safe_vel` and `sector_dist_penalty` appear in the episode sums and have reasonable values:

```bash
cd /home/t3chichi/Lidar_legged_gym && python -c "
import torch, sys
import isaacgym
from legged_gym.envs import task_registry
from legged_gym.utils.helpers import get_args

sys.argv = ['smoke', '--task=go2_cmd_safe', '--num_envs=4', '--headless']
args = get_args()
env_cfg, train_cfg = task_registry.get_cfgs(args.task)
env_cfg.env.num_envs = 4
env_cfg.terrain.num_rows = 1
env_cfg.terrain.num_cols = 1
env_cfg.terrain.mesh_type = 'plane'
env_cfg.terrain.curriculum = False
env_cfg.replay.enable_collision_replay = False
env_cfg.domain_rand.lidar_point_mask_ratio = 0.0
env_cfg.domain_rand.lidar_distance_noise_ratio = 0.0

env, _ = task_registry.make_env(args.task, env_cfg, args)
env.reset()
actions = torch.randn(4, env.num_actions, device=env.device)
obs, rewards, _, _ = env.step(actions.detach())

print('Episode sums after one step:')
for name in sorted(env.episode_sums.keys()):
    print(f'  {name}: {env.episode_sums[name].tolist()}')

# Verify new rewards exist
assert 'cmd_safe_vel' in env.episode_sums, 'cmd_safe_vel missing!'
assert 'sector_dist_penalty' in env.episode_sums, 'sector_dist_penalty missing!'
# Verify old rewards are NOT active
assert 'vel_avoid' not in env.episode_sums, 'vel_avoid should be inactive!'
assert 'rays' not in env.episode_sums, 'rays should be inactive!'

print('All reward checks passed')
"
```

- [ ] **Step 4: Commit any fixes if needed, then final commit**

If the smoke test passes, no additional commit needed. If fixes were required:

```bash
git add -A
git commit -m "fix: smoke test fixes for go2_cmd_safe"
```

---

### Task 7: Final verification — full pipeline sanity check

- [ ] **Step 1: Run multiple steps with plane terrain, verify no crashes**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -c "
import torch, sys
import isaacgym
from legged_gym.envs import task_registry
from legged_gym.utils.helpers import get_args

sys.argv = ['smoke', '--task=go2_cmd_safe', '--num_envs=16', '--headless']
args = get_args()
env_cfg, train_cfg = task_registry.get_cfgs(args.task)
env_cfg.env.num_envs = 16
env_cfg.terrain.num_rows = 1
env_cfg.terrain.num_cols = 1
env_cfg.terrain.mesh_type = 'plane'
env_cfg.terrain.curriculum = False
env_cfg.replay.enable_collision_replay = True  # test replay doesn't crash
env_cfg.domain_rand.lidar_point_mask_ratio = 0.02  # test with domain rand
env_cfg.domain_rand.lidar_distance_noise_ratio = 0.02

env, _ = task_registry.make_env(args.task, env_cfg, args)
env.reset()

for step in range(50):
    actions = torch.randn(16, env.num_actions, device=env.device)
    obs, rewards, done_buf, _ = env.step(actions.detach())
    if step % 10 == 0:
        alive = (~done_buf).sum().item()
        print(f'Step {step:3d}: alive={alive}/16, '
              f'r_cmd_safe={env.rew_buf.mean().item():.4f}')

print('50-step pipeline OK')
"
```

- [ ] **Step 2: Verify existing go2_lidar_pd_risknet task still works**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -c "
import torch, sys
import isaacgym
from legged_gym.envs import task_registry
from legged_gym.utils.helpers import get_args

sys.argv = ['compat', '--task=go2_lidar_pd_risknet', '--num_envs=4', '--headless']
args = get_args()
env_cfg, train_cfg = task_registry.get_cfgs(args.task)
env_cfg.env.num_envs = 4
env_cfg.terrain.num_rows = 1
env_cfg.terrain.num_cols = 1
env_cfg.terrain.mesh_type = 'plane'
env_cfg.terrain.curriculum = False
env_cfg.replay.enable_collision_replay = False
env_cfg.domain_rand.lidar_point_mask_ratio = 0.0
env_cfg.domain_rand.lidar_distance_noise_ratio = 0.0

env, _ = task_registry.make_env(args.task, env_cfg, args)
env.reset()

for step in range(10):
    actions = torch.randn(4, env.num_actions, device=env.device)
    obs, rewards, _, _ = env.step(actions.detach())

# Check old rewards still exist
assert 'vel_avoid' in env.episode_sums, 'vel_avoid should still exist in old task!'
assert 'rays' in env.episode_sums, 'rays should still exist in old task!'
assert 'cmd_safe_vel' not in env.episode_sums, 'cmd_safe_vel should NOT be in old task!'

print('Old task compatibility OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: add smoke tests for go2_cmd_safe pipeline"
```
