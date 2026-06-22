# Go2 走廊地形课程防遗忘回退机制 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Go2 走廊地形课程的 `_update_terrain_curriculum` 方法中插入防遗忘回退逻辑，到达最高等级并连续成功 N 次后随机回退到低级。

**Architecture:** 单文件修改，在现有 `_update_terrain_curriculum` 走廊分支中 `terrain_levels` 增减之后、clamp 之前，插入约 10 行回退判断代码。复用已有 `consecutive_upgrade_episodes` 配置，不引入新参数。

**Tech Stack:** Python, PyTorch

---

### Task 1: 插入防遗忘回退逻辑

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py:632-634`

- [ ] **Step 1: 在第 632 行（`self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down`）和第 633 行（`self.terrain_levels[env_ids] = torch.clip(...)`）之间插入回退逻辑**

将当前的：

```python
            self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
            self.terrain_levels[env_ids] = torch.clip(
                self.terrain_levels[env_ids], 0, self.max_terrain_level - 1)
```

改为：

```python
            self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down

            # 防遗忘回退：最高级连续成功 N 次后，随机回退到低级
            at_max = self.terrain_levels[env_ids] >= self.max_terrain_level - 1
            fallback = at_max & (self._consecutive_upgrade_count[env_ids] >= cons_up)
            self.terrain_levels[env_ids] = torch.where(
                fallback,
                torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level - 1),
                self.terrain_levels[env_ids],
            )
            self._consecutive_upgrade_count[env_ids] = torch.where(
                fallback,
                torch.zeros_like(self._consecutive_upgrade_count[env_ids]),
                self._consecutive_upgrade_count[env_ids],
            )

            self.terrain_levels[env_ids] = torch.clip(
                self.terrain_levels[env_ids], 0, self.max_terrain_level - 1)
```

- [ ] **Step 2: 验证语法正确性**

```bash
python -c "import ast; ast.parse(open('legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: 运行已有测试确保无回归**

```bash
python -m pytest legged_gym/legged_gym/tests/test_env.py -v
```

- [ ] **Step 4: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: add anti-forgetting fallback to corridor terrain curriculum"
```

---

### Task 2: 编写单元测试验证回退逻辑

**Files:**
- Create: `legged_gym/legged_gym/tests/test_terrain_curriculum_fallback.py`

- [ ] **Step 1: 创建测试文件，测试回退触发的各个条件**

```python
"""Tests for terrain curriculum fallback mechanism in Go2LidarPDRiskNet."""
import torch
import pytest


class MockCfg:
    """Minimal config mock for testing _update_terrain_curriculum fallback."""
    class terrain:
        num_rows = 5
        num_cols = 4
        terrain_length = 15
        terrain_width = 15
        goal_radius = 1.8
        curriculum = True
    
    class pd_risknet:
        move_down_ratio = 0.5
        consecutive_upgrade_episodes = 5
        consecutive_downgrade_episodes = 2
    
    class env:
        episode_length_s = 30


def build_dummy_goal_offsets(num_rows, num_cols, goal_dist=13.5):
    """Build goal_offsets that point north (+Y) from each cell."""
    off = torch.zeros(num_rows, num_cols, 2)
    off[:, :, 1] = goal_dist  # goal is goal_dist meters north of origin
    return off


def make_state(env_ids, terrain_levels, upgrade_counts, downgrade_counts,
               root_xy, env_origins_xy, channel_forward):
    """Return a dict mimicking the environment state needed by the method."""
    n = len(env_ids)
    device = torch.device("cpu")
    return {
        "env_ids": env_ids,
        "terrain_levels": terrain_levels.clone(),
        "_consecutive_upgrade_count": upgrade_counts.clone(),
        "_consecutive_downgrade_count": downgrade_counts.clone(),
        "root_states_xy": root_xy.clone(),
        "env_origins_xy": env_origins_xy.clone(),
        "_channel_forward": channel_forward.clone(),
        "max_terrain_level": 5,
        "_goal_offsets_table": build_dummy_goal_offsets(5, 4).to(device),
        "terrain_types": torch.zeros(len(env_ids), dtype=torch.long, device=device),
        "init_done": True,
        "cfg": MockCfg(),
    }


def simulate_curriculum_step(env_ids, state):
    """Simulate the corridor branch of _update_terrain_curriculum.

    Returns updated terrain_levels, upgrade_count, downgrade_count.
    """
    cons_up = state["cfg"].pd_risknet.consecutive_upgrade_episodes
    cons_down = state["cfg"].pd_risknet.consecutive_downgrade_episodes
    move_down_ratio = state["cfg"].pd_risknet.move_down_ratio
    goal_radius = state["cfg"].terrain.goal_radius

    terrain_levels = state["terrain_levels"].clone()
    upgrade_count = state["_consecutive_upgrade_count"].clone()
    downgrade_count = state["_consecutive_downgrade_count"].clone()
    max_level = state["max_terrain_level"]
    goal_offsets = state["_goal_offsets_table"]
    terrain_types = state["terrain_types"]

    # Compute forward_dist and goal_dist (same as real code)
    delta_xy = state["root_states_xy"] - state["env_origins_xy"]
    forward_dist = (delta_xy * state["_channel_forward"]).sum(dim=1)
    off = goal_offsets[terrain_levels, terrain_types]
    goal_dist = (off * state["_channel_forward"]).sum(dim=1) - goal_radius

    move_up_raw = forward_dist > goal_dist
    move_down_raw = (forward_dist < move_down_ratio * goal_dist) & ~move_up_raw

    # Upgrade counter
    upgrade_count = torch.where(
        move_up_raw, upgrade_count + 1, torch.zeros_like(upgrade_count))
    move_up = upgrade_count >= cons_up

    # Downgrade counter
    downgrade_count = torch.where(
        move_down_raw, downgrade_count + 1, torch.zeros_like(downgrade_count))
    downgrade_count = torch.where(
        move_up_raw, torch.zeros_like(downgrade_count), downgrade_count)
    move_down = downgrade_count >= cons_down

    # Reset downgrade on upgrade
    downgrade_count = torch.where(
        move_up, torch.zeros_like(downgrade_count), downgrade_count)

    terrain_levels += 1 * move_up - 1 * move_down

    # === NEW: fallback logic ===
    at_max = terrain_levels >= max_level - 1
    fallback = at_max & (upgrade_count >= cons_up)
    terrain_levels = torch.where(
        fallback,
        torch.randint_like(terrain_levels, max_level - 1),
        terrain_levels,
    )
    upgrade_count = torch.where(
        fallback,
        torch.zeros_like(upgrade_count),
        upgrade_count,
    )

    terrain_levels = torch.clip(terrain_levels, 0, max_level - 1)

    # Reset upgrade/downgrade counts on level changes
    upgrade_count = torch.where(
        move_up | move_down, torch.zeros_like(upgrade_count), upgrade_count)
    downgrade_count = torch.where(
        move_down, torch.zeros_like(downgrade_count), downgrade_count)

    return terrain_levels, upgrade_count, downgrade_count


class TestTerrainCurriculumFallback:
    """Test the anti-forgetting fallback mechanism."""

    def test_no_fallback_below_max_level(self):
        """Robots below max level should not trigger fallback."""
        n = 4
        env_ids = torch.arange(n)
        # All robots at level 2, upgrade count = 5 (enough to trigger if at max)
        terrain_levels = torch.full((n,), 2, dtype=torch.long)
        upgrade_counts = torch.full((n,), 5, dtype=torch.long)
        downgrade_counts = torch.zeros(n, dtype=torch.long)
        # Robots reached the goal (forward_dist > goal_dist)
        root_xy = torch.tensor([[0.0, 14.0], [0.0, 14.0], [0.0, 14.0], [0.0, 14.0]])
        env_origins_xy = torch.zeros(n, 2)
        channel_forward = torch.tensor([[0.0, 1.0]] * n)

        state = make_state(env_ids, terrain_levels, upgrade_counts,
                          downgrade_counts, root_xy, env_origins_xy, channel_forward)

        new_levels, new_up, new_down = simulate_curriculum_step(env_ids, state)

        # Should have upgraded to level 3, NOT randomly fallen back
        assert (new_levels == 3).all(), f"Expected level 3, got {new_levels}"
        # Upgrade count should be reset after upgrade
        assert (new_up == 0).all()

    def test_fallback_at_max_level_with_sufficient_upgrades(self):
        """Robot at max level with enough consecutive successes should fallback."""
        n = 4
        env_ids = torch.arange(n)
        max_level = 5  # num_rows
        terrain_levels = torch.full((n,), max_level - 1, dtype=torch.long)  # level 4
        # Already have 5 consecutive successes at max level
        upgrade_counts = torch.full((n,), 5, dtype=torch.long)
        downgrade_counts = torch.zeros(n, dtype=torch.long)
        # Robots reached the goal again
        root_xy = torch.tensor([[0.0, 14.0], [0.0, 14.0], [0.0, 14.0], [0.0, 14.0]])
        env_origins_xy = torch.zeros(n, 2)
        channel_forward = torch.tensor([[0.0, 1.0]] * n)

        state = make_state(env_ids, terrain_levels, upgrade_counts,
                          downgrade_counts, root_xy, env_origins_xy, channel_forward)

        # Run multiple times to verify fallback happens
        fallback_occurred = False
        for _ in range(20):
            new_levels, new_up, new_down = simulate_curriculum_step(env_ids, state)
            if (new_levels < max_level - 1).any():
                fallback_occurred = True
                # Upgrade count should be reset
                assert (new_up == 0).all()
                # All levels should be in valid range
                assert (new_levels >= 0).all() and (new_levels < max_level).all()
                break
            # Update state for next iteration
            state["terrain_levels"] = new_levels
            state["_consecutive_upgrade_count"] = new_up
            state["_consecutive_downgrade_count"] = new_down

        assert fallback_occurred, "Fallback should have occurred within 20 attempts"

    def test_fallback_excludes_max_level(self):
        """Fallback should never put robot at max level (max_level - 1 is excluded)."""
        n = 1
        env_ids = torch.arange(n)
        max_level = 5
        terrain_levels = torch.full((n,), max_level - 1, dtype=torch.long)
        upgrade_counts = torch.full((n,), 5, dtype=torch.long)
        downgrade_counts = torch.zeros(n, dtype=torch.long)
        root_xy = torch.tensor([[0.0, 14.0]])
        env_origins_xy = torch.zeros(n, 2)
        channel_forward = torch.tensor([[0.0, 1.0]])

        state = make_state(env_ids, terrain_levels, upgrade_counts,
                          downgrade_counts, root_xy, env_origins_xy, channel_forward)

        for _ in range(100):
            new_levels, new_up, new_down = simulate_curriculum_step(env_ids, state)
            assert (new_levels < max_level - 1).all() or (new_levels == max_level - 1).all(), \
                f"Level should never be >= max_level, got {new_levels}"
            state["terrain_levels"] = new_levels
            state["_consecutive_upgrade_count"] = new_up
            state["_consecutive_downgrade_count"] = new_down

    def test_fallback_resets_upgrade_count(self):
        """After fallback, consecutive upgrade count must be zero."""
        n = 1
        env_ids = torch.arange(n)
        terrain_levels = torch.full((n,), 4, dtype=torch.long)  # max level
        upgrade_counts = torch.full((n,), 5, dtype=torch.long)
        downgrade_counts = torch.zeros(n, dtype=torch.long)
        root_xy = torch.tensor([[0.0, 14.0]])
        env_origins_xy = torch.zeros(n, 2)
        channel_forward = torch.tensor([[0.0, 1.0]])

        state = make_state(env_ids, terrain_levels, upgrade_counts,
                          downgrade_counts, root_xy, env_origins_xy, channel_forward)

        for _ in range(30):
            new_levels, new_up, new_down = simulate_curriculum_step(env_ids, state)
            # If fallback occurred, upgrade count must be zero
            if (new_levels < 4).item():
                assert new_up.item() == 0, \
                    f"Upgrade count should be 0 after fallback, got {new_up.item()}"
                return
            state["terrain_levels"] = new_levels
            state["_consecutive_upgrade_count"] = new_up
            state["_consecutive_downgrade_count"] = new_down

        pytest.fail("Fallback never triggered within 30 iterations")

    def test_insufficient_upgrades_no_fallback(self):
        """Robot at max level with insufficient consecutive successes stays."""
        n = 1
        env_ids = torch.arange(n)
        terrain_levels = torch.full((n,), 4, dtype=torch.long)
        upgrade_counts = torch.full((n,), 3, dtype=torch.long)  # not enough
        downgrade_counts = torch.zeros(n, dtype=torch.long)
        root_xy = torch.tensor([[0.0, 14.0]])
        env_origins_xy = torch.zeros(n, 2)
        channel_forward = torch.tensor([[0.0, 1.0]])

        state = make_state(env_ids, terrain_levels, upgrade_counts,
                          downgrade_counts, root_xy, env_origins_xy, channel_forward)

        # Robot reached goal → upgrade_count becomes 4, still < 5
        new_levels, new_up, new_down = simulate_curriculum_step(env_ids, state)
        # Should not have fallen back (still at max level or moved up then clamped)
        assert 0 <= new_levels.item() < 5
        # Upgrade count should be 4 (was 3, succeeded once)
        assert new_up.item() == 4
```

- [ ] **Step 2: 运行新增测试确认通过**

```bash
python -m pytest legged_gym/legged_gym/tests/test_terrain_curriculum_fallback.py -v
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/tests/test_terrain_curriculum_fallback.py
git commit -m "test: add unit tests for terrain curriculum fallback logic"
```
