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

    # === NEW: fallback logic (with guard for max_terrain_level > 1) ===
    if max_level > 1:
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

        # Fallback is guaranteed on first call (upgrade_count >= cons_up at max level)
        new_levels, new_up, new_down = simulate_curriculum_step(env_ids, state)
        # Should have fallen back below max level
        assert (new_levels < max_level - 1).all(), \
            f"Expected levels below max, got {new_levels}"
        # Upgrade count should be reset
        assert (new_up == 0).all()
        # All levels should be in valid range
        assert (new_levels >= 0).all() and (new_levels < max_level).all()

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

        # Fallback is guaranteed on first call (upgrade_count >= cons_up at max level)
        new_levels, new_up, new_down = simulate_curriculum_step(env_ids, state)
        # Fallback should put robot BELOW max level (not at max_level - 1)
        assert (new_levels < max_level - 1).all(), \
            f"Fallback must put robot below max level, got {new_levels}"

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

        # Fallback is guaranteed on first call (upgrade_count >= cons_up at max level)
        new_levels, new_up, new_down = simulate_curriculum_step(env_ids, state)
        # Should have fallen back below max level
        assert new_levels.item() < 4, \
            f"Expected level < 4 after fallback, got {new_levels.item()}"
        # Upgrade count should be reset
        assert new_up.item() == 0, \
            f"Upgrade count should be 0 after fallback, got {new_up.item()}"

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

        # Robot reached goal -> upgrade_count becomes 4, still < 5
        new_levels, new_up, new_down = simulate_curriculum_step(env_ids, state)
        # Should not have fallen back (still at max level or moved up then clamped)
        assert 0 <= new_levels.item() < 5
        # Upgrade count should be 4 (was 3, succeeded once)
        assert new_up.item() == 4

    def test_max_terrain_level_one_does_not_crash(self):
        """When max_terrain_level == 1, fallback should be skipped gracefully."""
        n = 4
        env_ids = torch.arange(n)
        terrain_levels = torch.zeros(n, dtype=torch.long)
        upgrade_counts = torch.full((n,), 5, dtype=torch.long)
        downgrade_counts = torch.zeros(n, dtype=torch.long)
        root_xy = torch.tensor([[0.0, 14.0]] * n)
        env_origins_xy = torch.zeros(n, 2)
        channel_forward = torch.tensor([[0.0, 1.0]] * n)

        state = make_state(env_ids, terrain_levels, upgrade_counts,
                          downgrade_counts, root_xy, env_origins_xy, channel_forward)
        state["max_terrain_level"] = 1
        # Rebuild goal_offsets for max_level=1
        state["_goal_offsets_table"] = build_dummy_goal_offsets(1, 4)

        # Should not crash
        new_levels, new_up, new_down = simulate_curriculum_step(env_ids, state)
        # Levels should stay at 0 (clamped)
        assert (new_levels == 0).all()
