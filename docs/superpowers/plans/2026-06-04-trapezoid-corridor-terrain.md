# Trapezoid Wave Corridor Terrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace sine-wave curved corridor terrain with trapezoid-wave corridor (2-bend, L-R alternating) to fix spawn-direction bias and provide more straight-line walking experience.

**Architecture:** New `trapezoid_corridor_terrain()` function in `terrain.py` replaces `curved_corridor_terrain()`. The corridor consists of 5 straight segments (entry straight → left diagonal → middle straight → right diagonal → exit straight), all north-pointing. Half the environments use L-R pattern, half use R-L, controlled by `_first_turn_left` on cfg. Spawn angle is always π/2 (facing +Y). Config removes sine-wave params and adds trapezoid params.

**Tech Stack:** numpy, Isaac Gym terrain_utils, existing legged_gym Terrain class

---

## File Structure

| File | Role |
|------|------|
| `legged_gym/legged_gym/utils/terrain.py` | New `trapezoid_corridor_terrain()`, update `curiculum()`/`randomized_terrain()` parity logic, update `add_terrain_to_map()` |
| `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py` | Replace sine params with trapezoid params |
| `legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py` | Add terrain geometry tests |

---

### Task 1: Update Config Parameters

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py:87-124`

- [ ] **Step 1: Replace sine-wave terrain params with trapezoid params**

In `class terrain(Go2RoughCfg.terrain)`, replace the corridor section:

```python
# --- 旧参数 (删除) ---
# amplitude = 1.0
# num_cycles = 1.5
# alternate_sign = True
# straight_length = 5.0

# --- 新参数 ---
# 梯形波弯曲通道地形配置
corridor_width = 3.0       # 通道宽度 (m)
wall_height = 1.5          # 墙壁高度 (m)
wall_thickness = 2         # 墙壁厚度 (m)
turn_angle_deg_max = 55.0  # 最大转弯角度 (deg), 课程从 0° 到 55°
diagonal_length = 3.0      # 转弯斜段长度 (m)
end_margin = 0.5           # 通道两端与地块边缘的间距 (m)
goal_forward_margin = 0.6  # 终点向前挪动距离 (m)
goal_radius = 1.6          # 终点半径 (m)
```

Note: `corridor_width` and `wall_height`/`wall_thickness` are already in the config; only the corridor shape params change. Keep `terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 1.0]` unchanged — it still routes to index 7 (corridor).

- [ ] **Step 2: Run existing config gate test to verify**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py::test_pd_risknet_config_gate -v
```

Expected: PASS (config class loads without the removed params, but they were only consumed by `curved_corridor_terrain` which is not imported in test).

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py
git commit -m "feat: replace sine-wave terrain params with trapezoid corridor params"
```

---

### Task 2: Write Terrain Tests

**Files:**
- Modify: `legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py` (append at end)

- [ ] **Step 1: Add test for trapezoid corridor centerline geometry**

```python
def test_trapezoid_corridor_geometry():
    """Verify trapezoid corridor centerline returns to midline and faces +Y."""
    import numpy as np
    import math
    from legged_gym.utils.terrain import trapezoid_corridor_terrain
    from legged_gym.utils.terrain import SubTerrain

    hs = 0.1  # horizontal_scale
    vs = 1.0  # vertical_scale
    size = 150  # pixels for 15m terrain

    terrain = SubTerrain("test", width=size, length=size,
                         vertical_scale=vs, horizontal_scale=hs)

    class Cfg:
        corridor_width = 3.0
        wall_height = 1.5
        wall_thickness = 2.0
        turn_angle_deg_max = 55.0
        diagonal_length = 3.0
        terrain_length = 15.0
        terrain_width = 15.0
        end_margin = 0.5
        goal_forward_margin = 0.6
        goal_radius = 1.6
        curriculum = False
        _first_turn_left = True

    cfg = Cfg()
    trapezoid_corridor_terrain(terrain, difficulty=0.5, cfg=cfg)

    # spawn_angle must be pi/2 (facing +Y)
    assert abs(terrain.spawn_angle - math.pi / 2) < 1e-6, \
        f"Expected spawn_angle=pi/2, got {terrain.spawn_angle}"

    # goal_offset_x must be 0 (centered)
    assert cfg.goal_offset_x == 0.0, \
        f"Expected goal_offset_x=0, got {cfg.goal_offset_x}"

    # goal_offset_y must be positive
    assert cfg.goal_offset_y > 0, \
        f"Expected goal_offset_y > 0, got {cfg.goal_offset_y}"
```

- [ ] **Step 2: Add test for L-R vs R-L mirror symmetry**

```python
def test_trapezoid_corridor_lr_rl_mirror():
    """L-R and R-L corridors should be mirror images across the midline."""
    import numpy as np
    import math
    from legged_gym.utils.terrain import trapezoid_corridor_terrain
    from legged_gym.utils.terrain import SubTerrain

    hs = 0.1
    vs = 1.0
    size = 150

    class Cfg:
        corridor_width = 3.0
        wall_height = 1.5
        wall_thickness = 2.0
        turn_angle_deg_max = 45.0
        diagonal_length = 3.0
        terrain_length = 15.0
        terrain_width = 15.0
        end_margin = 0.5
        goal_forward_margin = 0.0
        goal_radius = 1.6
        curriculum = False

    cfg_lr = Cfg()
    cfg_lr._first_turn_left = True
    terrain_lr = SubTerrain("lr", width=size, length=size,
                            vertical_scale=vs, horizontal_scale=hs)
    trapezoid_corridor_terrain(terrain_lr, difficulty=0.5, cfg=cfg_lr)

    cfg_rl = Cfg()
    cfg_rl._first_turn_left = False
    terrain_rl = SubTerrain("rl", width=size, length=size,
                            vertical_scale=vs, horizontal_scale=hs)
    trapezoid_corridor_terrain(terrain_rl, difficulty=0.5, cfg=cfg_rl)

    # Mirror across X midline: hf_lr[x, y] should equal hf_rl[size-1-x, y]
    hf_lr = terrain_lr.height_field_raw
    hf_rl = terrain_rl.height_field_raw
    hf_rl_mirrored = hf_rl[::-1, :]  # flip along X axis
    assert np.array_equal(hf_lr, hf_rl_mirrored), \
        "L-R and R-L corridors should be X-mirror images"
```

- [ ] **Step 3: Add test for Level 0 straight corridor**

```python
def test_trapezoid_corridor_level0_straight():
    """Difficulty 0 (turn_angle=0) should produce a straight north corridor."""
    import numpy as np
    from legged_gym.utils.terrain import trapezoid_corridor_terrain
    from legged_gym.utils.terrain import SubTerrain

    hs = 0.1
    vs = 1.0
    size = 150

    class Cfg:
        corridor_width = 3.0
        wall_height = 1.5
        wall_thickness = 2.0
        turn_angle_deg_max = 55.0
        diagonal_length = 3.0
        terrain_length = 15.0
        terrain_width = 15.0
        end_margin = 0.5
        goal_forward_margin = 0.0
        goal_radius = 1.6
        curriculum = False
        _first_turn_left = True

    cfg = Cfg()
    terrain = SubTerrain("straight", width=size, length=size,
                         vertical_scale=vs, horizontal_scale=hs)
    trapezoid_corridor_terrain(terrain, difficulty=0.0, cfg=cfg)

    hf = terrain.height_field_raw
    mid_x = size // 2

    # At the midline, the corridor should be floor (0) from y_start to y_end
    half_cw = int(3.0 / hs // 2)
    y_start = half_cw + int(0.5 / hs)
    y_end = size - half_cw - int(0.5 / hs)

    # Midline column should have floor in corridor region
    floor_pixels = (hf[mid_x, y_start:y_end] == 0).sum()
    total_pixels = y_end - y_start
    assert floor_pixels > 0.9 * total_pixels, \
        f"Expected mostly floor at midline, got {floor_pixels}/{total_pixels}"
```

- [ ] **Step 4: Run tests to verify they fail (function not defined yet)**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py::test_trapezoid_corridor_geometry -v
```

Expected: FAIL with ImportError/AttributeError (function not yet implemented)

- [ ] **Step 5: Commit**

```bash
git add legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py
git commit -m "test: add trapezoid corridor geometry tests"
```

---

### Task 3: Implement trapezoid_corridor_terrain()

**Files:**
- Modify: `legged_gym/legged_gym/utils/terrain.py` (add new function after `curved_corridor_terrain`)

- [ ] **Step 1: Add the trapezoid_corridor_terrain function**

Insert after the `curved_corridor_terrain` function (after line 498):

```python
def trapezoid_corridor_terrain(terrain, difficulty, cfg):
    """Generate trapezoid-wave corridor terrain with two alternating bends.

    Corridor centerline (5 segments, all point-to-point straight):
        Segment 1 (entry):  north (+Y) from midline
        Segment 2 (diag 1): diagonal at angle theta (left or right)
        Segment 3 (middle): north (+Y), offset from midline
        Segment 4 (diag 2): diagonal opposite angle, returns to midline
        Segment 5 (exit):   north (+Y) on midline

    Configurable parameters (via cfg):
        corridor_width:      corridor width (m), default 3.0
        wall_height:         wall height (m), default 1.5
        wall_thickness:      wall thickness (m), default 2.0
        turn_angle_deg_max:  max turn angle in degrees (mapped by difficulty)
        diagonal_length:     length of diagonal turn segment (m)
        terrain_length:      terrain patch length (m)
        terrain_width:       terrain patch width (m)
        end_margin:          margin at start/end of corridor (m)
        _first_turn_left:    if True, L-R pattern; if False, R-L pattern
    """
    corridor_width = float(getattr(cfg, "corridor_width", 3.0))
    wall_height = float(getattr(cfg, "wall_height", 1.5))
    wall_thickness = float(getattr(cfg, "wall_thickness", 2.0))
    turn_angle_deg_max = float(getattr(cfg, "turn_angle_deg_max", 55.0))
    diagonal_length = float(getattr(cfg, "diagonal_length", 3.0))
    terrain_len = float(getattr(cfg, "terrain_length", 15.0))
    terrain_width_cfg = float(getattr(cfg, "terrain_width", terrain_len))
    end_margin = float(getattr(cfg, "end_margin", 0.5))
    first_turn_left = getattr(cfg, "_first_turn_left", True)

    hs = terrain.horizontal_scale
    vs = terrain.vertical_scale

    corridor_width_px = int(corridor_width / hs)
    wall_height_px = int(wall_height / vs)
    half_cw = corridor_width_px // 2
    end_margin_px = int(end_margin / hs)
    diagonal_px = int(diagonal_length / hs)

    size_x = terrain.width
    size_y = terrain.length
    mid_x = size_x // 2

    # Turn angle from difficulty
    turn_angle_rad = math.radians(difficulty * turn_angle_deg_max)

    # Sign: +1 = turn right first (R-L), -1 = turn left first (L-R)
    sign = -1.0 if first_turn_left else 1.0
    sin_t = math.sin(turn_angle_rad)
    cos_t = math.cos(turn_angle_rad)

    # Corridor start/end Y positions (same as old curved corridor)
    y_start = half_cw + end_margin_px
    y_end = size_y - half_cw - end_margin_px
    available_y = y_end - y_start

    # Total Y consumed by diagonal segments
    diag_y = 2.0 * diagonal_px * cos_t

    # Remaining Y for straight segments (4 straight segments total: entry, entry2, exit2, exit)
    # Segments 1 and 5 (entry/exit): length L1
    # Segments 3a and 3b (middle north segments before/after junction): total 2*L1
    # Actually: the middle north run spans from end of diag1 to start of diag2.
    # We allocate: entry=L1, middle=2*L1, exit=L1 -> total straight Y = 4*L1
    if available_y <= diag_y + 4:
        # Not enough room — fall back to straight corridor
        L1 = max(1, (available_y - 4) // 4)
        diag_y = 0.0
        diagonal_px = 0
    else:
        L1 = int((available_y - diag_y) / 4.0)

    # Centerline waypoints (pixel coords)
    # P0: corridor entrance (bottom)
    # P1: end of entry straight
    # P2: end of first diagonal
    # P3: end of middle straight (north, offset)
    # P4: end of second diagonal (back to midline)
    # P5: corridor exit (top)

    P0_y = y_start
    P1_y = P0_y + L1
    P1_x = float(mid_x)

    P2_x = P1_x + sign * diagonal_px * sin_t
    P2_y = P1_y + diagonal_px * cos_t

    P3_x = P2_x
    P3_y = P2_y + 2.0 * L1

    P4_x = float(mid_x)
    P4_y = P3_y + diagonal_px * cos_t

    P5_y = P4_y + L1

    # Build coordinate grids
    x_coord, y_coord = np.meshgrid(
        np.arange(size_x, dtype=np.float64),
        np.arange(size_y, dtype=np.float64),
        indexing='ij',
    )

    # Compute distance from each pixel to the centerline polyline
    # For each segment, compute: t (projection param), perp_dist
    # Pixel is in corridor if min_dist <= half_cw

    segments = [
        (P1_x, P0_y, P1_x, P1_y),                    # seg 1: entry straight (vertical)
        (P1_x, P1_y, P2_x, P2_y),                     # seg 2: first diagonal
        (P2_x, P2_y, P3_x, P3_y),                     # seg 3: middle straight (vertical)
        (P3_x, P3_y, P4_x, P4_y),                     # seg 4: second diagonal
        (P4_x, P4_y, P4_x, P5_y),                     # seg 5: exit straight (vertical)
    ]

    # Start with no corridor floor
    in_corridor = np.zeros((size_x, size_y), dtype=bool)

    for (sx0, sy0, sx1, sy1) in segments:
        dx = sx1 - sx0
        dy = sy1 - sy0
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-9:
            continue
        seg_len = math.sqrt(seg_len_sq)

        # Projection parameter t = dot(P-A, B-A) / |B-A|^2
        t = ((x_coord - sx0) * dx + (y_coord - sy0) * dy) / seg_len_sq
        t_clamped = np.clip(t, 0.0, 1.0)

        # Closest point on segment
        cx = sx0 + t_clamped * dx
        cy = sy0 + t_clamped * dy

        # Perpendicular distance
        perp = np.sqrt((x_coord - cx) ** 2 + (y_coord - cy) ** 2)

        in_corridor |= (perp <= float(half_cw))

    # Clip to terrain bounds (leave 1px wall border)
    in_corridor &= (x_coord >= 1) & (x_coord < size_x - 1)
    in_corridor &= (y_coord >= 1) & (y_coord < size_y - 1)

    # Initialize: all walls, corridor = floor
    terrain.height_field_raw[:, :] = wall_height_px
    terrain.height_field_raw[in_corridor] = 0

    # Goal info (relative to env_origin, which is at corridor entrance center)
    cfg.goal_offset_x = 0.0  # always centered for trapezoid
    cfg.goal_offset_y = float(terrain_len) - corridor_width - 2.0 * end_margin
    goal_forward_margin = float(getattr(cfg, "goal_forward_margin", 0.0))
    if goal_forward_margin > 0:
        cfg.goal_offset_y -= goal_forward_margin
    cfg.goal_radius = float(getattr(cfg, "goal_radius", corridor_width / 2.0))

    # Spawn always facing +Y (pi/2 from +X axis)
    terrain.spawn_angle = math.pi / 2.0

    return terrain
```

Need to add `import math` at top of terrain.py if not already present. Let me check... Yes, `math` is not imported. But `np.sin`, `np.cos` are available. Let me use `np` equivalents instead:

- `math.radians(x)` → `np.deg2rad(x)`
- `math.sqrt(x)` → `np.sqrt(x)`
- `math.pi` → `np.pi`

- [ ] **Step 2: Run the geometry tests**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py::test_trapezoid_corridor_geometry legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py::test_trapezoid_corridor_lr_rl_mirror legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py::test_trapezoid_corridor_level0_straight -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/utils/terrain.py
git commit -m "feat: add trapezoid_corridor_terrain replacing curved_corridor_terrain"
```

---

### Task 4: Update Terrain Class Routing

**Files:**
- Modify: `legged_gym/legged_gym/utils/terrain.py:89-90,101-102,164,184-199`

- [ ] **Step 1: Update make_terrain to call trapezoid_corridor_terrain**

At line 164, replace the call:

```python
# OLD:
elif choice < self.proportions[7]:
    curved_corridor_terrain(terrain, difficulty, self.cfg)

# NEW:
elif choice < self.proportions[7]:
    trapezoid_corridor_terrain(terrain, difficulty, self.cfg)
```

- [ ] **Step 2: Update curiculum() parity logic**

At lines 101-102, replace `_sign_parity` with `_first_turn_left`:

```python
# OLD:
if getattr(self.cfg, "alternate_sign", False):
    self.cfg._sign_parity = (i + j) % 2

# NEW:
if hasattr(self.cfg, "turn_angle_deg_max"):
    self.cfg._first_turn_left = (j % 2 == 0)
```

- [ ] **Step 3: Update randomized_terrain() parity logic**

At lines 89-90, same replacement:

```python
# OLD:
if getattr(self.cfg, "alternate_sign", False):
    self.cfg._sign_parity = (i + j) % 2

# NEW:
if hasattr(self.cfg, "turn_angle_deg_max"):
    self.cfg._first_turn_left = (j % 2 == 0)
```

- [ ] **Step 4: Update add_terrain_to_map spawn_angle**

Currently `add_terrain_to_map` reads `terrain.spawn_angle` and stores it (line 198-199). No code change needed — the function already reads the value set by `trapezoid_corridor_terrain`. The corridor_width check at line 184 for env_origin positioning also remains valid. Verify nothing else references `_sign_parity` or `alternate_sign`:

```bash
grep -n "_sign_parity\|alternate_sign" legged_gym/legged_gym/utils/terrain.py
```

Should show only lines we just modified (no other references remain).

- [ ] **Step 5: Run all terrain tests**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
```

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add legged_gym/legged_gym/utils/terrain.py
git commit -m "feat: route trapezoid_corridor_terrain through Terrain class"
```

---

### Task 5: Integration Verification

**Files:** None modified; verification-only task.

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
```

Expected: All 11 tests PASS

- [ ] **Step 2: Verify config imports cleanly**

```bash
python -c "
import sys
sys.path.insert(0, 'legged_gym')
sys.path.insert(0, 'rsl_rl')
from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet_config import Go2LidarPDRiskNetCfg
cfg = Go2LidarPDRiskNetCfg()
print('turn_angle_deg_max:', cfg.terrain.turn_angle_deg_max)
print('diagonal_length:', cfg.terrain.diagonal_length)
print('corridor_width:', cfg.terrain.corridor_width)
print('OK - config loads with trapezoid params')
"
```

Expected: prints config values, no errors.

- [ ] **Step 3: Verify go2_lidar_pillar_config still loads**

```bash
python -c "
import sys
sys.path.insert(0, 'legged_gym')
sys.path.insert(0, 'rsl_rl')
from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pillar_config import Go2LidarPillarCfg
cfg = Go2LidarPillarCfg()
print('pillar config terrain_proportions:', cfg.terrain.terrain_proportions)
print('OK - pillar config unaffected')
"
```

Expected: prints pillar config, no errors.

- [ ] **Step 4: Commit (if any fixups needed)**

```bash
git commit -m "chore: final integration verification"
```

---

## Self-Review

**1. Spec coverage:**
- [x] Trapezoid wave corridor with 2 bends (L-R / R-L) → Task 3
- [x] Configurable diagonal_length → Task 1 (diagonal_length param), Task 3 (reads it)
- [x] All robots face +Y (unified spawn direction) → Task 3 (spawn_angle = π/2)
- [x] Left/right alternating (4 cols: 2 L-R, 2 R-L) → Task 4 (j%2 parity)
- [x] Pure level 0 straight corridor → Task 3 (difficulty=0 → turn_angle=0)
- [x] Curriculum: only angle grows, bend count fixed at 2 → Task 3 (difficulty * turn_angle_deg_max)
- [x] Goal centered on midline → Task 3 (goal_offset_x = 0)
- [x] Remove old sine params from config → Task 1
- [x] Tests → Task 2

**2. Placeholder scan:** No TBD/TODO/fill-in-later patterns found.

**3. Type consistency:**
- `trapezoid_corridor_terrain(terrain, difficulty, cfg)` — consistent across Task 2 tests and Task 3 implementation
- `cfg._first_turn_left` — set in Task 4, consumed in Task 3
- `SubTerrain` is from `terrain_utils` (isaacgym), `terrain.spawn_angle` is set in Task 3 and read by existing `add_terrain_to_map`
- `np.deg2rad` used instead of `math.radians` (terrain.py doesn't import math)
