# commands[:,2] 修复回退与坐标系文档计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 回退 StandGo2._reward_tracking_ang_vel 的错误修复，更新文档记录两种坐标系约定。

**背景：** 初次调查时误判 StandGo2/StandAnymal/StandElSpider 的 `_reward_tracking_ang_vel` 中 `base_ang_vel[:, 0]` 为 bug。深入分析后确认：站立机器人 (body X 朝上) 与行走机器人 (body Z 朝上) 使用不同的坐标约定，`[:, 0]` 在站立姿态下是正确的"偏航"轴。只有 Fix 1 (_resample_commands) 需要保留。

**技术要点：**
- 行走: body Z = 世界 Z = 偏航轴, `base_ang_vel[:, 2]`
- 站立: body X = 世界 Z = 偏航轴, `base_ang_vel[:, 0]`
- 三个 Stand 类的方法内部自洽

---

### Task 1: 回退 StandGo2._reward_tracking_ang_vel

**文件:**
- 修改: `legged_gym/legged_gym/envs/go2/go2.py:282`

- [ ] **Step 1: 回退 `[:, 2]` → `[:, 0]`**

```python
# 修改前 (go2.py:282)
ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])

# 修改后
ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 0])
```

- [ ] **Step 2: 验证语法和已有测试**

```bash
conda run -n li_leggym python -c "import ast; ast.parse(open('legged_gym/legged_gym/envs/go2/go2.py').read()); print('OK')"
conda run -n li_leggym python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
```

预期: 语法 OK, 36 passed

- [ ] **Step 3: 确认修改正确 — 读取 StandGo2 三个方法验证一致性**

确认以下三个方法全部使用站立坐标系:
- `_reward_orientation` (line 265-268): `projected_gravity[:, 1:]` → 重力对齐 -X → body X = UP ✅
- `_reward_ang_vel_xy` (line 261-263): `base_ang_vel[:, 1:]` → 惩罚 body Y+Z ✅
- `_reward_tracking_ang_vel` (line 280-283): `base_ang_vel[:, 0]` → body X 追踪偏航 ✅

- [ ] **Step 4: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/go2.py
git commit -m "$(cat <<'EOF'
revert: undo incorrect StandGo2 _reward_tracking_ang_vel axis change

StandGo2 uses standing coordinate frame (body X = world Z = yaw axis),
so base_ang_vel[:, 0] is the correct axis for yaw tracking. This is
consistent with _reward_orientation (gravity → -X) and _reward_ang_vel_xy
(penalizing [:, 1:]).
EOF
)"
```

---

### Task 2: 更新修复文档

**文件:**
- 修改: `docs/superpowers/specs/2026-06-16-commands-fix.md`

- [ ] **Step 1: 移除 Fix 2 相关内容，补充坐标系说明**

将文档中 Fix 2 部分替换为坐标系约定说明:

```markdown
# commands[:,2] 补充与坐标系约定

日期: 2026-06-16 | 更新: 2026-06-17

## 修复: `_resample_commands` 补充 `commands[:,2]`

... (保留原有 Fix 1 内容) ...

## 坐标系约定 (无需修改)

本项目存在两种机体坐标系约定，取决于机器人姿态:

### 行走姿态 (标准)

```
body X = 前进, body Y = 左, body Z = 上 = 世界 Z
偏航轴 = body Z = base_ang_vel[:, 2]
```

所有行走类机器人使用此约定:
- LeggedRobot (基类)
- Go2LidarPDRiskNet
- 各 Pose 变体 (PoseGo2, PoseAnymal, PoseElSpider)

### 站立姿态 (Stand 变体)

```
body X = 上 = 世界 Z, body Y = 左, body Z = 后
偏航轴 = body X = base_ang_vel[:, 0]
```

所有 Stand 变体使用此约定，三个方法自洽:
- `_reward_orientation`: projected_gravity[:, 1:] → 0 → body X 朝上
- `_reward_ang_vel_xy`: base_ang_vel[:, 1:] → 惩罚 body Y+Z，允许绕 X 旋转
- `_reward_tracking_ang_vel`: base_ang_vel[:, 0] → body X 追踪世界偏航

涉及的类 (均正确，无需修改):
- StandGo2 (go2.py:280)
- StandAnymal (anymal.py:285)
- StandElSpider (elspider.py:708)

### 影响总结

| 任务 | 修复前 commands[:,2] | 修复后 |
|------|------|------|
| 走廊/梅花桩 (ang_vel=[0,0]) | 0 | 0 (不变) |
| 软预训练 (ang_vel=[-1,1]) | 0 | 随机 [-1,1] |
| 旧预训练 (ang_vel=[-1,1]) | 0 | 随机 [-1,1] |

## 修改范围

| 文件 | 改动 | 状态 |
|------|------|:---:|
| `go2_lidar_pd_risknet.py` | `_resample_commands` 补充 else 分支 | ✅ 已应用 |
| 其余 `_reward_tracking_ang_vel` 覆写 | 无改动 (站立坐标系，[:,0] 正确) | — |
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-16-commands-fix.md
git commit -m "$(cat <<'EOF'
docs: update commands fix spec with coordinate system conventions

Remove the incorrect Fix 2 for StandGo2 and document the two valid
coordinate conventions (walking vs standing pose).
EOF
)"
```

---

### Self-Review

1. **Spec 覆盖**: 两个任务覆盖了回退和文档更新
2. **无占位符**: 所有步骤包含具体代码和命令
3. **类型一致性**: 不涉及新类型或接口
