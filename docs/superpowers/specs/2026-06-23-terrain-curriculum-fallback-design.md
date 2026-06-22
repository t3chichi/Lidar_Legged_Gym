# Go2 走廊地形课程防遗忘回退机制

## 背景

原始 legged_gym 的地形课程设计中，机器人到达最高等级后会随机传送回低级课程，防止灾难性遗忘。当前 Go2 雷达训练的走廊地形课程分支缺少此机制——机器人到达最高等级（level 4，55° 最大转弯角）后仅被 clamp 在最高级，不再接触简单地形。

## 修改范围

- `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py`
  - 方法 `_update_terrain_curriculum`，走廊地形分支（`_goal_offsets_table is not None` 路径）

## 设计

### 回退逻辑

在现有升降级逻辑之后、`terrain_levels` clamp 之前插入回退判断：

```
当 terrain_level >= max_terrain_level - 1（最高等级）
  且 consecutive_upgrade_count >= consecutive_upgrade_episodes（连续成功达标）
→ terrain_level 随机回退到 [0, max_terrain_level - 1) 的任意等级
→ consecutive_upgrade_count 归零
```

### 设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 回退分布 | 均匀随机 | 与原始 legged_gym 一致 |
| 回退范围 | `[0, max_level - 1)` | 排除最高级，确保一定回到更低级 |
| 触发阈值 | 复用 `consecutive_upgrade_episodes`（默认 5） | 不引入新配置参数，保持简洁 |
| 回退后行为 | 通过现有机制重新逐级升级 | 无需额外逻辑 |

### 完整流程

```
低级 → 逐级升级 → 最高级 → 连续成功 N 次 → 随机回退到低级 → 循环
```

### 伪代码

```python
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
```

### 不修改的部分

- `consecutive_upgrade_episodes` 配置项（`go2_lidar_pd_risknet_config.py:87`）保持不变
- 降级逻辑不变
- 非走廊分支（旧距离导向逻辑）不变
- 配置文件中不新增参数

## 测试验证

1. 训练运行中观察 `terrain_level` 分布，确认最高级机器人会周期性地降到低级
2. 确认 `consecutive_upgrade_count` 在回退后被正确清零
3. 确认回退后机器人能正常重新逐级升级
4. 确保 `terrain_levels` 始终在 `[0, max_terrain_level - 1]` 范围内
