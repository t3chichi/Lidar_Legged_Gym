# Forward-Sector Rays 奖励设计

## 目标

将 `_reward_rays()` 从 36 扇区等权平均改为仅取机器人前方 12 扇区（±60°），增强方向性梯度，使机器人学会将前方朝向开阔地带，最终取代 `y_progress` 奖励。

## 动机

- 当前 36 扇区等权平均稀释了方向信号：后方扇区看到远处也会推高奖励，导致机器人朝墙时仍能获得中等奖励
- `y_progress` 依赖全局 Y 坐标系，不通用
- 网络输入 360° 完整点云，仅前方扇区参与奖励：网络知道全局信息，奖励聚焦前方

## 扇区几何

- 传感器坐标系：X = cos(az)·cos(el), Y = sin(az)·cos(el), Z = sin(el)
- 扇区 ID = floor((atan2(Y,X) + π) / (2π/36))，范围 0~35
- 传感器安装 RPY = [0, -165°, 180°]，传感器 +X → 机器人 +X（正前方）
- 扇区 18 = 机器人正前方

## 变更

### 文件 1：`go2_lidar_pd_risknet_config.py`

`pd_risknet` 类新增：

```python
ray_forward_sector_count = 12    # rays 奖励使用的前方扇区数（扇区18为中轴）
ray_forward_sector_center = 18   # 前方扇区中轴索引
```

### 文件 2：`go2_lidar_pd_risknet.py`

`_reward_rays()` 末尾改为：

```python
sector_mean = torch.stack(sector_means, dim=1)  # (N, 36)
n_fwd = int(self.cfg.pd_risknet.ray_forward_sector_count)
center = int(self.cfg.pd_risknet.ray_forward_sector_center)
start = center - n_fwd // 2
end = start + n_fwd
return sector_mean[:, start:end].mean(dim=1) / d_max
```

### 消融实验

`y_progress` scale 从 10.0 直接清零。

## 验证

1. `play.py` 不报错
2. 用旧 checkpoint infer，对比新旧 rays 值量级
3. 训练观察：机器人是否能前进（不转圈）
4. 若转圈：提高 rays scale 或增大 N
