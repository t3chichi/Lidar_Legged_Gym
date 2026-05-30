# LiDAR 点云密度提升与 Ray 奖励改造

## 参数汇总

| 参数 | 当前值 | 目标值 |
|------|--------|--------|
| `spherical_num_azimuth` | 36 | **80** |
| `spherical_num_elevation` | 24 | **50** |
| `num_lidar_points` | 864 | **4000** |
| `proximal_points` | 256 | **512** |
| `distal_points` | 96 | **256** |
| `ray_max_distance` | 10.0 | **50.0** |
| `raycaster.max_distance` | 10.0 | **50.0** |
| ray 奖励 | 全体远端 mean(d/10) | **36扇区 top-25%, 滤除d=50m** |
| 网络输入滤除 | — | **不滤除** |

## 改动文件

### go2_lidar_pd_risknet.py
- `_init_pd_risknet_buffers`: 新增 `_distal_ray_sector_ids` 预计算
- `_reward_rays`: 重写为 d_max=50 + 滤除 50m + 36 扇区 top-25%
- `_draw_debug_vis`: 适配新分辨率

### go2_lidar_pd_risknet_config.py (走廊)
- 模块常量: azimuth/elevation, proximal/distal_points
- pd_risknet: ray_max_distance, num_lidar_points
- raycaster: max_distance, spherical_num_*
- env: num_observations (推导)

### go2_lidar_pillar_config.py (梅花桩)
- 同上参数同步

### go2_pd_pretrain_config.py (预训练)
- 同上参数同步 (射线密度和网络参数须一致以保证 checkpoint 兼容)
- 预训练 rays=0，奖励公式改动对其无影响
