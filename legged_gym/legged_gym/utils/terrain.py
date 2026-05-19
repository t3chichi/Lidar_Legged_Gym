# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.spawn_angles = np.zeros((cfg.num_rows, cfg.num_cols))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            if hasattr(cfg, 'difficulty_scale'):
                self.curiculum(cfg.difficulty_scale)
            else:
                self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            if hasattr(cfg, 'difficulty_scale'):
                self.randomized_terrain(cfg.difficulty_scale)
            else:
                self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         self.cfg.slope_treshold)

    def randomized_terrain(self, difficulty_scale=1.0):
        for k in range(self.cfg.num_sub_terrains):
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9]) * difficulty_scale
            if getattr(self.cfg, "alternate_sign", False):
                self.cfg._sign_parity = (i + j) % 2
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        self._draw_goal_rings()

    def curiculum(self, difficulty_scale=1.0):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows * difficulty_scale
                choice = j / self.cfg.num_cols + 0.001

                if getattr(self.cfg, "alternate_sign", False):
                    self.cfg._sign_parity = (i + j) % 2
                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)
        self._draw_goal_rings()

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                                               width=self.width_per_env_pixels,
                                               length=self.width_per_env_pixels,
                                               vertical_scale=self.vertical_scale,
                                               horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
        self._draw_goal_rings()

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=self.width_per_env_pixels,
                                           length=self.width_per_env_pixels,
                                           vertical_scale=self.cfg.vertical_scale,
                                           horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        # Optional task-level override: sample obstacle height from a configurable range.
        if hasattr(self.cfg, "discrete_obstacle_height_range"):
            h_min, h_max = self.cfg.discrete_obstacle_height_range
            discrete_obstacles_height = np.random.uniform(h_min, h_max)
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height,
                                                     rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size,
                                                  stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)

        elif choice < self.proportions[7]:
            if hasattr(self.cfg, "corridor_width"):
                curved_corridor_terrain(terrain, difficulty, self.cfg)
            else:
                pillar_field_terrain(terrain, difficulty, self.cfg)
        
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        if hasattr(self.cfg, "corridor_width"):
            margin = float(getattr(self.cfg, "end_margin", 0.5))
            env_origin_x = i * self.env_length + self.cfg.terrain_width / 2.0
            env_origin_y = j * self.env_width + self.cfg.corridor_width / 2.0 + margin
            env_origin_z = 0.0
        else:
            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
            x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
            y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
            y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
            env_origin_z = np.min(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        if hasattr(terrain, "spawn_angle"):
            self.spawn_angles[i, j] = terrain.spawn_angle

    def _draw_goal_rings(self):
        """在地形高度图上绘制终点轮廓圈（走廊地形）。"""
        if not hasattr(self.cfg, "goal_offset_x") or not hasattr(self.cfg, "corridor_width"):
            return
        hs = self.cfg.horizontal_scale
        vs = self.cfg.vertical_scale
        ring_r_px = int(self.cfg.goal_radius / hs)
        ring_h_px = max(int(0.03 / vs), 1)
        N = max(ring_r_px * 6, 24)
        for i in range(self.cfg.num_rows):
            for j in range(self.cfg.num_cols):
                gx = self.env_origins[i, j, 0] + self.cfg.goal_offset_x
                gy = self.env_origins[i, j, 1] + self.cfg.goal_offset_y
                gx_px = int(gx / hs) + self.border
                gy_px = int(gy / hs) + self.border
                for k in range(N):
                    a = 2.0 * np.pi * k / N
                    px = gx_px + int(ring_r_px * np.cos(a))
                    py = gy_px + int(ring_r_px * np.sin(a))
                    if 0 <= px < self.tot_rows and 0 <= py < self.tot_cols:
                        self.height_field_raw[px, py] = max(
                            self.height_field_raw[px, py], ring_h_px)


def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x-x2: center_x + x2, center_y-y2: center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1: center_x + x1, center_y-y1: center_y + y1] = 0


def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def pillar_field_terrain(terrain, difficulty, cfg):
    """
    生成随机分布的四棱柱障碍物地形（矩形截面）。
    数量随 difficulty 线性增加，尺寸和间距可配置。
    """
    # 数量范围（随难度插值）
    count_min = getattr(cfg, "pillar_count_min", 5)
    count_max = getattr(cfg, "pillar_count_max", 25)
    # 矩形边长范围（米）
    size_x_min = getattr(cfg, "pillar_size_x_min", 0.15)
    size_x_max = getattr(cfg, "pillar_size_x_max", 0.30)
    size_y_min = getattr(cfg, "pillar_size_y_min", 0.15)
    size_y_max = getattr(cfg, "pillar_size_y_max", 0.30)
    # 高度范围（米）
    height_min = getattr(cfg, "pillar_height_min", 0.20)
    height_max = getattr(cfg, "pillar_height_max", 0.50)
    # 间距与放置
    min_separation = getattr(cfg, "pillar_min_separation", 1.2)          # 柱心最小间距（米）
    center_clear_radius = getattr(cfg, "pillar_center_clear_radius", 1.2)  # 出生点净空半径（米）
    spawn_radius = getattr(cfg, "pillar_spawn_radius", 4.0)              # 障碍物最大生成半径（米）
    allow_height_variation = getattr(cfg, "pillar_allow_height_variation", True)

    # 根据难度插值计算当前数量
    count = int(count_min + difficulty * (count_max - count_min))
    # 尺寸也可随难度略微增大（可选）
    size_x = size_x_min + difficulty * (size_x_max - size_x_min)
    size_y = size_y_min + difficulty * (size_y_max - size_y_min)
    height = height_min + difficulty * (height_max - height_min)

    # 转换为像素单位
    size_x_px = int(size_x / terrain.horizontal_scale)
    size_y_px = int(size_y / terrain.horizontal_scale)
    height_px = int(height / terrain.vertical_scale)
    min_sep_px = int(min_separation / terrain.horizontal_scale)
    clear_radius_px = int(center_clear_radius / terrain.horizontal_scale)
    spawn_radius_px = int(spawn_radius / terrain.horizontal_scale)

    # 地形中心
    center_x = terrain.width // 2
    center_y = terrain.length // 2

    # 生成满足约束的随机位置
    max_attempts = count * 100
    positions = []
    for _ in range(max_attempts):
        if len(positions) >= count:
            break
        # 在圆形区域内随机采样
        r = np.random.uniform(clear_radius_px, spawn_radius_px)
        theta = np.random.uniform(0, 2 * np.pi)
        cx = int(center_x + r * np.cos(theta))
        cy = int(center_y + r * np.sin(theta))

        # 边界检查
        if (cx - size_x_px//2 < 0 or cx + size_x_px//2 >= terrain.width or
            cy - size_y_px//2 < 0 or cy + size_y_px//2 >= terrain.length):
            continue

        # 检查与中心点距离
        if np.hypot(cx - center_x, cy - center_y) < clear_radius_px:
            continue

        # 检查与已有位置的间距
        valid = True
        for px, py in positions:
            if np.hypot(cx - px, cy - py) < min_sep_px:
                valid = False
                break
        if valid:
            positions.append((cx, cy))

    # 在高度图上绘制矩形棱柱
    for cx, cy in positions:
        if allow_height_variation:
            h_px = np.random.randint(int(height_px * 0.6), height_px + 1)
        else:
            h_px = height_px

        # 矩形区域
        x1 = cx - size_x_px // 2
        x2 = cx + size_x_px // 2
        y1 = cy - size_y_px // 2
        y2 = cy + size_y_px // 2
        # 确保不越界
        x1 = max(0, x1)
        x2 = min(terrain.width, x2)
        y1 = max(0, y1)
        y2 = min(terrain.length, y2)
        terrain.height_field_raw[x1:x2, y1:y2] = h_px

    return terrain


def curved_corridor_terrain(terrain, difficulty, cfg):
    """生成正弦曲线弯曲通道地形，两侧由墙壁围成，两端半圆形封口。

    坐标系约定:
        height_field_raw 形状 (size_x, size_y)，轴0=X方向，轴1=Y方向
        通道沿 Y 轴延伸，中心线在 X 方向正弦摆动。

    可配置参数（通过 cfg 传入）:
        corridor_width:  通道宽度 (m), 默认 3.0
        wall_height:     墙壁高度 (m), 默认 0.8
        wall_thickness:  墙壁厚度 (m), 默认 0.4
        amplitude:       正弦波振幅 (m), 默认 1.5
        num_cycles:      正弦波周期数, 默认 1.5
        terrain_length:  地块长度 (m), 默认从 cfg.terrain_length 读取
        end_margin:      通道两端距地块边缘的距离 (m), 默认 0.5
    """
    corridor_width = float(getattr(cfg, "corridor_width", 3.0))
    wall_height = float(getattr(cfg, "wall_height", 0.8))
    wall_thickness = float(getattr(cfg, "wall_thickness", 0.4))
    max_amplitude = float(getattr(cfg, "amplitude", 1.5))
    num_cycles = float(getattr(cfg, "num_cycles", 1.5))
    terrain_len = float(getattr(cfg, "terrain_length", 12.0))
    terrain_width_cfg = float(getattr(cfg, "terrain_width", terrain_len))
    end_margin = float(getattr(cfg, "end_margin", 0.5))

    # 振幅课程学习: difficulty 来自 curiculum() (row_i / num_rows), 映射为振幅比例
    if getattr(cfg, "curriculum", False):
        num_rows_cfg = int(getattr(cfg, "num_rows", 4))
        amplitude = difficulty * num_rows_cfg / max(num_rows_cfg - 1, 1) * max_amplitude
    else:
        amplitude = max_amplitude

    if getattr(cfg, "alternate_sign", False):
        if getattr(cfg, "_sign_parity", 0) == 1:
            amplitude = -amplitude
    elif getattr(cfg, "randomize_sign", False) and np.random.rand() > 0.5:
        amplitude = -amplitude

    hs = terrain.horizontal_scale
    vs = terrain.vertical_scale

    corridor_width_px = int(corridor_width / hs)
    amplitude_px = int(amplitude / hs)
    wall_height_px = int(wall_height / vs)

    # 像素尺寸: size_x=X方向(轴0), size_y=Y方向(轴1)
    size_x = terrain.width
    size_y = terrain.length
    mid_x = size_x // 2

    half_cw = corridor_width_px // 2
    end_margin_px = int(end_margin / hs)

    # 通道沿 Y 轴起止位置（含端部间距）
    y_start = half_cw + end_margin_px
    y_end = size_y - half_cw - end_margin_px
    corridor_len_px = max(y_end - y_start, 1)

    # 坐标网格，匹配 height_field_raw 形状 (size_x, size_y)
    x_coord, y_coord = np.meshgrid(np.arange(size_x), np.arange(size_y), indexing='ij')

    # 通道中心线 X 位置（随 Y 正弦变化）
    phase = 2.0 * np.pi * num_cycles * (y_coord.astype(np.float64) - y_start) / corridor_len_px
    center_x = mid_x + amplitude_px * np.sin(phase)
    dx = np.abs(x_coord - center_x)

    # 终点半圆形封口
    dist_end = np.sqrt((y_coord - y_end) ** 2 + (x_coord - mid_x) ** 2)

    # 起点切线直道 + 垂直封口
    straight_len_m = float(getattr(cfg, "straight_length", 0.5))
    straight_px = int(straight_len_m / hs)
    tangent_slope = amplitude_px * 2.0 * np.pi * num_cycles / float(corridor_len_px)
    sin_t = tangent_slope / np.sqrt(1.0 + tangent_slope**2)
    cos_t = 1.0 / np.sqrt(1.0 + tangent_slope**2)
    u = (x_coord.astype(np.float64) - mid_x) * sin_t + (y_coord.astype(np.float64) - y_start) * cos_t
    v = -(x_coord.astype(np.float64) - mid_x) * cos_t + (y_coord.astype(np.float64) - y_start) * sin_t
    # 底边 = 补齐短壁 + 延伸 straight_px
    u_back = -np.float64(half_cw) * np.abs(sin_t) - np.float64(straight_px)
    # 确保直线通道不超出子地块边界，四周保留至少 1 像素围墙
    x_in_bounds = (x_coord >= 1) & (x_coord < size_x - 1)
    y_in_bounds = y_coord >= 1
    straight_area = (u >= u_back) & (y_coord < y_start) & (np.abs(v) <= np.float64(half_cw) * cos_t) & x_in_bounds & y_in_bounds

    in_corridor = (
        ((y_coord >= y_start) & (y_coord <= y_end) & (dx <= half_cw) & x_in_bounds) |
        straight_area |
        ((y_coord > y_end) & (dist_end <= half_cw) & x_in_bounds)
    )

    # 初始化高度图：全图墙壁，通道区域覆盖为地板
    terrain.height_field_raw[:, :] = wall_height_px
    terrain.height_field_raw[in_corridor] = 0

    # --- 通道内随机方柱 ---
    pillar_count = int(getattr(cfg, "pillar_count", 0))
    if pillar_count > 0:
        pillar_hw = float(getattr(cfg, "pillar_half_width", 0.15))
        pillar_min_sep = float(getattr(cfg, "pillar_min_separation", 1.0))
        pillar_wall_margin = float(getattr(cfg, "pillar_wall_margin", 0.5))
        pillar_center_margin = float(getattr(cfg, "pillar_centerline_margin", 0.0))
        pillar_margin_end = float(getattr(cfg, "pillar_margin_end", 1.5))
        pillar_max_attempts = int(getattr(cfg, "pillar_max_attempts", 50))

        hw_px = int(pillar_hw / hs)
        min_sep_px = int(pillar_min_sep / hs)
        wall_margin_px = int(pillar_wall_margin / hs)
        center_margin_px = int(pillar_center_margin / hs)
        margin_end_px = int(pillar_margin_end / hs)

        placed = []
        for _ in range(pillar_count):
            for _ in range(pillar_max_attempts):
                # Y 坐标：在通道有效长度内随机
                py = np.random.randint(y_start + margin_end_px,
                                       max(y_start + margin_end_px + 1, y_end - margin_end_px))
                phase_p = 2.0 * np.pi * num_cycles * (py - y_start) / corridor_len_px
                cx = mid_x + int(amplitude_px * np.sin(phase_p))
                max_dx = max(1, half_cw - hw_px - wall_margin_px)

                # X 坐标：在中心线两侧随机，避开中心线禁区
                px = np.random.randint(cx - max_dx, cx + max_dx + 1)
                if abs(px - cx) < center_margin_px:
                    if cx - max_dx < cx - center_margin_px:
                        lo = cx - max_dx
                        hi = max(cx - center_margin_px, lo + 1)
                        px = np.random.randint(lo, hi + 1) if hi > lo else np.random.randint(cx + center_margin_px, cx + max_dx + 1)
                    else:
                        px = np.random.randint(cx + center_margin_px, cx + max_dx + 1)

                # 与已有方柱间距检查 (Chebyshev 距离，对应方形)
                too_close = False
                for epy, epx in placed:
                    if max(abs(py - epy), abs(px - epx)) < min_sep_px:
                        too_close = True
                        break
                if not too_close:
                    placed.append((py, px))
                    x1 = max(0, px - hw_px)
                    x2 = min(size_x, px + hw_px + 1)
                    y1 = max(0, py - hw_px)
                    y2 = min(size_y, py + hw_px + 1)
                    terrain.height_field_raw[x1:x2, y1:y2] = wall_height_px
                    break

    # 终点信息存 cfg 供环境读取（offset 相对 env_origin，即通道入口中心）
    cfg.goal_offset_x = float(terrain_len - terrain_width_cfg) / 2.0
    cfg.goal_offset_y = float(terrain_len) - corridor_width - 2.0 * end_margin
    goal_forward_margin = float(getattr(cfg, "goal_forward_margin", 0.0))
    if goal_forward_margin > 0:
        cfg.goal_offset_y -= goal_forward_margin
    cfg.goal_radius = float(getattr(cfg, "goal_radius", corridor_width / 2.0))

    # 起点切线方向
    terrain.spawn_angle = float(np.arctan2(1.0, tangent_slope))

    return terrain


def _draw_circle(cx, cy, radius, width, length):
    """在指定尺寸的画布上生成圆形区域的像素索引。"""
    y, x = np.ogrid[:width, :length]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    mask = dist <= radius
    rr, cc = np.nonzero(mask)
    return rr, cc