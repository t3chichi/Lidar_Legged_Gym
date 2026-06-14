"""Generate 3D pillar meshes for LiDAR rendering on flat terrain."""

import numpy as np
import torch


def generate_pillar_positions(center_x, center_y, spawn_radius, clear_radius,
                               min_separation, count, size_x, size_y,
                               size_x_range, size_y_range,
                               height_min, height_max,
                               allow_height_variation, rng):
    """Sample pillar positions via polar coordinates with rejection sampling.

    Reuses the logic from pillar_field_terrain but returns world-frame
    position/size/height tuples instead of modifying a height field.

    Args:
        center_x, center_y: sub-terrain center in world XY (meters).
        spawn_radius, clear_radius, min_separation: meters.
        count: number of pillars to place.
        size_x, size_y: base pillar half-sizes. Overridden by ranges if given.
        size_x_range, size_y_range: [min, max] tuples.
        height_min, height_max: pillar height range (meters).
        allow_height_variation: if True, randomize height in [0.6*h, h].
        rng: np.random.RandomState for deterministic placement.

    Returns:
        list of (cx, cy, sx, sy, h) in world meters.
    """
    if size_x_range is not None:
        size_x = rng.uniform(*size_x_range)
    if size_y_range is not None:
        size_y = rng.uniform(*size_y_range)

    max_attempts = count * 100
    positions = []
    for _ in range(max_attempts):
        if len(positions) >= count:
            break
        r = rng.uniform(clear_radius, spawn_radius)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        cx = center_x + r * np.cos(theta)
        cy = center_y + r * np.sin(theta)

        if np.hypot(cx - center_x, cy - center_y) < clear_radius:
            continue
        valid = True
        for px, py in positions:
            if np.hypot(cx - px, cy - py) < min_separation:
                valid = False
                break
        if valid:
            positions.append((cx, cy))

    pillars = []
    for cx, cy in positions:
        if allow_height_variation:
            h = rng.uniform(height_min * 0.6, height_max)
        else:
            h = rng.uniform(height_min, height_max)
        sx = rng.uniform(size_x_range[0], size_x_range[1]) if size_x_range else size_x
        sy = rng.uniform(size_y_range[0], size_y_range[1]) if size_y_range else size_y
        pillars.append((cx, cy, sx, sy, h))
    return pillars


def build_box_mesh(cx, cy, sx, sy, h):
    """Build 8 vertices + 12 triangles for an axis-aligned box.

    Returns (verts_8x3, tris_12x3) in world frame.
    """
    x0, x1 = cx - sx / 2.0, cx + sx / 2.0
    y0, y1 = cy - sy / 2.0, cy + sy / 2.0
    z0, z1 = 0.0, h

    verts = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)

    tris = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [1, 2, 6], [1, 6, 5],  # right
        [2, 3, 7], [2, 7, 6],  # back
        [3, 0, 4], [3, 4, 7],  # left
    ], dtype=np.int32)

    return verts, tris


def generate_pillar_lidar_mesh(terrain_cfg, pd_cfg, device='cuda:0'):
    """Build a ground plane + pillar clusters per sub-terrain cell.

    Uses terrain_cfg grid (num_rows, num_cols, terrain_length, terrain_width)
    so pillar placement matches the formal pillar config parameters.

    Args:
        terrain_cfg: terrain config with num_rows, num_cols, terrain_length,
                     terrain_width, border_size.
        pd_cfg: pd_risknet config with pillar_* parameters.
        device: torch device string.

    Returns:
        (vertices, triangles_np, pillar_boxes)
        pillar_boxes: list of (cell_id, cx, cy, sx, sy, h) where cell_id =
                      row * num_cols + col.
    """
    num_rows = terrain_cfg.num_rows
    num_cols = terrain_cfg.num_cols
    t_len = terrain_cfg.terrain_length
    t_wid = terrain_cfg.terrain_width
    border = getattr(terrain_cfg, 'border_size', 0.0)

    total_x = num_cols * t_len
    total_y = num_rows * t_wid

    # Ground plane
    plane_verts = np.array([
        [-border,             -border,              0.0],
        [total_x + border,    -border,              0.0],
        [total_x + border,     total_y + border,    0.0],
        [-border,              total_y + border,     0.0],
    ], dtype=np.float32)
    plane_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    all_verts = [plane_verts]
    all_tris = [plane_tris]
    vert_offset = 4
    pillar_boxes = []

    rng = np.random.RandomState(42)

    for row in range(num_rows):
        for col in range(num_cols):
            cell_id = row * num_cols + col
            center_x = col * t_len + t_len / 2.0
            center_y = row * t_wid + t_wid / 2.0

            pillars = generate_pillar_positions(
                center_x=center_x, center_y=center_y,
                spawn_radius=pd_cfg.pillar_spawn_radius,
                clear_radius=pd_cfg.pillar_center_clear_radius,
                min_separation=pd_cfg.pillar_min_separation,
                count=pd_cfg.pillar_count,
                size_x=None, size_y=None,
                size_x_range=[pd_cfg.pillar_size_x_min, pd_cfg.pillar_size_x_max],
                size_y_range=[pd_cfg.pillar_size_y_min, pd_cfg.pillar_size_y_max],
                height_min=pd_cfg.pillar_height_min,
                height_max=pd_cfg.pillar_height_max,
                allow_height_variation=pd_cfg.pillar_allow_height_variation,
                rng=rng,
            )

            for cx, cy, sx, sy, h in pillars:
                verts, tris = build_box_mesh(cx, cy, sx, sy, h)
                all_verts.append(verts)
                all_tris.append(tris + vert_offset)
                vert_offset += 8
                pillar_boxes.append((cell_id, cx, cy, sx, sy, h))

    vertices_np = np.concatenate(all_verts, axis=0).astype(np.float32)
    triangles_np = np.concatenate(all_tris, axis=0).astype(np.int32)
    vertices = torch.as_tensor(vertices_np, device=device, dtype=torch.float32)
    return vertices, triangles_np, pillar_boxes