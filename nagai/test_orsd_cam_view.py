"""Dual-camera view-collision demo for the or_sd_cam ORSD.

Ports the `is_view_collided` / `get_view_rays` idea from
from_wrs/viewpoint/orsd_dualcamera*.py into the `one` framework.

Two cameras are rigidly attached to the screwdriver body. A ray is cast from
each camera toward a fixed target point; if an obstacle blocks the line of
sight, the ray turns red, otherwise green. The obstacle oscillates across the
sight lines so you can watch the occlusion toggle live.

Run: uv run python nagai/test_orsd_cam_view.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one import ovw, ouc, ossop
from one_assembly.robots.or_sd_cam import or_sd

base = ovw.World(cam_pos=(0.55, 0.45, 0.45),
                 cam_lookat_pos=(0.15, 0, 0.18),
                 toggle_auto_cam_orbit=False)
ossop.frame().attach_to(base.scene)

screwdriver = or_sd.ORSD()
screwdriver.attach_to(base.scene)

# camera frames (for reference). the cameras aim at the screw acting point.
cam_tfs = screwdriver.camera_tfs()
for cam_tf in cam_tfs:
    ossop.frame(pos=cam_tf[:3, 3], rotmat=cam_tf[:3, :3],
                length_scale=0.4, radius_scale=0.4).attach_to(base.scene)

# view target: a bit beyond the screw acting point, along the driving axis
# (TCP +z == flange +x), i.e. a point on the workpiece ahead of the screw.
flange_tf = screwdriver.runtime_root_lnk.tf
screw_pos = (flange_tf @ np.append(screwdriver.tcp_pos, 1.0))[:3]
drive_dir = flange_tf[:3, :3] @ screwdriver.tcp_rotmat[:, 2]  # screw +z in world
tgt_pos = screw_pos + 0.08 * drive_dir
avg_cam_pos = np.mean([t[:3, 3] for t in cam_tfs], axis=0)

target = ossop.sphere(pos=tgt_pos, radius=0.006, rgb=ouc.BasicColor.GREEN)
target.attach_to(base.scene)

# obstacle: oscillates along world Y, crossing the sight lines near their middle
obstacle_center = 0.5 * (avg_cam_pos + tgt_pos)
obstacle = ossop.box(pos=obstacle_center,
                     half_extents=(0.012, 0.012, 0.03),
                     rgb=ouc.BasicColor.GRAY,
                     alpha=0.6,
                     collision_type=ouc.CollisionType.MESH,
                     is_free=True)
obstacle.attach_to(base.scene)

view_rays = []
phase = 0.0
shank = screwdriver.shank_range[0]
shank_flag = 'extend'


def update(dt):
    global view_rays, phase, shank, shank_flag

    # extend/retract the shank (cameras are flange-fixed, so they don't move)
    if shank_flag == 'extend':
        shank += 0.0005
        if shank >= screwdriver.shank_range[1]:
            shank = screwdriver.shank_range[1]
            shank_flag = 'retract'
    else:
        shank -= 0.0005
        if shank <= screwdriver.shank_range[0]:
            shank = screwdriver.shank_range[0]
            shank_flag = 'extend'
    screwdriver.fk(qs=[shank])

    # sweep the obstacle across the line of sight
    phase += dt
    obstacle.pos = obstacle_center + np.array([0.0, 0.05 * np.sin(phase * 1.5), 0.0])

    occluded = screwdriver.is_view_collided(tgt_pos, obstacle_list=[obstacle])
    obstacle.rgba = (1, 0, 0, 0.6) if occluded else (0.4, 0.4, 0.4, 0.6)

    # redraw the colored view rays
    for ray in view_rays:
        ray.detach_from(base.scene)
    view_rays = screwdriver.get_view_rays(tgt_pos)
    for ray in view_rays:
        ray.rgb = ouc.BasicColor.RED if occluded else ouc.BasicColor.GREEN
        ray.attach_to(base.scene)


base.schedule_interval(update, interval=0.02)
base.run()
