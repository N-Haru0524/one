import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one import ossop, ovw
from one_assembly.robots.khi_bunri.khi_bunri import KHIBunri


base = ovw.World(
    cam_pos=(3.2, 1.8, 2.0),
    cam_lookat_pos=(0.2, 0.0, 0.6),
    toggle_auto_cam_orbit=False,
)
ossop.frame(length_scale=0.25, radius_scale=1.2).attach_to(base.scene)

robot = KHIBunri()
robot.attach_to(base.scene)

lft_tcp_frame = ossop.frame(
    pos=robot.lft_tcp_tf[:3, 3],
    rotmat=robot.lft_tcp_tf[:3, :3],
    length_scale=0.18,
    radius_scale=0.8,
)
lft_tcp_frame.attach_to(base.scene)
rgt_tcp_frame = ossop.frame(
    pos=robot.rgt_tcp_tf[:3, 3],
    rotmat=robot.rgt_tcp_tf[:3, :3],
    length_scale=0.18,
    radius_scale=0.8,
)
rgt_tcp_frame.attach_to(base.scene)

# robot.lft_arm.fk(np.radians([0.0, -10.0, -95.0, 0.0, -80.0, -90.0]).astype(np.float32))
# robot.rgt_arm.fk(np.radians([25.0, 10.0, -100.0, 80.0, -105.0, 235.0]).astype(np.float32))
lft_tcp_frame.set_rotmat_pos(rotmat=robot.lft_tcp_tf[:3, :3], pos=robot.lft_tcp_tf[:3, 3])
rgt_tcp_frame.set_rotmat_pos(rotmat=robot.rgt_tcp_tf[:3, :3], pos=robot.rgt_tcp_tf[:3, 3])

base.run()
