import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from one import ovw, ouc, ossop, ocm
from one_assembly.robots.or_2fg7 import or_2fg7
from one.physics import mj_contact

base = ovw.World(cam_pos=(.5, .5, .5), cam_lookat_pos=(0, 0, .2),
                toggle_auto_cam_orbit=False)
oframe = ossop.frame().attach_to(base.scene)
gripper = or_2fg7.OR2FG7()
#
# gripper.toggle_render_collision = True
gripper.attach_to(base.scene)
tcpframe = ossop.frame(pos=gripper.fk()[-1][:3, 3], rotmat=gripper.fk()[-1][:3, :3])
tcpframe.attach_to(base.scene)
# base.run()

box = ossop.cylinder(spos=(.0, .055, 0.15), epos=(.0, .055, 0.25), radius=.03,
                         collision_type=ouc.CollisionType.AABB,
                         is_free=True)
box.attach_to(base.scene)
box.toggle_render_collision = True

collider = ocm.MJCollider()
collider.append(gripper)
collider.append(box)
collider.actors = [gripper]
collider.compile()

contact_viz = mj_contact.MJContactViz(base.scene, max_contacts=64, radius=0.003)
contact_force_viz = mj_contact.MjContactForceViz(base.scene, max_contacts=64)

flag = "open"
def update_finger(dt):
    global flag
    if flag == "open":
        qs = gripper.qs[0] + 0.0001
        if qs >= gripper.jaw_range[1] / 2 :
            qs = gripper.jaw_range[1] / 2
            flag = "close"
    else:
        qs = gripper.qs[0] - 0.0001
        if qs <= gripper.jaw_range[0] / 2 :
            qs = gripper.jaw_range[0] / 2
            flag = "open"
    gripper.set_jaw_width(qs * 2)
    print("jaw width:", gripper.qs[0] * 2)
    collided = collider.is_collided([qs, qs])
    contact_viz.update_from_data(collider._mjenv.data)
    contact_force_viz.update_from_data(collider._mjenv.model, collider._mjenv.data)
    if collided:
        box.rgba = (1, 0, 0, 0.5)
    else:
        box.rgba = (0, 1, 0, 0.5)

base.schedule_interval(update_finger, interval=.01)
base.run()
