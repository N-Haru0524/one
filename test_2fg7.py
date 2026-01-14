import numpy as np
from one import ovw, ouc, ossop, or_2fg7

base = ovw.World(cam_pos=(.5, .5, .5), cam_lookat_pos=(0, 0, .2))
oframe = ossop.gen_frame().attach_to(base.scene)
gripper = or_2fg7.OR2FG7()
#
# gripper.toggle_render_collision = True
gripper.attach_to(base.scene)
# base.run()

box = ossop.gen_cylinder(spos=(.3, 0, 0), epos=(.3, 0, .1), radius=.03,
                         collision_type=ouc.CollisionType.AABB,
                         is_free=True)
box.attach_to(base.scene)
# box.toggle_render_collision = True
# gripper.grasp(box)

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
    print(gripper.qs)

base.schedule_interval(update_finger, interval=.01)
base.run()