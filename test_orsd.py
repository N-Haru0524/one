import numpy as np
from one import ovw, ouc, ossop, orsd, ocm

base = ovw.World(cam_pos=(.5, .5, .5), cam_lookat_pos=(0, 0, .2))
oframe = ossop.gen_frame().attach_to(base.scene)
screwdriver = orsd.ORSD()

screwdriver.toggle_render_collision = True
screwdriver.attach_to(base.scene)
# base.run()

box = ossop.gen_cylinder(spos=(.0, .22, 0), epos=(.0, .22, .1), radius=.03,
                         collision_type=ouc.CollisionType.AABB,
                         is_free=True)
box.attach_to(base.scene)
box.toggle_render_collision = True
# gripper.grasp(box)

collider = ocm.MjCollider()
collider.append(screwdriver)
collider.append(box)
collider.actors = [screwdriver]
collider.compile()

flag = "extend"
qs = screwdriver.qs[0]
def update_finger(dt):
    global flag, qs
    if flag == "extend":
        qs = screwdriver.qs[0] + 0.0001
        if qs >= screwdriver.shank_range[1]:
            qs = screwdriver.shank_range[1]
            flag = "retract"
    else:
        qs = screwdriver.qs[0] - 0.0001
        if qs <= screwdriver.shank_range[0]:
            qs = screwdriver.shank_range[0]
            flag = "extend"
    screwdriver.set_shank_len(qs)
    print(screwdriver.qs)
    if collider.is_collided([qs]):
        box.rgba = (1, 0, 0, 0.5)
    else:
        box.rgba = (0, 1, 0, 0.5)

base.schedule_interval(update_finger, interval=.01)
base.run()