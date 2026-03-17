import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one import ovw, ouc, ossop, ocm
from one.physics import mj_contact
from one_assembly.robots.or_sd import or_sd

base = ovw.World(cam_pos=(0.5, 0.5, 0.5),
                 cam_lookat_pos=(0, 0, 0.2),
                 toggle_auto_cam_orbit=False)
ossop.frame().attach_to(base.scene)

screwdriver = or_sd.ORSD()
screwdriver.toggle_render_collision = True
screwdriver.attach_to(base.scene)

tcp_tf = screwdriver.gl_tcp_tf
tcpframe = ossop.frame(pos=tcp_tf[:3, 3], rotmat=tcp_tf[:3, :3], radius_scale=0.5)
tcpframe.attach_to(base.scene)

box = ossop.cylinder(spos=(0.2, -0.03, 0.095),
                     epos=(0.2, 0.03, 0.095),
                     radius=0.012,
                     collision_type=ouc.CollisionType.AABB,
                     is_free=True)
box.attach_to(base.scene)
box.toggle_render_collision = True

collider = ocm.MJCollider()
collider.append(screwdriver)
collider.append(box)
collider.actors = [screwdriver]
collider.compile()

contact_viz = mj_contact.MJContactViz(base.scene, max_contacts=64, radius=0.003)
contact_force_viz = mj_contact.MjContactForceViz(base.scene, max_contacts=64)

flag = 'extend'
def update_shank(dt):
    global flag

    qs = screwdriver.qs[0]
    if flag == 'extend':
        qs += 0.0005
        if qs >= screwdriver.shank_range[1]:
            qs = screwdriver.shank_range[1]
            flag = 'retract'
    else:
        qs -= 0.0005
        if qs <= screwdriver.shank_range[0]:
            qs = screwdriver.shank_range[0]
            flag = 'extend'

    screwdriver.fk(qs=[qs])
    tcpframe.set_rotmat_pos(rotmat=screwdriver._loc_tcp_tf[:3, :3], pos=screwdriver._loc_tcp_tf[:3, 3])

    collided = collider.is_collided([qs])
    contact_viz.update_from_data(collider._mjenv.data)
    contact_force_viz.update_from_data(collider._mjenv.model, collider._mjenv.data)
    if collided:
        box.rgba = (1, 0, 0, 0.5)
    else:
        box.rgba = (0, 1, 0, 0.5)


base.schedule_interval(update_shank, interval=0.01)
base.run()
