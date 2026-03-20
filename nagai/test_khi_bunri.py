import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one import ocm, ossop, ovw
from one.physics import mj_contact
from one_assembly.precise_collision import PreciseSIMDCollider
from one_assembly.robots.khi_bunri.khi_bunri import KHIBunri


def compose_robot_state(robot):
    return np.concatenate(
        [
            robot.lft_arm.qs[:robot.lft_arm.ndof],
            robot.lft_gripper.qs[:robot.lft_gripper.ndof],
            robot.rgt_arm.qs[:robot.rgt_arm.ndof],
            robot.rgt_screwdriver.qs[:robot.rgt_screwdriver.ndof],
        ]
    ).astype(np.float32)


def print_mj_contacts(collider, qs):
    collided = collider.is_collided(qs)
    print('mj_initial_collision =', collided)
    print('ncon =', collider._mjenv.data.ncon)
    contact_points = []
    for i in range(int(collider._mjenv.data.ncon)):
        contact = collider._mjenv.data.contact[i]
        body_id_1 = int(collider._mjenv.model.geom_bodyid[contact.geom1])
        body_id_2 = int(collider._mjenv.model.geom_bodyid[contact.geom2])
        body_1 = mujoco.mj_id2name(collider._mjenv.model, mujoco.mjtObj.mjOBJ_BODY, body_id_1)
        body_2 = mujoco.mj_id2name(collider._mjenv.model, mujoco.mjtObj.mjOBJ_BODY, body_id_2)
        pos = np.asarray(contact.pos, dtype=np.float32)
        contact_points.append(pos.copy())
        print(
            f'mj_contact[{i}] {body_1} <-> {body_2}, '
            f'pos={pos.tolist()}, dist={float(contact.dist):.6f}'
        )
    if len(contact_points) == 0:
        print('mj_contact[0] no hit pairs')
    return collided, contact_points


def _entity_label(entity):
    name = getattr(entity, 'name', None)
    if isinstance(name, str) and name:
        return name
    file_path = getattr(entity, 'file_path', None)
    if isinstance(file_path, str) and file_path:
        return Path(file_path).stem
    return type(entity).__name__


def print_simd_contacts(collider):
    collided = collider.is_collided(initial_state)
    print('simd_initial_collision =', collided)
    hit_count = 0
    contact_points = []
    pair_groups = (
        ('self', collider._self_collision_pairs),
        ('actor_sobj', collider._actor_sobj_pairs),
        ('actor_mecba', collider._actor_mecba_pairs),
        ('actor_actor', collider._actor_actor_pairs),
    )
    for group_name, pairs in pair_groups:
        for pair in pairs:
            if group_name == 'self':
                actor, lidx_i, lidx_j, col_i, col_j = pair
                obj_a = actor.runtime_lnks[lidx_i]
                obj_b = actor.runtime_lnks[lidx_j]
                points = collider._check_pair_direct(col_i, obj_a.tf, col_j, obj_b.tf)
            elif group_name == 'actor_sobj':
                actor, lidx, sobj, col_lnk, col_sobj = pair
                obj_a = actor.runtime_lnks[lidx]
                obj_b = sobj
                points = collider._check_pair_direct(col_lnk, obj_a.tf, col_sobj, obj_b.tf)
            elif group_name == 'actor_mecba':
                actor, actor_lidx, robot_obs, obs_lidx, col_a, col_o = pair
                obj_a = actor.runtime_lnks[actor_lidx]
                obj_b = robot_obs.runtime_lnks[obs_lidx]
                points = collider._check_pair_direct(col_a, obj_a.tf, col_o, obj_b.tf)
            else:
                actor_a, lidx_a, actor_b, lidx_b, col_a, col_b = pair
                obj_a = actor_a.runtime_lnks[lidx_a]
                obj_b = actor_b.runtime_lnks[lidx_b]
                points = collider._check_pair_direct(col_a, obj_a.tf, col_b, obj_b.tf)
            if points is None or len(points) == 0:
                continue
            hit_count += 1
            first_point = np.asarray(points[0], dtype=np.float32)
            contact_points.append(first_point.copy())
            print(
                f'simd_contact[{hit_count}] group={group_name} '
                f'{_entity_label(obj_a)} <-> {_entity_label(obj_b)}, '
                f'point={first_point.tolist()}, npts={len(points)}'
            )
    if hit_count == 0:
        print('simd_contact[0] no hit pairs')
    return contact_points


base = ovw.World(
    cam_pos=(3.2, 1.8, 2.0),
    cam_lookat_pos=(0.2, 0.0, 0.6),
    toggle_auto_cam_orbit=False,
)
ossop.frame(length_scale=0.25, radius_scale=1.2).attach_to(base.scene)

robot = KHIBunri()
robot.attach_to(base.scene)
robot.body.toggle_render_collision = True
robot.lft_arm.toggle_render_collision = True
robot.rgt_arm.toggle_render_collision = True

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

robot.lft_arm.fk([0.3810805380344391, 0.19793473184108734, -1.644507646560669, -0.2810658812522888, -1.745247721672058, -1.9072362184524536])
# robot.rgt_arm.fk(np.radians([25.0, 10.0, -100.0, 80.0, -105.0, 235.0]).astype(np.float32))
lft_tcp_frame.set_rotmat_pos(rotmat=robot.lft_tcp_tf[:3, :3], pos=robot.lft_tcp_tf[:3, 3])
rgt_tcp_frame.set_rotmat_pos(rotmat=robot.rgt_tcp_tf[:3, :3], pos=robot.rgt_tcp_tf[:3, 3])

collider = ocm.MJCollider()
collider.append(robot.body)
collider.append(robot.lft_arm)
collider.append(robot.lft_gripper)
collider.append(robot.rgt_arm)
collider.append(robot.rgt_screwdriver)
collider.actors = [robot.lft_arm, robot.lft_gripper, robot.rgt_arm, robot.rgt_screwdriver]
collider.compile()

contact_viz = mj_contact.MJContactViz(base.scene, max_contacts=128, radius=0.006)

initial_state = compose_robot_state(robot)
mj_collided, mj_contact_points = print_mj_contacts(collider, initial_state)
contact_viz.update_from_data(collider._mjenv.data)
for pos in mj_contact_points:
    marker = ossop.sphere(
        pos=pos,
        radius=0.005,
        rgb=(1.0, 0.2, 0.0),
        alpha=0.7,
        collision_type=None,
        is_fixed=True,
    )
    marker.attach_to(base.scene)

simd_collider = PreciseSIMDCollider(use_gpu=True, max_points=16)
simd_collider.append(robot.body)
simd_collider.append(robot.lft_arm)
simd_collider.append(robot.lft_gripper)
simd_collider.append(robot.rgt_arm)
simd_collider.append(robot.rgt_screwdriver)
simd_collider.actors = [robot.lft_arm, robot.lft_gripper, robot.rgt_arm, robot.rgt_screwdriver]
simd_collider.compile()
simd_contact_points = print_simd_contacts(simd_collider)
print('collision_summary =', {'mj': mj_collided, 'simd': bool(simd_contact_points)})
for pos in simd_contact_points:
    marker = ossop.sphere(
        pos=pos,
        radius=0.004,
        rgb=(0.0, 0.2, 1.0),
        alpha=0.7,
        collision_type=None,
        is_fixed=True,
    )
    marker.attach_to(base.scene)

base.run()
