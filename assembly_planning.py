import builtins
import numpy as np

from one import oum, ovw, ouc, ossop, osso, khi_rs007l, or_2fg7
from one.grasp.antipodal import antipodal
from planner.ppplanner import PickPlacePlanner


base = ovw.World(cam_pos=(-1.2, 1.2, 1.0), cam_lookat_pos=(0.3, 0.0, 0.3))
builtins.base = base
ossop.gen_frame().attach_to(base.scene)

robot = khi_rs007l.RS007L()
robot.set_rotmat_pos(pos=(0.0, 0.0, 0.01))
gripper = or_2fg7.OR2FG7()
robot.engage(gripper)
robot.attach_to(base.scene)

table = ossop.gen_box(half_extents=(0.4, 0.4, 0.02), pos=(0.4, 0.0, 0.01),
                      collision_type=ouc.CollisionType.AABB)
table.rgb = ouc.ExtendedColor.SILVER_GRAY
table.attach_to(base.scene)

belt = osso.SceneObject.from_file('meshes/belt.stl',
                                  collision_type=ouc.CollisionType.MESH)
belt.set_rotmat_pos(pos=(0.35, 0.0, 0.06))
belt.attach_to(base.scene)

target_pose = (np.array([0.55, 0.0, 0.06], dtype=np.float32), belt.rotmat)
target_marker = belt.clone()
target_marker.set_rotmat_pos(rotmat=target_pose[1], pos=target_pose[0])
target_marker.alpha = 0.2
target_marker.attach_to(base.scene)

print('Computing grasps for belt...')
grasps = antipodal(scene_obj=belt, gripper=gripper, obstacles=[belt],
                   density=0.01, normal_tol_deg=20, roll_step_deg=30,
                   max_grasps=10)
if not grasps:
    raise RuntimeError('No valid grasps found for meshes/belt.stl')

pick_pose_tf, jaw_width, score = grasps[0]
place_pose_tf = oum.tf_from_rotmat_pos(pick_pose_tf[:3, :3], target_pose[0])

planner = PickPlacePlanner(robot)
pick_plan = planner.plan_pick(pick_pose_tf, jaw_width, obstacles=[table])
if pick_plan is None:
    raise RuntimeError('Pick plan failed')
place_plan = planner.plan_place(place_pose_tf, start_qs=pick_plan.qs_list[-1],
                                obstacles=[table])
if place_plan is None:
    raise RuntimeError('Place plan failed')

plan_qs = pick_plan.qs_list + place_plan.qs_list[1:]
pick_end_idx = len(pick_plan.qs_list) - 1
place_end_idx = len(plan_qs) - 1

start_pose = (belt.pos.copy(), belt.rotmat.copy())


def reset_scene():
    if belt in gripper._mountings:
        gripper.release(belt)
    belt.set_rotmat_pos(rotmat=start_pose[1], pos=start_pose[0])
    gripper.open()


def update(dt, state):
    if state['idx'] >= len(plan_qs):
        reset_scene()
        state['idx'] = 0
        return
    robot.fk(qs=plan_qs[state['idx']])
    if state['idx'] == pick_end_idx:
        gripper.grasp(belt, jaw_width=jaw_width)
    if state['idx'] == place_end_idx:
        gripper.release(belt)
        gripper.open()
    state['idx'] += 1


reset_scene()
base.schedule_interval(update, interval=0.1, state={'idx': 0})
base.run()
