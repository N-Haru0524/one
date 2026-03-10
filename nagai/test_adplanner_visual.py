import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one import oum, ovw, ouc, ossop, ocm, omppc, khi_rs007l
from one_assembly.motion_planner import ADPlanner


base = ovw.World(
    cam_pos=(-2.2, 2.0, 1.8),
    cam_lookat_pos=(0.0, 0.0, 0.6),
    toggle_auto_cam_orbit=False,
)
ossop.frame().attach_to(base.scene)

robot = khi_rs007l.RS007L()
robot.is_free = True
robot.rotmat = oum.rotmat_from_euler(0, 0, -oum.pi / 2)
robot.attach_to(base.scene)

wall = ossop.box(
    half_extents=(0.8, 0.02, 0.18),
    pos=(0.0, -0.28, 0.92),
    collision_type=ouc.CollisionType.AABB,
)
wall.attach_to(base.scene)
pillar = ossop.box(
    half_extents=(0.08, 0.08, 0.45),
    pos=(0.32, 0.02, 0.45),
    collision_type=ouc.CollisionType.AABB,
)
pillar.attach_to(base.scene)

collider = ocm.MJCollider()
collider.append(robot)
collider.append(wall)
collider.append(pillar)
collider.actors = [robot]
collider.compile(margin=0.0)

planner = ADPlanner(robot, pln_ctx=omppc.PlanningContext(collider=collider))

start_qs = np.zeros(6, dtype=np.float32)
goal_qs = np.array(
    [-0.35, -0.0, 1.20, 0.05, 0.72, 0.10],
    dtype=np.float32,
)
end_qs = np.array(
    [0.35, -0.50, 0.95, -0.10, 0.55, -0.15],
    dtype=np.float32,
)
robot_goal_pose = robot.clone()
robot_goal_pose.fk(qs=goal_qs)
goal_tcp_tf = robot_goal_pose.gl_tcp_tf

print('\nStarting ADPlanner visual test...')
t0 = time.time()
state_plan = planner.gen_approach_depart(
    goal_tcp_pos=goal_tcp_tf[:3, 3],
    goal_tcp_rotmat=goal_tcp_tf[:3, :3],
    start_qs=start_qs,
    end_qs=end_qs,
    approach_direction=np.array([0.0, 0.0, -1.0], dtype=np.float32),
    approach_distance=0.05,
    depart_direction=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    depart_distance=0.05,
    linear_granularity=0.02,
    use_rrt=False,
)
t1 = time.time()

print(f"\n{'=' * 60}")
print(f'Planning completed in {t1 - t0:.3f}s')
if state_plan is None:
    print('No path found')
    print(f"{'=' * 60}")
    base.run()

state_list = state_plan.qs_list
print(f'Path found with {len(state_list)} waypoints')
print(f"{'=' * 60}")

robot_start = robot.clone()
robot_start.fk(qs=start_qs)
robot_start.rgba = (1.0, 0.0, 0.0, 0.45)
robot_start.attach_to(base.scene)

robot_goal = robot.clone()
robot_goal.fk(qs=goal_qs)
robot_goal.rgba = (0.0, 0.2, 1.0, 0.25)
robot_goal.attach_to(base.scene)

robot_end = robot.clone()
robot_end.fk(qs=end_qs)
robot_end.rgba = (0.0, 0.7, 0.1, 0.35)
robot_end.attach_to(base.scene)

counter = [0]


def update_pose(dt, counter):
    if counter[0] < len(state_list):
        robot.fk(qs=state_list[counter[0]])
        counter[0] += 1
    else:
        counter[0] = 0


base.schedule_interval(update_pose, interval=0.08, counter=counter)
base.run()
