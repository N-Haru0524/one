"""
Visual test for retime_global: plan an RRT path on RS007L, time-parameterize it,
and replay the motion in the viewer at real-time speed.

Run:
    ./dev_gui.sh ./codex_python.sh nagai/test_retime_global_visual.py
"""
import builtins
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one import ocm, omppc, ompr, ossop, ouc, oum, ovw, khi_rs007l
from one_assembly.motion_planner.time_param import retime_global

# ---------------------------------------------------------------------------
# RS007L velocity / acceleration limits (Kawasaki spec, rad/s and rad/s^2)
# J1-J3: 200 deg/s,  J4-J6: 300 deg/s
# ---------------------------------------------------------------------------
V_MAX = np.deg2rad([200, 200, 230, 300, 300, 550]).astype(np.float32)
A_MAX = (V_MAX * 2.0).astype(np.float32)
DT = 0.01  # 100 Hz playback

# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------
base = ovw.World(
    cam_pos=np.array([-2.0, 2.0, 2.0]),
    cam_lookat_pos=np.array([0.0, 0.0, 0.5]),
    toggle_auto_cam_orbit=False,
)
builtins.base = base

ossop.frame(length_scale=0.3, radius_scale=1.2).attach_to(base.scene)

robot = khi_rs007l.RS007L()
robot.rotmat = oum.rotmat_from_euler(0, 0, -oum.pi / 2)
robot.attach_to(base.scene)

tcp_frame = ossop.frame(
    pos=robot.gl_tcp_tf[:3, 3],
    rotmat=robot.gl_tcp_tf[:3, :3],
    length_scale=0.18,
    radius_scale=0.8,
)
tcp_frame.attach_to(base.scene)

# obstacles
wall = ossop.box(
    half_extents=(1.0, 0.01, 0.15),
    pos=(0.0, -0.30, 1.0),
    collision_type=ouc.CollisionType.AABB,
)
wall.attach_to(base.scene)

wall2 = ossop.box(
    half_extents=(0.15, 0.01, 1.0),
    pos=(-0.5, -0.30, 0.5),
    collision_type=ouc.CollisionType.AABB,
)
wall2.attach_to(base.scene)

wall3 = ossop.box(
    half_extents=(0.01, 1.0, 0.15),
    pos=(0.3, 0.0, 1.0),
    collision_type=ouc.CollisionType.AABB,
)
wall3.attach_to(base.scene)

# ---------------------------------------------------------------------------
# RRT planning  (probabilistic — retry on failure)
# ---------------------------------------------------------------------------
collider = ocm.MJCollider()
collider.append(robot)
collider.append(wall)
collider.append(wall2)
collider.append(wall3)
collider.actors = [robot]
collider.compile()

pln_ctx = omppc.PlanningContext(collider=collider, cd_step_size=np.pi / 180)
planner = ompr.RRTConnectPlanner(pln_ctx=pln_ctx, extend_step_size=np.pi / 36)

START = np.zeros(6, dtype=np.float32)
GOAL  = np.array([-np.pi / 2, -np.pi / 4, np.pi / 2,
                  -np.pi / 2,  np.pi / 4, np.pi / 3], dtype=np.float32)

raw_path = None
for attempt in range(5):
    print(f'planning RRT path (attempt {attempt + 1}/5)...')
    t0 = time.perf_counter()
    raw_path = planner.solve(start=START, goal=GOAL, max_iters=5000)
    t1 = time.perf_counter()
    if raw_path is not None:
        break

if raw_path is None:
    raise RuntimeError('RRT failed after 5 attempts')

raw_path = ompr.shortcut_path(raw_path, pln_ctx)
q_seq = np.asarray(raw_path, dtype=np.float32)
print(f'RRT: {len(q_seq)} waypoints in {t1 - t0:.2f}s')

# ---------------------------------------------------------------------------
# time parameterization
# ---------------------------------------------------------------------------
t_seq, q_out, qd_out, qdd_out = retime_global(q_seq, V_MAX, A_MAX, dt=DT)
print(f'retime_global: {len(t_seq)} samples, total {t_seq[-1]:.3f}s  '
      f'(vs trapezoidal baseline: segments×stop-start)')

# also run reverse (GOAL → START) for looping
t_seq_r, q_out_r, _, _ = retime_global(q_seq[::-1], V_MAX, A_MAX, dt=DT)
print(f'reverse path:  {len(t_seq_r)} samples, total {t_seq_r[-1]:.3f}s')

# ---------------------------------------------------------------------------
# playback
# ---------------------------------------------------------------------------
state = {
    'phase': 'fwd',   # 'fwd' | 'rev'
    'idx': 0,
    'paused': False,
}

PAUSE_FRAMES = int(0.5 / DT)  # 0.5 s pause at each end
pause_counter = {'n': 0}


def update(_dt):
    if state['paused']:
        pause_counter['n'] += 1
        if pause_counter['n'] >= PAUSE_FRAMES:
            state['paused'] = False
            pause_counter['n'] = 0
            state['idx'] = 0
            state['phase'] = 'rev' if state['phase'] == 'fwd' else 'fwd'
        return

    traj = q_out if state['phase'] == 'fwd' else q_out_r
    idx = state['idx']
    robot.fk(traj[idx])
    tcp_frame.set_rotmat_pos(rotmat=robot.gl_tcp_tf[:3, :3], pos=robot.gl_tcp_tf[:3, 3])

    state['idx'] += 1
    if state['idx'] >= len(traj):
        state['paused'] = True


robot.fk(START)
base.schedule_interval(update, interval=DT)
base.run()
