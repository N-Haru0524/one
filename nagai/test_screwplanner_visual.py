import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one import ovw, ouc, ossop, ocm, khi_rs007l
import one.robots.end_effectors.onrobot.or_sd.or_sd as oreorsd
from one_assembly.motion_planner import ScrewPlanner, utils as omp_utils
import one.utils.math as oum


def split_state(robot, screwdriver, qs):
    qs = oum.np.asarray(qs, dtype=oum.np.float32).reshape(-1)
    robot_qs = qs[:robot.ndof]
    screw_qs = qs[robot.ndof:robot.ndof + getattr(screwdriver, 'ndof', 0)]
    return robot_qs, screw_qs


def apply_screwdriver_state(screwdriver, active_qs):
    del screwdriver
    del active_qs


def main():
    base = ovw.World(
        cam_pos=(-1.8, 1.8, 1.35),
        cam_lookat_pos=(0.35, -0.02, 0.35),
        toggle_auto_cam_orbit=False,
    )
    scene = base.scene
    ossop.frame(length_scale=0.2, radius_scale=1.2).attach_to(scene)

    robot = khi_rs007l.RS007L()
    robot.set_rotmat_pos(pos=(0.0, 0.0, 0.01))
    screwdriver = oreorsd.ORSD()
    robot.engage(screwdriver, engage_tf=oum.tf_from_rotmat_pos(pos=(0.0, 0.0, 0.05)))
    robot.attach_to(scene)

    table = ossop.box(
        half_extents=(0.45, 0.35, 0.02),
        pos=(0.42, 0.0, -0.03),
        rgb=ouc.ExtendedColor.SILVER_GRAY,
        collision_type=ouc.CollisionType.AABB,
    )
    table.attach_to(scene)

    fixture = ossop.box(
        half_extents=(0.08, 0.08, 0.08),
        pos=(0.45, -0.10, 0.08),
        rgb=(0.72, 0.78, 0.85),
        collision_type=ouc.CollisionType.AABB,
    )
    fixture.attach_to(scene)

    wall = ossop.box(
        half_extents=(0.15, 0.02, 0.16),
        pos=(0.28, 0.18, 0.16),
        rgb=(0.85, 0.80, 0.76),
        collision_type=ouc.CollisionType.AABB,
    )
    wall.attach_to(scene)

    screw_pos = oum.vec(0.45, -0.10, 0.16).astype(oum.np.float32)
    screw_axis = oum.vec(0.0, 0.0, -1.0).astype(oum.np.float32)
    hole_axis = ossop.cylinder(
        spos=screw_pos,
        epos=screw_pos + screw_axis * 0.10,
        radius=0.006,
        rgb=(0.85, 0.25, 0.15),
        alpha=0.8,
    )
    hole_axis.attach_to(scene)
    ossop.frame(
        pos=screw_pos,
        rotmat=oum.frame_from_normal(screw_axis),
        length_scale=0.08,
        radius_scale=0.8,
        color_mat=ouc.CoordColor.DYO,
        alpha=0.8,
    ).attach_to(scene)

    start_rotmat = oum.rotmat_from_euler(oum.pi, 0.0, oum.pi)
    start_pos = oum.vec(0.30, -0.28, 0.24).astype(oum.np.float32)
    start_qs = robot.ik_tcp_nearest(
        tgt_rotmat=start_rotmat,
        tgt_pos=start_pos,
        ref_qs=robot.qs.copy(),
    )
    if start_qs is None:
        raise RuntimeError('failed to solve start pose IK')
    robot.fk(qs=start_qs)
    robot_start = robot.clone()
    robot_start.fk(qs=start_qs)
    robot_start.rgba = (0.0, 0.55, 0.15, 0.35)
    robot_start.attach_to(scene)

    home_pose = (
        oum.vec(0.35, -0.20, 0.24).astype(oum.np.float32),
        start_rotmat.copy(),
    )
    ossop.frame(
        pos=home_pose[0],
        rotmat=home_pose[1],
        length_scale=0.08,
        radius_scale=0.75,
        color_mat=ouc.CoordColor.MYC,
        alpha=0.35,
    ).attach_to(scene)

    collider = ocm.MJCollider()
    collider.append(robot)
    collider.append(screwdriver)
    collider.append(table)
    collider.append(fixture)
    collider.append(wall)
    collider.actors = [robot, screwdriver]
    collider.compile(margin=0.0)
    pln_ctx = omp_utils.build_planning_context(collider)
    planner = ScrewPlanner(robot, pln_ctx=pln_ctx, ee_actor=screwdriver)

    goal_pose_list = planner.gen_goal_pose_list(
        tgt_pos=screw_pos,
        tgt_vec=screw_axis,
        resolution=24,
        angle_offset=oum.np.deg2rad(7.5),
    )
    for sid, (pos, rotmat) in enumerate(goal_pose_list):
        color = ouc.CoordColor.DYO if sid % 2 == 0 else ouc.CoordColor.MYC
        ossop.frame(
            pos=pos,
            rotmat=rotmat,
            length_scale=0.05,
            radius_scale=0.45,
            color_mat=color,
            alpha=0.12,
        ).attach_to(scene)

    print('\nStarting ScrewPlanner visual test...')
    t0 = time.time()
    plan = planner.gen_screw(
        start_qs=start_qs,
        goal_pose_list=goal_pose_list,
        home_pose=home_pose,
        home_approach_direction=oum.vec(0.0, 0.0, -1.0).astype(oum.np.float32),
        home_approach_distance=0.05,
        approach_direction=screw_axis,
        approach_distance=0.06,
        depart_direction=oum.vec(0.0, 0.0, 1.0).astype(oum.np.float32),
        depart_distance=0.04,
        linear_granularity=0.01,
        use_rrt=True,
        toggle_dbg=True,
    )
    t1 = time.time()

    print(f"\n{'=' * 60}")
    print(f'Planning completed in {t1 - t0:.3f}s')
    if plan is None:
        print('No screw plan found')
        print(f"{'=' * 60}")
        base.run()
        return

    sid = plan.events.get('sid')
    if sid is not None:
        sid_pose = goal_pose_list[int(sid)]
        ossop.frame(
            pos=sid_pose[0],
            rotmat=sid_pose[1],
            length_scale=0.09,
            radius_scale=1.0,
            color_mat=ouc.CoordColor.RGB,
            alpha=0.85,
        ).attach_to(scene)

    state_list = plan.qs_list
    print(f'selected sid: {sid}')
    print(f'Path found with {len(state_list)} waypoints')
    print(f"{'=' * 60}")

    tcp_trace = []
    for qs in state_list:
        robot_qs, screw_qs = split_state(robot, screwdriver, qs)
        robot.fk(qs=robot_qs)
        apply_screwdriver_state(screwdriver, screw_qs)
        tcp_trace.append(screwdriver.gl_tcp_tf[:3, 3].copy())
    if len(tcp_trace) >= 2:
        segs = oum.np.stack([tcp_trace[:-1], tcp_trace[1:]], axis=1)
        ossop.linsegs(
            segs,
            radius=0.002,
            srgbs=oum.vec(0.1, 0.2, 0.9).astype(oum.np.float32),
            alpha=0.45,
        ).attach_to(scene)

    robot_goal = robot.clone()
    goal_robot_qs, _ = split_state(robot, screwdriver, state_list[-1])
    robot_goal.fk(qs=goal_robot_qs)
    robot_goal.rgba = (0.0, 0.15, 1.0, 0.22)
    robot_goal.attach_to(scene)

    robot.fk(qs=start_qs)
    counter = {'idx': 0}

    def update(_dt):
        if counter['idx'] >= len(state_list):
            counter['idx'] = 0
        robot_qs, screw_qs = split_state(robot, screwdriver, state_list[counter['idx']])
        robot.fk(qs=robot_qs)
        apply_screwdriver_state(screwdriver, screw_qs)
        counter['idx'] += 1

    base.schedule_interval(update, interval=0.08)
    base.run()


if __name__ == '__main__':
    main()
