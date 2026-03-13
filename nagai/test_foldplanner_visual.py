import argparse
import builtins
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one import ovw, ouc, ossop, osso, ocm, khi_rs007l, or_2fg7
from one.grasp.antipodal import antipodal
import one.utils.math as oum
from one_assembly.motion_planner import FoldPlanner, interpolate_fold, utils as omp_utils


def split_state(robot, gripper, qs):
    qs = oum.np.asarray(qs, dtype=oum.np.float32).reshape(-1)
    robot_qs = qs[:robot.ndof]
    gripper_qs = qs[robot.ndof:robot.ndof + gripper.ndof]
    return robot_qs, gripper_qs


def apply_gripper_state(gripper, active_qs):
    active_qs = oum.np.asarray(active_qs, dtype=oum.np.float32).reshape(-1)
    if active_qs.size == 0:
        return
    if active_qs.size == len(gripper.qs):
        gripper.fk(qs=active_qs)
        return
    if active_qs.size == getattr(gripper, 'ndof', 0):
        if hasattr(gripper, 'jaw_range'):
            half_width = float(active_qs[0])
            min_half = float(gripper.jaw_range[0]) * 0.5
            max_half = float(gripper.jaw_range[1]) * 0.5
            half_width = min(max(half_width, min_half), max_half)
            gripper.fk(qs=[half_width, half_width])
            return
        gripper.fk(qs=[float(active_qs[0]), float(active_qs[0])])
        return
    jaw_width = float(active_qs[0] * 2.0)
    gripper.set_jaw_width(jaw_width)


def lifted_fold_pose_list(start_pose, target_pose, n_steps=12, lift_height=0.10):
    pose_list = interpolate_fold(start_pose, target_pose, n_steps=n_steps)
    if len(pose_list) <= 2:
        return pose_list
    lifted = []
    for idx, (pos, rotmat) in enumerate(pose_list):
        t = idx / (len(pose_list) - 1)
        lift = lift_height * oum.np.sin(oum.pi * t)
        pos = oum.np.asarray(pos, dtype=oum.np.float32).copy()
        pos[2] += float(lift)
        lifted.append((pos, rotmat))
    return lifted


def candidate_target_poses(start_pos, start_rotmat):
    for rot_y_deg in (90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0):
        for rot_z_deg in (-12.0, 0.0, 12.0):
            for dx, dy, dz in (
                (0.02, 0.03, 0.06),
                (0.03, 0.04, 0.07),
                (0.04, 0.05, 0.08),
            ):
                target_rotmat = (
                    oum.rotmat_from_axangle(oum.vec(0.0, 1.0, 0.0), oum.np.deg2rad(rot_y_deg)) @
                    oum.rotmat_from_axangle(oum.vec(0.0, 0.0, 1.0), oum.np.deg2rad(rot_z_deg)) @
                    start_rotmat
                ).astype(oum.np.float32)
                target_pos = (start_pos + oum.vec(dx, dy, dz)).astype(oum.np.float32)
                yield target_pos, target_rotmat


def find_reachable_target_pose(planner, obj_model, grasp_collection, start_pose):
    best = None
    best_score = None
    for target_pose in candidate_target_poses(start_pose[0], start_pose[1]):
        goal_pose_list = lifted_fold_pose_list(start_pose, target_pose, n_steps=12, lift_height=0.12)
        common_gids = planner.reason_common_grasp_ids(
            obj_model=obj_model,
            grasp_collection=grasp_collection,
            goal_pose_list=goal_pose_list,
            pick_pose=oum.tf_from_rotmat_pos(obj_model.rotmat, obj_model.pos),
            linear_granularity=0.01,
            exclude_entities=[obj_model],
            toggle_dbg=False,
        )
        score = (
            float(oum.np.linalg.norm(target_pose[0] - start_pose[0])) +
            abs(float(oum.np.arccos(oum.np.clip((oum.np.trace(start_pose[1].T @ target_pose[1]) - 1.0) * 0.5, -1.0, 1.0))))
        )
        if common_gids:
            return target_pose, goal_pose_list, common_gids
        if best_score is None or score < best_score:
            best = (target_pose, goal_pose_list, common_gids)
            best_score = score
    return best


def build_world():
    base = ovw.World(
        cam_pos=(-1.45, 1.55, 1.15),
        cam_lookat_pos=(0.40, 0.02, 0.24),
        toggle_auto_cam_orbit=False,
    )
    builtins.base = base
    scene = base.scene
    ossop.frame(length_scale=0.2, radius_scale=1.1).attach_to(scene)

    robot = khi_rs007l.RS007L()
    robot.set_rotmat_pos(pos=(0.0, 0.0, 0.01))
    gripper = or_2fg7.OR2FG7()
    robot.engage(gripper)
    robot.attach_to(scene)

    table = ossop.box(
        half_extents=(0.42, 0.42, 0.02),
        pos=(0.42, 0.0, -0.03),
        rgb=ouc.ExtendedColor.SILVER_GRAY,
        collision_type=ouc.CollisionType.AABB,
    )
    table.attach_to(scene)

    wall = ossop.box(
        half_extents=(0.02, 0.18, 0.16),
        pos=(0.80, 0.0, 0.16),
        rgb=(0.78, 0.74, 0.70),
        collision_type=ouc.CollisionType.AABB,
    )
    wall.attach_to(scene)

    bunny = osso.SceneObject.from_file(
        str(ROOT / 'bunny.stl'),
        collision_type=ouc.CollisionType.MESH,
        is_free=True,
    )
    start_pos = oum.vec(0.4, -0.12, 0.3).astype(oum.np.float32)
    start_rotmat = oum.rotmat_from_euler(0.0, oum.np.deg2rad(10.0), oum.np.deg2rad(-15.0)).astype(oum.np.float32)
    bunny.set_rotmat_pos(rotmat=start_rotmat, pos=start_pos)
    bunny.rgb = (0.82, 0.72, 0.62)
    bunny.attach_to(scene)

    target_rotmat = (
        oum.rotmat_from_axangle(oum.vec(0.0, 1.0, 0.0), oum.np.deg2rad(24.0)) @
        start_rotmat
    ).astype(oum.np.float32)
    target_pos = (start_pos + oum.vec(0.02, 0.03, 0.06)).astype(oum.np.float32)
    target_marker = bunny.clone()
    target_marker.set_rotmat_pos(rotmat=target_rotmat, pos=target_pos)
    target_marker.alpha = 0.18
    target_marker.rgb = (0.20, 0.55, 0.95)
    target_marker.attach_to(scene)

    return base, robot, gripper, table, wall, bunny, target_marker, (target_pos, target_rotmat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless-smoke', action='store_true')
    args = parser.parse_args()

    base, robot, gripper, table, wall, bunny, target_marker, target_pose = build_world()
    scene = base.scene
    start_pose = (bunny.pos.copy(), bunny.rotmat.copy())

    collider = ocm.MJCollider()
    collider.append(robot)
    collider.append(gripper)
    collider.append(table)
    collider.append(wall)
    collider.append(bunny)
    collider.actors = [robot, gripper]
    collider.compile(margin=0.0)
    pln_ctx = omp_utils.build_planning_context(collider)
    planner = FoldPlanner(robot, pln_ctx=pln_ctx, ee_actor=gripper)

    print('building antipodal grasp set...')
    # antipodal() currently samples on collision geometry in its current world pose.
    # Build grasps at the canonical object pose so the planner can treat them as object-local grasps.
    bunny.set_rotmat_pos(
        rotmat=oum.np.eye(3, dtype=oum.np.float32),
        pos=oum.np.zeros(3, dtype=oum.np.float32),
    )
    grasp_collection = antipodal(
        gripper=gripper,
        target_sobj=bunny,
        density=0.01,
        normal_tol_deg=20,
        roll_step_deg=20,
        max_grasps=30,
    )
    bunny.set_rotmat_pos(rotmat=start_pose[1], pos=start_pose[0])
    found = find_reachable_target_pose(planner, bunny, grasp_collection, start_pose)
    if found is None:
        print('failed to find a candidate target pose')
        if args.headless_smoke:
            raise SystemExit(1)
        base.run()
        return
    target_pose, goal_pose_list, common_gids = found
    target_marker.set_rotmat_pos(rotmat=target_pose[1], pos=target_pose[0])
    print(f'prefilter common gids: {len(common_gids)}')
    if common_gids:
        print(f'seed gids: {common_gids[:10]}')
    for idx, (pos, rotmat) in enumerate(goal_pose_list[1:], start=1):
        ossop.frame(
            pos=pos,
            rotmat=rotmat,
            length_scale=0.05,
            radius_scale=0.55,
            color_mat=ouc.CoordColor.DYO if idx % 2 == 0 else ouc.CoordColor.MYC,
            alpha=0.14,
        ).attach_to(scene)

    start_qs = robot.qs.copy()
    print('\nStarting FoldPlanner visual test...')
    t0 = time.time()
    plan = planner.gen_pick_and_fold(
        obj_model=bunny,
        grasp_collection=grasp_collection if not common_gids else [grasp_collection[gid] for gid in common_gids],
        goal_pose_list=goal_pose_list,
        start_qs=start_qs,
        linear_granularity=0.01,
        reason_grasps=not bool(common_gids),
        exclude_entities=[bunny],
        use_rrt=True,
        toggle_dbg=True,
    )
    t1 = time.time()

    print(f"\n{'=' * 60}")
    print(f'Planning completed in {t1 - t0:.3f}s')
    if plan is None:
        print('No fold plan found')
        print(f"{'=' * 60}")
        if args.headless_smoke:
            raise SystemExit(1)
        base.run()
        return

    print(f'selected gid: {plan.events.get("gid")}')
    print(f'Path found with {len(plan.qs_list)} waypoints')
    print(f"{'=' * 60}")
    print(plan.qs_list)

    tcp_trace = []
    for qs in plan.qs_list:
        robot_qs, gripper_qs = split_state(robot, gripper, qs)
        robot.fk(qs=robot_qs)
        apply_gripper_state(gripper, gripper_qs)
        tcp_trace.append(gripper.gl_tcp_tf[:3, 3].copy())
    if len(tcp_trace) >= 2:
        segs = oum.np.stack([tcp_trace[:-1], tcp_trace[1:]], axis=1)
        lengths = oum.np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
        segs = segs[lengths > 1e-6]
        if len(segs) > 0:
            ossop.linsegs(
                segs,
                radius=0.002,
                srgbs=oum.vec(0.08, 0.28, 0.95).astype(oum.np.float32),
                alpha=0.45,
            ).attach_to(scene)

    robot_start = robot.clone()
    robot_start.fk(qs=start_qs)
    robot_start.rgba = (0.0, 0.55, 0.15, 0.30)
    robot_start.attach_to(scene)

    robot_goal = robot.clone()
    goal_robot_qs, _ = split_state(robot, gripper, plan.qs_list[-1])
    robot_goal.fk(qs=goal_robot_qs)
    robot_goal.rgba = (0.05, 0.18, 1.0, 0.22)
    robot_goal.attach_to(scene)

    attach_idx = plan.events.get('attach')
    counter = {'idx': 0}
    held = {'value': False}

    def reset_scene():
        if bunny in gripper._mountings:
            gripper.release(bunny)
        bunny.set_rotmat_pos(rotmat=start_pose[1], pos=start_pose[0])
        held['value'] = False
        robot.fk(qs=start_qs)

    def update(_dt):
        if counter['idx'] >= len(plan.qs_list):
            reset_scene()
            counter['idx'] = 0
            return

        robot_qs, gripper_qs = split_state(robot, gripper, plan.qs_list[counter['idx']])
        robot.fk(qs=robot_qs)
        apply_gripper_state(gripper, gripper_qs)

        jaw_width = float(gripper_qs[0] * 2.0)
        if not held['value'] and attach_idx is not None and counter['idx'] >= attach_idx:
            gripper.grasp(bunny, jaw_width=jaw_width)
            held['value'] = True

        counter['idx'] += 1

    reset_scene()
    if args.headless_smoke:
        for _ in range(min(5, len(plan.qs_list))):
            update(0.0)
        print('headless smoke passed')
        return

    base.schedule_interval(update, interval=0.08)
    base.run()


if __name__ == '__main__':
    main()
