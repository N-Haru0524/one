import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one import ovw, ouc, ossop, osso, ocm, khi_rs007l, or_2fg7
import one.collider.cpu_simd as occs
import one.geom.fitting as ogf
import one.geom.surface as ogs
import one.grasp.placement as ogp
from one_assembly.motion_planner import PickPlacePlanner, utils as omp_utils
from one.grasp.antipodal import antipodal
import one.utils.math as oum

def split_state(robot, gripper, qs):
    robot_ndof = robot.ndof
    return qs[:robot_ndof], qs[robot_ndof:robot_ndof + gripper.ndof]


def apply_gripper_state(gripper, active_qs):
    active_qs = oum.np.asarray(active_qs, dtype=oum.np.float32).reshape(-1)
    if active_qs.size == 0:
        return
    if active_qs.size == len(gripper.qs):
        gripper.fk(qs=active_qs)
        return
    jaw_width = float(active_qs[0] * 2.0)
    gripper.set_jaw_width(jaw_width)


def main():
    base = ovw.World(
        cam_pos=(-1.4, 1.4, 1.1),
        cam_lookat_pos=(0.38, 0.0, 0.28),
        toggle_auto_cam_orbit=False,
    )
    ossop.frame().attach_to(base.scene)

    robot = khi_rs007l.RS007L()
    robot.set_rotmat_pos(pos=(0.0, 0.0, 0.01))
    gripper = or_2fg7.OR2FG7()
    robot.engage(gripper)
    robot.attach_to(base.scene)

    table = ossop.box(
        half_extents=(0.4, 0.4, 0.02),
        pos=(0.4, 0.0, -0.03),
        collision_type=ouc.CollisionType.AABB,
    )
    table.rgb = ouc.ExtendedColor.SILVER_GRAY
    table.attach_to(base.scene)

    bunny = osso.SceneObject.from_file(
        str(ROOT / 'bunny.stl'),
        collision_type=ouc.CollisionType.MESH,
        is_free=True,
    )
    geom_hull = ogf.convex_hull(bunny.collisions[0].geom)
    facets = ogs.segment_surface(geom_hull)
    stable_poses = ogp.compute_stable_poses(
        geom_hull.vs,
        geom_hull.fs,
        facets,
        com=None,
        stable_thresh=10.0,
    )
    stable_pos, stable_rotmat, *_ = stable_poses[0]
    stable_pos = stable_pos + oum.vec(0.46, 0.0, 0.0).astype(oum.np.float32)
    bunny.set_rotmat_pos(rotmat=stable_rotmat, pos=stable_pos)
    bunny.rgb = (0.8, 0.7, 0.6)
    bunny.attach_to(base.scene)
    start_pose = (bunny.pos.copy(), bunny.rotmat.copy())

    target_pos = bunny.pos + oum.vec(0.08, 0.0, 0.0).astype(oum.np.float32)
    target_rotmat = bunny.rotmat.copy()
    target_marker = bunny.clone()
    target_marker.set_rotmat_pos(rotmat=target_rotmat, pos=target_pos)
    target_marker.alpha = 0.2
    target_marker.attach_to(base.scene)

    goal_bunny = bunny.clone()
    goal_bunny.set_rotmat_pos(rotmat=target_rotmat, pos=target_pos)

    collider = ocm.MJCollider()
    collider.append(robot)
    collider.append(gripper)
    collider.append(table)
    collider.append(bunny)
    collider.actors = [robot, gripper]
    collider.compile(margin=0.0)
    pln_ctx = omp_utils.build_planning_context(collider)
    planner = PickPlacePlanner(robot, pln_ctx=pln_ctx, ee_actor=gripper)

    print('\nBuilding grasp set...')
    t0 = time.time()
    grasp_collection = antipodal(
        gripper=gripper,
        target_sobj=bunny,
        density=0.01,
        normal_tol_deg=20,
        roll_step_deg=30,
        max_grasps=80,
    )
    
    plan = planner.gen_pick_and_place(
        obj_model=bunny,
        grasp_collection=grasp_collection,
        goal_pose_list=[(target_pos, target_rotmat)],
        linear_granularity=0.02,
        reason_grasps=True,
        pln_jnt=True,
        use_rrt=True,
        toggle_dbg=True,
    )
    t1 = time.time()

    print(f"\n{'=' * 60}")
    print(f'Planning completed in {t1 - t0:.3f}s')
    if plan is None:
        print('No pick-and-place plan found')
        print(f"{'=' * 60}")
        base.run()
        return

    state_list = plan.qs_list
    attach_idx = plan.events.get('attach')
    release_idx = plan.events.get('release')
    print(f'Path found with {len(state_list)} waypoints')
    print(f"{'=' * 60}")
    print(plan.qs_list)

    tcp_trace = []
    for qs in state_list:
        robot_qs, gripper_qs = split_state(robot, gripper, qs)
        robot.fk(qs=robot_qs)
        apply_gripper_state(gripper, gripper_qs)
        tcp_trace.append(gripper.gl_tcp_tf[:3, 3].copy())
    if len(tcp_trace) >= 2:
        segs = oum.np.stack([tcp_trace[:-1], tcp_trace[1:]], axis=1)
        ossop.linsegs(
            segs,
            radius=0.002,
            srgbs=oum.vec(0.1, 0.2, 0.9).astype(oum.np.float32),
            alpha=0.45,
        ).attach_to(base.scene)

    held = {'value': False}
    counter = {'idx': 0}

    def reset_scene():
        if bunny in gripper._mountings:
            gripper.release(bunny)
        bunny.set_rotmat_pos(rotmat=start_pose[1], pos=start_pose[0])
        held['value'] = False

    def update(_dt):
        if counter['idx'] >= len(state_list):
            reset_scene()
            counter['idx'] = 0
            return

        robot_qs, gripper_qs = split_state(robot, gripper, state_list[counter['idx']])
        robot.fk(qs=robot_qs)
        apply_gripper_state(gripper, gripper_qs)

        jaw_width = float(gripper_qs[0] * 2.0)
        if not held['value'] and bunny in gripper._mountings:
            held['value'] = True
        if not held['value'] and attach_idx is not None and counter['idx'] >= attach_idx:
            gripper.grasp(bunny, jaw_width=jaw_width)
            held['value'] = True
        if held['value'] and release_idx is not None and counter['idx'] >= release_idx:
            gripper.release(bunny)
            held['value'] = False

        counter['idx'] += 1

    reset_scene()
    base.schedule_interval(update, interval=0.1)
    base.run()


if __name__ == '__main__':
    main()
