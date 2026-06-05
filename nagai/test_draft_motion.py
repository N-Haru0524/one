"""Inspect each motion-planner *draft* one segment at a time, in a window that
stays open, using the shared ``nagai.viz`` dual-arm playback toolkit.

This is a visual harness for understanding what the three planner drafts actually
do:

    * PickPlacePlanner.gen_place_draft   (left arm + gripper)   -> 'place'
    * FoldPlanner.gen_fold_draft         (left arm + gripper)   -> 'fold'
    * ScrewPlanner.gen_screw_draft       (right arm + driver)   -> 'screw'

It plans one draft per symbolic action, splits each draft into its segments, and
replays them one by one.  Playback, grasping and scene reset are handled by the
shared ``nagai.viz`` package (see nagai/viz/__init__.py); this file only does the
planning and wires the pieces together.

Controls (focus the 3D window):
    SPACE / N / RIGHT : next segment    B / LEFT : prev    R : replay
    P : pause          ESC / Q : quit

Default sequence is ``blt-blt_fld-blt_fld_scrw`` (place / fold / screw on the
belt), which exercises all three planners on a single part.  Override with
--sequence.

Run:
    uv run python nagai/test_draft_motion.py
    uv run python nagai/test_draft_motion.py --sequence rly-rly_scrw --mode loop
"""

import argparse
import builtins
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one import ossop, ovw, ouc
import one.utils.math as oum
from one.grasp.antipodal import antipodal
from one_assembly.assembly_data import EEEvent
from one_assembly.assembly_planning import (
    AssemblyNode,
    execute_sequence_string,
    reset_work_state,
    sequence_labels,
)
from one_assembly.motion_planner import FoldPlanner, PickPlacePlanner, ScrewPlanner, interpolate_fold
from one_assembly.motion_planner import utils as omp_utils
from one_assembly.robots.khi_bunri.khi_bunri import KHIBunri
from one_assembly.worklist import WorkList

from nagai.viz import (
    Clip,
    EEController,
    KHIBunriView,
    Player,
    SceneResetter,
    TcpFrameDecorator,
    build_clip_frames,
    build_sync_segment_from_draft,
    normalize_action_draft,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Replay each planner draft segment one at a time; advance on key press.',
    )
    parser.add_argument('--layout', default='home')
    parser.add_argument(
        '--sequence',
        default='blt-blt_fld-blt_fld_scrw',
        help='Symbolic sequence; one draft is built per action. '
             'Default exercises place/fold/screw on the belt.',
    )
    parser.add_argument('--mode', choices=('step', 'loop', 'once'), default='step',
                        help='step: hold and wait per segment; loop: auto-advance; '
                             'once: play all then hold.')
    parser.add_argument('--interval', type=float, default=0.05,
                        help='Seconds between playback frames.')
    parser.add_argument('--no-trace', action='store_true',
                        help='Disable scene overlays: the per-clip TCP trajectory '
                             'polyline and the live TCP coordinate frames.')
    parser.add_argument(
        '--reasoning-backend', choices=('mujoco', 'simd'), default='simd',
    )
    parser.add_argument(
        '--transit-backend', choices=('mujoco', 'simd'), default='mujoco',
    )
    parser.add_argument('--screw-resolution', type=int, default=12)
    return parser.parse_args()


def sanitize_sync_token(raw: str) -> str:
    return raw.replace(':', '_').replace('/', '_').replace('-', '_')


# ---------------------------------------------------------------------------
# Planners (planning only; rendering/playback is delegated to nagai.viz)
# ---------------------------------------------------------------------------

def load_grasp_collection(work, gripper, cache):
    if work.name in cache:
        return cache[work.name]
    if work.grasp_collection:
        cache[work.name] = work.grasp_collection
        return work.grasp_collection
    saved_pose = (work.model.pos.copy(), work.model.rotmat.copy())
    work.model.set_rotmat_pos(rotmat=np.eye(3, dtype=np.float32), pos=np.zeros(3, dtype=np.float32))
    grasp_collection = antipodal(
        gripper=gripper,
        target_sobj=work.model,
        density=0.008,
        normal_tol_deg=20,
        roll_step_deg=30,
        clearance=0.002,
        max_grasps=60,
    )
    work.model.set_rotmat_pos(rotmat=saved_pose[1], pos=saved_pose[0])
    cache[work.name] = grasp_collection or []
    return cache[work.name]


def build_left_planner(robot, worklist, reasoning_backend, transit_backend, cls):
    obstacles = [robot.body, robot.rgt_arm, robot.rgt_screwdriver]
    if worklist.work_base is not None:
        obstacles.append(worklist.work_base)
    obstacles.extend(work.model for work in worklist.work)
    collider = omp_utils.build_collider([robot.lft_arm, robot.lft_gripper], obstacles=obstacles)
    return cls(
        robot.lft_arm,
        pln_ctx=omp_utils.build_planning_context(collider),
        ee_actor=robot.lft_gripper,
        reasoning_backend=reasoning_backend,
        transit_backend=transit_backend,
    )


def build_right_planner(robot, worklist, target_work):
    obstacles = [robot.body]
    if worklist.work_base is not None:
        obstacles.append(worklist.work_base)
    obstacles.extend(work.model for work in worklist.work if work is not target_work)
    collider = omp_utils.build_collider([robot.rgt_arm, robot.rgt_screwdriver], obstacles=obstacles)
    return ScrewPlanner(
        robot.rgt_arm,
        pln_ctx=omp_utils.build_planning_context(collider),
        ee_actor=robot.rgt_screwdriver,
    )


def plan_place_action(robot, worklist, view, state, action, grasp_cache,
                      reasoning_backend, transit_backend, keep_holding):
    target_work = worklist[action.work_idx]
    goal_pose = target_work.pose_after_action(action.action_idx)
    if goal_pose is None:
        return None
    view.apply(state)
    planner = build_left_planner(robot, worklist, reasoning_backend, transit_backend, PickPlacePlanner)
    grasp_collection = load_grasp_collection(target_work, robot.lft_gripper, grasp_cache)
    if not grasp_collection:
        print(f'no grasps for {target_work.name}')
        return None
    return planner.gen_place_draft(
        obj_model=target_work.model,
        work_name=target_work.name,
        grasp_collection=grasp_collection,
        goal_pose_list=[goal_pose],
        segment_label=f'place {target_work.name}',
        end_sync_label=action.label,
        pick_depart_direction=np.array([0, 0, -1], dtype=np.float32),
        approach_direction=np.array([0, 0, -1], dtype=np.float32),
        start_qs=view.compose_left(state),
        linear_granularity=0.02,
        reason_grasps=True,
        use_rrt=True,
        pln_jnt=False,
        keep_holding=keep_holding,
        toggle_dbg=False,
    )


def plan_fold_action(robot, worklist, view, state, action, grasp_cache,
                     reasoning_backend, transit_backend, held_grasp):
    target_work = worklist[action.work_idx]
    start_pose = target_work.current_pose
    goal_pose = target_work.pose_after_action(action.action_idx, start_pose=start_pose)
    if goal_pose is None:
        return None
    view.apply(state)
    planner = build_left_planner(robot, worklist, reasoning_backend, transit_backend, FoldPlanner)
    goal_pose_list = interpolate_fold(start_pose, goal_pose, n_steps=6)
    grasp_collection = load_grasp_collection(target_work, robot.lft_gripper, grasp_cache)
    if not grasp_collection:
        print(f'no grasps for {target_work.name}')
        return None
    return planner.gen_fold_draft(
        obj_model=target_work.model,
        work_name=target_work.name,
        grasp_collection=grasp_collection,
        goal_pose_list=goal_pose_list,
        held_grasp=held_grasp,
        segment_label=f'fold {target_work.name}',
        end_sync_label=action.label,
        start_qs=view.compose_left(state),
        linear_granularity=0.02,
        reason_grasps=True,
        use_rrt=True,
        pln_jnt=False,
        exclude_entities=[target_work.model],
        toggle_dbg=False,
    )


def resolve_screw_pick_pose_candidates(planner, state, pick_pose, roll_resolution=12):
    ref_qs = state.rgt_qs.astype(np.float32)
    roll_candidates = planner.gen_pose_roll_candidates(pick_pose, resolution=roll_resolution)
    valid_roll_poses = []
    prev_ee_qs = planner._active_ee_qs_for_ik
    planner._active_ee_qs_for_ik = planner._extended_ee_qs()
    try:
        for roll_pose in roll_candidates:
            candidate_state = planner._pick_state(pick_pose=roll_pose, ref_qs=ref_qs)
            if candidate_state is None:
                continue
            valid_roll_poses.append(roll_pose)
    finally:
        planner._active_ee_qs_for_ik = prev_ee_qs
    return valid_roll_poses


def plan_screw_action(robot, worklist, view, state, action, screw_resolution):
    target_work = worklist[action.work_idx]
    goal_pose = target_work.pose_after_action(action.action_idx, start_pose=target_work.current_pose)
    if goal_pose is None:
        return None
    screw_counter = int(worklist.screw_counter)
    pick_pose = worklist.get_screw_pose()
    worklist.screw_counter = screw_counter
    view.apply(state)
    planner = build_right_planner(robot, worklist, target_work)
    pick_pose_list = resolve_screw_pick_pose_candidates(
        planner, state, pick_pose, roll_resolution=screw_resolution,
    )
    if not pick_pose_list:
        print(f'no valid screw pickup pose for {target_work.name}')
        return None
    screw_axis = goal_pose[1][:, 2].astype(np.float32)
    draft = planner.gen_screw_draft(
        work_name=target_work.name,
        start_qs=view.compose_right(state),
        goal_pose_list=[goal_pose],
        resolution=screw_resolution,
        pick_pose_list=pick_pose_list,
        approach_direction=screw_axis,
        approach_distance=0.05,
        depart_direction=screw_axis,
        depart_distance=0.03,
        linear_granularity=0.02,
        use_rrt=True,
        pln_jnt=False,
        segment_label=f'screw {target_work.name}',
        end_sync_label=action.label,
        toggle_dbg=False,
    )
    if draft is None:
        failure = getattr(planner, '_last_plan_failure', None)
        if failure is not None:
            print(f'screw failed for {target_work.name}: {failure["stage"]} {failure["reason"]}')
    return draft


def apply_symbolic_action(worklist: WorkList, action):
    work = worklist[action.work_idx]
    work.apply_action(action.action_idx)
    if work.type[action.action_idx] == 'screw':
        worklist.screw_counter += 1


def manual_leaf(worklist: WorkList, sequence: str, layout_name: str) -> AssemblyNode:
    result = execute_sequence_string(worklist, sequence, layout_name=layout_name)
    actions = tuple(result.actions)
    node_id = 'manual' if not actions else 'manual/' + '/'.join(a.label for a in actions)
    return AssemblyNode(
        node_id=node_id,
        depth=len(actions),
        parent_id=None,
        action=None if not actions else actions[-1],
        sequence=actions,
        work_state=None if not result.work_states else result.work_states[-1],
    )


# ---------------------------------------------------------------------------
# Plan one draft per action, split into per-segment clips
# ---------------------------------------------------------------------------

def build_draft_clips(robot, worklist, view, leaf, layout_name,
                      reasoning_backend, transit_backend, screw_resolution) -> list[Clip]:
    grasp_cache = {}
    open_qs = view.open_gripper_qs()

    reset_work_state(worklist, layout_name=layout_name)
    view.home()
    current_state = view.capture()
    held_grasp = None
    clips: list[Clip] = []
    prefix_actions: list[tuple[int, int]] = []
    # EE side-effects accumulate across the WHOLE sequence so a clip that starts
    # while already holding a part (e.g. a held-mode fold inheriting the grasp
    # from a keep_holding place) is restored correctly when replayed in isolation.
    running_events: list[EEEvent] = []

    for idx, action in enumerate(leaf.sequence):
        reset_work_state(worklist, layout_name=layout_name, actions=prefix_actions)
        next_action = None if idx + 1 >= len(leaf.sequence) else leaf.sequence[idx + 1]
        keep_holding = (
            action.action_type == 'place' and action.immediate and
            next_action is not None and next_action.work_idx == action.work_idx and
            next_action.action_type == 'fold'
        )
        clip_prefix = list(prefix_actions)
        clip_start_state = current_state.copy()

        print(f'[plan {idx + 1}/{len(leaf.sequence)}] {action.action_type} "{action.label}" ...')
        if action.action_type == 'place':
            draft = plan_place_action(robot, worklist, view, current_state, action, grasp_cache,
                                      reasoning_backend, transit_backend, keep_holding)
        elif action.action_type == 'fold':
            draft = plan_fold_action(robot, worklist, view, current_state, action, grasp_cache,
                                     reasoning_backend, transit_backend, held_grasp)
        elif action.action_type == 'screw':
            draft = plan_screw_action(robot, worklist, view, current_state, action, screw_resolution)
        else:
            print(f'  unsupported action type: {action.action_type}')
            draft = None

        if draft is None or not normalize_action_draft(draft).segments:
            print(f'  -> draft FAILED/empty, skipping {action.label}')
            apply_symbolic_action(worklist, action)
            prefix_actions.append((action.work_idx, action.action_idx))
            held_grasp = None
            continue

        action_draft = normalize_action_draft(draft)
        sync_segments = [
            build_sync_segment_from_draft(
                segment_id=f'seg_{idx}_{seg_idx}_{sanitize_sync_token(seg_draft.segment_label)}',
                start_sync_id=f'sp_{idx}_{seg_idx}',
                end_sync_id=f'sp_{idx}_{seg_idx + 1}',
                draft=seg_draft,
            )
            for seg_idx, seg_draft in enumerate(action_draft.segments)
        ]
        seg_count = len(sync_segments)
        state = clip_start_state.copy()
        for seg_idx, segment in enumerate(sync_segments):
            seg_frames = build_clip_frames(state, [segment], open_qs)
            clips.append(Clip(
                label=segment.label,
                group=action.label,
                action_type=action.action_type,
                seg_index=seg_idx,
                seg_count=seg_count,
                initial_state=state.copy(),
                prefix_actions=clip_prefix,
                setup_events=list(running_events),
                frames=seg_frames,
            ))
            state = seg_frames[-1].state.copy()
            running_events.extend(segment.ee_events)
        print(f'  -> {seg_count} segment(s): {[seg.label for seg in sync_segments]}')

        current_state = state.copy()
        held_grasp = action_draft.held_after
        apply_symbolic_action(worklist, action)
        prefix_actions.append((action.work_idx, action.action_idx))

    reset_work_state(worklist, layout_name=layout_name)
    view.home()
    return clips


def main():
    args = parse_args()

    # Resolve the symbolic sequence on a headless worklist first.
    planning_worklist = WorkList(collision_type=ouc.CollisionType.MESH)
    planning_worklist.init_pose(seed=args.layout)
    leaf = manual_leaf(planning_worklist, args.sequence, args.layout)
    print(f'sequence: {sequence_labels(leaf)}')

    # Viewer + scene worklist + robot.
    base = ovw.World(
        cam_pos=(1.9, 0.0, 0.55),
        cam_lookat_pos=(0.18, 0.0, 0.55),
        toggle_auto_cam_orbit=False,
    )
    builtins.base = base
    ossop.frame(length_scale=0.5, radius_scale=1.2).attach_to(base.scene)

    worklist = WorkList(
        pos=oum.vec(0.2 + 0.09 + 0.035, 0, 0.11 + 0.018 + 0.0902),
        collision_type=ouc.CollisionType.MESH,
    )
    worklist.init_pose(seed=args.layout)
    worklist.attach_to(base.scene)

    robot = KHIBunri()
    robot.attach_to(base.scene)
    view = KHIBunriView(robot)

    clips = build_draft_clips(
        robot, worklist, view, leaf, args.layout,
        reasoning_backend=args.reasoning_backend,
        transit_backend=args.transit_backend,
        screw_resolution=args.screw_resolution,
    )
    if not clips:
        raise RuntimeError('No draft clips could be built for this sequence.')

    ee = EEController(robot, worklist, view)
    resetter = SceneResetter(robot, worklist, view, ee, args.layout)
    decorators = []
    if not args.no_trace:
        from nagai.viz import TrajectoryTrace
        decorators.append(TcpFrameDecorator(base, view))
        decorators.append(TrajectoryTrace(base, view, actor='left'))

    player = Player(
        base, view, ee, resetter, clips,
        mode=args.mode,
        interval=args.interval,
        decorators=decorators,
    )
    player.run()


if __name__ == '__main__':
    main()
