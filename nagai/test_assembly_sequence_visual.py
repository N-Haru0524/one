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
from one_assembly.assembly_data import (
    ArmSegment,
    DualRobotState,
    EEEvent,
    HeldGrasp,
    PlannerActionDraft,
    PlannerSegmentDraft,
    SyncPoint,
    SyncSegment,
    SynchronizedPlan,
)
from one_assembly.assembly_planning import (
    AssemblyNode,
    assembly_sequence_planning,
    execute_sequence_string,
    leaf_nodes,
    reset_work_state,
    sequence_labels,
)
from one_assembly.motion_planner import FoldPlanner, PickPlacePlanner, ScrewPlanner, interpolate_fold
from one_assembly.motion_planner import utils as omp_utils
from one_assembly.robots.khi_bunri.khi_bunri import KHIBunri
from one_assembly.worklist import WorkList


def parse_args():
    parser = argparse.ArgumentParser(
        description='Replay one feasible assembly sequence with actual motion planning on KHIBunri.',
    )
    parser.add_argument('--layout', default='home')
    parser.add_argument('--max-depth', type=int, default=2)
    parser.add_argument(
        '--sequence',
        default=None,
        help='Explicit symbolic sequence string such as rly-rly_scrw. If set, skip leaf search.',
    )
    parser.add_argument(
        '--leaf-index',
        type=int,
        default=None,
        help='If set, use this leaf index after printing the leaf list. Otherwise search the first feasible leaf.',
    )
    parser.add_argument('--interval', type=float, default=0.1)
    parser.add_argument(
        '--reasoning-backend',
        choices=('mujoco', 'simd'),
        default='simd',
        help='Collision backend used for detailed pick/place goal reasoning.',
    )
    parser.add_argument(
        '--transit-backend',
        choices=('mujoco', 'simd'),
        default='mujoco',
        help='Collision backend used for pre/depart-style reasoning screens.',
    )
    parser.add_argument(
        '--screw-resolution',
        type=int,
        default=12,
        help='Number of roll candidates around the screw axis for screw planning.',
    )
    parser.add_argument('--list-only', action='store_true')
    return parser.parse_args()


def capture_dual_state(robot: KHIBunri) -> DualRobotState:
    return DualRobotState(
        lft_qs=robot.lft_arm.qs.copy(),
        lft_ee_qs=robot.lft_gripper.qs[:robot.lft_gripper.ndof].copy(),
        rgt_qs=robot.rgt_arm.qs.copy(),
        rgt_ee_qs=robot.rgt_screwdriver.qs[:robot.rgt_screwdriver.ndof].copy(),
    )


def apply_dual_state(robot: KHIBunri, state: DualRobotState):
    robot.lft_arm.fk(state.lft_qs)
    robot.lft_gripper.fk(state.lft_ee_qs)
    robot.rgt_arm.fk(state.rgt_qs)
    robot.rgt_screwdriver.fk(state.rgt_ee_qs)


def compose_left_state(state: DualRobotState) -> np.ndarray:
    return np.concatenate([state.lft_qs, state.lft_ee_qs]).astype(np.float32)


def compose_right_state(state: DualRobotState) -> np.ndarray:
    return np.concatenate([state.rgt_qs, state.rgt_ee_qs]).astype(np.float32)


_ROBOT_REF = {'value': None}


def robot_ref():
    robot = _ROBOT_REF['value']
    if robot is None:
        raise RuntimeError('robot reference is not initialized')
    return robot


def load_grasp_collection(work, gripper, cache):
    if work.name in cache:
        return cache[work.name]
    if work.grasp_collection:
        cache[work.name] = work.grasp_collection
        return work.grasp_collection
    saved_pose = (work.model.pos.copy(), work.model.rotmat.copy())
    work.model.set_rotmat_pos(
        rotmat=np.eye(3, dtype=np.float32),
        pos=np.zeros(3, dtype=np.float32),
    )
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
    if not grasp_collection:
        cache[work.name] = []
        return []
    cache[work.name] = grasp_collection
    return grasp_collection


def build_left_planner(robot: KHIBunri, worklist: WorkList, reasoning_backend='simd', transit_backend='mujoco'):
    # obstacles = [robot.rgt_arm, robot.rgt_screwdriver]
    obstacles = [robot.body, robot.rgt_arm, robot.rgt_screwdriver]
    if worklist.work_base is not None:
        obstacles.append(worklist.work_base)
    obstacles.extend(work.model for work in worklist.work)
    collider = omp_utils.build_collider(
        [robot.lft_arm, robot.lft_gripper],
        obstacles=obstacles,
    )
    return PickPlacePlanner(
        robot.lft_arm,
        pln_ctx=omp_utils.build_planning_context(collider),
        ee_actor=robot.lft_gripper,
        reasoning_backend=reasoning_backend,
        transit_backend=transit_backend,
    )


def build_fold_planner(robot: KHIBunri, worklist: WorkList, reasoning_backend='simd', transit_backend='mujoco'):
    obstacles = [robot.body, robot.rgt_arm, robot.rgt_screwdriver]
    if worklist.work_base is not None:
        obstacles.append(worklist.work_base)
    obstacles.extend(work.model for work in worklist.work)
    collider = omp_utils.build_collider(
        [robot.lft_arm, robot.lft_gripper],
        obstacles=obstacles,
    )
    return FoldPlanner(
        robot.lft_arm,
        pln_ctx=omp_utils.build_planning_context(collider),
        ee_actor=robot.lft_gripper,
        reasoning_backend=reasoning_backend,
        transit_backend=transit_backend,
    )


def build_right_planner(robot: KHIBunri, worklist: WorkList, target_work):
    obstacles = [robot.body]
    if worklist.work_base is not None:
        obstacles.append(worklist.work_base)
    obstacles.extend(
        work.model for work in worklist.work
        if work is not target_work
    )
    collider = omp_utils.build_collider(
        [robot.rgt_arm, robot.rgt_screwdriver],
        obstacles=obstacles,
    )
    return ScrewPlanner(
        robot.rgt_arm,
        pln_ctx=omp_utils.build_planning_context(collider),
        ee_actor=robot.rgt_screwdriver,
    )


def print_attach_debug(robot: KHIBunri, work, gid, jaw_width, engage_tf):
    tcp_tf = robot.lft_tcp_tf.copy()
    obj_tf = oum.tf_from_rotmat_pos(work.model.rotmat, work.model.pos)
    print(
        f'[playback_attach {work.name}] '
        f'gid={gid}, '
        f'jaw_width={float(jaw_width):.6f}, '
        f'tcp_pos={tcp_tf[:3, 3].tolist()}, '
        f'tcp_rotmat={tcp_tf[:3, :3].tolist()}, '
        f'obj_pos={obj_tf[:3, 3].tolist()}, '
        f'obj_rotmat={obj_tf[:3, :3].tolist()}, '
        f'engage_pos={engage_tf[:3, 3].tolist()}, '
        f'engage_rotmat={engage_tf[:3, :3].tolist()}'
    )


def print_timing_report(planner, label, min_total_s=1e-4):
    report = planner.timing_report()
    stats = report.get('stats', {})
    metrics = report.get('metrics', {})
    if not stats:
        stats = {}
    print(f'[{label} timing]')
    for key, value in sorted(stats.items(), key=lambda item: (-item[1]['total_s'], item[0])):
        total_s = float(value['total_s'])
        if total_s < min_total_s:
            continue
        count = int(value['count'])
        avg_s = total_s / count if count > 0 else 0.0
        max_s = float(value['max_s'])
        print(
            f'  {key}: total={total_s:.3f}s count={count} '
            f'avg={avg_s:.3f}s max={max_s:.3f}s'
        )
    for key, value in sorted(metrics.items(), key=lambda item: item[0]):
        count = int(value['count'])
        if count <= 0:
            continue
        avg_value = float(value['sum']) / count
        min_value = float(value['min']) if value['min'] is not None else 0.0
        max_value = float(value['max']) if value['max'] is not None else 0.0
        last_value = float(value['last']) if value['last'] is not None else 0.0
        print(
            f'  {key}: count={count} avg={avg_value:.3f} '
            f'min={min_value:.3f} max={max_value:.3f} last={last_value:.3f}'
        )


def sanitize_sync_token(raw: str) -> str:
    return raw.replace(':', '_').replace('/', '_').replace('-', '_')


def open_gripper_qs() -> np.ndarray:
    open_half = float(robot_ref().lft_gripper.jaw_range[1] * 0.5)
    return np.array([open_half, open_half], dtype=np.float32)


def segment_arm_path(segment: SyncSegment, actor: str) -> list[np.ndarray]:
    for arm_segment in segment.arm_segments:
        if arm_segment.actor == actor:
            return arm_segment.qs_list
    return []


def apply_ee_event_to_state(state: DualRobotState, event: EEEvent):
    updated = state.copy()
    if event.actor == 'left_gripper':
        if event.action in ('open', 'release'):
            updated.lft_ee_qs[:] = open_gripper_qs()
        elif event.action in ('close', 'attach') and event.value is not None:
            width = float(event.value) * 0.5
            updated.lft_ee_qs[:] = np.array([width, width], dtype=np.float32)
    elif event.actor == 'right_driver':
        if event.action in ('extend', 'retract') and event.value is not None:
            updated.rgt_ee_qs[:] = np.array([float(event.value)], dtype=np.float32)
    return updated


def expand_segment_states(start_state: DualRobotState, segment: SyncSegment) -> list[DualRobotState]:
    left_path = segment_arm_path(segment, 'left_arm')
    right_path = segment_arm_path(segment, 'right_arm')
    sample_count = max(len(left_path), len(right_path), 1)
    states = []
    for idx in range(sample_count):
        state = start_state.copy() if idx == 0 else states[-1].copy()
        if left_path:
            state.lft_qs = np.asarray(left_path[min(idx, len(left_path) - 1)], dtype=np.float32).copy()
        if right_path:
            state.rgt_qs = np.asarray(right_path[min(idx, len(right_path) - 1)], dtype=np.float32).copy()
        states.append(state)
    for event in segment.ee_events:
        if event.timing == 'sample' and event.sample_index is not None:
            start_idx = max(0, min(int(event.sample_index), sample_count - 1))
        elif event.timing == 'end':
            start_idx = sample_count - 1
        else:
            start_idx = 0
        for idx in range(start_idx, sample_count):
            states[idx] = apply_ee_event_to_state(states[idx], event)
    return states


def end_state_after_segment(start_state: DualRobotState, segment: SyncSegment) -> DualRobotState:
    return expand_segment_states(start_state, segment)[-1].copy()


def boundary_state_for_segment(playback: SynchronizedPlan, segment_idx: int) -> DualRobotState:
    if playback.initial_state is None:
        raise ValueError('playback.initial_state is required for replay')
    state = playback.initial_state.copy()
    for prev_segment in playback.sync_segments[:segment_idx]:
        state = end_state_after_segment(state, prev_segment)
    return state


def build_sync_segment_from_draft(segment_id: str,
                                  start_sync_id: str,
                                  end_sync_id: str,
                                  draft: PlannerSegmentDraft) -> SyncSegment:
    left_path = draft.left_path or []
    right_path = draft.right_path or []
    ee_events = list(draft.ee_events or [])
    return SyncSegment(
        id=segment_id,
        label=draft.segment_label,
        start_sync_id=start_sync_id,
        end_sync_id=end_sync_id,
        arm_segments=[
            ArmSegment(actor='left_arm', qs_list=left_path, idle=len(left_path) == 0),
            ArmSegment(actor='right_arm', qs_list=right_path, idle=len(right_path) == 0),
        ],
        ee_events=ee_events,
    )


def normalize_action_draft(draft) -> PlannerActionDraft:
    if isinstance(draft, PlannerActionDraft):
        return draft
    if isinstance(draft, PlannerSegmentDraft):
        return PlannerActionDraft(
            segments=[draft],
            held_after=draft.held_after,
        )
    raise TypeError(f'Unsupported planner draft type: {type(draft)!r}')


def plan_place_action(robot: KHIBunri,
                      worklist: WorkList,
                      state: DualRobotState,
                      action,
                      grasp_cache,
                      reasoning_backend='simd',
                      transit_backend='mujoco',
                      keep_holding=False):
    target_work = worklist[action.work_idx]
    goal_pose = target_work.pose_after_action(action.action_idx)
    if goal_pose is None:
        return None
    apply_dual_state(robot, state)
    planner = build_left_planner(
        robot,
        worklist,
        reasoning_backend=reasoning_backend,
        transit_backend=transit_backend,
    )
    grasp_collection = load_grasp_collection(target_work, robot.lft_gripper, grasp_cache)
    if not grasp_collection:
        print(f'no grasps for {target_work.name}')
        return None
    draft = planner.gen_place_draft(
        obj_model=target_work.model,
        work_name=target_work.name,
        grasp_collection=grasp_collection,
        goal_pose_list=[goal_pose],
        segment_label=f'place {target_work.name}',
        end_sync_label=action.label,
        pick_depart_direction=np.array([0, 0, -1], dtype=np.float32),
        approach_direction=np.array([0, 0, -1], dtype=np.float32),
        start_qs=compose_left_state(state),
        linear_granularity=0.02,
        reason_grasps=True,
        use_rrt=True,
        pln_jnt=False,
        keep_holding=keep_holding,
        toggle_dbg=False,
    )
    # print_timing_report(planner, f'pick/place {target_work.name}')
    if draft is None:
        print(f'pick/place failed for {target_work.name}')
        return None
    return draft


def plan_fold_action(robot: KHIBunri,
                     worklist: WorkList,
                     state: DualRobotState,
                     action,
                     grasp_cache,
                     reasoning_backend='simd',
                     transit_backend='mujoco',
                     held_grasp: HeldGrasp | None = None):
    target_work = worklist[action.work_idx]
    start_pose = target_work.current_pose
    goal_pose = target_work.pose_after_action(action.action_idx, start_pose=start_pose)
    if goal_pose is None:
        return None
    apply_dual_state(robot, state)
    planner = build_fold_planner(
        robot,
        worklist,
        reasoning_backend=reasoning_backend,
        transit_backend=transit_backend,
    )
    goal_pose_list = interpolate_fold(start_pose, goal_pose, n_steps=6)
    grasp_collection = load_grasp_collection(target_work, robot.lft_gripper, grasp_cache)
    if not grasp_collection:
        print(f'no grasps for {target_work.name}')
        return None
    draft = planner.gen_fold_draft(
        obj_model=target_work.model,
        work_name=target_work.name,
        grasp_collection=grasp_collection,
        goal_pose_list=goal_pose_list,
        held_grasp=held_grasp,
        segment_label=f'fold {target_work.name}',
        end_sync_label=action.label,
        start_qs=compose_left_state(state),
        linear_granularity=0.02,
        reason_grasps=True,
        use_rrt=True,
        pln_jnt=False,
        exclude_entities=[target_work.model],
        toggle_dbg=False,
    )
    if draft is None:
        print(f'pick/fold failed for {target_work.name}')
        return None
    return draft


def plan_screw_action(robot: KHIBunri,
                      worklist: WorkList,
                      state: DualRobotState,
                      action,
                      screw_resolution=12):
    target_work = worklist[action.work_idx]
    goal_pose = target_work.pose_after_action(action.action_idx, start_pose=target_work.current_pose)
    if goal_pose is None:
        return None
    screw_counter = int(worklist.screw_counter)
    pick_pose = worklist.get_screw_pose()
    worklist.screw_counter = screw_counter
    apply_dual_state(robot, state)
    planner = build_right_planner(robot, worklist, target_work)
    pick_pose_list = resolve_screw_pick_pose_candidates(
        planner,
        state,
        pick_pose,
        roll_resolution=screw_resolution,
        toggle_dbg=False,
    )
    if not pick_pose_list:
        print(f'no valid screw pickup pose for {target_work.name}')
        return None
    screw_axis = goal_pose[1][:, 2].astype(np.float32)
    draft = planner.gen_screw_draft(
        work_name=target_work.name,
        start_qs=compose_right_state(state),
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
        print(f'screw failed for {target_work.name}')
        return None
    return draft


def resolve_screw_pick_pose_candidates(planner: ScrewPlanner,
                                       state: DualRobotState,
                                       pick_pose,
                                       roll_resolution=12,
                                       toggle_dbg=False):
    ref_qs = state.rgt_qs.astype(np.float32)
    roll_candidates = planner.gen_pose_roll_candidates(pick_pose, resolution=roll_resolution)
    valid_roll_poses = []
    for roll_idx, roll_pose in enumerate(roll_candidates):
        candidate_state = planner._pick_state(pick_pose=roll_pose, ref_qs=ref_qs)
        if candidate_state is None:
            if toggle_dbg:
                print(f'[screw_pick] roll_idx={roll_idx}, pick_state=None')
            continue
        if toggle_dbg:
            print(
                '[screw_pick] '
                f'roll_idx={roll_idx}, pick_state=1, '
                f'candidate_pos={roll_pose[0].tolist()}'
            )
        valid_roll_poses.append(roll_pose)
    return valid_roll_poses


def apply_symbolic_action(worklist: WorkList, action):
    work = worklist[action.work_idx]
    work.apply_action(action.action_idx)
    if work.type[action.action_idx] == 'screw':
        worklist.screw_counter += 1


def reset_prefix_work_state(worklist: WorkList, leaf, layout_name: str, action_idx: int):
    prefix_actions = [
        (prev_action.work_idx, prev_action.action_idx)
        for prev_action in leaf.sequence[:action_idx]
    ]
    reset_work_state(worklist, layout_name=layout_name, actions=prefix_actions)


def build_synchronized_plan(robot: KHIBunri,
                            worklist: WorkList,
                            leaf,
                            layout_name,
                            reasoning_backend='simd',
                            transit_backend='mujoco',
                            screw_resolution=12):
    grasp_cache = {}
    max_plan_attempts = 1
    _ROBOT_REF['value'] = robot

    reset_work_state(worklist, layout_name=layout_name)
    robot.goto_home_conf()
    robot.lft_gripper.open()
    robot.rgt_screwdriver.set_shank_len(robot.rgt_screwdriver.shank_range[0])
    current_state = capture_dual_state(robot)
    sync_points = [SyncPoint(id='sp_home', label='both home')]
    sync_segments = []
    held_grasp = None

    for idx, action in enumerate(leaf.sequence):
        reset_prefix_work_state(worklist, leaf, layout_name, idx)
        next_action = None if idx + 1 >= len(leaf.sequence) else leaf.sequence[idx + 1]
        keep_holding = (
            action.action_type == 'place' and
            action.immediate and
            next_action is not None and
            next_action.work_idx == action.work_idx and
            next_action.action_type == 'fold'
        )
        start_sync_id = sync_points[-1].id
        end_sync_id = f'sp_{idx + 1}_{sanitize_sync_token(action.label)}'
        draft = None
        for _attempt in range(max_plan_attempts):
            if action.action_type == 'place':
                draft = plan_place_action(
                    robot,
                    worklist,
                    current_state,
                    action,
                    grasp_cache,
                    reasoning_backend=reasoning_backend,
                    transit_backend=transit_backend,
                    keep_holding=keep_holding,
                )
            elif action.action_type == 'fold':
                draft = plan_fold_action(
                    robot,
                    worklist,
                    current_state,
                    action,
                    grasp_cache,
                    reasoning_backend=reasoning_backend,
                    transit_backend=transit_backend,
                    held_grasp=held_grasp,
                )
            elif action.action_type == 'screw':
                draft = plan_screw_action(
                    robot,
                    worklist,
                    current_state,
                    action,
                    screw_resolution=screw_resolution,
                )
            else:
                return None
            if draft is not None:
                break
        if draft is None:
            print(f'plan failed for {action.label}')
            return None
        action_draft = normalize_action_draft(draft)
        if not action_draft.segments:
            print(f'empty draft for {action.label}')
            return None
        segment_start_sync_id = start_sync_id
        for seg_idx, seg_draft in enumerate(action_draft.segments):
            seg_end_label = seg_draft.end_sync_label or action.label
            seg_end_sync_id = (
                end_sync_id if seg_idx == len(action_draft.segments) - 1
                else f'{end_sync_id}_k{seg_idx}'
            )
            segment = build_sync_segment_from_draft(
                segment_id=f'seg_{idx}_{seg_idx}_{sanitize_sync_token(seg_end_label)}',
                start_sync_id=segment_start_sync_id,
                end_sync_id=seg_end_sync_id,
                draft=seg_draft,
            )
            sync_segments.append(segment)
            current_state = end_state_after_segment(current_state, segment)
            sync_points.append(SyncPoint(id=seg_end_sync_id, label=seg_end_label))
            segment_start_sync_id = seg_end_sync_id
        held_grasp = action_draft.held_after
        apply_symbolic_action(worklist, action)

    reset_work_state(worklist, layout_name=layout_name)
    robot.goto_home_conf()
    robot.lft_gripper.open()
    robot.rgt_screwdriver.set_shank_len(robot.rgt_screwdriver.shank_range[0])
    return SynchronizedPlan(
        labels=sequence_labels(leaf),
        initial_state=capture_dual_state(robot),
        sync_points=sync_points,
        sync_segments=sync_segments,
    )


def sorted_leaves(nodes):
    leaves = leaf_nodes(nodes)
    return sorted(leaves, key=lambda node: (-node.depth, node.node_id))


def manual_leaf(worklist: WorkList, sequence: str, layout_name: str) -> AssemblyNode:
    result = execute_sequence_string(worklist, sequence, layout_name=layout_name)
    actions = tuple(result.actions)
    node_id = 'manual' if not actions else 'manual/' + '/'.join(action.label for action in actions)
    return AssemblyNode(
        node_id=node_id,
        depth=len(actions),
        parent_id=None,
        action=None if not actions else actions[-1],
        sequence=actions,
        work_state=None if not result.work_states else result.work_states[-1],
    )


def main():
    args = parse_args()

    planning_worklist = WorkList(collision_type=ouc.CollisionType.MESH)
    planning_worklist.init_pose(seed=args.layout)
    if args.sequence is not None:
        leaves = [manual_leaf(planning_worklist, args.sequence, args.layout)]
        print(f'manual seq={sequence_labels(leaves[0])}')
    else:
        nodes = assembly_sequence_planning(
            worklist=planning_worklist,
            initial_layout=args.layout,
            max_depth=args.max_depth,
        )
        leaves = sorted_leaves(nodes)
        for idx, leaf in enumerate(leaves):
            print(f'leaf[{idx}] depth={leaf.depth} seq={sequence_labels(leaf)}')
    if args.list_only:
        return

    base = ovw.World(
        cam_pos=(3.1, 1.9, 2.0),
        cam_lookat_pos=(0.18, 0.0, 0.55),
        toggle_auto_cam_orbit=False,
    )
    builtins.base = base
    ossop.frame(length_scale=0.25, radius_scale=1.2).attach_to(base.scene)

    worklist = WorkList(pos=oum.vec(0.2 + 0.09 + 0.035, 0, 0.11 + 0.018 + 0.0902), collision_type=ouc.CollisionType.MESH)
    worklist.init_pose(seed=args.layout)
    worklist.attach_to(base.scene)

    robot = KHIBunri()
    robot.attach_to(base.scene)
    # robot.body.alpha = 0.22
    # robot.lft_arm.alpha = 0.22
    # robot.rgt_arm.alpha = 0.22
    # robot.lft_gripper.alpha = 0.22
    # robot.rgt_screwdriver.alpha = 0.22
    # robot.lft_adapter.alpha = 0.22
    # robot.rgt_adapter.alpha = 0.22
    _ROBOT_REF['value'] = robot

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

    if args.sequence is not None:
        candidates = leaves
    elif args.leaf_index is not None:
        candidates = [leaves[args.leaf_index]]
    else:
        candidates = leaves

    selected_leaf = None
    playback = None
    for leaf in candidates:
        print(f'trying leaf: {sequence_labels(leaf)}')
        playback = build_synchronized_plan(
            robot,
            worklist,
            leaf,
            args.layout,
            reasoning_backend=args.reasoning_backend,
            transit_backend=args.transit_backend,
            screw_resolution=args.screw_resolution,
        )
        if playback is not None:
            selected_leaf = leaf
            break
    if playback is None or selected_leaf is None:
        raise RuntimeError('Failed to build a feasible playback plan for the selected leaves')

    print(f'selected leaf: {playback.labels}')
    held = {'name': None}
    counter = {'segment_idx': 0, 'sample_idx': 0}
    root_work_state = reset_work_state(worklist, layout_name=args.layout)
    root_free_states = {
        work.name: bool(getattr(work.model, 'is_free', False))
        for work in worklist.work
    }
    current_segment_cache = {'segment_id': None, 'states': []}

    def clear_left_mountings():
        for child in list(robot.lft_gripper._mountings.keys()):
            robot.lft_gripper.release(child)

    def apply_ee_event(event: EEEvent):
        work = worklist.get_work(event.work_name) if event.work_name is not None else None
        if event.actor == 'left_gripper':
            if event.action == 'open':
                robot.lft_gripper.fk(open_gripper_qs())
            elif event.action == 'attach' and work is not None:
                if held['name'] != event.work_name:
                    if hasattr(work.model, 'is_free'):
                        work.model.is_free = True
                    jaw_width = event.value
                    engage_tf = event.engage_tf
                    if jaw_width is None or engage_tf is None:
                        jaw_width = float(np.sum(robot.lft_gripper.qs[:robot.lft_gripper.ndof]))
                        robot.lft_gripper.grasp(work.model, jaw_width=jaw_width)
                    else:
                        # print_attach_debug(robot, work, event.grasp_id, jaw_width, engage_tf)
                        robot.lft_gripper.set_jaw_width(float(jaw_width))
                        robot.lft_gripper.mount(work.model, robot.lft_gripper.runtime_root_lnk, engage_tf)
                        robot.lft_gripper._update_mounting(robot.lft_gripper._mountings[work.model])
                    held['name'] = event.work_name
            elif event.action == 'release' and work is not None:
                if work.model in robot.lft_gripper._mountings:
                    robot.lft_gripper.release(work.model)
                robot.lft_gripper.fk(open_gripper_qs())
                held['name'] = None
        elif event.actor == 'right_driver' and event.action in ('extend', 'retract') and event.value is not None:
            robot.rgt_screwdriver.fk(np.array([float(event.value)], dtype=np.float32))

    def trigger_segment_events(segment: SyncSegment, sample_idx: int):
        is_last_sample = sample_idx == len(current_segment_cache['states']) - 1
        for event in segment.ee_events:
            if event.timing == 'start' and sample_idx == 0:
                apply_ee_event(event)
            elif event.timing == 'sample' and event.sample_index == sample_idx:
                apply_ee_event(event)
            elif event.timing == 'end' and is_last_sample:
                apply_ee_event(event)

    def segment_states(segment: SyncSegment, start_state: DualRobotState) -> list[DualRobotState]:
        if current_segment_cache['segment_id'] != segment.id:
            current_segment_cache['segment_id'] = segment.id
            current_segment_cache['states'] = expand_segment_states(start_state, segment)
        return current_segment_cache['states']

    def reset_scene():
        clear_left_mountings()
        held['name'] = None
        reset_work_state(worklist, state=root_work_state)
        for work in worklist.work:
            root_is_free = root_free_states.get(work.name)
            if root_is_free is not None and hasattr(work.model, 'is_free'):
                work.model.is_free = root_is_free
        if playback.initial_state is None:
            raise RuntimeError('playback.initial_state is not available')
        apply_dual_state(robot, playback.initial_state)
        current_segment_cache['segment_id'] = None
        current_segment_cache['states'] = []
        counter['segment_idx'] = 0
        counter['sample_idx'] = 0

    def update_tcp_frames():
        lft_tcp_frame.set_rotmat_pos(rotmat=robot.lft_tcp_tf[:3, :3], pos=robot.lft_tcp_tf[:3, 3])
        rgt_tcp_frame.set_rotmat_pos(rotmat=robot.rgt_tcp_tf[:3, :3], pos=robot.rgt_tcp_tf[:3, 3])

    def update(_dt):
        if counter['segment_idx'] >= len(playback.sync_segments):
            reset_scene()
            return
        segment = playback.sync_segments[counter['segment_idx']]
        boundary_state = boundary_state_for_segment(playback, counter['segment_idx'])
        states = segment_states(segment, boundary_state)
        apply_dual_state(robot, states[counter['sample_idx']])
        trigger_segment_events(segment, counter['sample_idx'])
        update_tcp_frames()
        counter['sample_idx'] += 1
        if counter['sample_idx'] >= len(states):
            counter['segment_idx'] += 1
            counter['sample_idx'] = 0
            current_segment_cache['segment_id'] = None
            current_segment_cache['states'] = []

    reset_scene()
    update(0.0)
    base.schedule_interval(update, interval=args.interval)
    base.run()


if __name__ == '__main__':
    main()
