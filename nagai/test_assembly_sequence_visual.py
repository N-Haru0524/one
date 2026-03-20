import argparse
import builtins
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one import ossop, ovw, ouc
import one.utils.math as oum
from one.grasp.antipodal import antipodal
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


@dataclass
class DualRobotState:
    lft_qs: np.ndarray
    lft_ee_qs: np.ndarray
    rgt_qs: np.ndarray
    rgt_ee_qs: np.ndarray

    def copy(self):
        return DualRobotState(
            lft_qs=self.lft_qs.copy(),
            lft_ee_qs=self.lft_ee_qs.copy(),
            rgt_qs=self.rgt_qs.copy(),
            rgt_ee_qs=self.rgt_ee_qs.copy(),
        )


@dataclass
class PlaybackPlan:
    labels: list[str]
    state_list: list[DualRobotState]
    event_map: dict[int, tuple]


@dataclass
class HeldGrasp:
    work_name: str
    gid: int


@dataclass
class PlannedSegment:
    state_list: list[DualRobotState]
    event_map: dict[int, tuple]
    held_after: HeldGrasp | None = None


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


def embed_left_path(state: DualRobotState, path) -> list[DualRobotState]:
    embedded = []
    for qs in path:
        qs = np.asarray(qs, dtype=np.float32)
        lft_qs = qs[:6]
        lft_ee_qs = qs[6:6 + robot_ref().lft_gripper.ndof]
        embedded.append(DualRobotState(
            lft_qs=lft_qs.copy(),
            lft_ee_qs=lft_ee_qs.copy(),
            rgt_qs=state.rgt_qs.copy(),
            rgt_ee_qs=state.rgt_ee_qs.copy(),
        ))
    return embedded


def embed_right_path(state: DualRobotState, path) -> list[DualRobotState]:
    embedded = []
    for qs in path:
        qs = np.asarray(qs, dtype=np.float32)
        rgt_qs = qs[:6]
        rgt_ee_qs = qs[6:6 + robot_ref().rgt_screwdriver.ndof]
        embedded.append(DualRobotState(
            lft_qs=state.lft_qs.copy(),
            lft_ee_qs=state.lft_ee_qs.copy(),
            rgt_qs=rgt_qs.copy(),
            rgt_ee_qs=rgt_ee_qs.copy(),
        ))
    return embedded


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


def append_release_state(state_list: list[DualRobotState], event_map, work_name):
    final_state = state_list[-1].copy()
    open_half = float(robot_ref().lft_gripper.jaw_range[1] * 0.5)
    final_state.lft_ee_qs[:] = np.array([open_half, open_half], dtype=np.float32)
    state_list.append(final_state)
    event_map[len(state_list) - 1] = ('release', work_name)

def grasp_event_payload(gripper, obj_pose, grasp):
    pose_tf = oum.ensure_tf(obj_pose) @ np.asarray(grasp[0], dtype=np.float32)
    pick_tf = oum.ensure_tf(obj_pose)
    jaw_width = float(np.asarray(grasp[2], dtype=np.float32).reshape(-1)[0])
    ee_base_tf = pose_tf @ np.linalg.inv(gripper.loc_tcp_tf)
    engage_tf = np.linalg.inv(ee_base_tf) @ pick_tf
    return jaw_width, engage_tf.astype(np.float32)


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


def plan_place_action(robot: KHIBunri,
                      worklist: WorkList,
                      state: DualRobotState,
                      action,
                      grasp_cache,
                      reasoning_backend='simd',
                      transit_backend='mujoco',
                      keep_holding=False,
                      release_work_name=None):
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
    plan = planner.gen_pick_and_place(
        obj_model=target_work.model,
        grasp_collection=grasp_collection,
        goal_pose_list=[goal_pose],
        start_qs=compose_left_state(state),
        linear_granularity=0.02,
        reason_grasps=True,
        use_rrt=True,
        pln_jnt=False,
        release_at_end=not keep_holding,
        toggle_dbg=True,
    )
    print_timing_report(planner, f'pick/place {target_work.name}')
    if plan is None:
        print(f'pick/place failed for {target_work.name}')
        return None
    state_list = embed_left_path(state, plan.qs_list)
    event_map = {}
    selected_gid = int(plan.events['gid']) if 'gid' in plan.events else None
    if selected_gid is not None and 0 <= selected_gid < len(grasp_collection):
        pick_pose_tf = oum.tf_from_rotmat_pos(target_work.model.rotmat, target_work.model.pos)
        jaw_width, engage_tf = grasp_event_payload(
            robot.lft_gripper,
            pick_pose_tf,
            grasp_collection[selected_gid],
        )
    else:
        jaw_width = None
        engage_tf = None
    if 'attach' in plan.events:
        event_map[int(plan.events['attach'])] = (
            'attach',
            target_work.name,
            selected_gid,
            jaw_width,
            engage_tf,
        )
    if 'release' in plan.events:
        event_map[int(plan.events['release'])] = ('release', target_work.name)
    held_after = None
    if keep_holding and selected_gid is not None:
        held_after = HeldGrasp(work_name=target_work.name, gid=selected_gid)
    return PlannedSegment(state_list=state_list, event_map=event_map, held_after=held_after)


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
    if held_grasp is not None and held_grasp.work_name == target_work.name:
        if held_grasp.gid < 0 or held_grasp.gid >= len(grasp_collection):
            return None
        plan = planner.gen_hold_and_fold(
            obj_model=target_work.model,
            grasp=grasp_collection[held_grasp.gid],
            goal_pose_list=goal_pose_list,
            start_qs=compose_left_state(state),
            linear_granularity=0.02,
            pln_jnt=False,
            exclude_entities=[target_work.model],
            gid=held_grasp.gid,
            toggle_dbg=True,
        )
    else:
        plan = planner.gen_pick_and_fold(
            obj_model=target_work.model,
            grasp_collection=grasp_collection,
            goal_pose_list=goal_pose_list,
            start_qs=compose_left_state(state),
            linear_granularity=0.02,
            reason_grasps=True,
            use_rrt=True,
            pln_jnt=False,
            exclude_entities=[target_work.model],
            toggle_dbg=True,
        )
    if plan is None:
        print(f'pick/fold failed for {target_work.name}')
        return None
    state_list = embed_left_path(state, plan.qs_list)
    event_map = {}
    if 'attach' in plan.events:
        event_map[int(plan.events['attach'])] = ('attach', target_work.name)
    append_release_state(state_list, event_map, target_work.name)
    return PlannedSegment(state_list=state_list, event_map=event_map, held_after=None)


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
    home_pose = worklist.get_screw_pose()
    worklist.screw_counter = screw_counter
    apply_dual_state(robot, state)
    planner = build_right_planner(robot, worklist, target_work)
    home_pose = resolve_screw_home_pose(
        planner,
        state,
        home_pose,
        roll_resolution=screw_resolution,
        toggle_dbg=True,
    )
    if home_pose is None:
        print(f'no valid screw pickup pose for {target_work.name}')
        return None
    screw_axis = goal_pose[1][:, 2].astype(np.float32)
    plan = planner.gen_screw(
        start_qs=compose_right_state(state),
        goal_pose_list=[goal_pose],
        resolution=screw_resolution,
        home_pose=home_pose,
        approach_direction=screw_axis,
        approach_distance=0.05,
        depart_direction=-screw_axis,
        depart_distance=0.03,
        linear_granularity=0.02,
        use_rrt=True,
        pln_jnt=False,
        toggle_dbg=True,
    )
    if plan is None:
        print(f'screw failed for {target_work.name}')
        return None
    state_list = embed_right_path(state, plan.qs_list)
    event_map = {}
    retract_state = state_list[-1].copy()
    retract_state.rgt_ee_qs[:] = np.array([robot.rgt_screwdriver.shank_range[0]], dtype=np.float32)
    state_list.append(retract_state)
    return PlannedSegment(state_list=state_list, event_map=event_map, held_after=None)


def resolve_screw_home_pose(planner: ScrewPlanner,
                            state: DualRobotState,
                            home_pose,
                            roll_resolution=12,
                            toggle_dbg=False):
    ref_qs = state.rgt_qs.astype(np.float32)
    roll_candidates = planner.gen_pose_roll_candidates(home_pose, resolution=roll_resolution)
    for roll_idx, roll_pose in enumerate(roll_candidates):
        candidate_state = planner._home_state(home_pose=roll_pose, ref_qs=ref_qs)
        if candidate_state is None:
            if toggle_dbg:
                print(f'[screw_home] roll_idx={roll_idx}, home_state=None')
            continue
        is_valid = planner.pln_ctx.is_state_valid(candidate_state)
        if toggle_dbg:
            print(
                '[screw_home] '
                f'roll_idx={roll_idx}, state_valid={int(is_valid)}, '
                f'candidate_pos={roll_pose[0].tolist()}'
            )
        if is_valid:
            return roll_pose
    return None


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


def flatten_segment(global_states, global_events, state_list, event_map):
    start_idx = len(global_states)
    skipped_prefix = 0
    for idx, state in enumerate(state_list):
        if start_idx > 0 and idx == 0 and np.allclose(
                global_states[-1].lft_qs, state.lft_qs) and np.allclose(
                global_states[-1].lft_ee_qs, state.lft_ee_qs) and np.allclose(
                global_states[-1].rgt_qs, state.rgt_qs) and np.allclose(
                global_states[-1].rgt_ee_qs, state.rgt_ee_qs):
            skipped_prefix = 1
            continue
        global_states.append(state.copy())
    for idx, event in event_map.items():
        global_events[start_idx + idx - skipped_prefix] = event


def build_playback_plan(robot: KHIBunri,
                        worklist: WorkList,
                        leaf,
                        layout_name,
                        reasoning_backend='simd',
                        transit_backend='mujoco',
                        screw_resolution=12):
    grasp_cache = {}
    global_states = []
    global_events = {}
    max_plan_attempts = 1
    _ROBOT_REF['value'] = robot

    reset_work_state(worklist, layout_name=layout_name)
    robot.goto_home_conf()
    robot.lft_gripper.open()
    robot.rgt_screwdriver.set_shank_len(robot.rgt_screwdriver.shank_range[0])
    current_state = capture_dual_state(robot)
    global_states.append(current_state.copy())
    held_grasp = None
    pending_release_work = None

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
        planned = None
        for _attempt in range(max_plan_attempts):
            if action.action_type == 'place':
                planned = plan_place_action(
                    robot,
                    worklist,
                    current_state,
                    action,
                    grasp_cache,
                    reasoning_backend=reasoning_backend,
                    transit_backend=transit_backend,
                    keep_holding=keep_holding,
                    release_work_name=pending_release_work,
                )
            elif action.action_type == 'fold':
                planned = plan_fold_action(
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
                planned = plan_screw_action(
                    robot,
                    worklist,
                    current_state,
                    action,
                    screw_resolution=screw_resolution,
                )
            else:
                return None
            if planned is not None:
                break
        if planned is None:
            print(f'plan failed for {action.label}')
            return None

        if pending_release_work is not None and len(global_states) > 0:
            global_events[len(global_states) - 1] = ('release', pending_release_work)
            pending_release_work = None
        flatten_segment(global_states, global_events, planned.state_list, planned.event_map)
        current_state = global_states[-1].copy()
        held_grasp = planned.held_after
        if action.action_type == 'place' and not keep_holding:
            pending_release_work = worklist[action.work_idx].name
        apply_symbolic_action(worklist, action)

    reset_work_state(worklist, layout_name=layout_name)
    robot.goto_home_conf()
    robot.lft_gripper.open()
    robot.rgt_screwdriver.set_shank_len(robot.rgt_screwdriver.shank_range[0])
    return PlaybackPlan(
        labels=sequence_labels(leaf),
        state_list=global_states,
        event_map=global_events,
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
        playback = build_playback_plan(
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
    counter = {'idx': 0}
    root_state = capture_dual_state(robot)
    root_work_state = reset_work_state(worklist, layout_name=args.layout)
    root_free_states = {
        work.name: bool(getattr(work.model, 'is_free', False))
        for work in worklist.work
    }

    def apply_event(idx):
        event = playback.event_map.get(idx)
        if event is None:
            return
        kind = event[0]
        work_name = event[1]
        work = worklist.get_work(work_name)
        if work is None:
            return
        if kind == 'attach':
            if held['name'] != work_name:
                if hasattr(work.model, 'is_free'):
                    work.model.is_free = True
                _gid, jaw_width, engage_tf = event[2], event[3], event[4]
                if jaw_width is None or engage_tf is None:
                    jaw_width = float(np.sum(robot.lft_gripper.qs[:robot.lft_gripper.ndof]))
                    robot.lft_gripper.grasp(work.model, jaw_width=jaw_width)
                else:
                    print_attach_debug(robot, work, _gid, jaw_width, engage_tf)
                    robot.lft_gripper.set_jaw_width(float(jaw_width))
                    robot.lft_gripper.mount(work.model, robot.lft_gripper.runtime_root_lnk, engage_tf)
                    robot.lft_gripper._update_mounting(robot.lft_gripper._mountings[work.model])
                held['name'] = work_name
        elif kind == 'release':
            if work.model in robot.lft_gripper._mountings:
                robot.lft_gripper.release(work.model)
            held['name'] = None

    def reset_scene():
        for child in list(robot.lft_gripper._mountings.keys()):
            robot.lft_gripper.release(child)
        held['name'] = None
        reset_work_state(worklist, state=root_work_state)
        for work in worklist.work:
            root_is_free = root_free_states.get(work.name)
            if root_is_free is not None and hasattr(work.model, 'is_free'):
                work.model.is_free = root_is_free
        apply_dual_state(robot, root_state)
        counter['idx'] = 0

    def update_tcp_frames():
        lft_tcp_frame.set_rotmat_pos(rotmat=robot.lft_tcp_tf[:3, :3], pos=robot.lft_tcp_tf[:3, 3])
        rgt_tcp_frame.set_rotmat_pos(rotmat=robot.rgt_tcp_tf[:3, :3], pos=robot.rgt_tcp_tf[:3, 3])

    def update(_dt):
        if counter['idx'] >= len(playback.state_list):
            reset_scene()
            return
        apply_dual_state(robot, playback.state_list[counter['idx']])
        apply_event(counter['idx'])
        update_tcp_frames()
        counter['idx'] += 1

    reset_scene()
    update(0.0)
    base.schedule_interval(update, interval=args.interval)
    base.run()


if __name__ == '__main__':
    main()
