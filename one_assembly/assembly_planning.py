from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .worklist import Pose, WorkList


@dataclass(frozen=True)
class AssemblyAction:
    work_idx: int
    action_idx: int
    work_name: str
    action_type: str
    immediate: bool
    label: str
    token: str


@dataclass
class WorkState:
    layout_name: Optional[str]
    screw_counter: int
    part_poses: Dict[str, Pose]
    work_base_pose: Optional[Pose] = None


@dataclass
class RobotState:
    qs: np.ndarray
    ee_qs: Optional[np.ndarray] = None


@dataclass
class AssemblyNode:
    node_id: str
    depth: int
    parent_id: Optional[str]
    action: Optional[AssemblyAction]
    sequence: Tuple[AssemblyAction, ...]
    children: List[str] = field(default_factory=list)
    work_state: Optional[WorkState] = None


@dataclass
class ExecutionResult:
    actions: List[AssemblyAction]
    work_states: List[WorkState]


_WORK_ALIASES = {
    'workbench': 'wrkbnch',
    'bracket': 'brckt',
    'capacitor': 'cpctr',
    'relay': 'rly',
    'belt': 'blt',
    'terminal': 'trmnl',
}

_ACTION_SUFFIXES = {
    'place': '',
    'fold': 'fld',
    'screw': 'scrw',
}


def capture_robot_state(robot, ee_actor=None) -> RobotState:
    qs = np.asarray(getattr(robot, 'qs'), dtype=np.float32).copy()
    ee_qs = None
    if ee_actor is not None:
        ee_ndof = getattr(ee_actor, 'ndof', None)
        raw_ee_qs = np.asarray(getattr(ee_actor, 'qs'), dtype=np.float32)
        ee_qs = raw_ee_qs.copy() if ee_ndof is None else raw_ee_qs[:ee_ndof].copy()
    return RobotState(qs=qs, ee_qs=ee_qs)


def reset_robot_state(robot, state: RobotState, ee_actor=None) -> RobotState:
    robot.fk(np.asarray(state.qs, dtype=np.float32))
    if ee_actor is not None and state.ee_qs is not None:
        ee_actor.fk(np.asarray(state.ee_qs, dtype=np.float32))
    return capture_robot_state(robot, ee_actor=ee_actor)


def capture_work_state(worklist: WorkList) -> WorkState:
    work_base_pose = None
    if worklist.work_base is not None:
        work_base_pose = (
            worklist.work_base.pos.copy(),
            worklist.work_base.rotmat.copy(),
        )
    return WorkState(
        layout_name=worklist.layout_name,
        screw_counter=int(worklist.screw_counter),
        part_poses=worklist.current_part_poses(),
        work_base_pose=work_base_pose,
    )


def _restore_work_state(worklist: WorkList, state: WorkState) -> WorkState:
    worklist.layout_name = state.layout_name
    worklist.set_part_poses(state.part_poses)
    worklist.screw_counter = int(state.screw_counter)
    if state.work_base_pose is not None:
        work_base = worklist._ensure_work_base()
        work_base.pos = np.asarray(state.work_base_pose[0], dtype=np.float32)
        work_base.rotmat = np.asarray(state.work_base_pose[1], dtype=np.float32)
    return capture_work_state(worklist)


def _apply_symbolic_action(worklist: WorkList, action: AssemblyAction):
    work = worklist[action.work_idx]
    work.apply_action(action.action_idx)
    if action.action_type == 'screw':
        worklist.screw_counter += 1


def reset_work_state(worklist: WorkList,
                     layout_name: str = 'home',
                     actions: Optional[Sequence[Tuple[int, int]]] = None,
                     state: Optional[WorkState] = None) -> WorkState:
    if state is not None:
        return _restore_work_state(worklist, state)

    worklist.init_pose(seed=layout_name)
    if actions is not None:
        for work_idx, action_idx in actions:
            _apply_symbolic_action(worklist, make_action(worklist, int(work_idx), int(action_idx)))
    return capture_work_state(worklist)


def _work_alias(work_name: str) -> str:
    return _WORK_ALIASES.get(work_name, work_name.replace('_', '').lower())


def _action_token(work, action_idx: int) -> str:
    alias = _work_alias(work.name)
    suffixes = []
    for step_type in work.type[:action_idx + 1]:
        suffix = _ACTION_SUFFIXES.get(step_type, step_type)
        if suffix:
            suffixes.append(suffix)
    if not suffixes:
        return alias
    return f"{alias}_{'_'.join(suffixes)}"


def make_action(worklist: WorkList, work_idx: int, action_idx: int) -> AssemblyAction:
    work = worklist[work_idx]
    step = work.step(action_idx)
    return AssemblyAction(
        work_idx=work_idx,
        action_idx=action_idx,
        work_name=work.name,
        action_type=step.action_type,
        immediate=step.immediate,
        label=f'{work.name}:time{action_idx}',
        token=_action_token(work, action_idx),
    )


def _executed_action_counts(sequence: Sequence[AssemblyAction], n_works: int) -> List[int]:
    counts = [0] * n_works
    for action in sequence:
        counts[action.work_idx] = max(counts[action.work_idx], action.action_idx + 1)
    return counts


def _initial_action_counts(worklist: WorkList, layout_name: str) -> List[int]:
    counts = [0] * len(worklist)
    layout = worklist.layout_specs.get(layout_name)
    if layout is None:
        return counts
    for entry in layout.part_entries:
        if not entry.preplace:
            continue
        if entry.work_idx >= len(worklist):
            continue
        work = worklist[entry.work_idx]
        if len(work) == 0:
            continue
        first_step = work.step(0)
        if first_step.action_type == 'place':
            counts[entry.work_idx] = 1
    return counts


def _candidate_actions(worklist: WorkList,
                       sequence: Sequence[AssemblyAction],
                       initial_counts: Optional[Sequence[int]] = None) -> List[AssemblyAction]:
    counts = list(initial_counts) if initial_counts is not None else [0] * len(worklist)
    executed_counts = _executed_action_counts(sequence, len(worklist))
    for work_idx, executed in enumerate(executed_counts):
        counts[work_idx] = max(counts[work_idx], executed)
    if sequence:
        last_action = sequence[-1]
        if last_action.immediate:
            next_idx = counts[last_action.work_idx]
            if next_idx < len(worklist[last_action.work_idx]):
                return [make_action(worklist, last_action.work_idx, next_idx)]
            return []

    candidates = []
    for work_idx, next_idx in enumerate(counts):
        if next_idx >= len(worklist[work_idx]):
            continue
        candidates.append(make_action(worklist, work_idx, next_idx))
    return candidates


def _sequence_node_id(sequence: Sequence[AssemblyAction]) -> str:
    if not sequence:
        return 'root'
    return 'root/' + '/'.join(action.label for action in sequence)


def assembly_sequence_planning(worklist: WorkList,
                               initial_layout: str = 'home',
                               max_depth: Optional[int] = None) -> Dict[str, AssemblyNode]:
    total_actions = sum(len(work) for work in worklist)
    if max_depth is None:
        max_depth = total_actions
    max_depth = max(0, min(int(max_depth), total_actions))

    initial_counts = _initial_action_counts(worklist, initial_layout)
    root_state = reset_work_state(worklist, layout_name=initial_layout)
    root = AssemblyNode(
        node_id='root',
        depth=0,
        parent_id=None,
        action=None,
        sequence=tuple(),
        work_state=root_state,
    )
    nodes = {root.node_id: root}
    frontier = [root]

    while frontier:
        node = frontier.pop(0)
        if node.depth >= max_depth:
            continue
        for action in _candidate_actions(worklist, node.sequence, initial_counts=initial_counts):
            child_sequence = node.sequence + (action,)
            child_id = _sequence_node_id(child_sequence)
            if child_id in nodes:
                if child_id not in node.children:
                    node.children.append(child_id)
                continue
            child_state = reset_work_state(
                worklist,
                layout_name=initial_layout,
                actions=[(item.work_idx, item.action_idx) for item in child_sequence],
            )
            child = AssemblyNode(
                node_id=child_id,
                depth=node.depth + 1,
                parent_id=node.node_id,
                action=action,
                sequence=child_sequence,
                work_state=child_state,
            )
            node.children.append(child_id)
            nodes[child_id] = child
            frontier.append(child)

    reset_work_state(worklist, state=root_state)
    return nodes


def sequence_labels(node: AssemblyNode) -> List[str]:
    return [action.label for action in node.sequence]


def leaf_nodes(nodes: Dict[str, AssemblyNode]) -> List[AssemblyNode]:
    return [node for node in nodes.values() if not node.children]


def _label_maps(worklist: WorkList):
    label_to_action = {}
    token_to_actions = {}
    work_to_actions = {}
    for work_idx, work in enumerate(worklist):
        actions = []
        for action_idx in range(len(work)):
            action = make_action(worklist, work_idx, action_idx)
            label_to_action[action.label] = action
            token_to_actions.setdefault(action.token, []).append(action)
            actions.append(action)
        work_to_actions[work.name] = actions
        work_to_actions[_work_alias(work.name)] = actions
    return label_to_action, token_to_actions, work_to_actions


def parse_sequence_string(worklist: WorkList,
                          sequence: str,
                          layout_name: str = 'home') -> List[AssemblyAction]:
    _, token_to_actions, work_to_actions = _label_maps(worklist)
    initial_counts = _initial_action_counts(worklist, layout_name)
    counts = list(initial_counts)
    skipped_precompleted = set()
    parsed = []
    for raw_token in sequence.split('-'):
        token = raw_token.strip().lower()
        if not token:
            continue
        action = None
        if token in token_to_actions:
            for candidate in token_to_actions[token]:
                if counts[candidate.work_idx] == candidate.action_idx:
                    action = candidate
                    break
        if action is None and token in work_to_actions:
            for candidate in work_to_actions[token]:
                if counts[candidate.work_idx] == candidate.action_idx:
                    action = candidate
                    break
                if initial_counts[candidate.work_idx] > candidate.action_idx:
                    skipped_precompleted.add(candidate.label)
                    action = candidate
                    break
        if action is None:
            raise ValueError(f'Unknown or out-of-order assembly token: {raw_token}')
        if action.label in skipped_precompleted:
            continue
        if parsed and parsed[-1].immediate:
            expected = _candidate_actions(worklist, parsed, initial_counts=initial_counts)
            if len(expected) != 1 or expected[0].label != action.label:
                raise ValueError(f'Immediate continuation required after {parsed[-1].label}')
        parsed.append(action)
        counts[action.work_idx] = action.action_idx + 1
    return parsed


def execute_sequence_string(worklist: WorkList,
                            sequence: str,
                            layout_name: str = 'home') -> ExecutionResult:
    actions = parse_sequence_string(worklist, sequence, layout_name=layout_name)
    reset_work_state(worklist, layout_name=layout_name)
    work_states = []
    for action in actions:
        _apply_symbolic_action(worklist, action)
        work_states.append(capture_work_state(worklist))
    return ExecutionResult(actions=actions, work_states=work_states)
