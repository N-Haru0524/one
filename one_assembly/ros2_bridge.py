"""ROS 2 bridge dispatcher for the one planner. Requires a sourced ROS 2 (rclpy) environment; not installed in one's local venv."""

from __future__ import annotations

import json
import time
from typing import Any, Iterable, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from one_assembly.assembly_data import (
    DualRobotState,
    EEEvent,
    SyncSegment,
    SynchronizedPlan,
)


_DEFAULT_PLAN_TOPIC = '/one_planner_bridge/plan'
_DEFAULT_ACTION_TOPIC = '/one_planner_bridge/action'
_DEFAULT_STATUS_TOPIC = '/one_planner_bridge/status'


class BridgeClient:
    def __init__(
        self,
        node_name: str = 'one_bridge_client',
        plan_topic: str = _DEFAULT_PLAN_TOPIC,
        action_topic: str = _DEFAULT_ACTION_TOPIC,
        status_topic: str = _DEFAULT_STATUS_TOPIC,
    ) -> None:
        self._node_name = node_name
        self._plan_topic = plan_topic
        self._action_topic = action_topic
        self._status_topic = status_topic
        self._node: Optional[Node] = None
        self._plan_pub = None
        self._action_pub = None
        self._status_sub = None
        self._owns_rclpy = False
        self.latest_status: Optional[dict] = None

    def __enter__(self) -> 'BridgeClient':
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        self._node = rclpy.create_node(self._node_name)
        self._plan_pub = self._node.create_publisher(String, self._plan_topic, 10)
        self._action_pub = self._node.create_publisher(String, self._action_topic, 10)
        self._status_sub = self._node.create_subscription(
            String, self._status_topic, self._on_status, 10
        )
        return self

    def __exit__(self, *exc) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
            self._owns_rclpy = False

    def _on_status(self, msg: String) -> None:
        try:
            self.latest_status = json.loads(msg.data)
        except (ValueError, TypeError):
            self.latest_status = None

    def send_plan(self, plan_dict: dict) -> None:
        if self._plan_pub is None:
            raise RuntimeError('BridgeClient must be used as a context manager.')
        msg = String()
        msg.data = json.dumps(plan_dict, ensure_ascii=False)
        self._plan_pub.publish(msg)

    def send_action(self, action_dict: dict) -> None:
        if self._action_pub is None:
            raise RuntimeError('BridgeClient must be used as a context manager.')
        msg = String()
        msg.data = json.dumps(action_dict, ensure_ascii=False)
        self._action_pub.publish(msg)

    def stream_actions(self, action_iter: Iterable[dict], rate_hz: float = 30.0) -> None:
        period = 1.0 / float(rate_hz) if rate_hz > 0 else 0.0
        for action in action_iter:
            self.send_action(action)
            if self._node is not None:
                rclpy.spin_once(self._node, timeout_sec=0.0)
            if period > 0.0:
                time.sleep(period)

    def wait_for_status(self, target: str, timeout: Optional[float] = None) -> bool:
        if self._node is None:
            raise RuntimeError('BridgeClient must be used as a context manager.')
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            rclpy.spin_once(self._node, timeout_sec=0.05)
            status = self.latest_status
            if isinstance(status, dict) and status.get('status') == target:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False


def _qs_to_list(qs) -> Optional[list]:
    if qs is None:
        return None
    arr = np.asarray(qs)
    return [float(x) for x in arr.reshape(-1)]


def _initial_state_dict(state: DualRobotState) -> dict:
    return {
        'lft_qs': _qs_to_list(state.lft_qs),
        'lft_ee_qs': _qs_to_list(state.lft_ee_qs),
        'rgt_qs': _qs_to_list(state.rgt_qs),
        'rgt_ee_qs': _qs_to_list(state.rgt_ee_qs),
    }


def _segment_arm_path(segment: SyncSegment, actor: str) -> list:
    for arm in segment.arm_segments:
        if arm.actor == actor:
            return list(arm.qs_list)
    return []


def _apply_ee_event(state: DualRobotState, event: EEEvent) -> DualRobotState:
    updated = state.copy()
    if event.actor == 'left_gripper':
        if event.action in ('open', 'release'):
            current_width = float(updated.lft_ee_qs.reshape(-1)[0]) * 2.0
            half = current_width * 0.5 if current_width > 0.0 else 0.02
            updated.lft_ee_qs = np.array([half, half], dtype=np.float32)
        elif event.action in ('close', 'attach') and event.value is not None:
            half = float(event.value) * 0.5
            updated.lft_ee_qs = np.array([half, half], dtype=np.float32)
    elif event.actor == 'right_driver':
        if event.action in ('extend', 'retract') and event.value is not None:
            updated.rgt_ee_qs = np.array([float(event.value)], dtype=np.float32)
    return updated


def _expand_segment_states(start: DualRobotState, segment: SyncSegment) -> list[DualRobotState]:
    left_path = _segment_arm_path(segment, 'left_arm')
    right_path = _segment_arm_path(segment, 'right_arm')
    sample_count = max(len(left_path), len(right_path), 1)
    states: list[DualRobotState] = []
    for idx in range(sample_count):
        state = start.copy() if idx == 0 else states[-1].copy()
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
            states[idx] = _apply_ee_event(states[idx], event)
    return states


def _ee_event_to_array(event: EEEvent) -> Optional[list]:
    if event.actor == 'left_gripper':
        if event.action in ('close', 'attach'):
            tf = None
            if event.engage_tf is not None:
                tf = np.asarray(event.engage_tf).reshape(-1).tolist()
            return [
                'attach',
                event.work_name,
                int(event.grasp_id) if event.grasp_id is not None else None,
                float(event.value) if event.value is not None else None,
                tf,
            ]
        if event.action in ('open', 'release'):
            return ['release', event.work_name]
    return None


def _ee_event_to_dict(event: EEEvent) -> dict:
    payload: dict[str, Any] = {
        'kind': event.action,
        'actor': event.actor,
        'side': 'left' if event.actor == 'left_gripper' else 'right',
        'timing': event.timing,
    }
    if event.sample_index is not None:
        payload['sample_index'] = int(event.sample_index)
    if event.value is not None:
        payload['value'] = float(event.value)
    if event.work_name is not None:
        payload['work_name'] = event.work_name
    if event.grasp_id is not None:
        payload['grasp_id'] = int(event.grasp_id)
    if event.engage_tf is not None:
        payload['engage_tf'] = np.asarray(event.engage_tf).reshape(-1).tolist()
    if event.label is not None:
        payload['label'] = event.label
    return payload


def _event_waypoint_index(event: EEEvent, sample_count: int) -> int:
    if event.timing == 'sample' and event.sample_index is not None:
        return max(0, min(int(event.sample_index), sample_count - 1))
    if event.timing == 'end':
        return max(sample_count - 1, 0)
    return 0


def _state_to_waypoint(state: DualRobotState, time_from_start: Optional[float]) -> dict:
    entry: dict[str, Any] = {
        'lft_qs': _qs_to_list(state.lft_qs),
        'lft_ee_qs': _qs_to_list(state.lft_ee_qs),
        'rgt_qs': _qs_to_list(state.rgt_qs),
        'rgt_ee_qs': _qs_to_list(state.rgt_ee_qs),
    }
    if time_from_start is not None:
        entry['time_from_start'] = float(time_from_start)
    return entry


def synchronized_plan_to_dict(
    plan: SynchronizedPlan,
    policy_after_indices: Optional[set[int]] = None,
    plan_id: str = 'sync_plan',
    waypoint_dt: Optional[float] = None,
) -> dict:
    if plan.initial_state is None:
        raise ValueError('SynchronizedPlan.initial_state is required to build a plan dict.')
    policy_after_indices = policy_after_indices or set()

    state = plan.initial_state.copy()
    planned_segments: list[dict] = []
    for seg_idx, segment in enumerate(plan.sync_segments):
        states = _expand_segment_states(state, segment)
        sample_count = len(states)

        state_list: list[dict] = []
        for sample_idx, sample_state in enumerate(states):
            t = float(sample_idx * waypoint_dt) if waypoint_dt is not None else None
            state_list.append(_state_to_waypoint(sample_state, t))

        event_map: dict[str, Any] = {}
        for event in segment.ee_events:
            wp_idx = _event_waypoint_index(event, sample_count)
            array_form = _ee_event_to_array(event)
            event_map[str(wp_idx)] = array_form if array_form is not None else _ee_event_to_dict(event)

        planned_segments.append({
            'state_list': state_list,
            'event_map': event_map,
            'policy_after': seg_idx in policy_after_indices,
        })
        state = states[-1].copy()

    return {
        'plan_id': plan_id,
        'labels': [seg.label for seg in plan.sync_segments],
        'initial_state': _initial_state_dict(plan.initial_state),
        'planned_segments': planned_segments,
    }
