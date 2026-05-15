"""ROS2 I/O helpers for the screw-correction policy phase.

Wraps `one_assembly.ros2_bridge.BridgeClient` with:
- /left/joint_states and /right/joint_states subscriptions for IK seed
- status-topic monitoring to detect waiting_for_policy / executing transitions
- send_action / send_done convenience methods

Designed to be run in the same process as the correction loop. Calls to
spin_once() are made by `pump()` so the caller (e.g. a viewer schedule_interval
callback) controls cadence and the GUI event loop is not blocked.
"""
from __future__ import annotations

import json
from typing import Optional, Sequence

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from sensor_msgs.msg import JointState
    _HAS_RCLPY = True
except ImportError:  # allow import-only environments without ROS
    rclpy = None  # type: ignore
    Node = object  # type: ignore
    String = object  # type: ignore
    JointState = object  # type: ignore
    _HAS_RCLPY = False

# These mirror the defaults inside one_assembly.ros2_bridge. Duplicated here
# so this module can be imported without ROS sourced (one_assembly.ros2_bridge
# hard-imports rclpy at module load).
_DEFAULT_PLAN_TOPIC = "/one_planner_bridge/plan"
_DEFAULT_ACTION_TOPIC = "/one_planner_bridge/action"
_DEFAULT_STATUS_TOPIC = "/one_planner_bridge/status"


def _ensure_rclpy():
    if not _HAS_RCLPY:
        raise RuntimeError(
            "rclpy is not available in this environment. "
            "Source ROS2 before launching the correction client."
        )


DEFAULT_LEFT_JOINT_NAMES = [f"left_joint{i}" for i in range(1, 7)]
DEFAULT_RIGHT_JOINT_NAMES = [f"right_joint{i}" for i in range(1, 7)]


class CorrectionBridgeClient:
    """Combined plan publisher + action publisher + status & joint subscribers."""

    def __init__(
        self,
        node_name: str = "screw_correction_client",
        plan_topic: str = _DEFAULT_PLAN_TOPIC,
        action_topic: str = _DEFAULT_ACTION_TOPIC,
        status_topic: str = _DEFAULT_STATUS_TOPIC,
        left_joint_state_topic: str = "/left/joint_states",
        right_joint_state_topic: str = "/right/joint_states",
        left_joint_names: Sequence[str] = DEFAULT_LEFT_JOINT_NAMES,
        right_joint_names: Sequence[str] = DEFAULT_RIGHT_JOINT_NAMES,
    ) -> None:
        self._node_name = node_name
        self._plan_topic = plan_topic
        self._action_topic = action_topic
        self._status_topic = status_topic
        self._left_joint_state_topic = left_joint_state_topic
        self._right_joint_state_topic = right_joint_state_topic
        self._left_joint_names = list(left_joint_names)
        self._right_joint_names = list(right_joint_names)
        self._node: Optional[Node] = None
        self._plan_pub = None
        self._action_pub = None
        self._status_sub = None
        self._lft_sub = None
        self._rgt_sub = None
        self._owns_rclpy = False
        self.latest_status: Optional[dict] = None
        self.latest_lft_qs: Optional[np.ndarray] = None
        self.latest_rgt_qs: Optional[np.ndarray] = None
        self.latest_lft_stamp_ns: Optional[int] = None
        self.latest_rgt_stamp_ns: Optional[int] = None

    def __enter__(self) -> "CorrectionBridgeClient":
        _ensure_rclpy()
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        self._node = rclpy.create_node(self._node_name)
        self._plan_pub = self._node.create_publisher(String, self._plan_topic, 10)
        self._action_pub = self._node.create_publisher(String, self._action_topic, 10)
        self._status_sub = self._node.create_subscription(
            String, self._status_topic, self._on_status, 10
        )
        self._lft_sub = self._node.create_subscription(
            JointState, self._left_joint_state_topic, self._on_lft_joint, 10
        )
        self._rgt_sub = self._node.create_subscription(
            JointState, self._right_joint_state_topic, self._on_rgt_joint, 10
        )
        return self

    def __exit__(self, *exc):
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
            self._owns_rclpy = False

    def _on_status(self, msg) -> None:
        try:
            self.latest_status = json.loads(msg.data)
        except (ValueError, TypeError):
            self.latest_status = None

    def _reorder_qs(self, msg, expected_names: Sequence[str]) -> Optional[np.ndarray]:
        name_to_pos = dict(zip(msg.name, msg.position))
        try:
            return np.asarray(
                [float(name_to_pos[n]) for n in expected_names], dtype=np.float32
            )
        except KeyError:
            return None

    def _stamp_ns(self, msg) -> int:
        h = msg.header.stamp
        return int(h.sec) * 1_000_000_000 + int(h.nanosec)

    def _on_lft_joint(self, msg) -> None:
        qs = self._reorder_qs(msg, self._left_joint_names)
        if qs is not None:
            self.latest_lft_qs = qs
            self.latest_lft_stamp_ns = self._stamp_ns(msg)

    def _on_rgt_joint(self, msg) -> None:
        qs = self._reorder_qs(msg, self._right_joint_names)
        if qs is not None:
            self.latest_rgt_qs = qs
            self.latest_rgt_stamp_ns = self._stamp_ns(msg)

    def pump(self, timeout_sec: float = 0.0) -> None:
        """Spin once to drain pending callbacks. Call from your main loop / GUI tick."""
        if self._node is not None and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def send_plan(self, plan_dict: dict) -> None:
        if self._plan_pub is None:
            raise RuntimeError("CorrectionBridgeClient must be used as a context manager.")
        msg = String()
        msg.data = json.dumps(plan_dict, ensure_ascii=False)
        self._plan_pub.publish(msg)

    def send_action(
        self,
        *,
        side: str = "right",
        lft_qs: Optional[Sequence[float]] = None,
        rgt_qs: Optional[Sequence[float]] = None,
        lft_ee_qs: Optional[Sequence[float]] = None,
        rgt_ee_qs: Optional[Sequence[float]] = None,
        done: bool = False,
    ) -> None:
        if self._action_pub is None:
            raise RuntimeError("CorrectionBridgeClient must be used as a context manager.")
        payload: dict = {"side": side}
        if lft_qs is not None:
            payload["lft_qs"] = [float(x) for x in lft_qs]
        if rgt_qs is not None:
            payload["rgt_qs"] = [float(x) for x in rgt_qs]
        if lft_ee_qs is not None:
            payload["lft_ee_qs"] = [float(x) for x in lft_ee_qs]
        if rgt_ee_qs is not None:
            payload["rgt_ee_qs"] = [float(x) for x in rgt_ee_qs]
        if done:
            payload["done"] = True
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._action_pub.publish(msg)

    def send_done(self, side: str = "right") -> None:
        self.send_action(side=side, done=True)

    def wait_for_status(self, target: str, timeout: float | None = None, poll: float = 0.05) -> bool:
        import time
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            self.pump(poll)
            st = self.latest_status
            if isinstance(st, dict) and st.get("status") == target:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False

    def wait_for_joint_state(self, side: str = "right", timeout: float = 5.0, poll: float = 0.05) -> bool:
        import time
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            self.pump(poll)
            qs = self.latest_rgt_qs if side == "right" else self.latest_lft_qs
            if qs is not None:
                return True
        return False
