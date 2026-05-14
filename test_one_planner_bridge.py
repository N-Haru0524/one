#!/usr/bin/env python3

import argparse
import math
import sys
import time
from typing import List, Optional

import rclpy
from action_msgs.msg import GoalStatusArray
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory


class PlannerBridgeTester(Node):
    def __init__(self, bridge_node_name: str, joint_names: List[str]) -> None:
        super().__init__('one_planner_bridge_tester')

        self._bridge_node_name = bridge_node_name
        self._joint_names = joint_names
        self._latest_joint_state: Optional[JointState] = None
        self._latest_trajectory: Optional[JointTrajectory] = None
        self._latest_move_action_status: Optional[GoalStatusArray] = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        action_status_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(JointState, '/joint_states', self._on_joint_state, qos)
        self.create_subscription(
            JointTrajectory,
            '/khi_controller/joint_trajectory',
            self._on_joint_trajectory,
            qos,
        )
        self.create_subscription(
            GoalStatusArray,
            '/move_action/_action/status',
            self._on_move_action_status,
            action_status_qos,
        )

        self._joint_goal_pub = self.create_publisher(Float64MultiArray, '/one_planner_bridge/joint_goal', qos)
        param_service = '/%s/set_parameters' % self._bridge_node_name.lstrip('/')
        self._set_params_client = self.create_client(SetParameters, param_service)

    def _on_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def _on_joint_trajectory(self, msg: JointTrajectory) -> None:
        self._latest_trajectory = msg

    def _on_move_action_status(self, msg: GoalStatusArray) -> None:
        self._latest_move_action_status = msg

    def spin_until(self, predicate, timeout_sec: float, tick_sec: float = 0.05) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=tick_sec)
            if predicate():
                return True
        return False

    def ensure_bridge_node_exists(self, timeout_sec: float) -> bool:
        def _exists() -> bool:
            target = self._bridge_node_name.lstrip('/')
            return target in self.get_node_names()

        return self.spin_until(_exists, timeout_sec)

    def ensure_joint_goal_subscriber(self, timeout_sec: float) -> bool:
        return self.spin_until(lambda: self._joint_goal_pub.get_subscription_count() > 0, timeout_sec)

    def ensure_joint_state(self, timeout_sec: float) -> bool:
        return self.spin_until(lambda: self._latest_joint_state is not None, timeout_sec)

    def set_use_moveit(self, enabled: bool, timeout_sec: float) -> bool:
        if not self._set_params_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error('parameter service for %s is not available.' % self._bridge_node_name)
            return False

        param = ParameterMsg(
            name='use_moveit',
            value=ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=enabled),
        )
        request = SetParameters.Request()
        request.parameters = [param]
        future = self._set_params_client.call_async(request)

        if not self.spin_until(lambda: future.done(), timeout_sec):
            self.get_logger().error('set_parameters timeout for use_moveit=%s' % enabled)
            return False

        result = future.result()
        if result is None or len(result.results) != 1:
            self.get_logger().error('unexpected set_parameters response: %s' % result)
            return False

        ok = bool(result.results[0].successful)
        if not ok:
            self.get_logger().error(
                'failed to set use_moveit=%s: %s' % (enabled, result.results[0].reason)
            )
            return False

        self.get_logger().info('set use_moveit=%s' % enabled)
        return True

    def publish_joint_goal(self, target: List[float]) -> None:
        msg = Float64MultiArray()
        msg.data = target
        self._joint_goal_pub.publish(msg)

    def wait_trajectory(self, timeout_sec: float) -> Optional[JointTrajectory]:
        self._latest_trajectory = None
        ok = self.spin_until(lambda: self._latest_trajectory is not None, timeout_sec)
        return self._latest_trajectory if ok else None

    def wait_move_action_status(self, timeout_sec: float) -> Optional[GoalStatusArray]:
        self._latest_move_action_status = None
        ok = self.spin_until(lambda: self._latest_move_action_status is not None, timeout_sec)
        return self._latest_move_action_status if ok else None

    def has_move_action_server(self) -> bool:
        action_names_types = self.get_action_names_and_types()
        for name, _types in action_names_types:
            if name == '/move_action':
                return True
        return False


def almost_equal_list(a: List[float], b: List[float], eps: float = 1e-6) -> bool:
    if len(a) != len(b):
        return False
    return all(math.isclose(x, y, abs_tol=eps) for x, y in zip(a, b))


def main() -> int:
    parser = argparse.ArgumentParser(description='one_planner_bridge integration tester')
    parser.add_argument('--bridge-node', default='one_planner_bridge')
    parser.add_argument('--timeout', type=float, default=5.0)
    parser.add_argument('--joint-goal', nargs=6, type=float, default=[0.1, -0.1, 0.2, 0.0, 0.0, 0.0])
    parser.add_argument('--skip-moveit', action='store_true')
    parser.add_argument(
        '--joint-names',
        nargs=6,
        default=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
    )
    args = parser.parse_args()

    rclpy.init()
    tester = PlannerBridgeTester(args.bridge_node, args.joint_names)

    failures = []

    try:
        print('[TEST] wait bridge node ...')
        if not tester.ensure_bridge_node_exists(args.timeout):
            failures.append('bridge node not found: %s' % args.bridge_node)

        print('[TEST] wait /joint_states ...')
        if not tester.ensure_joint_state(args.timeout):
            failures.append('no /joint_states received')

        print('[TEST] direct mode publish check ...')
        if not tester.ensure_joint_goal_subscriber(args.timeout):
            failures.append('no subscriber on /one_planner_bridge/joint_goal')

        if tester.set_use_moveit(False, args.timeout):
            tester.publish_joint_goal(args.joint_goal)
            traj = tester.wait_trajectory(args.timeout)
            if traj is None:
                failures.append('no trajectory published to /khi_controller/joint_trajectory')
            else:
                got_names = list(traj.joint_names)
                got_pos = list(traj.points[0].positions) if traj.points else []
                if got_names != args.joint_names:
                    failures.append('joint_names mismatch: expected=%s got=%s' % (args.joint_names, got_names))
                if not almost_equal_list(got_pos, args.joint_goal):
                    failures.append('joint positions mismatch: expected=%s got=%s' % (args.joint_goal, got_pos))
        else:
            failures.append('cannot set use_moveit=false')

        if not args.skip_moveit:
            print('[TEST] moveit mode goal send check ...')
            if not tester.has_move_action_server():
                failures.append('/move_action action server not found')
            elif tester.set_use_moveit(True, args.timeout):
                tester.publish_joint_goal(args.joint_goal)
                status = tester.wait_move_action_status(args.timeout)
                if status is None:
                    failures.append('no status received on /move_action/_action/status after goal publish')
            else:
                failures.append('cannot set use_moveit=true')

    finally:
        tester.destroy_node()
        rclpy.shutdown()

    if failures:
        print('\n[RESULT] FAILED')
        for item in failures:
            print(' - %s' % item)
        return 1

    print('\n[RESULT] PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
