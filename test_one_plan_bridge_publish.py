#!/usr/bin/env python3

import argparse

import numpy as np
import rclpy

from one_assembly.assembly_data import ArmSegment, DualRobotState, SyncPoint, SyncSegment, SynchronizedPlan
from one_assembly.ros2_bridge import OnePlanPublisher


def build_demo_plan() -> SynchronizedPlan:
    state0 = DualRobotState(
        lft_qs=np.zeros(6, dtype=float),
        lft_ee_qs=np.array([0.04, 0.04], dtype=float),
        rgt_qs=np.zeros(6, dtype=float),
        rgt_ee_qs=np.array([0.0], dtype=float),
    )
    state1 = DualRobotState(
        lft_qs=np.zeros(6, dtype=float),
        lft_ee_qs=np.array([0.04, 0.04], dtype=float),
        rgt_qs=np.zeros(6, dtype=float),
        rgt_ee_qs=np.array([0.0], dtype=float),
    )
    return SynchronizedPlan(
        labels=['demo_segment'],
        initial_state=state0,
        sync_points=[
            SyncPoint(id='sp0', label='home'),
            SyncPoint(id='sp1', label='done'),
        ],
        sync_segments=[
            SyncSegment(
                id='seg0',
                label='demo_segment',
                start_sync_id='sp0',
                end_sync_id='sp1',
                arm_segments=[
                    ArmSegment(actor='left_arm', qs_list=[state0.lft_qs, state1.lft_qs]),
                    ArmSegment(actor='right_arm', qs_list=[state0.rgt_qs, state1.rgt_qs]),
                ],
                ee_events=[],
            )
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description='Publish one planned_segments payload to one_planner_bridge')
    parser.add_argument('--plan-id', default='one-demo-plan')
    parser.add_argument('--wait-status', default='')
    parser.add_argument('--timeout', type=float, default=5.0)
    parser.add_argument('--no-auto-start', action='store_true')
    args = parser.parse_args()

    rclpy.init()
    publisher = OnePlanPublisher()
    try:
        publisher.publish_synchronized_plan(
            build_demo_plan(),
            plan_id=args.plan_id,
            auto_start=not args.no_auto_start,
        )
        if args.wait_status:
            ok = publisher.wait_for_status(args.wait_status, timeout_sec=args.timeout)
            print(f'wait_status={args.wait_status} ok={ok} latest={publisher.latest_status()}')
        else:
            print(f'published plan_id={args.plan_id}')
    finally:
        publisher.close()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
