"""Visible end-to-end demo of the ScrewOperation bridge stack against the
live one_planner_bridge node.

Goal: let a human watch RViz while we exercise the full pipeline:

  current_qs
    --[seg0: pre-prescrew]--> nudge_A_qs
        --[seg1: prescrew (policy_after=True)]--> nudge_B_qs
            == enter policy phase ==
            send N small corrections via /one_planner_bridge/action
            send done
        --[seg2: return]--> current_qs
    == completed ==

The motion is built RELATIVE to whatever pose the right arm is currently
in (from /right/joint_states), so it works regardless of where you left
the arm. Each segment moves ONE joint by --magnitude radians (default
0.3 rad). The policy-phase corrections oscillate the same joint by
±--correction_magnitude (default 0.04 rad).

Usage (must be run inside the one_ros2 container with ROS sourced):

  docker exec -w /home/wrs/nagai/one one_ros2 \\
      bash -lc "source /opt/ros/humble/setup.bash && python3 nagai/test_screwoperation_rviz_demo.py"

  # Larger swing, slower playback, more corrections:
  ... python3 nagai/test_screwoperation_rviz_demo.py \\
      --magnitude 0.5 --waypoint_dt 2.0 \\
      --num_corrections 6 --correction_interval 1.0

  # Dry-run: build & validate the plan, don't publish
  ... python3 nagai/test_screwoperation_rviz_demo.py --dry_run

What to watch in RViz:
  - Right arm slowly moves to nudge_A, then to nudge_B (seg0 → seg1)
  - On reaching nudge_B, bridge logs 'Entered policy phase ...'
  - The arm wiggles ±correction_magnitude for ~num_corrections steps
  - Arm returns to the original pose (seg2)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# We bypass our package's session/approach_plan helpers here on purpose:
# this is a deliberately self-contained demo so a reader can map every
# byte of the plan to what RViz shows.

def _require_rclpy_env():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "rclpy not importable. Run this script INSIDE the one_ros2 container "
            "(or source /opt/ros/humble/setup.bash first).\n"
        )
        sys.exit(2)


def build_plan_dict(
    *,
    current_rgt: np.ndarray,
    current_lft: np.ndarray,
    nudge_joint: int,
    magnitude: float,
    waypoint_dt: float,
    plan_id: str,
) -> dict:
    """Build a 3-segment SyncPlan dict directly (no helpers).

    The plan's right-arm path is:
        current  -> A  -> B  -> (policy phase) -> current
    where A and B are the same joint nudged by +magnitude and +2*magnitude.
    Left arm holds. The middle segment has policy_after=True.
    """
    from one_assembly.assembly_data import (
        ArmSegment, DualRobotState, SyncPoint, SyncSegment, SynchronizedPlan,
    )
    from one_assembly.ScrewOperation.approach_plan import plan_to_bridge_dict_with_indices

    n = current_rgt.shape[0]
    if not (0 <= nudge_joint < n):
        raise ValueError(f"--nudge_joint must be in [0,{n - 1}], got {nudge_joint}")
    delta = np.zeros(n, dtype=np.float32)
    delta[nudge_joint] = float(magnitude)
    a = current_rgt + delta
    b = current_rgt + 2.0 * delta

    initial_state = DualRobotState(
        lft_qs=current_lft.astype(np.float32),
        lft_ee_qs=np.zeros(2, dtype=np.float32),
        rgt_qs=current_rgt.astype(np.float32),
        rgt_ee_qs=np.zeros(1, dtype=np.float32),
    )

    def _seg(idx: int, label: str, start_id: str, end_id: str, rgt_start, rgt_end) -> SyncSegment:
        # 3 waypoints (start, middle interp, end) so the bridge gets a smoother trajectory
        mid = 0.5 * (np.asarray(rgt_start) + np.asarray(rgt_end))
        return SyncSegment(
            id=f"seg{idx}_{label}",
            label=label,
            start_sync_id=start_id,
            end_sync_id=end_id,
            arm_segments=[
                ArmSegment(actor="left_arm", qs_list=[current_lft, current_lft], idle=True),
                ArmSegment(actor="right_arm", qs_list=[np.asarray(rgt_start, dtype=np.float32),
                                                       mid.astype(np.float32),
                                                       np.asarray(rgt_end, dtype=np.float32)]),
            ],
            ee_events=[],
        )

    plan = SynchronizedPlan(
        labels=["rviz_demo_seg_pre", "rviz_demo_prescrew", "rviz_demo_return"],
        initial_state=initial_state,
        sync_points=[
            SyncPoint(id="sp_home", label="start"),
            SyncPoint(id="sp_A",    label="nudge_A"),
            SyncPoint(id="sp_B",    label="prescrew_B"),
            SyncPoint(id="sp_home2", label="back_to_start"),
        ],
        sync_segments=[
            _seg(0, "rviz_demo_seg_pre",  "sp_home", "sp_A",    current_rgt, a),
            _seg(1, "rviz_demo_prescrew", "sp_A",   "sp_B",     a,           b),
            _seg(2, "rviz_demo_return",   "sp_B",   "sp_home2", b,           current_rgt),
        ],
    )
    # policy_after on seg 1 only (the "prescrew" segment)
    return plan_to_bridge_dict_with_indices(plan, {1}, plan_id=plan_id, waypoint_dt=waypoint_dt)


@dataclass
class StatusObserver:
    """Records every distinct status string emitted, with timestamps."""
    timeline: list

    def push(self, t: float, status: Optional[str]):
        if status is None:
            return
        if not self.timeline or self.timeline[-1][1] != status:
            self.timeline.append((round(t, 3), status))
            print(f"  t={self.timeline[-1][0]:>6.2f}s  status={status}")


def stream_corrections(
    bridge,
    *,
    nudge_joint: int,
    correction_magnitude: float,
    num_corrections: int,
    correction_interval: float,
):
    """During the policy phase, oscillate the chosen joint and send each
    resulting rgt_qs as an action. Final action carries done=True."""
    base_rgt = (
        bridge.latest_rgt_qs.copy()
        if bridge.latest_rgt_qs is not None
        else None
    )
    if base_rgt is None:
        print("  WARN: no /right/joint_states; sending done with no corrections")
        bridge.send_action(side="right", done=True)
        return

    print(f"  streaming {num_corrections} corrections on joint {nudge_joint}, "
          f"±{correction_magnitude} rad, {correction_interval}s apart")
    for k in range(num_corrections):
        sign = 1.0 if (k % 2 == 0) else -1.0
        offset = np.zeros_like(base_rgt)
        offset[nudge_joint] = sign * float(correction_magnitude)
        target = base_rgt + offset
        bridge.send_action(side="right", rgt_qs=target.tolist())
        print(f"    correction {k + 1}/{num_corrections}: joint{nudge_joint} "
              f"{('+' if sign > 0 else '-')}{correction_magnitude:.3f}")
        # pump + sleep so the status/joint_state callbacks have time to fire
        end = time.monotonic() + correction_interval
        while time.monotonic() < end:
            bridge.pump(0.05)

    print("  sending done")
    bridge.send_action(side="right", done=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nudge_joint", type=int, default=5,
                    help="Right-arm joint index (0..5) to nudge for visibility (default: 5 = wrist roll)")
    ap.add_argument("--magnitude", type=float, default=0.30,
                    help="Per-segment joint delta in radians (default: 0.30)")
    ap.add_argument("--waypoint_dt", type=float, default=1.5,
                    help="Bridge-default waypoint dt; bigger = slower playback (default: 1.5 s)")
    ap.add_argument("--num_corrections", type=int, default=5,
                    help="Number of action messages streamed during the policy phase (default: 5)")
    ap.add_argument("--correction_magnitude", type=float, default=0.04,
                    help="Joint amplitude of each correction in radians (default: 0.04)")
    ap.add_argument("--correction_interval", type=float, default=0.8,
                    help="Spacing between correction actions in seconds (default: 0.8)")
    ap.add_argument("--policy_timeout", type=float, default=60.0,
                    help="Max wait for the bridge to enter waiting_for_policy (default: 60 s)")
    ap.add_argument("--completion_timeout", type=float, default=60.0,
                    help="Max wait for the bridge to reach completed after done (default: 60 s)")
    ap.add_argument("--plan_id", type=str, default="rviz_demo")
    ap.add_argument("--dry_run", action="store_true",
                    help="Build & validate the plan locally; do NOT publish or talk to ROS")
    args = ap.parse_args()

    if args.dry_run:
        print("== DRY RUN ==")
        # Use synthetic 'current' values so no rclpy is needed
        current_rgt = np.zeros(6, dtype=np.float32)
        current_lft = np.zeros(6, dtype=np.float32)
        d = build_plan_dict(
            current_rgt=current_rgt, current_lft=current_lft,
            nudge_joint=args.nudge_joint, magnitude=args.magnitude,
            waypoint_dt=args.waypoint_dt, plan_id=args.plan_id,
        )
        print(f"plan_id      : {d['plan_id']}")
        print(f"labels       : {d['labels']}")
        print(f"segments     : {len(d['planned_segments'])}")
        for i, seg in enumerate(d['planned_segments']):
            n_wp = len(seg['state_list'])
            print(f"  seg[{i}] wpts={n_wp} policy_after={seg['policy_after']} "
                  f"label={d['labels'][i]!r}")
        print(f"plan JSON size: {len(json.dumps(d))} bytes")
        # Roundtrip through the bridge's parser if available
        try:
            sys.path.insert(0, '/home/wrs/nagai/one_ros2/ws/src/one_planner_bridge')
            from one_planner_bridge.plan_data import PlanConverter
            conv = PlanConverter(default_event_side='right')
            exec_plan = conv.convert_payload_to_plan(d, default_plan_id='fallback')
            print(f"PlanConverter OK: {len(exec_plan.segments)} segments parsed")
        except Exception as exc:
            print(f"(skipping PlanConverter roundtrip: {exc!s})")
        return

    _require_rclpy_env()
    from one_assembly.ScrewOperation.bridge_io import CorrectionBridgeClient

    with CorrectionBridgeClient(node_name="screwop_rviz_demo") as bridge:
        # Need real joint_states to build the plan
        print("waiting for joint_states...")
        if not bridge.wait_for_joint_state("right", timeout=5.0):
            print("ERROR: no /right/joint_states within 5 s")
            sys.exit(2)
        if not bridge.wait_for_joint_state("left", timeout=5.0):
            print("WARN: no /left/joint_states within 5 s; using zeros")

        current_rgt = bridge.latest_rgt_qs.copy()
        current_lft = (
            bridge.latest_lft_qs.copy()
            if bridge.latest_lft_qs is not None
            else np.zeros(6, dtype=np.float32)
        )
        print(f"current rgt: {np.round(current_rgt, 4).tolist()}")
        print(f"current lft: {np.round(current_lft, 4).tolist()}")

        plan_dict = build_plan_dict(
            current_rgt=current_rgt, current_lft=current_lft,
            nudge_joint=args.nudge_joint, magnitude=args.magnitude,
            waypoint_dt=args.waypoint_dt, plan_id=args.plan_id,
        )

        observer = StatusObserver(timeline=[])
        t_start = time.monotonic()

        print(f"\npublishing plan id={args.plan_id!r} "
              f"(joint{args.nudge_joint} ±{args.magnitude} rad, "
              f"waypoint_dt={args.waypoint_dt}s)\n")
        bridge.send_plan(plan_dict)

        # Phase 1: wait for the bridge to hit waiting_for_policy
        print("waiting for waiting_for_policy...")
        deadline = time.monotonic() + args.policy_timeout
        while time.monotonic() < deadline:
            bridge.pump(0.1)
            observer.push(time.monotonic() - t_start,
                          bridge.latest_status.get('status') if bridge.latest_status else None)
            st = bridge.latest_status.get('status') if bridge.latest_status else None
            if st == 'waiting_for_policy':
                break
            if st in ('completed', 'aborted', 'failed'):
                print(f"ERROR: bridge reached {st!r} before policy phase")
                sys.exit(3)
        else:
            print("ERROR: bridge never reached waiting_for_policy")
            sys.exit(4)

        # Phase 2: drive corrections
        print("\n== entering policy phase ==")
        stream_corrections(
            bridge,
            nudge_joint=args.nudge_joint,
            correction_magnitude=args.correction_magnitude,
            num_corrections=args.num_corrections,
            correction_interval=args.correction_interval,
        )

        # Phase 3: wait for completion
        print("\nwaiting for completed...")
        deadline = time.monotonic() + args.completion_timeout
        while time.monotonic() < deadline:
            bridge.pump(0.1)
            observer.push(time.monotonic() - t_start,
                          bridge.latest_status.get('status') if bridge.latest_status else None)
            st = bridge.latest_status.get('status') if bridge.latest_status else None
            if st in ('completed', 'aborted', 'failed'):
                break
        else:
            print("ERROR: bridge did not finish within timeout")
            sys.exit(5)

        print("\n== status timeline ==")
        for t, s in observer.timeline:
            print(f"  t={t:>6.2f}s  status={s}")


if __name__ == "__main__":
    main()
