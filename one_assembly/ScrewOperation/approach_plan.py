"""Build the SynchronizedPlan that brings the right arm to the prescrew pose
and lets the bridge enter policy phase for closed-loop correction.

Two flavours:

- `build_correction_approach_plan(...)` — minimal single-segment SyncPlan that
  publishes one right-arm trajectory ending at the prescrew joint configuration
  and flips `policy_after=True` on that segment. Use this when the upstream
  approach (pick screw, extend shank, position above the hole) has already
  been executed via a separate plan and you just want to hand off to the
  correction client.

- `build_full_screw_plan_with_correction(...)` — placeholder that delegates to
  `ScrewPlanner.gen_screw` and wraps the resulting waypoint list as a SyncPlan.
  The actual ScrewPlanner integration depends on the scene / collider /
  screwdriver setup the caller owns; left here as a documented call site.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from one_assembly.assembly_data import (
    ArmSegment,
    DualRobotState,
    EEEvent,
    SyncPoint,
    SyncSegment,
    SynchronizedPlan,
)


def _synchronized_plan_to_dict(*args, **kwargs):
    # Lazy: `one_assembly.ros2_bridge` imports rclpy at module-load. Defer it.
    from one_assembly.ros2_bridge import synchronized_plan_to_dict as _impl
    return _impl(*args, **kwargs)


def _as_qs(qs) -> np.ndarray:
    arr = np.asarray(qs, dtype=np.float32).reshape(-1)
    return arr


def build_correction_approach_plan(
    *,
    initial_state: DualRobotState,
    prescrew_rgt_qs: np.ndarray,
    rgt_intermediate_qs: Optional[Sequence[np.ndarray]] = None,
    rgt_ee_qs_at_start: Optional[np.ndarray] = None,
    rgt_ee_qs_at_end: Optional[np.ndarray] = None,
    label: str = "approach_prescrew",
) -> SynchronizedPlan:
    """Minimal one-segment SyncPlan ending at prescrew_rgt_qs with policy_after.

    The bridge will execute this single right-arm trajectory then enter
    `waiting_for_policy`. The correction client (`screw_correction_run`) then
    feeds incremental rgt_qs via /one_planner_bridge/action.
    """
    start_rgt = _as_qs(initial_state.rgt_qs)
    end_rgt = _as_qs(prescrew_rgt_qs)
    qs_list: list[np.ndarray] = [start_rgt]
    if rgt_intermediate_qs:
        for qs in rgt_intermediate_qs:
            qs_list.append(_as_qs(qs))
    qs_list.append(end_rgt)

    lft_idle_path = [_as_qs(initial_state.lft_qs), _as_qs(initial_state.lft_qs)]

    ee_events: list[EEEvent] = []
    if rgt_ee_qs_at_start is not None:
        ee_events.append(
            EEEvent(
                actor="right_driver",
                action="extend",
                timing="start",
                value=float(np.asarray(rgt_ee_qs_at_start, dtype=np.float32).reshape(-1)[0]),
                label="extend shank before approach",
            )
        )
    if rgt_ee_qs_at_end is not None:
        ee_events.append(
            EEEvent(
                actor="right_driver",
                action="extend",
                timing="end",
                value=float(np.asarray(rgt_ee_qs_at_end, dtype=np.float32).reshape(-1)[0]),
                label="hold shank at prescrew",
            )
        )

    segment = SyncSegment(
        id="seg_correction_approach",
        label=label,
        start_sync_id="sp_home",
        end_sync_id="sp_prescrew",
        arm_segments=[
            ArmSegment(actor="left_arm", qs_list=lft_idle_path, idle=True),
            ArmSegment(actor="right_arm", qs_list=qs_list),
        ],
        ee_events=ee_events,
    )

    return SynchronizedPlan(
        labels=[label],
        initial_state=initial_state.copy(),
        sync_points=[
            SyncPoint(id="sp_home", label="home"),
            SyncPoint(id="sp_prescrew", label="prescrew"),
        ],
        sync_segments=[segment],
    )


def plan_to_bridge_dict(
    plan: SynchronizedPlan,
    *,
    plan_id: str = "screw_correction",
    waypoint_dt: float = 0.2,
) -> dict:
    """Serialize a SyncPlan with policy_after=True on the LAST segment."""
    policy_after_indices = {len(plan.sync_segments) - 1} if plan.sync_segments else set()
    return _synchronized_plan_to_dict(
        plan,
        policy_after_indices=policy_after_indices,
        plan_id=plan_id,
        waypoint_dt=waypoint_dt,
    )


def build_full_screw_plan_with_correction(
    screw_planner,
    *,
    start_qs: np.ndarray,
    goal_pose_list: Iterable,
    pick_pose,
    pick_approach_direction: np.ndarray,
    pick_approach_distance: float,
    approach_direction: np.ndarray,
    approach_distance: float,
    depart_direction: np.ndarray,
    depart_distance: float,
    linear_granularity: float = 0.01,
    pln_jnt: bool = True,
    use_rrt: bool = True,
    initial_state: Optional[DualRobotState] = None,
    label_prefix: str = "screw",
):
    """Wrap ScrewPlanner.gen_screw output. The returned object is the raw plan
    from ScrewPlanner; callers are expected to convert it into a SyncPlan with
    one segment per approach phase and mark the final approach segment with
    policy_after=True when serialising via plan_to_bridge_dict.

    Note: the assembly/dual-arm layer that maps a single-arm ScrewPlanner output
    to a SyncPlan lives in the user's HierarchicalPlannerBase subclass.
    This helper is intentionally thin so it slots into that pipeline without
    forcing a particular shape on it.
    """
    plan = screw_planner.gen_screw(
        start_qs=start_qs,
        goal_pose_list=list(goal_pose_list),
        pick_pose=pick_pose,
        pick_approach_direction=pick_approach_direction,
        pick_approach_distance=pick_approach_distance,
        approach_direction=approach_direction,
        approach_distance=approach_distance,
        depart_direction=depart_direction,
        depart_distance=depart_distance,
        linear_granularity=linear_granularity,
        pln_jnt=pln_jnt,
        use_rrt=use_rrt,
        toggle_dbg=False,
    )
    return plan
