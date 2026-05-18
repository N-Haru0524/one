"""Build the SynchronizedPlan that brings the right arm to the prescrew pose
and lets the bridge enter policy phase for closed-loop correction.

Three flavours:

- `build_correction_approach_plan(...)` — minimal single-segment SyncPlan that
  publishes one right-arm trajectory ending at the prescrew joint configuration
  and flips `policy_after=True` on that segment. Use this when the upstream
  approach (pick screw, extend shank, position above the hole) has already
  been executed via a separate plan and you just want to hand off to the
  correction client.

- `screw_draft_to_sync_plan(...)` — convert a `PlannerActionDraft` produced by
  `ScrewPlanner.gen_screw_draft` into a `SynchronizedPlan`. Truncates segments
  after the prescrew phase (since the correction loop replaces them) and marks
  the final retained segment with policy_after=True (via plan_to_bridge_dict).

- `build_full_screw_plan_with_correction(...)` — thin wrapper around
  `ScrewPlanner.gen_screw_draft`, then runs the converter above to produce a
  ready-to-publish SyncPlan.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from dataclasses import dataclass

from one_assembly.assembly_data import (
    ArmSegment,
    DualRobotState,
    EEEvent,
    PlannerActionDraft,
    PlannerSegmentDraft,
    SyncPoint,
    SyncSegment,
    SynchronizedPlan,
)


# Default substring used to identify segments after which the bridge should
# enter policy phase. Matched against PlannerSegmentDraft.end_sync_label or
# segment_label (case-insensitive). ScrewPlanner.gen_screw_draft emits both a
# pickup-side prescrew and a place-side prescrew when configured for full
# screw operations, and BOTH need correction; the matcher returns indices for
# all matching segments, not just the first.
DEFAULT_PRESCREW_MATCHER_SUBSTRING = "prescrew"


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


@dataclass
class PrescrewPhase:
    """One correction handoff inside a multi-phase plan."""

    rgt_qs: np.ndarray
    label: str = "prescrew"
    rgt_ee_qs: Optional[np.ndarray] = None
    rgt_intermediate_qs: Optional[Sequence[np.ndarray]] = None


def build_multi_phase_correction_plan(
    *,
    initial_state: DualRobotState,
    phases: Sequence[PrescrewPhase],
) -> tuple[SynchronizedPlan, set[int]]:
    """Build a SyncPlan with one segment per phase, each ending at the phase's
    prescrew TCP joint configuration with policy_after=True.

    The bridge will execute segment 0, enter policy phase (phase 0 correction),
    on `done` move to segment 1, enter policy phase again (phase 1 correction),
    etc. Use this for the typical pick → place flow without needing a full
    ScrewPlanner setup.
    """
    if not phases:
        raise ValueError("build_multi_phase_correction_plan: phases must be non-empty.")

    sync_points = [SyncPoint(id="sp_home", label="home")]
    sync_segments: list[SyncSegment] = []
    prev_lft = _as_qs(initial_state.lft_qs)
    prev_rgt = _as_qs(initial_state.rgt_qs)

    for idx, phase in enumerate(phases):
        end_rgt = _as_qs(phase.rgt_qs)
        qs_list: list[np.ndarray] = [prev_rgt]
        if phase.rgt_intermediate_qs:
            for qs in phase.rgt_intermediate_qs:
                qs_list.append(_as_qs(qs))
        qs_list.append(end_rgt)
        ee_events: list[EEEvent] = []
        if phase.rgt_ee_qs is not None:
            ee_events.append(EEEvent(
                actor="right_driver",
                action="extend",
                timing="end",
                value=float(np.asarray(phase.rgt_ee_qs, dtype=np.float32).reshape(-1)[0]),
                label=f"shank position for {phase.label}",
            ))
        end_sync_id = f"sp_{idx}_{phase.label}"
        sync_segments.append(SyncSegment(
            id=f"seg_{idx}_{phase.label}",
            label=phase.label,
            start_sync_id=sync_points[-1].id,
            end_sync_id=end_sync_id,
            arm_segments=[
                ArmSegment(actor="left_arm", qs_list=[prev_lft, prev_lft], idle=True),
                ArmSegment(actor="right_arm", qs_list=qs_list),
            ],
            ee_events=ee_events,
        ))
        sync_points.append(SyncPoint(id=end_sync_id, label=phase.label))
        prev_rgt = end_rgt
        # prev_lft stays — left arm is idle in this lightweight plan

    plan = SynchronizedPlan(
        labels=[seg.label for seg in sync_segments],
        initial_state=initial_state.copy(),
        sync_points=sync_points,
        sync_segments=sync_segments,
    )
    policy_after_indices = set(range(len(sync_segments)))
    return plan, policy_after_indices


def _sanitize_token(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in (text or "").lower()).strip("_") or "x"


def _match_policy_after_indices(
    segments: Sequence[PlannerSegmentDraft],
    *,
    matcher_substring: Optional[str],
    matcher_predicate: Optional[Callable[[PlannerSegmentDraft, int], bool]],
) -> set[int]:
    """Return the set of segment indices to mark with policy_after=True."""
    if matcher_predicate is not None:
        return {idx for idx, seg in enumerate(segments) if matcher_predicate(seg, idx)}
    needle = (matcher_substring or "").lower()
    if not needle:
        return set()
    matches: set[int] = set()
    for idx, seg in enumerate(segments):
        end_label = (seg.end_sync_label or "").lower()
        seg_label = (seg.segment_label or "").lower()
        if needle in end_label or needle in seg_label:
            matches.add(idx)
    return matches


def screw_draft_to_sync_plan(
    draft: PlannerActionDraft,
    *,
    initial_state: DualRobotState,
    policy_after_substring: Optional[str] = DEFAULT_PRESCREW_MATCHER_SUBSTRING,
    policy_after_predicate: Optional[Callable[[PlannerSegmentDraft, int], bool]] = None,
    truncate_after_last_match: bool = False,
    label_prefix: str = "screw",
    plan_labels: Optional[Sequence[str]] = None,
) -> tuple[SynchronizedPlan, set[int]]:
    """Convert a ScrewPlanner draft into a SyncPlan with correction handoffs.

    All segments are retained by default — the bridge executes them in order,
    pausing into ``waiting_for_policy`` after each segment whose label matches
    the policy_after matcher. ScrewPlanner.gen_screw_draft yields both a
    pickup prescrew and a hole prescrew, so a single plan can carry two
    correction loops (pick + place) without re-publishing.

    Set ``truncate_after_last_match=True`` to drop trailing segments (the
    previous behaviour, kept for the case where the screwdrive/retract phase
    is handled by a separate plan).

    Returns ``(plan, policy_after_indices)`` for
    :func:`plan_to_bridge_dict_with_indices`.
    """
    if not isinstance(draft, PlannerActionDraft) or not draft.segments:
        raise ValueError("screw_draft_to_sync_plan: draft must have at least one segment.")

    policy_after_indices = _match_policy_after_indices(
        draft.segments,
        matcher_substring=policy_after_substring,
        matcher_predicate=policy_after_predicate,
    )
    if not policy_after_indices:
        raise ValueError(
            "screw_draft_to_sync_plan: no segment matched the policy_after matcher; "
            "the resulting plan would have no correction handoff."
        )
    if truncate_after_last_match:
        last_match = max(policy_after_indices)
        retained = list(draft.segments[: last_match + 1])
        # Recompute indices on the retained slice; same values since we kept the prefix.
        policy_after_indices = {i for i in policy_after_indices if i <= last_match}
    else:
        retained = list(draft.segments)

    sync_points: list[SyncPoint] = [SyncPoint(id="sp_start", label="start")]
    sync_segments: list[SyncSegment] = []
    prev_sync_id = "sp_start"
    for seg_idx, seg in enumerate(retained):
        end_label = seg.end_sync_label or seg.segment_label or f"seg_{seg_idx}"
        end_sync_id = f"sp_{seg_idx}_{_sanitize_token(end_label)}"
        sync_segments.append(
            SyncSegment(
                id=f"seg_{seg_idx}_{_sanitize_token(end_label)}",
                label=seg.segment_label or end_label,
                start_sync_id=prev_sync_id,
                end_sync_id=end_sync_id,
                arm_segments=[
                    ArmSegment(
                        actor="left_arm",
                        qs_list=list(seg.left_path or []),
                        idle=not (seg.left_path or []),
                    ),
                    ArmSegment(
                        actor="right_arm",
                        qs_list=list(seg.right_path or []),
                        idle=not (seg.right_path or []),
                    ),
                ],
                ee_events=list(seg.ee_events or []),
            )
        )
        sync_points.append(SyncPoint(id=end_sync_id, label=end_label))
        prev_sync_id = end_sync_id

    labels = list(plan_labels) if plan_labels is not None else [seg.label for seg in sync_segments]
    plan = SynchronizedPlan(
        labels=labels,
        initial_state=initial_state.copy(),
        sync_points=sync_points,
        sync_segments=sync_segments,
    )
    return plan, policy_after_indices


def plan_to_bridge_dict_with_indices(
    plan: SynchronizedPlan,
    policy_after_indices: set[int],
    *,
    plan_id: str = "screw_correction",
    waypoint_dt: float = 0.2,
) -> dict:
    """Like :func:`plan_to_bridge_dict` but with explicit policy_after indices.

    Used by :func:`screw_draft_to_sync_plan` since the cut may not always be
    the final segment.
    """
    return _synchronized_plan_to_dict(
        plan,
        policy_after_indices=set(policy_after_indices),
        plan_id=plan_id,
        waypoint_dt=waypoint_dt,
    )


def build_full_screw_plan_with_correction(
    screw_planner,
    *,
    work_name: str,
    initial_state: DualRobotState,
    start_qs: np.ndarray,
    goal_pose_list: Iterable,
    pick_pose=None,
    pick_pose_list=None,
    pick_approach_direction: np.ndarray | None = None,
    pick_approach_distance: float = 0.1,
    approach_direction: np.ndarray | None = None,
    approach_distance: float = 0.05,
    depart_direction: np.ndarray | None = None,
    depart_distance: float = 0.03,
    linear_granularity: float = 0.02,
    use_rrt: bool = True,
    pln_jnt: bool = False,
    resolution: int = 20,
    segment_label: Optional[str] = None,
    end_sync_label: Optional[str] = None,
    policy_after_substring: Optional[str] = DEFAULT_PRESCREW_MATCHER_SUBSTRING,
    policy_after_predicate: Optional[Callable[[PlannerSegmentDraft, int], bool]] = None,
    truncate_after_last_match: bool = False,
    toggle_dbg: bool = False,
) -> Optional[tuple[SynchronizedPlan, set[int]]]:
    """Run ScrewPlanner.gen_screw_draft and convert into a SyncPlan.

    Returns ``(plan, policy_after_indices)`` ready for
    :func:`plan_to_bridge_dict_with_indices`, or None if planning failed.
    """
    draft = screw_planner.gen_screw_draft(
        work_name=work_name,
        start_qs=start_qs,
        goal_pose_list=list(goal_pose_list),
        resolution=resolution,
        pick_pose_list=pick_pose_list,
        pick_pose=pick_pose,
        pick_approach_direction=pick_approach_direction,
        pick_approach_distance=pick_approach_distance,
        approach_direction=approach_direction,
        approach_distance=approach_distance,
        depart_direction=depart_direction,
        depart_distance=depart_distance,
        linear_granularity=linear_granularity,
        use_rrt=use_rrt,
        pln_jnt=pln_jnt,
        segment_label=segment_label,
        end_sync_label=end_sync_label,
        toggle_dbg=toggle_dbg,
    )
    if draft is None:
        return None
    return screw_draft_to_sync_plan(
        draft,
        initial_state=initial_state,
        policy_after_substring=policy_after_substring,
        policy_after_predicate=policy_after_predicate,
        truncate_after_last_match=truncate_after_last_match,
    )
