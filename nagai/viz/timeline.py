"""Timeline data model: normalize any draft into a flat list of replayable frames.

Everything that gets played back reduces to:

    Frame    one render step = a DualRobotState plus the EE events firing on it
    Clip     a labelled run of Frames + the info needed to restore the scene
             before replaying it in isolation (initial_state / prefix_actions /
             setup_events)

The joint-level baking here (gripper jaw width / driver shank baked into the
state) is deliberately separate from the *scene* side-effects (mounting a held
part onto the gripper), which live in ee_control.EEController.
"""

from dataclasses import dataclass, field

import numpy as np

from one_assembly.assembly_data import (
    ArmSegment,
    DualRobotState,
    EEEvent,
    PlannerActionDraft,
    PlannerSegmentDraft,
    SyncSegment,
)


@dataclass
class Frame:
    """One playback step."""
    state: DualRobotState
    events: list[EEEvent] = field(default_factory=list)


@dataclass
class Clip:
    """One replayable unit (typically a single draft segment)."""
    label: str                              # segment label, e.g. 'grasp belt'
    group: str                              # owning action label, e.g. 'belt:time0'
    action_type: str                        # 'place' / 'fold' / 'screw'
    seg_index: int                          # index within the owning draft
    seg_count: int                          # total segments in the owning draft
    initial_state: DualRobotState           # joint state at the first frame
    prefix_actions: list[tuple[int, int]]   # symbolic actions applied before this draft
    setup_events: list[EEEvent]             # EE events fired before this clip (whole sequence)
    frames: list[Frame]


# ---------------------------------------------------------------------------
# Joint-level baking (state only; no scene mounting)
# ---------------------------------------------------------------------------

def apply_ee_event_to_state(state: DualRobotState,
                            event: EEEvent,
                            open_gripper_qs: np.ndarray) -> DualRobotState:
    """Bake an EE event's joint effect (jaw width / driver shank) into a state."""
    updated = state.copy()
    if event.actor == 'left_gripper':
        if event.action in ('open', 'release'):
            updated.lft_ee_qs[:] = np.asarray(open_gripper_qs, dtype=np.float32)
        elif event.action in ('close', 'attach') and event.value is not None:
            width = float(event.value) * 0.5
            updated.lft_ee_qs[:] = np.array([width, width], dtype=np.float32)
    elif event.actor == 'right_driver':
        if event.action in ('extend', 'retract') and event.value is not None:
            updated.rgt_ee_qs[:] = np.array([float(event.value)], dtype=np.float32)
    return updated


def _segment_arm_path(segment: SyncSegment, actor: str) -> list[np.ndarray]:
    for arm_segment in segment.arm_segments:
        if arm_segment.actor == actor:
            return arm_segment.qs_list
    return []


def expand_segment_states(start_state: DualRobotState,
                          segment: SyncSegment,
                          open_gripper_qs: np.ndarray) -> list[DualRobotState]:
    """Expand a SyncSegment into one DualRobotState per sample, EE qs baked in."""
    left_path = _segment_arm_path(segment, 'left_arm')
    right_path = _segment_arm_path(segment, 'right_arm')
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
            states[idx] = apply_ee_event_to_state(states[idx], event, open_gripper_qs)
    return states


def build_sync_segment_from_draft(segment_id: str,
                                  start_sync_id: str,
                                  end_sync_id: str,
                                  draft: PlannerSegmentDraft) -> SyncSegment:
    left_path = draft.left_path or []
    right_path = draft.right_path or []
    return SyncSegment(
        id=segment_id,
        label=draft.segment_label,
        start_sync_id=start_sync_id,
        end_sync_id=end_sync_id,
        arm_segments=[
            ArmSegment(actor='left_arm', qs_list=left_path, idle=len(left_path) == 0),
            ArmSegment(actor='right_arm', qs_list=right_path, idle=len(right_path) == 0),
        ],
        ee_events=list(draft.ee_events or []),
    )


def build_clip_frames(initial_state: DualRobotState,
                      sync_segments: list[SyncSegment],
                      open_gripper_qs: np.ndarray) -> list[Frame]:
    """Flatten segments into per-frame (joint state + EE events to fire that frame)."""
    frames: list[Frame] = []
    state = initial_state.copy()
    for segment in sync_segments:
        states = expand_segment_states(state, segment, open_gripper_qs)
        n = len(states)
        for idx, sampled in enumerate(states):
            evs = []
            for event in segment.ee_events:
                if event.timing == 'start' and idx == 0:
                    evs.append(event)
                elif event.timing == 'sample' and event.sample_index == idx:
                    evs.append(event)
                elif event.timing == 'end' and idx == n - 1:
                    evs.append(event)
            frames.append(Frame(state=sampled, events=evs))
        state = states[-1].copy()
    return frames


def normalize_action_draft(draft) -> PlannerActionDraft:
    if isinstance(draft, PlannerActionDraft):
        return draft
    if isinstance(draft, PlannerSegmentDraft):
        return PlannerActionDraft(segments=[draft], held_after=draft.held_after)
    raise TypeError(f'Unsupported planner draft type: {type(draft)!r}')
