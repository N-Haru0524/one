from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np

"""
Data containers for synchronized dual-arm execution.

Synchronized execution model:
    sync_points + sync_segments
    - Suited for MoveIt-style segment planning with stop-and-sync barriers.
    - Each SyncSegment spans exactly one synchronization interval.
    - Left and right arms may move simultaneously inside the same SyncSegment.
    - End-effector actions are attached to the segment at its start, end, or a sample index.

    Example:
        sync_points = [
            SyncPoint(id='sp_home', label='both home'),
            SyncPoint(id='sp_relay_pregrasp', label='relay pre-grasp'),
            SyncPoint(id='sp_relay_grasped', label='relay grasped'),
            SyncPoint(id='sp_relay_pre_screw', label='relay placed, screwdriver aligned'),
        ]

        sync_segments = [
            SyncSegment(
                id='seg_home_to_pregrasp',
                label='left reaches relay while right waits',
                start_sync_id='sp_home',
                end_sync_id='sp_relay_pregrasp',
                arm_segments=[
                    ArmSegment(
                        actor='left_arm',
                        qs_list=[np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32)],
                    ),
                    ArmSegment(
                        actor='right_arm',
                        qs_list=[],
                        idle=True,
                    ),
                ],
            ),
            SyncSegment(
                id='seg_pregrasp_to_grasped',
                label='left closes gripper at grasp pose',
                start_sync_id='sp_relay_pregrasp',
                end_sync_id='sp_relay_grasped',
                arm_segments=[
                    ArmSegment(actor='left_arm', qs_list=[np.ones(6, dtype=np.float32)]),
                    ArmSegment(actor='right_arm', qs_list=[], idle=True),
                ],
                ee_events=[
                    EEEvent(
                        actor='left_gripper',
                        action='close',
                        timing='end',
                        value=0.032,
                        work_name='relay',
                        grasp_id=4,
                    ),
                ],
            ),
            SyncSegment(
                id='seg_grasped_to_pre_screw',
                label='left transports relay while right approaches with screwdriver',
                start_sync_id='sp_relay_grasped',
                end_sync_id='sp_relay_pre_screw',
                arm_segments=[
                    ArmSegment(actor='left_arm', qs_list=[np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)]),
                    ArmSegment(actor='right_arm', qs_list=[np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)]),
                ],
                ee_events=[
                    EEEvent(actor='right_driver', action='extend', timing='end', value=0.03),
                ],
            ),
        ]
"""

ArmActor: TypeAlias = Literal['left_arm', 'right_arm']
EEActor: TypeAlias = Literal['left_gripper', 'right_driver']
EEAction: TypeAlias = Literal[
    'open',
    'close',
    'attach',
    'release',
    'extend',
    'retract',
    'screw_start',
    'screw_stop',
]
EventTiming: TypeAlias = Literal['start', 'end', 'sample']


@dataclass
class DualRobotState:
    lft_qs: np.ndarray  # left arm joint vector, e.g. np.zeros(6, dtype=np.float32)
    lft_ee_qs: np.ndarray  # left gripper jaw state, e.g. np.array([0.02, 0.02], dtype=np.float32)
    rgt_qs: np.ndarray  # right arm joint vector, e.g. np.zeros(6, dtype=np.float32)
    rgt_ee_qs: np.ndarray  # right screwdriver state, e.g. np.array([0.03], dtype=np.float32)

    def copy(self):
        return DualRobotState(
            lft_qs=self.lft_qs.copy(),
            lft_ee_qs=self.lft_ee_qs.copy(),
            rgt_qs=self.rgt_qs.copy(),
            rgt_ee_qs=self.rgt_ee_qs.copy(),
        )


@dataclass(frozen=True)
class SyncPoint:
    id: str  # e.g. 'sp_relay_pregrasp'
    label: str  # e.g. 'relay pre-grasp'


@dataclass
class ArmSegment:
    actor: ArmActor  # e.g. 'left_arm'
    qs_list: list[np.ndarray]  # e.g. [np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32)]
    idle: bool = False  # e.g. True when the opposite arm moves and this arm waits at the barrier


@dataclass(frozen=True)
class EEEvent:
    actor: EEActor  # e.g. 'left_gripper' or 'right_driver'
    action: EEAction  # e.g. 'close', 'attach', 'extend', 'screw_start'
    timing: EventTiming  # e.g. 'start', 'end', or 'sample'
    sample_index: int | None = None  # e.g. 5 when timing == 'sample'
    value: float | None = None  # e.g. jaw_width=0.032 or screwdriver_extension=0.03
    work_name: str | None = None  # e.g. 'relay'
    grasp_id: int | None = None  # e.g. 4
    engage_tf: np.ndarray | None = None  # e.g. 4x4 tf used for object mounting on attach
    label: str | None = None  # e.g. 'close on contact'


@dataclass(frozen=True)
class HeldGrasp:
    work_name: str
    gid: int


@dataclass
class PlannerSegmentDraft:
    segment_label: str
    left_path: list[np.ndarray] | None = None
    right_path: list[np.ndarray] | None = None
    ee_events: list[EEEvent] | None = None
    end_sync_label: str | None = None
    held_after: HeldGrasp | None = None


@dataclass
class PlannerActionDraft:
    segments: list[PlannerSegmentDraft] = field(default_factory=list)
    held_after: HeldGrasp | None = None


@dataclass
class SyncSegment:
    id: str  # e.g. 'seg_grasped_to_pre_screw'
    label: str  # e.g. 'left transports relay while right approaches with screwdriver'
    start_sync_id: str  # e.g. 'sp_relay_grasped'
    end_sync_id: str  # e.g. 'sp_relay_pre_screw'
    arm_segments: list[ArmSegment] = field(default_factory=list)
    ee_events: list[EEEvent] = field(default_factory=list)


@dataclass
class SynchronizedPlan:
    labels: list[str]  # e.g. ['relay:time0', 'relay:time1', 'relay:time2']
    initial_state: DualRobotState | None = None  # e.g. both arms at home before the first segment
    sync_points: list[SyncPoint] = field(default_factory=list)
    sync_segments: list[SyncSegment] = field(default_factory=list)
