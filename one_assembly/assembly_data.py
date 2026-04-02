from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

AttachEvent: TypeAlias = tuple[str, str, int | None, float | None, np.ndarray | None]  # ('attach', work_name, grasp_id, jaw_width, engage_tf)
ReleaseEvent: TypeAlias = tuple[str, str]  # ('release', work_name)
AssemblyEvent: TypeAlias = AttachEvent | ReleaseEvent
EventMap: TypeAlias = dict[int, AssemblyEvent]  # {12: ('attach', 'relay', 4, 0.032, engage_tf), 25: ('release', 'relay')}


@dataclass
class DualRobotState:
    lft_qs: np.ndarray  # left arm joint vector
    lft_ee_qs: np.ndarray  # left gripper jaw state, e.g. [0.02, 0.02]
    rgt_qs: np.ndarray  # right arm joint vector
    rgt_ee_qs: np.ndarray  # right screwdriver state, e.g. [0.03]

    def copy(self):
        return DualRobotState(
            lft_qs=self.lft_qs.copy(),
            lft_ee_qs=self.lft_ee_qs.copy(),
            rgt_qs=self.rgt_qs.copy(),
            rgt_ee_qs=self.rgt_ee_qs.copy(),
        )


@dataclass
class PlaybackPlan:
    labels: list[str]  # e.g. ['relay', 'relay_fold', 'relay_scrw']
    state_list: list[DualRobotState]
    event_map: EventMap


@dataclass
class HeldGrasp:
    work_name: str  # e.g. 'relay'
    gid: int  # e.g. 4


@dataclass
class PlannedSegment:
    state_list: list[DualRobotState]  # one action worth of states before flattening
    event_map: EventMap
    held_after: HeldGrasp | None = None  # e.g. HeldGrasp(work_name='relay', gid=4) or None
