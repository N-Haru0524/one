from typing import TYPE_CHECKING, Any

from one_assembly.assembly_data import (
    ArmActor,
    ArmSegment,
    DualRobotState,
    EEAction,
    EEActor,
    EEEvent,
    HeldGrasp,
    PlannerActionDraft,
    PlannerSegmentDraft,
    SyncPoint,
    SyncSegment,
    SynchronizedPlan,
)
if TYPE_CHECKING:
    from one_assembly.work import Work
    from one_assembly.worklist import WorkList


__all__ = [
    'ArmActor',
    'PlannerActionDraft',
    'ArmSegment',
    'DualRobotState',
    'EEAction',
    'EEActor',
    'EEEvent',
    'HeldGrasp',
    'SyncPoint',
    'PlannerSegmentDraft',
    'SyncSegment',
    'PlannerActionDraft',
    'SynchronizedPlan',
    'Work',
    'WorkList',
]


def __getattr__(name: str) -> Any:
    if name == 'Work':
        from one_assembly.work import Work

        return Work
    if name == 'WorkList':
        from one_assembly.worklist import WorkList

        return WorkList
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
