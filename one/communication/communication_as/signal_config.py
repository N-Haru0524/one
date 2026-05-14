from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class InputSignalConfig:
    motion_complete: int
    workspace_occupied: List[int] = field(default_factory=list)


@dataclass
class OutputSignalConfig:
    motion_complete: int
    workspace_occupied: List[int] = field(default_factory=list)


@dataclass
class SignalConfig:
    inputs: InputSignalConfig
    outputs: OutputSignalConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalConfig':
        inputs = data.get('inputs', {})
        outputs = data.get('outputs', {})
        return cls(
            inputs=InputSignalConfig(
                motion_complete=int(inputs.get('motion_complete', 0)),
                workspace_occupied=list(inputs.get('workspace_occupied', [])),
            ),
            outputs=OutputSignalConfig(
                motion_complete=int(outputs.get('motion_complete', 0)),
                workspace_occupied=list(outputs.get('workspace_occupied', [])),
            ),
        )
