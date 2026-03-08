from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class InputSignalConfig(BaseModel):
    """Input signal settings."""

    motion_complete: int
    workspace_occupied: list[int]


class OutputSignalConfig(BaseModel):
    """Output signal settings."""

    motion_complete: int
    workspace_occupied: list[int]


class SignalConfig(BaseModel):
    """Bidirectional signal settings."""

    inputs: InputSignalConfig
    outputs: OutputSignalConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'SignalConfig':
        if hasattr(cls, 'model_validate'):
            return cls.model_validate(data)
        return cls.parse_obj(data)
