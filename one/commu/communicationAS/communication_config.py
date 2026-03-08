from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from one.commu.communicationAS.signal_config import SignalConfig


class CommunicationConfig(BaseModel):
    """Communication settings."""

    ip_address: str
    port: int
    timeout_ms: int
    connection_type: str
    simulator_mode: bool
    signals: SignalConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'CommunicationConfig':
        if hasattr(cls, 'model_validate'):
            return cls.model_validate(data)
        return cls.parse_obj(data)
