from dataclasses import dataclass
from typing import Any, Dict

from one.communication.communication_as.signal_config import SignalConfig


@dataclass
class CommunicationASConfig:
    ip_address: str
    port: int
    timeout_ms: int
    connection_type: str
    simulator_mode: bool
    signals: SignalConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommunicationASConfig':
        return cls(
            ip_address=str(data['ip_address']),
            port=int(data['port']),
            timeout_ms=int(data['timeout_ms']),
            connection_type=str(data['connection_type']),
            simulator_mode=bool(data.get('simulator_mode', False)),
            signals=SignalConfig.from_dict(data.get('signals', {})),
        )
