from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple


class ICommu(ABC):
    @abstractmethod
    def connect(self, port: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def command(self, cmd: Optional[str] = None, tmo: Optional[int] = None) -> Tuple[int, str]:
        raise NotImplementedError

    @abstractmethod
    def load(self, fname: str, qual: Optional[str] = None) -> int:
        raise NotImplementedError

    @abstractmethod
    def save(self, fname: str, prog: Optional[str] = None, qual: Optional[str] = None) -> int:
        raise NotImplementedError

    @abstractmethod
    def startLog(self, log_fname: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def stopLog(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def IsConnected(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def LoadSaveMsg(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def TimeoutValue(self) -> int:
        raise NotImplementedError

    @TimeoutValue.setter
    @abstractmethod
    def TimeoutValue(self, ms: int) -> None:
        raise NotImplementedError
