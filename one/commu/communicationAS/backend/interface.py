from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class ICommu(ABC):
    @abstractmethod
    def connect(self, port: str) -> int:
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        pass

    @abstractmethod
    def command(self, cmd: Optional[str] = None, tmo: Optional[int] = None) -> tuple[int, str]:
        pass

    @abstractmethod
    def load(self, fname: str, qual: Optional[str] = None) -> int:
        pass

    @abstractmethod
    def save(self, fname: str, prog: Optional[str] = None, qual: Optional[str] = None) -> int:
        pass

    @abstractmethod
    def startLog(self, log_fname: str) -> bool:
        pass

    @abstractmethod
    def stopLog(self) -> bool:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def IsConnected(self) -> bool:
        pass

    @property
    @abstractmethod
    def LoadSaveMsg(self) -> str:
        pass

    @property
    @abstractmethod
    def TimeoutValue(self) -> int:
        pass

    @TimeoutValue.setter
    @abstractmethod
    def TimeoutValue(self, ms: int) -> None:
        pass
