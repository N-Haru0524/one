import os
from typing import Optional

from .interface import ICommu


def _load_krcc(dll_root: str):
    import clr

    dll_path = os.path.join(dll_root, 'win', 'krcc64.dll')
    if not os.path.exists(dll_path):
        raise FileNotFoundError(f'Windows communication library is missing: {dll_path}')
    clr.AddReference(dll_path)
    import KRcc

    return KRcc


class CommuWin(ICommu):
    def __init__(self, port: str, dll_root: str):
        krcc = _load_krcc(dll_root)
        self._c = krcc.Commu(port)

    def connect(self, port: str) -> int:
        return int(self._c.connect(port))

    def disconnect(self) -> bool:
        return bool(self._c.disconnect())

    def command(self, cmd: Optional[str] = None, tmo: Optional[int] = None) -> tuple[int, str]:
        if cmd is None:
            arr = self._c.command()
        elif tmo is None:
            arr = self._c.command(cmd)
        else:
            arr = self._c.command(cmd, int(tmo))
        status = int(arr[0])
        msg = str(arr[1]) if len(arr) > 1 and arr[1] is not None else ''
        return status, msg

    def load(self, fname: str, qual: Optional[str] = None) -> int:
        return int(self._c.load(fname) if not qual else self._c.load(fname, qual))

    def save(self, fname: str, prog: Optional[str] = None, qual: Optional[str] = None) -> int:
        if prog is None and qual is None:
            return int(self._c.save(fname))
        if qual is None:
            return int(self._c.save(fname, prog))
        return int(self._c.save(fname, prog, qual))

    def startLog(self, log_fname: str) -> bool:
        return bool(self._c.startLog(log_fname))

    def stopLog(self) -> bool:
        return bool(self._c.stopLog())

    def name(self) -> str:
        return str(self._c.name())

    @property
    def IsConnected(self) -> bool:
        return bool(self._c.IsConnected)

    @property
    def LoadSaveMsg(self) -> str:
        return str(self._c.LoadSaveMsg)

    @property
    def TimeoutValue(self) -> int:
        return int(self._c.TimeoutValue)

    @TimeoutValue.setter
    def TimeoutValue(self, ms: int) -> None:
        self._c.TimeoutValue = int(ms)
