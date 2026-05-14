import os
from ctypes import CDLL, c_char_p, c_int, c_uint, c_void_p, create_string_buffer
from typing import Optional, Tuple

from one.communication.communication_as.backend.interface import ICommu

_ENCODING = 'utf-8'


def _b(s: Optional[str]) -> Optional[bytes]:
    return None if s is None else s.encode(_ENCODING, errors='ignore')


class CommuLinux(ICommu):
    def __init__(self, port: str, dll_root: Optional[str] = None):
        libdir = os.path.join(dll_root, 'linux')
        lib_path = os.path.join(libdir, 'librcc.so')
        if not os.path.exists(lib_path):
            alt_path = os.path.join(libdir, 'librcc.so.0.0.0')
            if os.path.exists(alt_path):
                lib_path = alt_path
        self._lib = CDLL(lib_path)

        self._lib.rcc_create.argtypes = [c_char_p]
        self._lib.rcc_create.restype = c_void_p
        self._lib.rcc_destroy.argtypes = [c_void_p]
        self._lib.rcc_destroy.restype = None

        self._lib.rcc_connect.argtypes = [c_void_p, c_char_p]
        self._lib.rcc_connect.restype = c_int
        self._lib.rcc_disconnect.argtypes = [c_void_p]
        self._lib.rcc_disconnect.restype = c_int

        self._lib.rcc_command_tmo.argtypes = [c_void_p, c_char_p, c_int, c_char_p, c_int]
        self._lib.rcc_command_tmo.restype = c_int
        self._lib.rcc_command.argtypes = [c_void_p, c_char_p, c_char_p, c_int]
        self._lib.rcc_command.restype = c_int

        self._lib.rcc_load.argtypes = [c_void_p, c_char_p, c_char_p]
        self._lib.rcc_load.restype = c_int
        self._lib.rcc_save.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p]
        self._lib.rcc_save.restype = c_int

        self._lib.rcc_start_log.argtypes = [c_void_p, c_char_p]
        self._lib.rcc_start_log.restype = c_int
        self._lib.rcc_stop_log.argtypes = [c_void_p]
        self._lib.rcc_stop_log.restype = c_int

        self._lib.rcc_is_connected.argtypes = [c_void_p]
        self._lib.rcc_is_connected.restype = c_int

        self._lib.rcc_set_timeout.argtypes = [c_void_p, c_uint]
        self._lib.rcc_set_timeout.restype = None
        self._lib.rcc_get_timeout.argtypes = [c_void_p]
        self._lib.rcc_get_timeout.restype = c_int

        self._lib.rcc_name.argtypes = [c_void_p, c_char_p, c_int]
        self._lib.rcc_name.restype = c_int
        self._lib.rcc_loadsavemsg.argtypes = [c_void_p, c_char_p, c_int]
        self._lib.rcc_loadsavemsg.restype = c_int

        self._h = self._lib.rcc_create(_b(port) if port else None)
        if not self._h:
            raise RuntimeError('rcc_create() failed')

    def connect(self, port: str) -> int:
        return self._lib.rcc_connect(self._h, _b(port))

    def disconnect(self) -> bool:
        return self._lib.rcc_disconnect(self._h) != 0

    def command(self, cmd: Optional[str] = None, tmo: Optional[int] = None) -> Tuple[int, str]:
        buf = create_string_buffer(1_000_000)
        if tmo is None:
            rc = self._lib.rcc_command(self._h, _b(cmd or ''), buf, len(buf))
        else:
            rc = self._lib.rcc_command_tmo(self._h, _b(cmd or ''), int(tmo), buf, len(buf))
        return rc, buf.value.decode(errors='ignore')

    def load(self, fname: str, qual: Optional[str] = None) -> int:
        return self._lib.rcc_load(self._h, _b(fname), None if qual is None else _b(qual))

    def save(self, fname: str, prog: Optional[str] = None, qual: Optional[str] = None) -> int:
        return self._lib.rcc_save(self._h, _b(fname), _b(prog), _b(qual))

    def startLog(self, log_fname: str) -> bool:
        return self._lib.rcc_start_log(self._h, _b(log_fname)) != 0

    def stopLog(self) -> bool:
        return self._lib.rcc_stop_log(self._h) != 0

    def name(self) -> str:
        buf = create_string_buffer(4096)
        self._lib.rcc_name(self._h, buf, len(buf))
        return buf.value.decode(_ENCODING, errors='ignore')

    @property
    def IsConnected(self) -> bool:
        return self._lib.rcc_is_connected(self._h) != 0

    @property
    def LoadSaveMsg(self) -> str:
        buf = create_string_buffer(1_000_000)
        self._lib.rcc_loadsavemsg(self._h, buf, len(buf))
        return buf.value.decode(_ENCODING, errors='ignore')

    @property
    def TimeoutValue(self) -> int:
        return int(self._lib.rcc_get_timeout(self._h))

    @TimeoutValue.setter
    def TimeoutValue(self, ms: int) -> None:
        self._lib.rcc_set_timeout(self._h, int(ms))

    def __del__(self):
        try:
            if getattr(self, '_h', None):
                self._lib.rcc_destroy(self._h)
        except Exception:
            pass
