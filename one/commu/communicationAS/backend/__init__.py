import sys


def create_commu(port: str, dll_root: str):
    if sys.platform.startswith('win'):
        from .windows_dotnet import CommuWin

        return CommuWin(port, dll_root)
    if sys.platform.startswith('linux'):
        from .linux_ctypes import CommuLinux

        return CommuLinux(port, dll_root)
    raise RuntimeError(f'unsupported platform: {sys.platform}')
