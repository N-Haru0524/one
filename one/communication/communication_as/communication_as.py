import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from one.communication.communication_as.backend import create_commu
from one.communication.communication_as.communication_config import CommunicationASConfig


class ASWriter:
    def __init__(self, file_path: str):
        self.file_path = file_path
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.file = open(self.file_path, 'w', encoding='utf-8')

    def write_header(self) -> None:
        name = os.path.splitext(os.path.basename(self.file_path))[0]
        self.file.write('.PROGRAM %s()\n' % name)

    def write_footer(self) -> None:
        self.file.write('.END\n')

    def write_line(self, line: str) -> None:
        self.file.write('  %s\n' % line)

    def close(self) -> None:
        if self.file:
            self.file.close()
            self.file = None

    def __del__(self):
        self.close()


class CommunicationAS:
    def __init__(
        self,
        config: Union[CommunicationASConfig, Dict[str, Any]],
        as_dir: Optional[str] = None,
        dll_root: Optional[str] = None,
    ):
        if isinstance(config, CommunicationASConfig):
            self.config = config
        else:
            self.config = CommunicationASConfig.from_dict(config)

        base_dir = os.path.dirname(__file__)
        self.as_dir = as_dir or os.path.join(base_dir, 'AS')
        self.dll_root = dll_root or os.path.join(base_dir, 'dll')

        self.comm = None
        self.writer: Optional[ASWriter] = None

    def exec_time(self, func, *args, **kwargs):
        start_time = time.monotonic()
        ret = func(*args, **kwargs)
        end_time = time.monotonic()
        print('Time for %s: %.4f seconds' % (func.__name__, end_time - start_time))
        return ret

    def _connection_string(self) -> str:
        return '%s as@%s %d %d 1' % (
            self.config.connection_type,
            self.config.ip_address,
            self.config.port,
            self.config.timeout_ms,
        )

    def connect(self) -> bool:
        self.comm = create_commu(self._connection_string(), self.dll_root)
        self.command('KILL', toggle_dbg=False)
        self.command('1', toggle_dbg=False)
        return self.comm is not None

    def disconnect(self) -> bool:
        if self.comm is None:
            return True
        return self.comm.disconnect()

    def command(self, command_str: str, toggle_dbg: bool = True) -> Tuple[int, str]:
        if self.comm is None:
            raise RuntimeError('connect() is required before command().')
        ret, msg = self.comm.command(command_str)
        if toggle_dbg:
            print(command_str)
            print(ret, msg)
        return ret, msg

    def load(self, file_path: str, toggle_dbg: bool = True) -> int:
        self.command('KILL', toggle_dbg=False)
        self.command('1', toggle_dbg=False)
        self.command('DELETE/P %s' % os.path.splitext(os.path.basename(file_path))[0], toggle_dbg=False)
        self.command('1', toggle_dbg=False)
        ret = self.comm.load(file_path)
        if toggle_dbg:
            print('[Load] %s: %s' % (file_path, ret))
        return ret

    def save(self, file_path: str, program_name: str, qual: str = '/P', toggle_dbg: bool = True) -> int:
        ret = self.comm.save(file_path, program_name, qual)
        if toggle_dbg:
            print('[Save] %s was saved in %s: %s' % (program_name, file_path, ret))
        return ret

    def start_edit(
        self,
        program_name: str = 'temp',
        init_setting: Optional[Iterable[str]] = None,
        toggle_continue: bool = False,
    ) -> None:
        if init_setting is None:
            init_setting = ['ACCURACY 1', 'SPEED 3']

        self.writer = ASWriter(os.path.join(self.as_dir, '%s.as' % program_name))
        self.writer.write_header()

        if not self.config.simulator_mode:
            self.writer.write_line('CALL or_init.pc(101)')

        if not toggle_continue:
            self.writer.write_line('SPEED 40MM/S')
            self.writer.write_line('JDEPART 100')
            self.writer.write_line('SPEED 150MM/S')
            self.writer.write_line('JMOVE #hm')
            self.writer.write_line('BREAK')

        for setting in init_setting:
            self.writer.write_line(setting)

    def end_edit(self, program_name: str = 'temp') -> Tuple[int, str]:
        if self.writer is None:
            raise RuntimeError('start_edit() is required before end_edit().')

        self.writer.write_footer()
        self.writer.close()
        self.writer = None

        file_path = os.path.join(self.as_dir, '%s.as' % program_name)
        ret = self.load(file_path)
        save_msg = self.comm.LoadSaveMsg if hasattr(self.comm, 'LoadSaveMsg') else ''
        return ret, save_msg

    def execute_program(self, program_name: str = 'temp') -> Tuple[int, str]:
        ret, msg = self.command('EXECUTE %s' % program_name)
        if 'already running' in msg:
            return -1, msg
        return ret, msg

    def move_joint_once(self, joints_rad: List[float]) -> None:
        if self.writer is None:
            raise RuntimeError('start_edit() is required before move_joint_once().')

        deg = [j * 180.0 / 3.141592653589793 for j in joints_rad]
        joint_text = ', '.join(['%.3f' % value for value in deg])
        self.writer.write_line('JMOVE #[%s]' % joint_text)

    def move_joint(self, joints_rad_list: List[List[float]]) -> None:
        for joints in joints_rad_list:
            self.move_joint_once(joints)

    def grip(self, width: float = 0.001, force: int = 10, speed: int = 80, wait: int = 1) -> None:
        if self.writer is None:
            raise RuntimeError('start_edit() is required before grip().')
        self.writer.write_line('; grip')
        if self.config.simulator_mode:
            return
        self.writer.write_line('TWAIT 0.1')
        self.writer.write_line('CALL OR_TwoFG_Grip.pc(1,%.2f,%d,%d,%d)' % (width * 1000.0 + 25.0, force, speed, wait))

    def enable_interlock(self, area_ids: List[int]) -> None:
        if self.writer is None or not area_ids:
            return
        inputs = [str(self.config.signals.inputs.workspace_occupied[idx]) for idx in area_ids]
        outputs = [str(-1 * self.config.signals.outputs.workspace_occupied[idx]) for idx in area_ids]
        self.writer.write_line('; interlock section start')
        self.writer.write_line('SWAIT %s' % ', '.join(inputs))
        self.writer.write_line('SIGNAL %s' % ', '.join(outputs))

    def disable_interlock(self, area_ids: List[int]) -> None:
        if self.writer is None or not area_ids:
            return
        outputs = [str(self.config.signals.outputs.workspace_occupied[idx]) for idx in area_ids]
        self.writer.write_line('; interlock section end')
        self.writer.write_line('SIGNAL %s' % ', '.join(outputs))

    def reset_interlock(self) -> None:
        if self.writer is None:
            return
        values = self.config.signals.outputs.workspace_occupied
        if not values:
            self.writer.write_line('; no interlock signals to reset')
            return
        reset_signals = [str(-1 * val) for val in values]
        self.writer.write_line('; reset interlock signals')
        self.writer.write_line('SIGNAL %s' % ', '.join(reset_signals))

    def notify_motion_complete(self) -> None:
        if self.writer is None:
            return
        val = int(-1 * self.config.signals.outputs.motion_complete)
        self.writer.write_line('; motion complete signal notification')
        self.writer.write_line('SIGNAL %d' % val)

    def wait_for_motion_complete(self) -> None:
        if self.writer is None:
            return
        val = int(-1 * self.config.signals.inputs.motion_complete)
        self.writer.write_line('; wait for motion complete')
        self.writer.write_line('SWAIT %d' % val)

    def reset_motion_complete(self) -> None:
        if self.writer is None:
            return
        val = int(self.config.signals.outputs.motion_complete)
        self.writer.write_line('; reset motion complete signal')
        self.writer.write_line('SIGNAL %d' % val)

    def break_motion(self) -> None:
        if self.writer is None:
            return
        self.writer.write_line('BREAK')

    def get_joint_values(self) -> List[float]:
        ret, text = self.command('WHERE', toggle_dbg=False)
        if ret < 0:
            raise ValueError('WHERE failed: %s' % text)
        lines = text.strip().splitlines()
        if len(lines) < 2:
            raise ValueError('Unexpected WHERE response: %s' % text)
        joint_deg = [float(x) for x in lines[1].split()]
        return [v * 3.141592653589793 / 180.0 for v in joint_deg]

    def get_status(self) -> Dict[str, Any]:
        ret, text = self.command('STATUS', toggle_dbg=False)
        if ret < 0:
            raise ValueError('STATUS failed: %s' % text)
        return self.parse_robot_status(text)

    def parse_robot_status(self, text: str) -> Dict[str, Any]:
        status: Dict[str, Any] = {}
        status['motor_power'] = 'OFF' if 'Motor power OFF' in text else 'ON'
        status['mode'] = 'TEACH mode' if 'TEACH mode' in text else 'REPEAT mode'

        status['environment'] = {}
        match = re.search(r'Monitor speed\(%\)\s*=\s*([0-9.]+)', text)
        if match:
            status['environment']['monitor_speed_percent'] = float(match.group(1))

        match = re.search(r'Program speed\(%\) ALWAYS\s*=\s*([0-9.]+)\s+([0-9.]+)', text)
        if match:
            status['environment']['program_speed_percent'] = {
                'always': float(match.group(1)),
                'current': float(match.group(2)),
            }

        match = re.search(r'ALWAYS Accu\.\[mm\]\s*=\s*([0-9.]+)', text)
        if match:
            status['environment']['always_accuracy_mm'] = float(match.group(1))

        status['stepper_status'] = {
            'program_running': 'Program is not running.' not in text,
        }

        status['execution_cycles'] = {}
        match = re.search(r'Completed cycles:\s+([0-9]+)', text)
        if match:
            status['execution_cycles']['completed'] = int(match.group(1))
        match = re.search(r'Remaining cycles:\s+([0-9]+)', text)
        if match:
            status['execution_cycles']['remaining'] = int(match.group(1))

        if 'No program is running.' in text:
            status['current_program'] = {'name': None, 'priority': None, 'step_no': None}
        else:
            prog_match = re.search(r'\n\s*(\w+)\s+(\d+)\s+(\d+)\s+', text)
            if prog_match:
                status['current_program'] = {
                    'name': prog_match.group(1),
                    'priority': int(prog_match.group(2)),
                    'step_no': int(prog_match.group(3)),
                }
            else:
                status['current_program'] = {'name': 'UNKNOWN', 'priority': None, 'step_no': None}

        return status
