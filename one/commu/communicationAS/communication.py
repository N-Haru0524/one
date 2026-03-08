from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from one.commu.communicationAS.backend import create_commu
from one.commu.communicationAS.communication_config import CommunicationConfig


_AS_DIR = os.path.join(os.path.dirname(__file__), 'as_programs')
_DLL_DIR = os.path.join(os.path.dirname(__file__), 'dll')


@dataclass(slots=True)
class _SimpleMotionData:
    jv_list: list[Any]
    ev_list: list[Any]

    def __len__(self) -> int:
        return len(self.jv_list)

    def extend(self, jv_items: Sequence[Any], ev_items: Sequence[Any]) -> None:
        self.jv_list.extend(jv_items)
        self.ev_list.extend(ev_items)


class ASWriter:
    def __init__(self, file_path: str):
        self.file_path = file_path
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.file = open(self.file_path, 'w', encoding='utf-8')

    def write_header(self) -> None:
        self.file.write(f'.PROGRAM {os.path.splitext(os.path.basename(self.file_path))[0]}()\n')

    def write_footer(self) -> None:
        self.file.write('.END\n')

    def write_line(self, line: str) -> None:
        self.file.write(f'  {line}\n')

    def close(self) -> None:
        if self.file:
            self.file.close()
            self.file = None

    def __del__(self):
        self.close()


class Communication:
    def __init__(self, robot: Any, config: dict[str, Any] | CommunicationConfig):
        self.config = config if isinstance(config, CommunicationConfig) else CommunicationConfig.from_dict(config)
        self.robot = robot
        self.comm = None
        self.writer: ASWriter | None = None

    def exec_time(self, func, *args, **kwargs):
        start_time = time.monotonic()
        ret = func(*args, **kwargs)
        print(f'Time for {func.__name__}: {time.monotonic() - start_time:.4f} seconds')
        return ret

    def command(self, command_str: str, toggle_dbg: bool = True) -> tuple[int, str]:
        ret, message = self.comm.command(command_str)
        if toggle_dbg:
            print(command_str)
            print(ret, message)
        return ret, message

    def connect(self) -> bool:
        port_spec = (
            f'{self.config.connection_type} '
            f'as@{self.config.ip_address} '
            f'{self.config.port} {self.config.timeout_ms} 1'
        )
        self.comm = create_commu(port_spec, _DLL_DIR)
        self.command('KILL')
        self.command('1')
        return self.comm is not None

    def load(self, file_path: str = os.path.join(_AS_DIR, 'temp.as'), toggle_dbg: bool = True) -> int:
        self.command('KILL')
        self.command('1')
        program_name = os.path.splitext(os.path.basename(file_path))[0]
        self.command(f'DELETE/P {program_name}')
        self.command('1')
        ret = self.comm.load(file_path)
        if toggle_dbg:
            print(f'[Load] {file_path}: {ret}')
        return ret

    def save(
        self,
        file_name: str = os.path.join(_AS_DIR, 'temp.as'),
        program_name: str = 'temp',
        qual: str = '/P',
        toggle_dbg: bool = True,
    ) -> int:
        ret = self.comm.save(file_name, program_name, qual)
        if toggle_dbg:
            print(f'[Save] {program_name} was saved in {file_name}: {ret}')
        return ret

    def set_robot_parameters(self, param: str, value: Any, always: bool = True) -> None:
        command_str = f'{param} {value} ALWAYS' if always else f'{param} {value}'
        self.writer.write_line(command_str)

    def move_joint(self, mot_data: Any) -> None:
        for jnts in mot_data.jv_list:
            self.move_joint_once(jnts)

    def move_joint_once(self, jnts: Sequence[float]) -> None:
        joints = ', '.join(f'{np.degrees(jv):.3f}' for jv in jnts)
        self.writer.write_line(f'JMOVE #[{joints}]')

    def grip(self, width: float = 0.001, force: int = 10, speed: int = 80, wait: int = 1) -> None:
        self.writer.write_line('; grip')
        if self.config.simulator_mode:
            return
        self.writer.write_line('TWAIT 0.1')
        self.writer.write_line(f'CALL OR_TwoFG_Grip.pc(1,{width * 1000 + 25:.2f},{force},{speed},{wait})')

    def pick_screw(self, start_pos: int = 40, force: int = 18, length: int = 6, wait: int = 1) -> None:
        self.writer.write_line('; pick screw')
        if self.config.simulator_mode:
            return
        instance = 1
        self.writer.write_line(f'CALL OR_SD_moveShank.pc({instance},{start_pos},{wait})')
        self.writer.write_line(f'CALL OR_SD_pickupScrew.pc({instance},{force},{length},{wait})')

    def screw(
        self,
        start_pos: int = 4,
        force: int = 18,
        length: int = 6,
        torque: float = 0.17,
        wait_for_stop: float = 0.5,
        wait: int = 1,
    ) -> None:
        self.writer.write_line('; screw')
        if self.config.simulator_mode:
            return
        instance = 1
        no_wait = 0
        self.writer.write_line(f'CALL OR_SD_moveShank.pc({instance},{start_pos},{wait})')
        self.writer.write_line(f'CALL OR_SD_tighten.pc({instance},{force},{length},{torque},{no_wait})')
        self.writer.write_line(f'TWAIT {wait_for_stop}')
        self.writer.write_line(f'CALL OR_SD_Stop.pc({instance})')
        self.writer.write_line(f'CALL OR_SD_moveShank.pc({instance},{start_pos},{wait})')

    def start_edit(
        self,
        program_name: str = 'temp',
        init_setting: Sequence[str] = ('ACCURACY 1', 'SPEED 3'),
        toggle_continue: bool = False,
    ) -> tuple[None, None]:
        self.writer = ASWriter(os.path.join(_AS_DIR, f'{program_name}.as'))
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
            self.set_robot_parameters(*setting.split())
        return None, None

    def end_edit(self, program_name: str = 'temp') -> tuple[int, str]:
        self.writer.write_footer()
        self.writer.close()
        self.writer = None
        ret = self.load(os.path.join(_AS_DIR, f'{program_name}.as'))
        save_msg = self.comm.LoadSaveMsg if hasattr(self.comm, 'LoadSaveMsg') else self.comm.loadsavemsg()
        return ret, save_msg

    def execute_program(self, program_name: str = 'temp') -> tuple[int, str]:
        ret, message = self.command(f'EXECUTE {program_name}')
        if 'already running' in message:
            ret = -1
        return ret, message

    def convert_to_as(self, mot_data: Any, toggle_mode: str = 'gripper') -> int:
        if toggle_mode == 'gripper':
            separated_mot = self.separate_motions(mot_data, toggle_mode)
            for mot in separated_mot:
                if mot.ev_list[0] is not None:
                    self.grip(mot.ev_list[0])
                self.move_joint(mot)
        elif toggle_mode == 'screw':
            separated_mot = self.separate_motions(mot_data, toggle_mode)
            for mot in separated_mot:
                self.move_joint(mot)
                if mot.ev_list[0] is not None:
                    if mot.ev_list[0] == 0:
                        self.pick_screw()
                    else:
                        self.screw()
        else:
            raise ValueError("toggle_mode must be either 'gripper' or 'screw'.")
        return 0

    def convert_to_as_with_interlock(self, mot_data: Any, interlock: list[list[int]], toggle_mode: str = 'gripper'):
        prev_locked = set()
        if toggle_mode == 'gripper':
            for i in range(len(mot_data)):
                if mot_data.jv_list[i] is None:
                    continue
                curr_locked = set(interlock[i]) if interlock[i] else set()
                to_lock = curr_locked - prev_locked
                to_unlock = prev_locked - curr_locked
                if to_unlock:
                    self.disable_interlock(sorted(to_unlock))
                if to_lock:
                    self.enable_interlock(sorted(to_lock))
                prev_locked = curr_locked
                if i == 0 or mot_data.ev_list[i] != mot_data.ev_list[i - 1]:
                    self.grip(mot_data.ev_list[i])
                self.move_joint_once(mot_data.jv_list[i])
        elif toggle_mode == 'screw':
            for i in range(len(mot_data)):
                if mot_data.jv_list[i] is None:
                    continue
                curr_locked = set(interlock[i]) if interlock[i] else set()
                to_lock = curr_locked - prev_locked
                to_unlock = prev_locked - curr_locked
                if to_unlock:
                    self.disable_interlock(sorted(to_unlock))
                if to_lock:
                    self.enable_interlock(sorted(to_lock))
                prev_locked = curr_locked
                self.move_joint_once(mot_data.jv_list[i])
                if mot_data.ev_list[i] is not None:
                    if i != 0 and mot_data.ev_list[i] == 1 and mot_data.ev_list[i - 1] == 0:
                        self.pick_screw()
                    elif i == len(mot_data.ev_list) - 1 and mot_data.ev_list[i] == 1:
                        self.screw()
        if prev_locked:
            self.disable_interlock(sorted(prev_locked))
        self.break_motion()
        return 0

    def separate_motions(self, mot_data: Any, toggle_mode: str = 'gripper') -> list[_SimpleMotionData]:
        if toggle_mode not in ['gripper', 'screw']:
            raise ValueError("toggle_mode must be either 'gripper' or 'screw'.")

        if len(mot_data) < 2:
            return [_SimpleMotionData(jv_list=list(mot_data.jv_list), ev_list=list(mot_data.ev_list))]

        result: list[_SimpleMotionData] = []
        temp = _SimpleMotionData(jv_list=[], ev_list=[])

        if toggle_mode == 'gripper':
            clamp_flag = False
            for i in range(len(mot_data)):
                if i == 0:
                    temp.extend([mot_data.jv_list[i]], [mot_data.ev_list[i]])
                    continue
                if mot_data.ev_list[i - 1] != mot_data.ev_list[i]:
                    if not clamp_flag:
                        clamp_flag = True
                        result.append(temp)
                        temp = _SimpleMotionData(jv_list=[mot_data.jv_list[i]], ev_list=[mot_data.ev_list[i]])
                else:
                    clamp_flag = False
                    temp.extend([mot_data.jv_list[i]], [mot_data.ev_list[i]])

        if toggle_mode == 'screw':
            screw_flag = False
            for i in range(len(mot_data)):
                if i == 0:
                    temp.extend([mot_data.jv_list[i]], [mot_data.ev_list[i]])
                    continue
                if mot_data.ev_list[i - 1] == (not mot_data.ev_list[i]):
                    if not screw_flag:
                        screw_flag = True
                        result.append(temp)
                        temp = _SimpleMotionData(jv_list=[mot_data.jv_list[i]], ev_list=[mot_data.ev_list[i]])
                else:
                    screw_flag = False
                    temp.extend([mot_data.jv_list[i]], [mot_data.ev_list[i]])

        result.append(temp)
        return result

    def get_joint_values(self) -> list[float]:
        _, message = self.command('WHERE', toggle_dbg=False)
        lines = message.strip().splitlines()
        jv = [float(x) for x in lines[1].split()]
        return [float(np.radians(j)) for j in jv]

    def get_ee_value(self) -> float:
        self.load(os.path.join(_AS_DIR, 'get_ee_value.as'))
        self.execute_program(program_name='get_ee_value')
        _, ee_value = self.command('PRINT "2FG current width: ", ret')
        values = re.findall(r'-?\d+\.?\d*', ee_value)
        if values:
            return float(values[-1]) / 1000.0
        raise ValueError('Failed to extract ee_value from the command output.')

    def get_status(self) -> dict[str, Any]:
        ret, message = self.command('STATUS', toggle_dbg=False)
        if ret < 0:
            raise ValueError(f'Failed to get status. Error code: {ret}, Message: {message}')
        return self.parse_robot_status(message)

    def test_gripper_connection(self) -> int:
        ret, _ = self.command('EXE test2fg')
        return ret

    def parse_robot_status(self, text: str) -> dict[str, Any]:
        status: dict[str, Any] = {}

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

        status['stepper_status'] = {'program_running': 'Program is not running.' not in text}

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

    def initialize_interlock(self, priority_areas: list[int], secondary_areas: list[int]) -> None:
        self.writer.write_line('; initialize interlock signals (priority areas)')
        if priority_areas:
            priority_outputs = [
                str(-1 * self.config.signals.outputs.workspace_occupied[area]) for area in priority_areas
            ]
            self.writer.write_line(f'SIGNAL {", ".join(priority_outputs)}')
        self.writer.write_line('; initialize interlock signals (secondary areas)')
        if secondary_areas:
            secondary_outputs = [
                str(self.config.signals.outputs.workspace_occupied[area]) for area in secondary_areas
            ]
            self.writer.write_line(f'SIGNAL {", ".join(secondary_outputs)}')

    def enable_interlock(self, area_ids: list[int]) -> None:
        if not area_ids:
            return
        inputs = [str(self.config.signals.inputs.workspace_occupied[area_id]) for area_id in area_ids]
        outputs = [str(-1 * self.config.signals.outputs.workspace_occupied[area_id]) for area_id in area_ids]
        self.writer.write_line('; interlock section start')
        self.writer.write_line(f'SWAIT {", ".join(inputs)}')
        self.writer.write_line(f'SIGNAL {", ".join(outputs)}')

    def disable_interlock(self, area_ids: list[int]) -> None:
        if not area_ids:
            return
        outputs = [str(self.config.signals.outputs.workspace_occupied[area_id]) for area_id in area_ids]
        self.writer.write_line('; interlock section end')
        self.writer.write_line(f'SIGNAL {", ".join(outputs)}')

    def reset_interlock(self) -> None:
        if not self.config.signals.outputs.workspace_occupied:
            self.writer.write_line('; no interlock signals to reset')
            return
        reset_signals = [str(-1 * signal) for signal in self.config.signals.outputs.workspace_occupied]
        self.writer.write_line('; reset interlock signals')
        self.writer.write_line(f'SIGNAL {", ".join(reset_signals)}')

    def notify_motion_complete(self) -> None:
        output = self.config.signals.outputs.motion_complete
        self.writer.write_line('; motion complete signal notification')
        self.writer.write_line(f'SIGNAL {int(-1 * output)}')

    def wait_for_motion_complete(self) -> None:
        input_signal = self.config.signals.inputs.motion_complete
        self.writer.write_line('; wait for motion complete')
        self.writer.write_line(f'SWAIT {int(-1 * input_signal)}')

    def reset_motion_complete(self) -> None:
        output = self.config.signals.outputs.motion_complete
        self.writer.write_line('; reset motion complete signal')
        self.writer.write_line(f'SIGNAL {int(output)}')

    def wait_for_reset_motion_complete(self) -> None:
        input_signal = self.config.signals.inputs.motion_complete
        self.writer.write_line('; wait for reset motion complete signal')
        self.writer.write_line(f'SWAIT {int(input_signal)}')

    def break_motion(self) -> None:
        self.writer.write_line('BREAK')

    def initialize_timer(self, timer_id: int) -> None:
        self.writer.write_line(f'; initialize timer {timer_id}')
        self.writer.write_line(f'TIMER {timer_id}=0')

    def print_timer(self, timer_id: int) -> None:
        self.writer.write_line(f'; print timer {timer_id}')
        self.writer.write_line(f'PRINT TIMER({timer_id})')

    def print(self, message: str) -> None:
        self.writer.write_line(f'PRINT "{message}"')
