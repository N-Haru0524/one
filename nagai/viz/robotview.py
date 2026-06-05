"""KHIBunriView: the dual-arm state codec.

All KHIBunri-specific knowledge (how to apply / capture a DualRobotState, how to
compose per-arm planner inputs, where the TCP frames are) is confined here so the
player and decorators stay robot-agnostic.
"""

import numpy as np

from one_assembly.assembly_data import DualRobotState
from one_assembly.robots.khi_bunri.khi_bunri import KHIBunri


class KHIBunriView:
    def __init__(self, robot: KHIBunri):
        self.robot = robot

    # --- state apply / capture --------------------------------------------
    def apply(self, state: DualRobotState):
        r = self.robot
        r.lft_arm.fk(state.lft_qs)
        r.lft_gripper.fk(state.lft_ee_qs)
        r.rgt_arm.fk(state.rgt_qs)
        r.rgt_screwdriver.fk(state.rgt_ee_qs)

    def capture(self) -> DualRobotState:
        r = self.robot
        return DualRobotState(
            lft_qs=r.lft_arm.qs.copy(),
            lft_ee_qs=r.lft_gripper.qs[:r.lft_gripper.ndof].copy(),
            rgt_qs=r.rgt_arm.qs.copy(),
            rgt_ee_qs=r.rgt_screwdriver.qs[:r.rgt_screwdriver.ndof].copy(),
        )

    # --- planner input composition ----------------------------------------
    def compose_left(self, state: DualRobotState) -> np.ndarray:
        return np.concatenate([state.lft_qs, state.lft_ee_qs]).astype(np.float32)

    def compose_right(self, state: DualRobotState) -> np.ndarray:
        return np.concatenate([state.rgt_qs, state.rgt_ee_qs]).astype(np.float32)

    # --- gripper / driver helpers -----------------------------------------
    def open_gripper_qs(self) -> np.ndarray:
        open_half = float(self.robot.lft_gripper.jaw_range[1] * 0.5)
        return np.array([open_half, open_half], dtype=np.float32)

    def reset_driver_shank(self):
        self.robot.rgt_screwdriver.set_shank_len(self.robot.rgt_screwdriver.shank_range[0])

    def home(self):
        self.robot.goto_home_conf()
        self.robot.lft_gripper.open()
        self.reset_driver_shank()

    # --- TCP frames -------------------------------------------------------
    def tcp_tfs(self):
        return self.robot.lft_tcp_tf, self.robot.rgt_tcp_tf
