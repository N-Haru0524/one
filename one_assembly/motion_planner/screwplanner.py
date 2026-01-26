import numpy as np
from motion_planner import utils


class ScrewPlanner:
    def __init__(self, robot, collider=None):
        self.robot = robot
        self._collider = collider

    def _ensure_collider(self, obstacles=None):
        if self._collider is not None:
            return self._collider
        return utils.build_collider([self.robot], obstacles=obstacles)

    def plan_screw(self, tgt_pose_tf, start_qs=None,
                   obstacles=None,
                   step_size=np.pi / 36,
                   max_iters=2000,
                   time_limit=None):
        if start_qs is None:
            start_qs = self.robot.qs
        pos = tgt_pose_tf[:3, 3]
        rotmat = tgt_pose_tf[:3, :3]
        ik_solutions = self.robot.ik_tcp(rotmat, pos)
        goal_qs = utils.select_ik_solution(ik_solutions, start_qs)
        if goal_qs is None:
            return None
        collider = self._ensure_collider(obstacles=obstacles)
        path = utils.plan_rrt(self.robot, start_qs, goal_qs, collider,
                              step_size=step_size,
                              max_iters=max_iters,
                              time_limit=time_limit)
        if path is None:
            return None
        return utils.MotionPlan(path)
