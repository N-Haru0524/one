import numpy as np
import one.utils.math as oum
from motion_planner import utils


class PickPlacePlanner:
    def __init__(self, robot, collider=None):
        self.robot = robot
        self._collider = collider

    def _ensure_collider(self, obstacles=None):
        if self._collider is not None:
            return self._collider
        return utils.build_collider([self.robot], obstacles=obstacles)

    def _plan_to_tcp(self, tgt_pos, tgt_rotmat, start_qs,
                     obstacles=None,
                     step_size=np.pi / 36,
                     max_iters=2000,
                     time_limit=None):
        ik_solutions = self.robot.ik_tcp(tgt_rotmat, tgt_pos)
        if not ik_solutions:
            return None
        collider = self._ensure_collider(obstacles=obstacles)
        start_qs = np.asarray(start_qs, dtype=np.float32)
        candidates = sorted(
            (np.asarray(qs, dtype=np.float32) for qs in ik_solutions),
            key=lambda qs: np.linalg.norm(qs - start_qs))
        for goal_qs in candidates:
            if collider.is_collided(goal_qs):
                continue
            path = utils.plan_rrt(self.robot, start_qs, goal_qs, collider,
                                  step_size=step_size,
                                  max_iters=max_iters,
                                  time_limit=time_limit)
            if path is None:
                continue
            return utils.MotionPlan(path)
        return None

    def plan_pick(self, grasp_pose_tf, jaw_width, start_qs=None,
                  obstacles=None,
                  step_size=np.pi / 36,
                  max_iters=2000,
                  time_limit=None):
        if start_qs is None:
            start_qs = self.robot.qs
        pos = grasp_pose_tf[:3, 3]
        rotmat = grasp_pose_tf[:3, :3]
        plan = self._plan_to_tcp(pos, rotmat, start_qs,
                                 obstacles=obstacles,
                                 step_size=step_size,
                                 max_iters=max_iters,
                                 time_limit=time_limit)
        if plan is None:
            return None
        plan.jaw_widths = [jaw_width] * len(plan)
        return plan

    def plan_place(self, place_pose_tf, start_qs=None,
                   obstacles=None,
                   step_size=np.pi / 36,
                   max_iters=2000,
                   time_limit=None):
        if start_qs is None:
            start_qs = self.robot.qs
        pos = place_pose_tf[:3, 3]
        rotmat = place_pose_tf[:3, :3]
        return self._plan_to_tcp(pos, rotmat, start_qs,
                                 obstacles=obstacles,
                                 step_size=step_size,
                                 max_iters=max_iters,
                                 time_limit=time_limit)

    def plan_pick_and_place(self, grasp_pose_tf, jaw_width, place_pose_tf,
                            start_qs=None,
                            obstacles=None,
                            step_size=np.pi / 36,
                            max_iters=2000,
                            time_limit=None):
        if start_qs is None:
            start_qs = self.robot.qs
        pick_plan = self.plan_pick(grasp_pose_tf, jaw_width,
                                   start_qs=start_qs,
                                   obstacles=obstacles,
                                   step_size=step_size,
                                   max_iters=max_iters,
                                   time_limit=time_limit)
        if pick_plan is None:
            return None
        last_qs = pick_plan.qs_list[-1]
        place_plan = self.plan_place(place_pose_tf,
                                     start_qs=last_qs,
                                     obstacles=obstacles,
                                     step_size=step_size,
                                     max_iters=max_iters,
                                     time_limit=time_limit)
        if place_plan is None:
            return None
        full_plan = utils.MotionPlan(pick_plan.qs_list, pick_plan.jaw_widths)
        full_plan.extend(place_plan.qs_list,
                         jaw_widths=[jaw_width] * len(place_plan))
        return full_plan
