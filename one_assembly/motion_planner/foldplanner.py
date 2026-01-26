import numpy as np
import one.utils.math as oum


def interpolate_fold(start_pose, goal_pose, n_steps=20):
    start_pos, start_rotmat = start_pose
    goal_pos, goal_rotmat = goal_pose
    start_pos = np.asarray(start_pos, dtype=np.float32)
    goal_pos = np.asarray(goal_pos, dtype=np.float32)
    start_quat = oum.quat_from_rotmat(start_rotmat)
    goal_quat = oum.quat_from_rotmat(goal_rotmat)
    poses = []
    for t in np.linspace(0.0, 1.0, n_steps):
        pos = (1.0 - t) * start_pos + t * goal_pos
        quat = oum.slerp_quat(start_quat, goal_quat, t)
        rotmat = oum.rotmat_from_quat(quat)
        poses.append((pos, rotmat))
    return poses


class FoldPlanner:
    def __init__(self, robot=None):
        self.robot = robot

    def plan_fold(self, start_pose, goal_pose, n_steps=20):
        return interpolate_fold(start_pose, goal_pose, n_steps=n_steps)
