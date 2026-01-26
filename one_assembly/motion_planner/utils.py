import numpy as np
import one.motion.probabilistic.space_provider as ompsp
import one.motion.probabilistic.rrt as ompr
import one.collider.mj_collider as ocm


class MotionPlan:
    def __init__(self, qs_list=None, jaw_widths=None):
        self.qs_list = [] if qs_list is None else [
            np.asarray(qs, dtype=np.float32) for qs in qs_list]
        self.jaw_widths = None if jaw_widths is None else list(jaw_widths)

    def __len__(self):
        return len(self.qs_list)

    def extend(self, qs_list, jaw_widths=None):
        self.qs_list.extend(
            np.asarray(qs, dtype=np.float32) for qs in qs_list)
        if self.jaw_widths is None:
            return
        if jaw_widths is None:
            raise ValueError("jaw_widths is required once initialized")
        self.jaw_widths.extend(jaw_widths)


def build_collider(actors, obstacles=None, margin=0.0):
    collider = ocm.MJCollider()
    for actor in actors:
        collider.append(actor)
    if obstacles:
        for obstacle in obstacles:
            collider.append(obstacle)
    collider.actors = actors
    collider.compile(margin=margin)
    return collider


def build_space_provider(robot, collider, max_edge_step=np.pi / 180):
    jlmt_low = robot.structure.compiled.jlmt_low_by_idx
    jlmt_high = robot.structure.compiled.jlmt_high_by_idx
    return ompsp.SpaceProvider.from_box_bounds(
        lmt_low=jlmt_low,
        lmt_high=jlmt_high,
        collider=collider,
        max_edge_step=max_edge_step)


def plan_rrt(robot, start_qs, goal_qs, collider,
             step_size=np.pi / 36,
             max_iters=2000,
             time_limit=None,
             max_edge_step=np.pi / 180):
    sspp = build_space_provider(robot, collider, max_edge_step=max_edge_step)
    planner = ompr.RRTConnectPlanner(ssp_provider=sspp, step_size=step_size)
    return planner.solve(start=start_qs, goal=goal_qs,
                         max_iters=max_iters, time_limit=time_limit)


def select_ik_solution(solutions, ref_qs):
    if not solutions:
        return None
    ref_qs = np.asarray(ref_qs, dtype=np.float32)
    candidates = [np.asarray(qs, dtype=np.float32) for qs in solutions]
    dists = [np.linalg.norm(qs - ref_qs) for qs in candidates]
    return candidates[int(np.argmin(dists))]
