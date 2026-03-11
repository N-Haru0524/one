import one.utils.math as oum
import one.collider.mj_collider as ocm
import one.motion.probabilistic.planning_context as omppc
import one.motion.probabilistic.rrt as ompr


class MotionData:
    def __init__(self, qs_list=None):
        self.qs_list = [] if qs_list is None else [
            oum.np.asarray(qs, dtype=oum.np.float32) for qs in qs_list]
        self.events = {}

    def __len__(self):
        return len(self.qs_list)
    
    def __add__(self, other):
        self.qs_list.extend(other.qs_list)
        return self

    def extend(self, qs_list):
        self.qs_list.extend(
            oum.np.asarray(qs, dtype=oum.np.float32) for qs in qs_list)

    def copy(self):
        new = MotionData(self.qs_list)
        new.events = dict(self.events)
        return new


def build_collider(actors, obstacles=None, aux_mecbas=None, margin=0.0):
    collider = ocm.MJCollider()
    for actor in actors:
        collider.append(actor)
    if aux_mecbas:
        for mecba in aux_mecbas:
            collider.append(mecba)
    if obstacles:
        for obstacle in obstacles:
            collider.append(obstacle)
    collider.actors = actors
    collider.compile(margin=margin)
    return collider


def build_planning_context(collider, aux_mecbas=None, max_edge_step=oum.pi / 180):
    joint_limits = None
    actors = getattr(collider, 'actors', ())
    if actors:
        lmt_low_list = []
        lmt_high_list = []
        for actor in actors:
            structure = getattr(actor, 'structure', None)
            compiled = None if structure is None else getattr(structure, 'compiled', None)
            ndof = getattr(actor, 'ndof', None)
            if compiled is None or ndof is None:
                continue
            lmt_low_list.append(oum.np.asarray(compiled.jlmt_low_by_idx[:ndof], dtype=oum.np.float32))
            lmt_high_list.append(oum.np.asarray(compiled.jlmt_high_by_idx[:ndof], dtype=oum.np.float32))
        if lmt_low_list:
            joint_limits = (
                oum.np.concatenate(lmt_low_list).astype(oum.np.float32),
                oum.np.concatenate(lmt_high_list).astype(oum.np.float32),
            )
    return omppc.PlanningContext(
        collider=collider,
        aux_mecbas=aux_mecbas,
        joint_limits=joint_limits,
        cd_step_size=max_edge_step,
    )


def plan_rrt(robot, start_qs, goal_qs, pln_ctx,
             step_size=oum.pi / 120,
             max_iters=2000,
             time_limit=None,
             max_edge_step=oum.pi / 180,
             aux_mecbas=None):
    del robot
    del max_edge_step, aux_mecbas
    planner = ompr.RRTConnectPlanner(
        pln_ctx=pln_ctx,
        extend_step_size=step_size,
    )
    return planner.solve(start=start_qs, goal=goal_qs,
                         max_iters=max_iters, time_limit=time_limit)


def interpolate_qs(start_qs, goal_qs, step_size=oum.pi / 36):
    start_qs = oum.np.asarray(start_qs, dtype=oum.np.float32)
    goal_qs = oum.np.asarray(goal_qs, dtype=oum.np.float32)
    dist = float(oum.np.linalg.norm(goal_qs - start_qs))
    if dist == 0.0:
        return [start_qs.copy()]
    n_steps = max(1, int(oum.np.ceil(dist / float(step_size))))
    path = []
    for i in range(n_steps + 1):
        t = i / n_steps
        path.append(((1.0 - t) * start_qs + t * goal_qs).astype(oum.np.float32))
    return path


def path_is_valid(path, pln_ctx):
    if not path:
        return False
    if not pln_ctx.is_state_valid(path[0]):
        return False
    for qs0, qs1 in zip(path[:-1], path[1:]):
        if not pln_ctx.is_motion_valid(qs0, qs1):
            return False
    return True


def plan_joint_path(start_qs, goal_qs, pln_ctx,
                    use_rrt=True,
                    step_size=oum.pi / 36,
                    max_iters=2000,
                    time_limit=None,
                    max_edge_step=oum.pi / 180):
    start_qs = oum.np.asarray(start_qs, dtype=oum.np.float32)
    goal_qs = oum.np.asarray(goal_qs, dtype=oum.np.float32)
    if use_rrt:
        return plan_rrt(
            None,
            start_qs,
            goal_qs,
            pln_ctx,
            step_size=step_size,
            max_iters=max_iters,
            time_limit=time_limit,
        )
    path = interpolate_qs(start_qs, goal_qs, step_size=step_size)
    del max_edge_step
    if not path_is_valid(path, pln_ctx):
        return None
    return path


def select_ik_solution(solutions, ref_qs):
    if not solutions:
        return None
    ref_qs = oum.np.asarray(ref_qs, dtype=oum.np.float32)
    candidates = [oum.np.asarray(qs, dtype=oum.np.float32) for qs in solutions]
    dists = [oum.np.linalg.norm(qs - ref_qs) for qs in candidates]
    return candidates[int(oum.np.argmin(dists))]
