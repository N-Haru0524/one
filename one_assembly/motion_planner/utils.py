import one.utils.math as oum
import time
import one.collider.mj_collider as ocm
import one.motion.probabilistic.planning_context as omppc
import one.motion.probabilistic.rrt as ompr
import one.robots.base.mech_base as orbmb
import one.scene.scene_object as osso
from one_assembly.precise_collision import PreciseSIMDCollider

DEFAULT_COLLISION_MARGIN = 0.0


class AssemblyFilteredMJCollider:
    def __init__(self):
        self._base = ocm.MJCollider()
        self._relevant_body_ids = set()
        self._actor_link_masks = {}
        self._actor_include_mounted = {}

    def append(self, entity):
        self._base.append(entity)

    def compile(self, margin=0.0):
        self._base.compile(margin=margin)
        self._rebuild_relevant_body_ids()

    def is_collided(self, qs):
        self._base.is_collided(qs)
        return any(True for _ in self.iter_collision_contact_indices())

    def get_slice(self, actor):
        return self._base.get_slice(actor)

    def set_mecba_qpos(self, mecba, qs):
        self._base.set_mecba_qpos(mecba, qs)

    def save(self, filepath, encoding='utf-8'):
        self._base.save(filepath, encoding=encoding)

    @property
    def scene(self):
        return self._base.scene

    @property
    def actors(self):
        return self._base.actors

    @actors.setter
    def actors(self, actors):
        self._base.actors = actors
        self._rebuild_relevant_body_ids()

    @property
    def _mjenv(self):
        return self._base._mjenv

    def iter_collision_contact_indices(self):
        if self._base._mjenv is None:
            return iter(())
        data = self._base._mjenv.data
        model = self._base._mjenv.model
        for cidx in range(int(data.ncon)):
            contact = data.contact[cidx]
            body_id_1 = int(model.geom_bodyid[contact.geom1])
            body_id_2 = int(model.geom_bodyid[contact.geom2])
            if self._is_relevant_contact(body_id_1, body_id_2):
                yield cidx

    def _rebuild_relevant_body_ids(self):
        self._relevant_body_ids.clear()
        if self._base._mjenv is None:
            return
        sync = self._base._mjenv.sync
        model = self._base._mjenv.model
        for actor in self._base.actors:
            self._collect_entity_body_ids(actor, sync, model, self._relevant_body_ids)

    def _collect_entity_body_ids(self, entity, sync, model, body_ids):
        if isinstance(entity, orbmb.MechBase):
            allowed = self._actor_link_masks.get(entity, None)
            for idx, lnk in enumerate(entity.runtime_lnks):
                if allowed is not None and idx not in allowed:
                    continue
                body = sync.rutl2bdy.get(lnk)
                if body is not None:
                    body_ids.add(model.body(body.name).id)
            if self._actor_include_mounted.get(entity, True):
                for child in getattr(entity, '_mountings', {}).keys():
                    self._collect_entity_body_ids(child, sync, model, body_ids)
            return
        if isinstance(entity, osso.SceneObject):
            body = sync.sobj2bdy.get(entity)
            if body is not None:
                body_ids.add(model.body(body.name).id)

    def _is_relevant_contact(self, body_id_1, body_id_2):
        return (
            body_id_1 in self._relevant_body_ids or
            body_id_2 in self._relevant_body_ids
        )


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


def _scene_entities_excluding_actors(collider):
    actors = tuple(getattr(collider, 'actors', ()))
    actor_ids = {id(actor) for actor in actors}
    entities = []
    for sobj in collider.scene.sobjs:
        entities.append(sobj)
    for mecba in collider.scene.mecbas:
        if id(mecba) not in actor_ids:
            entities.append(mecba)
    return actors, entities


def all_inclusive_mujoco_pln_ctx(pln_ctx):
    cached = getattr(pln_ctx, '_all_inclusive_mujoco_pln_ctx', None)
    if cached is not None:
        return cached
    collider = getattr(pln_ctx, 'collider', None)
    if collider is None:
        return pln_ctx
    actors, obstacles = _scene_entities_excluding_actors(collider)
    if not actors:
        return pln_ctx
    rebuilt = build_collider(
        actors=list(actors),
        obstacles=obstacles,
        backend='mujoco',
    )
    cached = build_planning_context(rebuilt, max_edge_step=pln_ctx.cd_step_size)
    pln_ctx._all_inclusive_mujoco_pln_ctx = cached
    return cached


def build_collider(actors,
                   obstacles=None,
                   aux_mecbas=None,
                   margin=DEFAULT_COLLISION_MARGIN,
                   backend='mujoco',
                   actor_link_masks=None,
                   actor_include_mounted=None):
    if backend == 'mujoco':
        collider = AssemblyFilteredMJCollider()
    elif backend == 'simd':
        collider = PreciseSIMDCollider(use_gpu=True)
    else:
        raise ValueError(f'Unsupported collider backend: {backend}')
    if actor_link_masks is not None:
        collider._actor_link_masks = {
            actor: None if mask is None else set(mask)
            for actor, mask in actor_link_masks.items()
        }
    if actor_include_mounted is not None:
        collider._actor_include_mounted = dict(actor_include_mounted)
    for actor in actors:
        collider.append(actor)
    if aux_mecbas:
        for mecba in aux_mecbas:
            collider.append(mecba)
    if obstacles:
        for obstacle in obstacles:
            collider.append(obstacle)
    collider.actors = actors
    if backend == 'mujoco':
        collider.compile(margin=margin)
    else:
        collider.compile()
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
             aux_mecbas=None,
             timing_hook=None,
             timing_label='joint_path.rrt_connect.solve'):
    del robot
    del max_edge_step, aux_mecbas
    pln_ctx = all_inclusive_mujoco_pln_ctx(pln_ctx)
    planner = ompr.RRTConnectPlanner(
        pln_ctx=pln_ctx,
        extend_step_size=step_size,
    )
    start_time = time.perf_counter()
    path = planner.solve(start=start_qs, goal=goal_qs,
                         max_iters=max_iters, time_limit=time_limit)
    if timing_hook is not None:
        timing_hook(timing_label, time.perf_counter() - start_time)
    return path


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
                    max_edge_step=oum.pi / 180,
                    timing_hook=None,
                    timing_label='joint_path.rrt_connect.solve'):
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
            timing_hook=timing_hook,
            timing_label=timing_label,
        )
    interp_start_time = time.perf_counter()
    path = interpolate_qs(start_qs, goal_qs, step_size=step_size)
    if timing_hook is not None:
        timing_hook('joint_path.interpolate', time.perf_counter() - interp_start_time)
    del max_edge_step
    validate_start_time = time.perf_counter()
    if not path_is_valid(path, pln_ctx):
        if timing_hook is not None:
            timing_hook('joint_path.interpolate_validate', time.perf_counter() - validate_start_time)
        return None
    if timing_hook is not None:
        timing_hook('joint_path.interpolate_validate', time.perf_counter() - validate_start_time)
    return path


def select_ik_solution(solutions, ref_qs):
    if not solutions:
        return None
    ref_qs = oum.np.asarray(ref_qs, dtype=oum.np.float32)
    candidates = [oum.np.asarray(qs, dtype=oum.np.float32) for qs in solutions]
    dists = [oum.np.linalg.norm(qs - ref_qs) for qs in candidates]
    return candidates[int(oum.np.argmin(dists))]
