import importlib.util
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUM_SPEC = importlib.util.spec_from_file_location('_test_oum', ROOT / 'one' / 'utils' / 'math.py')
oum = importlib.util.module_from_spec(OUM_SPEC)
OUM_SPEC.loader.exec_module(oum)

UTILS_PATH = ROOT / 'one_assembly' / 'motion_planner' / 'utils.py'
PLANNER_PATH = ROOT / 'one_assembly' / 'motion_planner' / 'approach_depart_planner.py'
HIER_PATH = ROOT / 'one_assembly' / 'motion_planner' / 'hierarchical.py'
PP_PATH = ROOT / 'one_assembly' / 'motion_planner' / 'ppplanner.py'
SCREW_PATH = ROOT / 'one_assembly' / 'motion_planner' / 'screwplanner.py'
FOLD_PATH = ROOT / 'one_assembly' / 'motion_planner' / 'foldplanner.py'


def _reset_modules():
    for name in list(sys.modules):
        if name == 'one' or name.startswith('one.') or name == 'one_assembly' or name.startswith('one_assembly.'):
            sys.modules.pop(name, None)


def _make_package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


class FakeMJCollider:
    def __init__(self):
        self.entities = []
        self._actors = ()
        self.scene = types.SimpleNamespace(sobjs=[], mecbas=[])
        self.collision_predicate = lambda qs: False

    def append(self, entity):
        self.entities.append(entity)
        if hasattr(entity, 'ndof'):
            self.scene.mecbas.append(entity)
        else:
            self.scene.sobjs.append(entity)

    def compile(self, margin=0.0):
        self.margin = margin

    def is_collided(self, qs):
        return bool(self.collision_predicate(oum.np.asarray(qs, dtype=oum.np.float32)))

    @property
    def actors(self):
        return self._actors

    @actors.setter
    def actors(self, value):
        self._actors = tuple(value)


class FakePlanningContext:
    def __init__(self, collider, planning_mecbas=None, aux_mecbas=None,
                 joint_limits=None, cd_step_size=oum.pi / 180, cache_size=10000):
        del planning_mecbas, joint_limits, cache_size
        self.collider = collider
        self.aux_mecbas = aux_mecbas or {}
        self.cd_step_size = cd_step_size

    def is_state_valid(self, state):
        state = oum.np.asarray(state, dtype=oum.np.float32)
        return bool(oum.np.all(oum.np.abs(state) <= oum.pi)) and not self.collider.is_collided(state)

    def is_motion_valid(self, state1, state2):
        return self.is_state_valid(state1) and self.is_state_valid(state2)

    def distance(self, state1, state2):
        return float(oum.np.linalg.norm(oum.np.asarray(state2) - oum.np.asarray(state1)))

    def interpolate(self, state1, state2, t):
        state1 = oum.np.asarray(state1, dtype=oum.np.float32)
        state2 = oum.np.asarray(state2, dtype=oum.np.float32)
        return ((1.0 - t) * state1 + t * state2).astype(oum.np.float32)

    def enforce_bounds(self, state):
        return oum.np.asarray(state, dtype=oum.np.float32)

    def clear_cache(self):
        return None

    def sample_uniform(self):
        return oum.np.zeros(3, dtype=oum.np.float32)

    def states_equal(self, state1, state2, tol=1e-4):
        return self.distance(state1, state2) <= tol


class FakeRRTConnectPlanner:
    def __init__(self, pln_ctx, extend_step_size=oum.pi / 36, goal_bias=0.7):
        del goal_bias
        self._pln_ctx = pln_ctx
        self._step = float(oum.np.asarray(extend_step_size).reshape(-1)[0])

    def solve(self, start, goal, max_iters=1000, time_limit=None):
        del max_iters, time_limit
        start = oum.np.asarray(start, dtype=oum.np.float32)
        goal = oum.np.asarray(goal, dtype=oum.np.float32)
        dist = self._pln_ctx.distance(start, goal)
        n_steps = max(1, int(oum.np.ceil(dist / self._step)))
        return [
            self._pln_ctx.interpolate(start, goal, i / n_steps)
            for i in range(n_steps + 1)
        ]


def fake_cartesian_to_jtraj(robot,
                            start_rotmat,
                            start_pos,
                            goal_rotmat=None,
                            goal_pos=None,
                            pos_step=0.01,
                            rot_step=oum.np.deg2rad(2.0),
                            ref_qs=None):
    del robot, start_rotmat, goal_rotmat, rot_step, ref_qs
    start_pos = oum.np.asarray(start_pos, dtype=oum.np.float32)
    goal_pos = start_pos if goal_pos is None else oum.np.asarray(goal_pos, dtype=oum.np.float32)
    dist = float(oum.np.linalg.norm(goal_pos - start_pos))
    n_steps = max(1, int(oum.np.ceil(dist / pos_step)))
    q_seq = oum.np.linspace(start_pos, goal_pos, n_steps + 1, dtype=oum.np.float32)
    rot_seq = oum.np.repeat(oum.np.eye(3, dtype=oum.np.float32)[None, :, :], len(q_seq), axis=0)
    return q_seq, (q_seq, rot_seq)


def fake_tf_from_rotmat_pos(rotmat, pos):
    tf = oum.np.eye(4, dtype=oum.np.float32)
    tf[:3, :3] = oum.np.asarray(rotmat, dtype=oum.np.float32)
    tf[:3, 3] = oum.np.asarray(pos, dtype=oum.np.float32)
    return tf


def fake_orth_vec(vec, toggle_unit=True):
    vec = oum.np.asarray(vec, dtype=oum.np.float32).reshape(-1)
    ref = oum.vec(1.0, 0.0, 0.0).astype(oum.np.float32)
    if abs(float(oum.np.dot(vec, ref))) > 0.9:
        ref = oum.vec(0.0, 1.0, 0.0).astype(oum.np.float32)
    out = oum.np.cross(vec, ref).astype(oum.np.float32)
    if not toggle_unit:
        return out
    norm = float(oum.np.linalg.norm(out))
    return out if norm <= 0.0 else (out / norm).astype(oum.np.float32)


def fake_rotmat_from_axangle(ax, angle):
    ax = oum.np.asarray(ax, dtype=oum.np.float32).reshape(-1)
    ax_norm = float(oum.np.linalg.norm(ax))
    if ax_norm <= 0.0:
        return oum.np.eye(3, dtype=oum.np.float32)
    ax = ax / ax_norm
    x, y, z = ax
    c = float(oum.np.cos(angle))
    s = float(oum.np.sin(angle))
    one_c = 1.0 - c
    return oum.np.array([
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ], dtype=oum.np.float32)


def fake_quat_from_rotmat(rotmat):
    return oum.quat_from_rotmat(rotmat)


def fake_rotmat_from_quat(quat):
    return oum.rotmat_from_quat(quat).astype(oum.np.float32)


def fake_slerp_quat(quat0, quat1, fraction, spin=0, shortestpath=True):
    return oum.slerp_quat(quat0, quat1, fraction, spin=spin, shortestpath=shortestpath)


def fake_axangle_between_rotmat(rotmati, rotmatj):
    return oum.axangle_between_rotmat(rotmati, rotmatj)


def fake_rotmat_slerp(rotmat0, rotmat1, n):
    return oum.rotmat_slerp(rotmat0, rotmat1, n).astype(oum.np.float32)


class FakeRobot:
    clone_count = 0

    def __init__(self):
        self.ndof = 3
        self.qs = oum.np.zeros(3, dtype=oum.np.float32)
        self.gl_tcp_tf = oum.np.eye(4, dtype=oum.np.float32)

    def clone(self):
        FakeRobot.clone_count += 1
        new = FakeRobot()
        new.qs = self.qs.copy()
        new.gl_tcp_tf = self.gl_tcp_tf.copy()
        return new

    def fk(self, qs=None):
        if qs is not None:
            self.qs = oum.np.asarray(qs, dtype=oum.np.float32)
        self.gl_tcp_tf = oum.np.eye(4, dtype=oum.np.float32)
        self.gl_tcp_tf[:3, 3] = self.qs
        return self.gl_tcp_tf

    def ik_tcp_nearest(self, tgt_rotmat, tgt_pos, ref_qs=None):
        del tgt_rotmat, ref_qs
        return oum.np.asarray(tgt_pos, dtype=oum.np.float32)

    def ik_tcp(self, tgt_rotmat, tgt_pos):
        del tgt_rotmat
        tgt_pos = oum.np.asarray(tgt_pos, dtype=oum.np.float32)
        return [
            tgt_pos.copy(),
            (tgt_pos + oum.vec(0.2, 0.0, 0.0).astype(oum.np.float32)).astype(oum.np.float32),
        ]


class FakeEEActor:
    def __init__(self):
        self.ndof = 2
        self.qs = oum.np.zeros(2, dtype=oum.np.float32)

    def clone(self):
        new = FakeEEActor()
        new.qs = self.qs.copy()
        return new

    def set_jaw_width(self, width):
        self.qs[:] = width * 0.5


def load_modules():
    _reset_modules()
    _make_package('one')
    _make_package('one.collider')
    _make_package('one.motion')
    _make_package('one.motion.probabilistic')
    _make_package('one.motion.trajectory')
    _make_package('one.robots')
    _make_package('one.robots.manipulators')
    _make_package('one.utils')
    _make_package('one_assembly')
    _make_package('one_assembly.motion_planner')

    oum_mod = types.ModuleType('one.oum')
    oum_mod.tf_from_rotmat_pos = fake_tf_from_rotmat_pos
    oum_mod.orth_vec = fake_orth_vec
    oum_mod.rotmat_from_axangle = fake_rotmat_from_axangle
    oum_mod.quat_from_rotmat = fake_quat_from_rotmat
    oum_mod.rotmat_from_quat = fake_rotmat_from_quat
    oum_mod.slerp_quat = fake_slerp_quat
    oum_mod.axangle_between_rotmat = fake_axangle_between_rotmat
    oum_mod.rotmat_slerp = fake_rotmat_slerp
    oum_mod.np = oum.np
    oum_mod.pi = oum.pi
    sys.modules[oum_mod.__name__] = oum_mod
    sys.modules['one'].oum = oum_mod

    utils_math_mod = types.ModuleType('one.utils.math')
    utils_math_mod.tf_from_rotmat_pos = fake_tf_from_rotmat_pos
    utils_math_mod.orth_vec = fake_orth_vec
    utils_math_mod.rotmat_from_axangle = fake_rotmat_from_axangle
    utils_math_mod.quat_from_rotmat = fake_quat_from_rotmat
    utils_math_mod.rotmat_from_quat = fake_rotmat_from_quat
    utils_math_mod.slerp_quat = fake_slerp_quat
    utils_math_mod.axangle_between_rotmat = fake_axangle_between_rotmat
    utils_math_mod.rotmat_slerp = fake_rotmat_slerp
    utils_math_mod.np = oum.np
    utils_math_mod.pi = oum.pi
    sys.modules[utils_math_mod.__name__] = utils_math_mod
    sys.modules['one.utils'].math = utils_math_mod

    mj_mod = types.ModuleType('one.collider.mj_collider')
    mj_mod.MJCollider = FakeMJCollider
    sys.modules[mj_mod.__name__] = mj_mod

    pc_mod = types.ModuleType('one.motion.probabilistic.planning_context')
    pc_mod.PlanningContext = FakePlanningContext
    sys.modules[pc_mod.__name__] = pc_mod

    rrt_mod = types.ModuleType('one.motion.probabilistic.rrt')
    rrt_mod.RRTConnectPlanner = FakeRRTConnectPlanner
    sys.modules[rrt_mod.__name__] = rrt_mod

    cart_mod = types.ModuleType('one.motion.trajectory.cartesian')
    cart_mod.cartesian_to_jtraj = fake_cartesian_to_jtraj
    sys.modules[cart_mod.__name__] = cart_mod

    manip_mod = types.ModuleType('one.robots.manipulators.manipulator_base')
    manip_mod.ManipulatorBase = FakeRobot
    sys.modules[manip_mod.__name__] = manip_mod

    utils_spec = importlib.util.spec_from_file_location(
        'one_assembly.motion_planner.utils',
        UTILS_PATH,
    )
    utils_mod = importlib.util.module_from_spec(utils_spec)
    sys.modules[utils_spec.name] = utils_mod
    utils_spec.loader.exec_module(utils_mod)

    planner_spec = importlib.util.spec_from_file_location(
        'one_assembly.motion_planner.approach_depart_planner',
        PLANNER_PATH,
    )
    planner_mod = importlib.util.module_from_spec(planner_spec)
    sys.modules[planner_spec.name] = planner_mod
    planner_spec.loader.exec_module(planner_mod)

    hier_spec = importlib.util.spec_from_file_location(
        'one_assembly.motion_planner.hierarchical',
        HIER_PATH,
    )
    hier_mod = importlib.util.module_from_spec(hier_spec)
    sys.modules[hier_spec.name] = hier_mod
    hier_spec.loader.exec_module(hier_mod)

    pp_spec = importlib.util.spec_from_file_location(
        'one_assembly.motion_planner.ppplanner',
        PP_PATH,
    )
    pp_mod = importlib.util.module_from_spec(pp_spec)
    sys.modules[pp_spec.name] = pp_mod
    pp_spec.loader.exec_module(pp_mod)

    screw_spec = importlib.util.spec_from_file_location(
        'one_assembly.motion_planner.screwplanner',
        SCREW_PATH,
    )
    screw_mod = importlib.util.module_from_spec(screw_spec)
    sys.modules[screw_spec.name] = screw_mod
    screw_spec.loader.exec_module(screw_mod)

    fold_spec = importlib.util.spec_from_file_location(
        'one_assembly.motion_planner.foldplanner',
        FOLD_PATH,
    )
    fold_mod = importlib.util.module_from_spec(fold_spec)
    sys.modules[fold_spec.name] = fold_mod
    fold_spec.loader.exec_module(fold_mod)
    return utils_mod, planner_mod, pp_mod, screw_mod, fold_mod


class ADPlannerSmokeTest(unittest.TestCase):
    def _make_grasp(self, pos, jaw_width=0.04):
        pose_tf = oum.np.eye(4, dtype=oum.np.float32)
        pose_tf[:3, 3] = oum.np.asarray(pos, dtype=oum.np.float32)
        pre_pose_tf = pose_tf.copy()
        pre_pose_tf[2, 3] += 0.05
        return pose_tf, pre_pose_tf, float(jaw_width), 1.0

    def setUp(self):
        self.utils_mod, self.planner_mod, self.pp_mod, self.screw_mod, self.fold_mod = load_modules()
        FakeRobot.clone_count = 0
        self.robot = FakeRobot()
        self.ee_actor = FakeEEActor()
        self.pln_ctx = self.utils_mod.build_planning_context(
            self.utils_mod.build_collider([self.robot, self.ee_actor])
        )
        self.planner = self.planner_mod.ADPlanner(
            robot=self.robot,
            pln_ctx=self.pln_ctx,
            ee_actor=self.ee_actor,
        )

    def test_linear_approach_generates_motion_plan(self):
        plan = self.planner.gen_linear_approach(
            goal_tcp_pos=oum.vec(0.3, 0.0, 0.0).astype(oum.np.float32),
            goal_tcp_rotmat=oum.np.eye(3, dtype=oum.np.float32),
            distance=0.1,
            granularity=0.05,
            ee_values=oum.np.array([0.01, 0.01], dtype=oum.np.float32),
        )
        self.assertIsNotNone(plan)
        self.assertGreaterEqual(len(plan), 3)
        oum.np.testing.assert_allclose(plan.qs_list[0], [0.3, 0.0, -0.1, 0.01, 0.01], atol=1e-6)
        oum.np.testing.assert_allclose(plan.qs_list[-1], [0.3, 0.0, 0.0, 0.01, 0.01], atol=1e-6)

    def test_interpolated_linear_motion_generates_motion_plan(self):
        planner = self.fold_mod.FoldPlanner(self.robot, pln_ctx=self.pln_ctx, ee_actor=self.ee_actor)
        plan = planner.gen_linear_motion(
            start_tcp_pos=oum.vec(0.0, 0.0, 0.0).astype(oum.np.float32),
            start_tcp_rotmat=oum.np.eye(3, dtype=oum.np.float32),
            goal_tcp_pos=oum.vec(0.2, 0.0, 0.0).astype(oum.np.float32),
            goal_tcp_rotmat=oum.np.eye(3, dtype=oum.np.float32),
            granularity=0.05,
            ee_values=oum.np.array([0.01, 0.01], dtype=oum.np.float32),
        )
        self.assertIsNotNone(plan)
        self.assertGreaterEqual(len(plan.qs_list), 2)
        oum.np.testing.assert_allclose(plan.qs_list[0], [0.0, 0.0, 0.0, 0.01, 0.01], atol=1e-6)
        oum.np.testing.assert_allclose(plan.qs_list[-1], [0.2, 0.0, 0.0, 0.01, 0.01], atol=1e-6)

    def test_interpolated_piecewise_motion_connects_all_waypoints(self):
        planner = self.fold_mod.FoldPlanner(self.robot, pln_ctx=self.pln_ctx, ee_actor=self.ee_actor)
        plan = planner.gen_piecewise_motion(
            start_tcp_pos=oum.vec(0.0, 0.0, 0.0).astype(oum.np.float32),
            start_tcp_rotmat=oum.np.eye(3, dtype=oum.np.float32),
            goal_tcp_pos_list=[
                oum.vec(0.1, 0.0, 0.0).astype(oum.np.float32),
                oum.vec(0.2, 0.1, 0.0).astype(oum.np.float32),
            ],
            goal_tcp_rotmat_list=[
                oum.np.eye(3, dtype=oum.np.float32),
                oum.np.eye(3, dtype=oum.np.float32),
            ],
            granularity=0.05,
            ee_values=oum.np.array([0.02, 0.02], dtype=oum.np.float32),
        )
        self.assertIsNotNone(plan)
        oum.np.testing.assert_allclose(plan.qs_list[-1], [0.2, 0.1, 0.0, 0.02, 0.02], atol=1e-6)

    def test_interpolated_piecewise_motion_starts_from_seed_qs(self):
        planner = self.fold_mod.FoldPlanner(self.robot, pln_ctx=self.pln_ctx, ee_actor=self.ee_actor)
        plan = planner.gen_piecewise_motion(
            start_tcp_pos=oum.vec(0.0, 0.0, 0.0).astype(oum.np.float32),
            start_tcp_rotmat=oum.np.eye(3, dtype=oum.np.float32),
            goal_tcp_pos_list=[oum.vec(0.1, 0.0, 0.0).astype(oum.np.float32)],
            goal_tcp_rotmat_list=[oum.np.eye(3, dtype=oum.np.float32)],
            granularity=0.05,
            ee_values=oum.np.array([0.02, 0.02], dtype=oum.np.float32),
            ref_qs=oum.np.array([0.8, 0.1, -0.2], dtype=oum.np.float32),
        )
        self.assertIsNotNone(plan)
        oum.np.testing.assert_allclose(plan.qs_list[0], [0.8, 0.1, -0.2, 0.02, 0.02], atol=1e-6)

    def test_interpolated_piecewise_motion_uses_given_ref_qs(self):
        planner = self.fold_mod.FoldPlanner(self.robot, pln_ctx=self.pln_ctx, ee_actor=self.ee_actor)
        captured = {}
        original_cartesian = self.fold_mod.omtc.cartesian_to_jtraj

        def record_cartesian_to_jtraj(*args, **kwargs):
            if 'first_ref_qs' not in captured:
                captured['first_ref_qs'] = oum.np.asarray(kwargs['ref_qs'], dtype=oum.np.float32).copy()
            return original_cartesian(*args, **kwargs)

        self.fold_mod.omtc.cartesian_to_jtraj = record_cartesian_to_jtraj
        planner.gen_piecewise_motion(
            start_tcp_pos=oum.vec(0.0, 0.0, 0.0).astype(oum.np.float32),
            start_tcp_rotmat=oum.np.eye(3, dtype=oum.np.float32),
            goal_tcp_pos_list=[oum.vec(0.1, 0.0, 0.0).astype(oum.np.float32)],
            goal_tcp_rotmat_list=[oum.np.eye(3, dtype=oum.np.float32)],
            granularity=0.05,
            ref_qs=oum.np.array([0.8, 0.1, -0.2], dtype=oum.np.float32),
        )
        self.fold_mod.omtc.cartesian_to_jtraj = original_cartesian
        oum.np.testing.assert_allclose(captured['first_ref_qs'], [0.8, 0.1, -0.2], atol=1e-6)

    def test_approach_depart_runs_end_to_end(self):
        plan = self.planner.gen_approach_depart(
            goal_qs=oum.np.array([0.3, 0.0, 0.0, 0.02, 0.02], dtype=oum.np.float32),
            start_qs=oum.np.array([0.0, 0.0, 0.0, 0.04, 0.04], dtype=oum.np.float32),
            end_qs=oum.np.array([0.5, 0.2, 0.0, 0.01, 0.01], dtype=oum.np.float32),
            approach_distance=0.1,
            depart_distance=0.1,
            approach_linear=True,
            depart_linear=True,
            linear_granularity=0.05,
            use_rrt=False,
        )
        self.assertIsNotNone(plan)
        oum.np.testing.assert_allclose(plan.qs_list[0], [0.0, 0.0, 0.0, 0.04, 0.04], atol=1e-6)
        oum.np.testing.assert_allclose(plan.qs_list[-1], [0.5, 0.2, 0.0, 0.01, 0.01], atol=1e-6)

    def test_connect_motion_reorients_reversed_path(self):
        planner = self.planner_mod.ADPlanner(
            robot=self.robot,
            pln_ctx=self.pln_ctx,
            ee_actor=self.ee_actor,
        )
        original_plan_joint_path = self.utils_mod.plan_joint_path

        def reversed_plan_joint_path(*args, **kwargs):
            path = original_plan_joint_path(*args, **kwargs)
            return list(reversed(path))

        self.utils_mod.plan_joint_path = reversed_plan_joint_path
        plan = planner._connect_motion(
            start_qs=oum.np.array([0.0, 0.0, 0.0, 0.01, 0.01], dtype=oum.np.float32),
            goal_qs=oum.np.array([0.2, 0.0, 0.0, 0.01, 0.01], dtype=oum.np.float32),
            pln_ctx=self.pln_ctx,
            use_rrt=False,
        )
        self.utils_mod.plan_joint_path = original_plan_joint_path
        self.assertIsNotNone(plan)
        oum.np.testing.assert_allclose(plan.qs_list[0], [0.0, 0.0, 0.0, 0.01, 0.01], atol=1e-6)
        oum.np.testing.assert_allclose(plan.qs_list[-1], [0.2, 0.0, 0.0, 0.01, 0.01], atol=1e-6)

    def test_utils_build_collider_includes_aux_mecbas(self):
        aux_actor = object()
        collider = self.utils_mod.build_collider(
            actors=[self.robot],
            obstacles=['obstacle'],
            aux_mecbas={aux_actor: oum.np.array([0.01, 0.01], dtype=oum.np.float32)},
        )
        self.assertEqual(collider.entities[0], self.robot)
        self.assertEqual(collider.entities[1], aux_actor)
        self.assertEqual(collider.entities[2], 'obstacle')

    def test_pickplace_reason_common_grasp_ids_uses_unified_collider(self):
        grasp0 = self._make_grasp([0.1, 0.0, 0.0])
        grasp1 = self._make_grasp([0.3, 0.0, 0.0])
        collider = self.utils_mod.build_collider([self.robot, self.ee_actor])
        collider.collision_predicate = lambda qs: bool(oum.np.asarray(qs, dtype=oum.np.float32)[0] < 0.2)
        planner = self.pp_mod.PickPlacePlanner(
            self.robot,
            pln_ctx=self.utils_mod.build_planning_context(collider),
            ee_actor=self.ee_actor,
        )
        self.robot.ik_tcp = lambda tgt_rotmat, tgt_pos: [oum.np.asarray(tgt_pos, dtype=oum.np.float32)]
        common_gids = planner.reason_common_grasp_ids(
            grasp_collection=[grasp0, grasp1],
            goal_pose_list=[(oum.np.zeros(3, dtype=oum.np.float32), oum.np.eye(3, dtype=oum.np.float32))],
            pick_pose=(oum.np.zeros(3, dtype=oum.np.float32), oum.np.eye(3, dtype=oum.np.float32)),
        )
        self.assertEqual(common_gids, [1])

    def test_screw_reason_common_sids_uses_unified_collider(self):
        collider = self.utils_mod.build_collider([self.robot])
        collider.collision_predicate = lambda qs: bool(oum.np.asarray(qs, dtype=oum.np.float32)[0] < 0.2)
        planner = self.screw_mod.ScrewPlanner(
            self.robot,
            pln_ctx=self.utils_mod.build_planning_context(collider),
        )
        self.robot.ik_tcp = lambda tgt_rotmat, tgt_pos: [oum.np.asarray(tgt_pos, dtype=oum.np.float32)]
        goal_pose_list = [
            (oum.vec(0.1, 0.0, 0.0).astype(oum.np.float32), oum.np.eye(3, dtype=oum.np.float32)),
            (oum.vec(0.3, 0.0, 0.0).astype(oum.np.float32), oum.np.eye(3, dtype=oum.np.float32)),
        ]
        common_sids = planner.reason_common_sids(
            goal_pose_list=goal_pose_list,
        )
        self.assertEqual(common_sids, [1])

    def test_screw_reason_common_sids_passes_exclude_entities(self):
        planner = self.screw_mod.ScrewPlanner(self.robot, pln_ctx=self.pln_ctx)
        sentinel = object()
        captured = {}
        original_screen_pose_with_stats = planner._screen_pose_with_stats

        def fake_screen_pose_with_stats(*args, **kwargs):
            captured.update(kwargs)
            return original_screen_pose_with_stats(*args, **kwargs)

        planner._screen_pose_with_stats = fake_screen_pose_with_stats
        common_sids = planner.reason_common_sids(
            goal_pose_list=[(oum.vec(0.3, 0.0, 0.0).astype(oum.np.float32), oum.np.eye(3, dtype=oum.np.float32))],
            exclude_entities=[sentinel],
        )
        self.assertEqual(common_sids, [0])
        self.assertIn('exclude_entities', captured)
        self.assertIn('survived_sids', planner._last_reason_common_screw_report)

    def test_screw_reason_common_sids_records_failure_location(self):
        collider = self.utils_mod.build_collider([self.robot])
        collider.collision_predicate = lambda qs: bool(oum.np.asarray(qs, dtype=oum.np.float32)[0] < 0.2)
        planner = self.screw_mod.ScrewPlanner(
            self.robot,
            pln_ctx=self.utils_mod.build_planning_context(collider),
        )
        self.robot.ik_tcp = lambda tgt_rotmat, tgt_pos: [oum.np.asarray(tgt_pos, dtype=oum.np.float32)]
        goal_pose_list = [
            (oum.vec(0.1, 0.0, 0.0).astype(oum.np.float32), oum.np.eye(3, dtype=oum.np.float32)),
            (oum.vec(0.3, 0.0, 0.0).astype(oum.np.float32), oum.np.eye(3, dtype=oum.np.float32)),
        ]
        common_sids = planner.reason_common_sids(
            goal_pose_list=goal_pose_list,
        )
        self.assertEqual(common_sids, [1])
        failure = planner._last_reason_common_screw_report['failures'][0]
        self.assertEqual(failure['label'], 'screw_goal')
        self.assertEqual(failure['reason'], 'goal_in_collision')

    def test_screw_goal_pose_generation_spans_requested_resolution(self):
        planner = self.screw_mod.ScrewPlanner(
            self.robot,
            pln_ctx=self.pln_ctx,
        )
        goal_pose_list = planner.gen_goal_pose_list(
            tgt_pos=oum.vec(0.2, 0.1, 0.3).astype(oum.np.float32),
            tgt_vec=oum.vec(0.0, 0.0, 1.0).astype(oum.np.float32),
            resolution=8,
        )
        self.assertEqual(len(goal_pose_list), 8)
        for pos, rotmat in goal_pose_list:
            oum.np.testing.assert_allclose(pos, [0.2, 0.1, 0.3], atol=1e-6)
            oum.np.testing.assert_allclose(rotmat[:, 2], [0.0, 0.0, 1.0], atol=1e-6)

    def test_gen_screw_accepts_axis_input_and_records_sid(self):
        planner = self.screw_mod.ScrewPlanner(
            self.robot,
            pln_ctx=self.pln_ctx,
        )
        plan = planner.gen_screw(
            start_qs=oum.np.zeros(3, dtype=oum.np.float32),
            tgt_pos=oum.vec(0.3, 0.0, 0.0).astype(oum.np.float32),
            tgt_vec=oum.vec(0.0, 0.0, 1.0).astype(oum.np.float32),
            resolution=6,
            approach_distance=0.05,
            depart_distance=0.0,
            linear_granularity=0.05,
            use_rrt=False,
        )
        self.assertIsNotNone(plan)
        self.assertIn('sid', plan.events)
        oum.np.testing.assert_allclose(plan.qs_list[0], [0.0, 0.0, 0.0], atol=1e-6)

    def test_pickplace_accepts_planning_context_directly(self):
        grasp = self._make_grasp([0.3, 0.0, 0.0])
        collider = self.utils_mod.build_collider([self.robot, self.ee_actor])
        pln_ctx = self.utils_mod.build_planning_context(collider)
        planner = self.pp_mod.PickPlacePlanner(self.robot, pln_ctx=pln_ctx, ee_actor=self.ee_actor)
        self.robot.ik_tcp = lambda tgt_rotmat, tgt_pos: [oum.np.asarray(tgt_pos, dtype=oum.np.float32)]
        common_gids = planner.reason_common_grasp_ids(
            grasp_collection=[grasp],
            goal_pose_list=[(oum.np.zeros(3, dtype=oum.np.float32), oum.np.eye(3, dtype=oum.np.float32))],
            pick_pose=(oum.np.zeros(3, dtype=oum.np.float32), oum.np.eye(3, dtype=oum.np.float32)),
        )
        self.assertEqual(common_gids, [0])

    def test_pickplace_reason_common_grasp_ids_passes_exclude_entities(self):
        planner = self.pp_mod.PickPlacePlanner(self.robot, pln_ctx=self.pln_ctx, ee_actor=self.ee_actor)
        grasp = self._make_grasp([0.3, 0.0, 0.0])
        sentinel = object()
        captured = {}
        original_screen_pose_with_stats = planner._screen_pose_with_stats

        def fake_screen_pose_with_stats(*args, **kwargs):
            captured.update(kwargs)
            return original_screen_pose_with_stats(*args, **kwargs)

        planner._screen_pose_with_stats = fake_screen_pose_with_stats
        common_gids = planner.reason_common_grasp_ids(
            grasp_collection=[grasp],
            goal_pose_list=[(oum.np.zeros(3, dtype=oum.np.float32), oum.np.eye(3, dtype=oum.np.float32))],
            pick_pose=(oum.np.zeros(3, dtype=oum.np.float32), oum.np.eye(3, dtype=oum.np.float32)),
            exclude_entities=[sentinel],
        )
        self.assertEqual(common_gids, [0])
        self.assertIn('survived_gids', planner._last_reason_common_grasp_report)

    def test_pickplace_reason_common_grasp_ids_records_failure_location(self):
        grasp0 = self._make_grasp([0.1, 0.0, 0.0])
        grasp1 = self._make_grasp([0.3, 0.0, 0.0])
        collider = self.utils_mod.build_collider([self.robot, self.ee_actor])
        collider.collision_predicate = lambda qs: bool(oum.np.asarray(qs, dtype=oum.np.float32)[0] < 0.2)
        planner = self.pp_mod.PickPlacePlanner(
            self.robot,
            pln_ctx=self.utils_mod.build_planning_context(collider),
            ee_actor=self.ee_actor,
        )
        self.robot.ik_tcp = lambda tgt_rotmat, tgt_pos: [oum.np.asarray(tgt_pos, dtype=oum.np.float32)]
        common_gids = planner.reason_common_grasp_ids(
            grasp_collection=[grasp0, grasp1],
            goal_pose_list=[(oum.np.zeros(3, dtype=oum.np.float32), oum.np.eye(3, dtype=oum.np.float32))],
            pick_pose=(oum.np.zeros(3, dtype=oum.np.float32), oum.np.eye(3, dtype=oum.np.float32)),
        )
        self.assertEqual(common_gids, [1])
        failure = planner._last_reason_common_grasp_report['failures'][0]
        self.assertEqual(failure['label'], 'pick_approach_start')
        self.assertEqual(failure['reason'], 'goal_in_collision')

    def test_filtered_pln_ctx_is_cached(self):
        planner = self.pp_mod.PickPlacePlanner(self.robot, pln_ctx=self.pln_ctx, ee_actor=self.ee_actor)
        obj = object()
        first = planner._filtered_pln_ctx(exclude_entities=[obj])
        second = planner._filtered_pln_ctx(exclude_entities=[obj])
        self.assertIs(first, second)

    def test_tcp_pose_query_reuses_scratch_robot(self):
        _ = self.planner._tcp_pose_from_qs(oum.np.array([0.1, 0.0, 0.0, 0.01, 0.01], dtype=oum.np.float32))
        _ = self.planner._tcp_pose_from_qs(oum.np.array([0.2, 0.0, 0.0, 0.01, 0.01], dtype=oum.np.float32))
        self.assertEqual(FakeRobot.clone_count, 1)

    def test_pickplace_reuses_reasoned_place_candidate(self):
        planner = self.pp_mod.PickPlacePlanner(self.robot, pln_ctx=self.pln_ctx, ee_actor=self.ee_actor)
        obj = types.SimpleNamespace(
            pos=oum.np.zeros(3, dtype=oum.np.float32),
            rotmat=oum.np.eye(3, dtype=oum.np.float32),
        )
        grasp = self._make_grasp([0.3, 0.0, 0.0])
        screen_calls = {'count': 0}
        original_screen_pose = planner._screen_pose_with_stats
        original_plan = planner.gen_approach_via_pose

        def counting_screen_pose(*args, **kwargs):
            screen_calls['count'] += 1
            return original_screen_pose(*args, **kwargs)

        def fake_plan(*args, **kwargs):
            del args, kwargs
            return self.utils_mod.MotionData([
                oum.np.zeros(5, dtype=oum.np.float32),
                oum.np.ones(5, dtype=oum.np.float32),
            ])

        planner._screen_pose_with_stats = counting_screen_pose
        planner.gen_approach_via_pose = fake_plan
        plan = planner.gen_pick_and_place(
            obj_model=obj,
            grasp_collection=[grasp],
            goal_pose_list=[(oum.np.zeros(3, dtype=oum.np.float32), oum.np.eye(3, dtype=oum.np.float32))],
            reason_grasps=True,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(screen_calls['count'], 6)

    def test_fold_reason_common_grasp_ids_uses_unified_collider(self):
        grasp0 = self._make_grasp([0.1, 0.0, 0.0])
        grasp1 = self._make_grasp([0.3, 0.0, 0.0])
        collider = self.utils_mod.build_collider([self.robot, self.ee_actor])
        collider.collision_predicate = lambda qs: bool(oum.np.asarray(qs, dtype=oum.np.float32)[0] < 0.2)
        planner = self.fold_mod.FoldPlanner(
            self.robot,
            pln_ctx=self.utils_mod.build_planning_context(collider),
            ee_actor=self.ee_actor,
        )
        self.robot.ik_tcp = lambda tgt_rotmat, tgt_pos: [oum.np.asarray(tgt_pos, dtype=oum.np.float32)]
        obj = types.SimpleNamespace(
            pos=oum.np.zeros(3, dtype=oum.np.float32),
            rotmat=oum.np.eye(3, dtype=oum.np.float32),
        )
        common_gids = planner.reason_common_grasp_ids(
            obj_model=obj,
            grasp_collection=[grasp0, grasp1],
            goal_pose_list=[(oum.np.zeros(3, dtype=oum.np.float32), oum.np.eye(3, dtype=oum.np.float32))],
            pick_pose=(oum.np.zeros(3, dtype=oum.np.float32), oum.np.eye(3, dtype=oum.np.float32)),
        )
        self.assertEqual(common_gids, [1])

    def test_fold_gen_pick_and_fold_records_attach_and_gid(self):
        planner = self.fold_mod.FoldPlanner(self.robot, pln_ctx=self.pln_ctx, ee_actor=self.ee_actor)
        obj = types.SimpleNamespace(
            pos=oum.np.zeros(3, dtype=oum.np.float32),
            rotmat=oum.np.eye(3, dtype=oum.np.float32),
        )
        grasp = self._make_grasp([0.3, 0.0, 0.0])
        plan = planner.gen_pick_and_fold(
            obj_model=obj,
            grasp_collection=[grasp],
            goal_pose_list=[
                (oum.np.zeros(3, dtype=oum.np.float32), oum.np.eye(3, dtype=oum.np.float32)),
                (oum.vec(0.1, 0.1, 0.0).astype(oum.np.float32), oum.np.eye(3, dtype=oum.np.float32)),
            ],
            linear_granularity=0.05,
            reason_grasps=True,
            use_rrt=False,
        )
        self.assertIsNotNone(plan)
        self.assertIn('attach', plan.events)
        self.assertEqual(plan.events['gid'], 0)
        self.assertGreater(len(plan.qs_list), 2)

    def test_interpolate_fold_returns_requested_resolution(self):
        start_pose = (oum.np.zeros(3, dtype=oum.np.float32), oum.np.eye(3, dtype=oum.np.float32))
        goal_pose = (
            oum.vec(0.0, 0.2, 0.0).astype(oum.np.float32),
            oum.rotmat_from_axangle(oum.vec(0.0, 0.0, 1.0), oum.pi / 2.0),
        )
        poses = self.fold_mod.interpolate_fold(start_pose, goal_pose, n_steps=5)
        self.assertEqual(len(poses), 5)
        oum.np.testing.assert_allclose(poses[0][0], start_pose[0], atol=1e-6)
        oum.np.testing.assert_allclose(poses[-1][0], goal_pose[0], atol=1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
