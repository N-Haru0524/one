import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
UTILS_PATH = ROOT / 'one_assembly' / 'motion_planner' / 'utils.py'
PLANNER_PATH = ROOT / 'one_assembly' / 'motion_planner' / 'approach_depart_planner.py'
HIER_PATH = ROOT / 'one_assembly' / 'motion_planner' / 'hierarchical.py'
PP_PATH = ROOT / 'one_assembly' / 'motion_planner' / 'ppplanner.py'
SCREW_PATH = ROOT / 'one_assembly' / 'motion_planner' / 'screwplanner.py'


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
        return bool(self.collision_predicate(np.asarray(qs, dtype=np.float32)))

    @property
    def actors(self):
        return self._actors

    @actors.setter
    def actors(self, value):
        self._actors = tuple(value)


class FakePlanningContext:
    def __init__(self, collider, planning_mecbas=None, aux_mecbas=None,
                 joint_limits=None, cd_step_size=np.pi / 180, cache_size=10000):
        del planning_mecbas, joint_limits, cache_size
        self.collider = collider
        self.aux_mecbas = aux_mecbas or {}
        self.cd_step_size = cd_step_size

    def is_state_valid(self, state):
        state = np.asarray(state, dtype=np.float32)
        return bool(np.all(np.abs(state) <= np.pi)) and not self.collider.is_collided(state)

    def is_motion_valid(self, state1, state2):
        return self.is_state_valid(state1) and self.is_state_valid(state2)

    def distance(self, state1, state2):
        return float(np.linalg.norm(np.asarray(state2) - np.asarray(state1)))

    def interpolate(self, state1, state2, t):
        state1 = np.asarray(state1, dtype=np.float32)
        state2 = np.asarray(state2, dtype=np.float32)
        return ((1.0 - t) * state1 + t * state2).astype(np.float32)

    def enforce_bounds(self, state):
        return np.asarray(state, dtype=np.float32)

    def clear_cache(self):
        return None

    def sample_uniform(self):
        return np.zeros(3, dtype=np.float32)

    def states_equal(self, state1, state2, tol=1e-4):
        return self.distance(state1, state2) <= tol


class FakeRRTConnectPlanner:
    def __init__(self, pln_ctx, extend_step_size=np.pi / 36, goal_bias=0.7):
        del goal_bias
        self._pln_ctx = pln_ctx
        self._step = float(np.asarray(extend_step_size).reshape(-1)[0])

    def solve(self, start, goal, max_iters=1000, time_limit=None):
        del max_iters, time_limit
        start = np.asarray(start, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        dist = self._pln_ctx.distance(start, goal)
        n_steps = max(1, int(np.ceil(dist / self._step)))
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
                            rot_step=np.deg2rad(2.0),
                            ref_qs=None):
    del robot, start_rotmat, goal_rotmat, rot_step, ref_qs
    start_pos = np.asarray(start_pos, dtype=np.float32)
    goal_pos = start_pos if goal_pos is None else np.asarray(goal_pos, dtype=np.float32)
    dist = float(np.linalg.norm(goal_pos - start_pos))
    n_steps = max(1, int(np.ceil(dist / pos_step)))
    q_seq = np.linspace(start_pos, goal_pos, n_steps + 1, dtype=np.float32)
    rot_seq = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], len(q_seq), axis=0)
    return q_seq, (q_seq, rot_seq)


def fake_tf_from_rotmat_pos(rotmat, pos):
    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] = np.asarray(rotmat, dtype=np.float32)
    tf[:3, 3] = np.asarray(pos, dtype=np.float32)
    return tf


class FakeRobot:
    clone_count = 0

    def __init__(self):
        self.ndof = 3
        self.qs = np.zeros(3, dtype=np.float32)
        self.gl_tcp_tf = np.eye(4, dtype=np.float32)

    def clone(self):
        FakeRobot.clone_count += 1
        new = FakeRobot()
        new.qs = self.qs.copy()
        new.gl_tcp_tf = self.gl_tcp_tf.copy()
        return new

    def fk(self, qs=None):
        if qs is not None:
            self.qs = np.asarray(qs, dtype=np.float32)
        self.gl_tcp_tf = np.eye(4, dtype=np.float32)
        self.gl_tcp_tf[:3, 3] = self.qs
        return self.gl_tcp_tf

    def ik_tcp_nearest(self, tgt_rotmat, tgt_pos, ref_qs=None):
        del tgt_rotmat, ref_qs
        return np.asarray(tgt_pos, dtype=np.float32)

    def ik_tcp(self, tgt_rotmat, tgt_pos):
        del tgt_rotmat
        tgt_pos = np.asarray(tgt_pos, dtype=np.float32)
        return [
            tgt_pos.copy(),
            (tgt_pos + np.array([0.2, 0.0, 0.0], dtype=np.float32)).astype(np.float32),
        ]


class FakeEEActor:
    def __init__(self):
        self.ndof = 2
        self.qs = np.zeros(2, dtype=np.float32)

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
    _make_package('one_assembly')
    _make_package('one_assembly.motion_planner')

    oum_mod = types.ModuleType('one.oum')
    oum_mod.tf_from_rotmat_pos = fake_tf_from_rotmat_pos
    sys.modules[oum_mod.__name__] = oum_mod
    sys.modules['one'].oum = oum_mod

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
    return utils_mod, planner_mod, pp_mod, screw_mod


class ADPlannerSmokeTest(unittest.TestCase):
    def _make_grasp(self, pos, jaw_width=0.04):
        pose_tf = np.eye(4, dtype=np.float32)
        pose_tf[:3, 3] = np.asarray(pos, dtype=np.float32)
        pre_pose_tf = pose_tf.copy()
        pre_pose_tf[2, 3] += 0.05
        return pose_tf, pre_pose_tf, float(jaw_width), 1.0

    def setUp(self):
        self.utils_mod, self.planner_mod, self.pp_mod, self.screw_mod = load_modules()
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
            goal_tcp_pos=np.array([0.3, 0.0, 0.0], dtype=np.float32),
            goal_tcp_rotmat=np.eye(3, dtype=np.float32),
            distance=0.1,
            granularity=0.05,
            ee_values=np.array([0.01, 0.01], dtype=np.float32),
        )
        self.assertIsNotNone(plan)
        self.assertGreaterEqual(len(plan), 3)
        np.testing.assert_allclose(plan.qs_list[0], [0.3, 0.0, -0.1, 0.01, 0.01], atol=1e-6)
        np.testing.assert_allclose(plan.qs_list[-1], [0.3, 0.0, 0.0, 0.01, 0.01], atol=1e-6)

    def test_approach_depart_runs_end_to_end(self):
        plan = self.planner.gen_approach_depart(
            goal_qs=np.array([0.3, 0.0, 0.0, 0.02, 0.02], dtype=np.float32),
            start_qs=np.array([0.0, 0.0, 0.0, 0.04, 0.04], dtype=np.float32),
            end_qs=np.array([0.5, 0.2, 0.0, 0.01, 0.01], dtype=np.float32),
            approach_distance=0.1,
            depart_distance=0.1,
            approach_linear=True,
            depart_linear=True,
            linear_granularity=0.05,
            use_rrt=False,
        )
        self.assertIsNotNone(plan)
        np.testing.assert_allclose(plan.qs_list[0], [0.0, 0.0, 0.0, 0.04, 0.04], atol=1e-6)
        np.testing.assert_allclose(plan.qs_list[-1], [0.5, 0.2, 0.0, 0.01, 0.01], atol=1e-6)

    def test_utils_build_collider_includes_aux_mecbas(self):
        aux_actor = object()
        collider = self.utils_mod.build_collider(
            actors=[self.robot],
            obstacles=['obstacle'],
            aux_mecbas={aux_actor: np.array([0.01, 0.01], dtype=np.float32)},
        )
        self.assertEqual(collider.entities[0], self.robot)
        self.assertEqual(collider.entities[1], aux_actor)
        self.assertEqual(collider.entities[2], 'obstacle')

    def test_pickplace_reason_common_grasp_ids_uses_unified_collider(self):
        grasp0 = self._make_grasp([0.1, 0.0, 0.0])
        grasp1 = self._make_grasp([0.3, 0.0, 0.0])
        collider = self.utils_mod.build_collider([self.robot, self.ee_actor])
        collider.collision_predicate = lambda qs: bool(np.asarray(qs, dtype=np.float32)[0] < 0.2)
        planner = self.pp_mod.PickPlacePlanner(
            self.robot,
            pln_ctx=self.utils_mod.build_planning_context(collider),
            ee_actor=self.ee_actor,
        )
        self.robot.ik_tcp = lambda tgt_rotmat, tgt_pos: [np.asarray(tgt_pos, dtype=np.float32)]
        common_gids = planner.reason_common_grasp_ids(
            grasp_collection=[grasp0, grasp1],
            goal_pose_list=[(np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32))],
            pick_pose=(np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32)),
        )
        self.assertEqual(common_gids, [1])

    def test_screw_reason_common_sids_uses_unified_collider(self):
        collider = self.utils_mod.build_collider([self.robot])
        collider.collision_predicate = lambda qs: bool(np.asarray(qs, dtype=np.float32)[0] < 0.2)
        planner = self.screw_mod.ScrewPlanner(
            self.robot,
            pln_ctx=self.utils_mod.build_planning_context(collider),
        )
        self.robot.ik_tcp = lambda tgt_rotmat, tgt_pos: [np.asarray(tgt_pos, dtype=np.float32)]
        goal_pose_list = [
            (np.array([0.1, 0.0, 0.0], dtype=np.float32), np.eye(3, dtype=np.float32)),
            (np.array([0.3, 0.0, 0.0], dtype=np.float32), np.eye(3, dtype=np.float32)),
        ]
        common_sids = planner.reason_common_sids(
            goal_pose_list=goal_pose_list,
        )
        self.assertEqual(common_sids, [1])

    def test_pickplace_accepts_planning_context_directly(self):
        grasp = self._make_grasp([0.3, 0.0, 0.0])
        collider = self.utils_mod.build_collider([self.robot, self.ee_actor])
        pln_ctx = self.utils_mod.build_planning_context(collider)
        planner = self.pp_mod.PickPlacePlanner(self.robot, pln_ctx=pln_ctx, ee_actor=self.ee_actor)
        self.robot.ik_tcp = lambda tgt_rotmat, tgt_pos: [np.asarray(tgt_pos, dtype=np.float32)]
        common_gids = planner.reason_common_grasp_ids(
            grasp_collection=[grasp],
            goal_pose_list=[(np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32))],
            pick_pose=(np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32)),
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
            goal_pose_list=[(np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32))],
            pick_pose=(np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32)),
            exclude_entities=[sentinel],
        )
        self.assertEqual(common_gids, [0])
        self.assertIn('survived_gids', planner._last_reason_common_grasp_report)

    def test_pickplace_reason_common_grasp_ids_records_failure_location(self):
        grasp0 = self._make_grasp([0.1, 0.0, 0.0])
        grasp1 = self._make_grasp([0.3, 0.0, 0.0])
        collider = self.utils_mod.build_collider([self.robot, self.ee_actor])
        collider.collision_predicate = lambda qs: bool(np.asarray(qs, dtype=np.float32)[0] < 0.2)
        planner = self.pp_mod.PickPlacePlanner(
            self.robot,
            pln_ctx=self.utils_mod.build_planning_context(collider),
            ee_actor=self.ee_actor,
        )
        self.robot.ik_tcp = lambda tgt_rotmat, tgt_pos: [np.asarray(tgt_pos, dtype=np.float32)]
        common_gids = planner.reason_common_grasp_ids(
            grasp_collection=[grasp0, grasp1],
            goal_pose_list=[(np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32))],
            pick_pose=(np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32)),
        )
        self.assertEqual(common_gids, [1])
        failure = planner._last_reason_common_grasp_report['failures'][0]
        self.assertEqual(failure['label'], 'pick_goal')
        self.assertEqual(failure['reason'], 'goal_in_collision')

    def test_filtered_pln_ctx_is_cached(self):
        planner = self.pp_mod.PickPlacePlanner(self.robot, pln_ctx=self.pln_ctx, ee_actor=self.ee_actor)
        obj = object()
        first = planner._filtered_pln_ctx(exclude_entities=[obj])
        second = planner._filtered_pln_ctx(exclude_entities=[obj])
        self.assertIs(first, second)

    def test_tcp_pose_query_reuses_scratch_robot(self):
        _ = self.planner._tcp_pose_from_qs(np.array([0.1, 0.0, 0.0, 0.01, 0.01], dtype=np.float32))
        _ = self.planner._tcp_pose_from_qs(np.array([0.2, 0.0, 0.0, 0.01, 0.01], dtype=np.float32))
        self.assertEqual(FakeRobot.clone_count, 1)

    def test_pickplace_reuses_reasoned_place_candidate(self):
        planner = self.pp_mod.PickPlacePlanner(self.robot, pln_ctx=self.pln_ctx, ee_actor=self.ee_actor)
        obj = types.SimpleNamespace(
            pos=np.zeros(3, dtype=np.float32),
            rotmat=np.eye(3, dtype=np.float32),
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
            return self.utils_mod.MotionData([np.zeros(5, dtype=np.float32), np.ones(5, dtype=np.float32)])

        planner._screen_pose_with_stats = counting_screen_pose
        planner.gen_approach_via_pose = fake_plan
        plan = planner.gen_pick_and_place(
            obj_model=obj,
            grasp_collection=[grasp],
            goal_pose_list=[(np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32))],
            reason_grasps=True,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(screen_calls['count'], 6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
