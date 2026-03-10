import contextlib
import io
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def bootstrap_one_headless():
    if 'one' in sys.modules:
        return
    pyglet_mod = types.ModuleType('pyglet')
    pyglet_mod.__path__ = []
    window_mod = types.ModuleType('pyglet.window')
    key_mod = types.ModuleType('pyglet.window.key')
    gl_mod = types.ModuleType('pyglet.gl')
    math_mod = types.ModuleType('pyglet.math')
    graphics_mod = types.ModuleType('pyglet.graphics')
    window_mod.key = key_mod
    pyglet_mod.window = window_mod
    pyglet_mod.gl = gl_mod
    pyglet_mod.math = math_mod
    pyglet_mod.graphics = graphics_mod
    sys.modules['pyglet'] = pyglet_mod
    sys.modules['pyglet.window'] = window_mod
    sys.modules['pyglet.window.key'] = key_mod
    sys.modules['pyglet.gl'] = gl_mod
    sys.modules['pyglet.math'] = math_mod
    sys.modules['pyglet.graphics'] = graphics_mod

    viewer_pkg = types.ModuleType('one.viewer')
    viewer_pkg.__path__ = []
    sys.modules['one.viewer'] = viewer_pkg

    world_mod = types.ModuleType('one.viewer.world')
    world_mod.World = type('World', (), {})
    sys.modules['one.viewer.world'] = world_mod

    device_buffer_mod = types.ModuleType('one.viewer.device_buffer')
    device_buffer_mod.MeshBuffer = type(
        'MeshBuffer', (), {'__init__': lambda self, *args, **kwargs: None}
    )
    device_buffer_mod.PointCloudBuffer = type(
        'PointCloudBuffer', (), {'__init__': lambda self, *args, **kwargs: None}
    )
    sys.modules['one.viewer.device_buffer'] = device_buffer_mod


bootstrap_one_headless()

from one import khi_rs007l, ocm, omppc, or_2fg7  # noqa: E402
from one_assembly.motion_planner import ADPlanner  # noqa: E402


class ADPlannerIntegrationTest(unittest.TestCase):
    def _run_quietly(self, func, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return func(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.robot = khi_rs007l.RS007L()
        cls.collider = ocm.MJCollider()
        cls.collider.append(cls.robot)
        cls.collider.actors = [cls.robot]
        cls.collider.compile(margin=0.0)
        cls.pln_ctx = omppc.PlanningContext(collider=cls.collider)
        cls.planner = ADPlanner(cls.robot, pln_ctx=cls.pln_ctx)
        cls.goal_qs = np.array(
            [0.0, -np.pi / 4, np.pi / 2, 0.0, np.pi / 4, 0.0],
            dtype=np.float32,
        )
        cls.end_qs = np.array(
            [0.2, -0.6, 1.0, 0.1, 0.5, 0.0],
            dtype=np.float32,
        )

    def test_linear_approach_to_goal_qs(self):
        plan = self._run_quietly(
            self.planner.gen_approach,
            goal_qs=self.goal_qs,
            approach_direction=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            approach_distance=0.05,
            linear=True,
            linear_granularity=0.02,
            use_rrt=False,
        )
        self.assertIsNotNone(plan)
        self.assertGreaterEqual(len(plan), 2)
        np.testing.assert_allclose(plan.qs_list[-1], self.goal_qs, atol=1e-4)

    def test_approach_depart_with_goal_qs(self):
        plan = self._run_quietly(
            self.planner.gen_approach_depart,
            goal_qs=self.goal_qs,
            start_qs=self.robot.qs.copy(),
            end_qs=self.end_qs,
            approach_direction=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            depart_direction=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            approach_distance=0.03,
            depart_distance=0.03,
            approach_linear=True,
            depart_linear=True,
            linear_granularity=0.02,
            use_rrt=False,
        )
        self.assertIsNotNone(plan)
        self.assertGreaterEqual(len(plan), 3)
        np.testing.assert_allclose(plan.qs_list[0], self.robot.qs, atol=1e-6)
        np.testing.assert_allclose(plan.qs_list[-1], self.end_qs, atol=1e-4)

    def test_combined_robot_and_ee_qs(self):
        robot = khi_rs007l.RS007L()
        gripper = or_2fg7.OR2FG7()
        robot.engage(gripper)
        collider = ocm.MJCollider()
        collider.append(robot)
        collider.append(gripper)
        collider.actors = [robot, gripper]
        collider.compile(margin=0.0)
        planner = ADPlanner(robot, pln_ctx=omppc.PlanningContext(collider=collider), ee_actor=gripper)
        goal_state = np.concatenate(
            [
                self.goal_qs,
                np.array([0.01], dtype=np.float32),
            ]
        ).astype(np.float32)
        plan = self._run_quietly(
            planner.gen_approach,
            goal_qs=goal_state,
            approach_direction=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            approach_distance=0.03,
            linear=True,
            linear_granularity=0.02,
            use_rrt=False,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.qs_list[-1].shape[0], robot.ndof + gripper.ndof)
        np.testing.assert_allclose(plan.qs_list[-1], goal_state, atol=1e-4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
