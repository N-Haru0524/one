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
    mujoco_mod = types.ModuleType('mujoco')
    mujoco_mod.MjModel = type('MjModel', (), {'from_xml_string': staticmethod(lambda xml: None)})
    mujoco_mod.MjData = type('MjData', (), {'__init__': lambda self, model: None})
    mujoco_mod.mj_step = lambda *args, **kwargs: None
    mujoco_mod.mj_forward = lambda *args, **kwargs: None
    mujoco_mod.mj_kinematics = lambda *args, **kwargs: None
    mujoco_mod.mj_collision = lambda *args, **kwargs: None
    sys.modules['mujoco'] = mujoco_mod

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

from one import ouc  # noqa: E402
import one.scene.scene_object as osso  # noqa: E402
from one_assembly.assembly_planning import (  # noqa: E402
    assembly_sequence_planning,
    capture_robot_state,
    capture_work_state,
    execute_sequence_string,
    parse_sequence_string,
    reset_robot_state,
    reset_work_state,
    sequence_labels,
)
from one_assembly.worklist import WorkList  # noqa: E402


class FakeRobot:
    def __init__(self, qs):
        self.qs = np.asarray(qs, dtype=np.float32).copy()

    def fk(self, qs):
        self.qs = np.asarray(qs, dtype=np.float32).copy()


class FakeEE:
    def __init__(self, qs):
        self.qs = np.asarray(qs, dtype=np.float32).copy()

    def fk(self, qs):
        self.qs = np.asarray(qs, dtype=np.float32).copy()


class AssemblySequencePlanningTest(unittest.TestCase):
    def test_robot_state_roundtrip(self):
        robot = FakeRobot([0, 1, 2])
        ee_actor = FakeEE([0.01, 0.02])
        state = capture_robot_state(robot, ee_actor=ee_actor)

        robot.fk([3, 4, 5])
        ee_actor.fk([0.03, 0.04])
        reset_robot_state(robot, state, ee_actor=ee_actor)

        np.testing.assert_allclose(robot.qs, [0, 1, 2], atol=1e-6)
        np.testing.assert_allclose(ee_actor.qs, [0.01, 0.02], atol=1e-6)

    def test_work_state_roundtrip(self):
        worklist = WorkList()
        root_state = reset_work_state(worklist, layout_name='home')
        mutated_state = reset_work_state(
            worklist,
            layout_name='home',
            actions=[(4, 0), (4, 1), (5, 0), (5, 1)],
        )

        self.assertEqual(mutated_state.screw_counter, 1)
        reset_work_state(worklist, state=root_state)
        restored = capture_work_state(worklist)

        self.assertEqual(restored.screw_counter, root_state.screw_counter)
        for name, pose in root_state.part_poses.items():
            np.testing.assert_allclose(restored.part_poses[name][0], pose[0], atol=1e-6)
            np.testing.assert_allclose(restored.part_poses[name][1], pose[1], atol=1e-6)

    def test_work_base_uses_worklist_collision_type(self):
        worklist = WorkList(collision_type=ouc.CollisionType.MESH)
        worklist.init_pose(seed='home')

        self.assertIsNotNone(worklist.work_base)
        self.assertEqual(worklist.work_base._collision_type, ouc.CollisionType.MESH)

    def test_collision_group_override_survives_mount_style_state_change(self):
        obj = osso.SceneObject(collision_type=None, is_free=True)
        self.assertEqual(obj.collision_group, ouc.CollisionGroup.OBJECT)

        obj.collision_group = ouc.CollisionGroup.OBJECT
        obj.is_free = False

        self.assertEqual(obj.collision_group, ouc.CollisionGroup.OBJECT)

    def test_sequence_planning_respects_immediate(self):
        worklist = WorkList()
        nodes = assembly_sequence_planning(worklist, max_depth=2)

        relay_node = nodes['root/relay:time0']
        self.assertEqual(relay_node.children, ['root/relay:time0/relay:time1'])

        self.assertNotIn('root/workbench:time0', nodes)

        capacitor_node = nodes['root/capacitor:time0']
        child_sequences = [sequence_labels(nodes[node_id]) for node_id in capacitor_node.children]
        self.assertGreater(len(child_sequences), 1)
        self.assertIn(['capacitor:time0', 'belt:time0'], child_sequences)
        self.assertNotIn(['capacitor:time0', 'bracket:time0'], child_sequences)

    def test_parse_and_execute_sequence_string(self):
        worklist = WorkList()
        actions = parse_sequence_string(
            worklist,
            'wrkbnch-brckt-cpctr-blt-blt_fld-blt_fld_scrw-trmnl-trmnl_scrw',
        )
        self.assertEqual(
            [action.label for action in actions],
            [
                'capacitor:time0',
                'belt:time0',
                'belt:time1',
                'belt:time2',
                'terminal:time0',
                'terminal:time1',
            ],
        )

        result = execute_sequence_string(
            worklist,
            'wrkbnch-brckt-cpctr-blt-blt_fld',
        )
        self.assertEqual(len(result.actions), 3)
        self.assertEqual(len(result.work_states), 3)
        self.assertEqual(result.actions[-1].label, 'belt:time1')
        self.assertEqual(result.work_states[-1].screw_counter, 0)


if __name__ == '__main__':
    unittest.main()
