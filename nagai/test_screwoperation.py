"""Unit tests for the one_assembly.ScrewOperation package.

Headless / no-ROS / no-camera. Run with:
    uv run python -m unittest nagai.test_screwoperation -v
or directly:
    uv run python nagai/test_screwoperation.py
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import unittest
from dataclasses import dataclass

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# spiral_metry
# ---------------------------------------------------------------------------

class SpiralMetryTest(unittest.TestCase):
    def test_hex_ring_abs_returns_origin_first(self):
        from one_assembly.ScrewOperation.spiral_metry import hex_ring_abs
        coords = hex_ring_abs(7, step=0.001)
        self.assertEqual(coords.shape, (7, 2))
        self.assertTrue(np.allclose(coords[0], [0.0, 0.0]))

    def test_hex_ring_abs_outer_radius_matches_ring_count(self):
        # 91 classes ⇒ outer ring 5 (1+6+12+18+24+30=91), radius = 5 * step
        from one_assembly.ScrewOperation.spiral_metry import hex_ring_abs
        coords = hex_ring_abs(91, step=0.0008)
        r = np.linalg.norm(coords, axis=1)
        self.assertAlmostEqual(r.max(), 0.0008 * 5, places=6)

    def test_hex_ring_abs_step_scales_linearly(self):
        from one_assembly.ScrewOperation.spiral_metry import hex_ring_abs
        a = hex_ring_abs(13, step=0.001)
        b = hex_ring_abs(13, step=0.002)
        self.assertTrue(np.allclose(b, 2.0 * a))


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

class UtilsTest(unittest.TestCase):
    def test_rot6d_roundtrip_identity(self):
        from one_assembly.ScrewOperation.utils import rotmat_to_rot6d, rot6d_to_rotmat
        R = np.eye(3, dtype=np.float32)
        r6 = rotmat_to_rot6d(R)
        self.assertEqual(r6.shape, (6,))
        R2 = rot6d_to_rotmat(r6)
        self.assertTrue(np.allclose(R2, R, atol=1e-6))

    def test_rot6d_roundtrip_random_rotmat(self):
        import one.utils.math as oum
        from one_assembly.ScrewOperation.utils import rotmat_to_rot6d, rot6d_to_rotmat
        rng = np.random.default_rng(42)
        for _ in range(8):
            axis = rng.standard_normal(3).astype(np.float32)
            axis /= np.linalg.norm(axis)
            angle = float(rng.uniform(-np.pi, np.pi))
            R = oum.rotmat_from_axangle(axis, angle).astype(np.float32)
            r6 = rotmat_to_rot6d(R)
            R2 = rot6d_to_rotmat(r6)
            self.assertTrue(np.allclose(R2, R, atol=1e-5),
                            msg=f'mismatch axis={axis} angle={angle}')

    def test_hexagon_vertex_3d_radius_and_count(self):
        from one_assembly.ScrewOperation.utils import hexagon_vertex_3d
        center = np.array([1.0, 2.0, 3.0])
        radius = 0.05
        verts = np.stack([hexagon_vertex_3d(center=center, radius=radius, idx=i) for i in range(6)])
        # All vertices equidistant from center
        d = np.linalg.norm(verts - center, axis=1)
        self.assertTrue(np.allclose(d, radius, atol=1e-6))
        # Distinct
        self.assertEqual(len({tuple(np.round(v, 6)) for v in verts}), 6)

    def test_hexagon_vertex_3d_idx_out_of_range_raises(self):
        from one_assembly.ScrewOperation.utils import hexagon_vertex_3d
        with self.assertRaises(ValueError):
            hexagon_vertex_3d(idx=6)

    def test_make_mode_dir_creates_numbered_subdirs(self):
        from one_assembly.ScrewOperation.utils import make_mode_dir
        with tempfile.TemporaryDirectory() as tmp:
            a = make_mode_dir(tmp, 'train', sequence='seq', mode='pick')
            b = make_mode_dir(tmp, 'train', sequence='seq', mode='pick')
            self.assertNotEqual(a, b)
            self.assertTrue(a.endswith('001'))
            self.assertTrue(b.endswith('002'))
            self.assertTrue(os.path.isdir(os.path.join(a, 'images')))

    def test_csv_writer_writes_header_then_rows(self):
        from one_assembly.ScrewOperation.utils import csv_writer
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'samples.csv')
            w = csv_writer(path, fieldnames=['a', 'b'])
            w.write({'a': 1, 'b': 2})
            w.write({'a': 3, 'b': 4})
            w.close()
            with open(path) as f:
                content = f.read()
        self.assertEqual(content.splitlines()[0], 'a,b')
        self.assertEqual(content.splitlines()[1], '1,2')
        self.assertEqual(content.splitlines()[2], '3,4')


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

class ConfigTest(unittest.TestCase):
    def test_yaml_roundtrip(self):
        from one_assembly.ScrewOperation.config import ScrewConfig, save_config, load_config
        c = ScrewConfig(num_classes=37, spiral_step=0.0012, roi1=(1, 2, 3, 4))
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'c.yaml')
            save_config(c, p)
            c2 = load_config(p)
        self.assertEqual(c.num_classes, c2.num_classes)
        self.assertAlmostEqual(c.spiral_step, c2.spiral_step)
        self.assertEqual(c.roi1, c2.roi1)

    def test_merge_cli_args_overrides(self):
        import argparse
        from one_assembly.ScrewOperation.config import ScrewConfig, merge_cli_args
        base = ScrewConfig(num_classes=7, spiral_step=0.001)
        ns = argparse.Namespace(num_classes=91, spiral_step=None)
        merged = merge_cli_args(base, ns)
        self.assertEqual(merged.num_classes, 91)
        self.assertAlmostEqual(merged.spiral_step, 0.001)  # untouched

    def test_image_size_property(self):
        from one_assembly.ScrewOperation.config import ScrewConfig
        c = ScrewConfig(resize_per_cam=(45, 40))
        self.assertEqual(c.image_size, (45, 80))


# ---------------------------------------------------------------------------
# prescrew
# ---------------------------------------------------------------------------

class _MockArm:
    """Fake arm that returns ref_qs perturbed by tgt_pos[0]."""

    def __init__(self):
        self.qs = np.zeros(6, dtype=np.float32)

    def ik_tcp_nearest(self, tgt_rotmat, tgt_pos, ref_qs):
        return np.asarray(ref_qs, dtype=np.float32) + np.array(
            [tgt_pos[0], 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        )


class PrescrewTest(unittest.TestCase):
    def test_prescrew_pose_offset_along_screw_z(self):
        from one_assembly.ScrewOperation.prescrew import prescrew_pose_from_screw_pose
        screw_pos = np.array([0.45, -0.10, 0.16], dtype=np.float32)
        # Identity rotmat => screw z is world z (advance direction = +z)
        pre_pos, _ = prescrew_pose_from_screw_pose(screw_pos, np.eye(3, dtype=np.float32),
                                                    prescrew_offset=0.005)
        self.assertTrue(np.allclose(pre_pos - screw_pos, [0, 0, -0.005], atol=1e-6))

    def test_prescrew_pose_flip_axis(self):
        from one_assembly.ScrewOperation.prescrew import prescrew_pose_from_screw_pose
        screw_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pre_pos, _ = prescrew_pose_from_screw_pose(screw_pos, np.eye(3, dtype=np.float32),
                                                    prescrew_offset=0.005, flip_axis=True)
        self.assertTrue(np.allclose(pre_pos, [0, 0, 0.005], atol=1e-6))

    def test_prescrew_pose_for_screw_pointing_down(self):
        from one_assembly.ScrewOperation.prescrew import prescrew_pose_from_screw_pose
        # Screw z = -world_z (typical top-down screws)
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        pre_pos, _ = prescrew_pose_from_screw_pose(
            np.array([0.5, 0.0, 0.2], dtype=np.float32), R, prescrew_offset=0.01,
        )
        # Offset should be ABOVE the screw (back along screw advance = +world_z)
        self.assertTrue(np.allclose(pre_pos, [0.5, 0.0, 0.21], atol=1e-6))

    def test_prescrew_qs_from_screw_pose_ik_returns_solution(self):
        from one_assembly.ScrewOperation.prescrew import prescrew_qs_from_screw_pose, PrescrewSolution
        sol = prescrew_qs_from_screw_pose(
            _MockArm(),
            np.array([0.45, -0.10, 0.16], dtype=np.float32),
            np.eye(3, dtype=np.float32),
            prescrew_offset=0.005,
        )
        self.assertIsInstance(sol, PrescrewSolution)
        self.assertEqual(sol.rgt_qs.shape, (6,))

    def test_yaml_roundtrip_with_ee(self):
        from one_assembly.ScrewOperation.prescrew import save_prescrew_yaml, load_prescrew_yaml
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'sub', 'p.yaml')
            qs = np.array([0.1, 0.2, -0.3, 0.4, -0.5, 0.6], dtype=np.float32)
            save_prescrew_yaml(out, qs, rgt_ee_qs=[-0.02])
            qs_back, ee_back = load_prescrew_yaml(out)
        self.assertTrue(np.allclose(qs_back, qs))
        self.assertIsNotNone(ee_back)
        self.assertAlmostEqual(float(ee_back[0]), -0.02)

    def test_load_prescrew_yaml_missing_rgt_qs(self):
        from one_assembly.ScrewOperation.prescrew import load_prescrew_yaml
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'bad.yaml')
            with open(p, 'w') as f:
                f.write("some_other_key: 1\n")
            with self.assertRaises(KeyError):
                load_prescrew_yaml(p)

    def test_resolve_prescrew_yaml_priority(self):
        from one_assembly.ScrewOperation.prescrew import save_prescrew_yaml, resolve_prescrew
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'p.yaml')
            qs = np.array([0.1] * 6, dtype=np.float32)
            save_prescrew_yaml(out, qs, rgt_ee_qs=[-0.01])
            sol = resolve_prescrew(yaml_path=out)
        self.assertTrue(np.allclose(sol.rgt_qs, qs))
        self.assertIsNotNone(sol.rgt_ee_qs)

    def test_resolve_prescrew_requires_source(self):
        from one_assembly.ScrewOperation.prescrew import resolve_prescrew
        with self.assertRaises(ValueError):
            resolve_prescrew()


# ---------------------------------------------------------------------------
# approach_plan
# ---------------------------------------------------------------------------

def _make_dual_state():
    from one_assembly.assembly_data import DualRobotState
    return DualRobotState(
        lft_qs=np.zeros(6, dtype=np.float32),
        lft_ee_qs=np.array([0.04, 0.04], dtype=np.float32),
        rgt_qs=np.zeros(6, dtype=np.float32),
        rgt_ee_qs=np.array([0.0], dtype=np.float32),
    )


class ApproachPlanTest(unittest.TestCase):
    def test_correction_approach_plan_minimal(self):
        from one_assembly.ScrewOperation.approach_plan import build_correction_approach_plan
        plan = build_correction_approach_plan(
            initial_state=_make_dual_state(),
            prescrew_rgt_qs=np.ones(6, dtype=np.float32) * 0.1,
        )
        self.assertEqual(len(plan.sync_segments), 1)
        seg = plan.sync_segments[0]
        actors = [arm.actor for arm in seg.arm_segments]
        self.assertEqual(actors, ['left_arm', 'right_arm'])
        # Right path starts at home, ends at prescrew
        rgt = [arm.qs_list for arm in seg.arm_segments if arm.actor == 'right_arm'][0]
        self.assertTrue(np.allclose(rgt[0], 0.0))
        self.assertTrue(np.allclose(rgt[-1], 0.1))

    def test_correction_approach_plan_with_intermediates_and_ee(self):
        from one_assembly.ScrewOperation.approach_plan import build_correction_approach_plan
        plan = build_correction_approach_plan(
            initial_state=_make_dual_state(),
            prescrew_rgt_qs=np.full(6, 0.5, dtype=np.float32),
            rgt_intermediate_qs=[np.full(6, 0.25, dtype=np.float32)],
            rgt_ee_qs_at_end=np.array([-0.02], dtype=np.float32),
        )
        seg = plan.sync_segments[0]
        rgt = [arm.qs_list for arm in seg.arm_segments if arm.actor == 'right_arm'][0]
        self.assertEqual(len(rgt), 3)
        # ee event at end exists
        self.assertEqual(len(seg.ee_events), 1)
        ev = seg.ee_events[0]
        self.assertEqual(ev.kind if hasattr(ev, 'kind') else ev.action, 'extend')
        self.assertEqual(ev.timing, 'end')
        self.assertAlmostEqual(ev.value, -0.02, places=6)

    def test_plan_to_bridge_dict_marks_policy_after(self):
        from one_assembly.ScrewOperation.approach_plan import (
            build_correction_approach_plan, plan_to_bridge_dict,
        )
        plan = build_correction_approach_plan(
            initial_state=_make_dual_state(),
            prescrew_rgt_qs=np.ones(6, dtype=np.float32) * 0.1,
        )
        d = plan_to_bridge_dict(plan, plan_id='unit-test')
        self.assertEqual(d['plan_id'], 'unit-test')
        self.assertEqual(len(d['planned_segments']), 1)
        self.assertTrue(d['planned_segments'][-1]['policy_after'])

    def _make_screw_draft(self):
        from one_assembly.assembly_data import PlannerSegmentDraft, PlannerActionDraft, EEEvent
        home = np.zeros(6, dtype=np.float32)
        def lin(a, b, n=3):
            return [(1 - t) * a + t * b for t in np.linspace(0, 1, n)]
        a = home
        b = home + np.array([0.1, 0, 0, 0, 0, 0], dtype=np.float32)
        c = home + np.array([0.1, 0.2, 0, 0, 0, 0], dtype=np.float32)
        d = home + np.array([0.1, 0.2, -0.3, 0, 0, 0], dtype=np.float32)
        e = home + np.array([0.1, 0.2, -0.3, 0.5, 0, 0], dtype=np.float32)
        f = home + np.array([0.1, 0.2, -0.3, 0.5, 0.7, 0], dtype=np.float32)
        segments = [
            PlannerSegmentDraft(segment_label='approach prepick', right_path=lin(a, b), left_path=[],
                                end_sync_label='rly prepick'),
            PlannerSegmentDraft(segment_label='pick prescrew', right_path=lin(b, c), left_path=[],
                                ee_events=[EEEvent(actor='right_driver', action='extend',
                                                    timing='end', value=-0.005)],
                                end_sync_label='rly pick_prescrew'),
            PlannerSegmentDraft(segment_label='place prescrew', right_path=lin(c, d), left_path=[],
                                ee_events=[EEEvent(actor='right_driver', action='extend',
                                                    timing='end', value=-0.02)],
                                end_sync_label='rly prescrew'),
            PlannerSegmentDraft(segment_label='screw', right_path=lin(d, e), left_path=[],
                                end_sync_label='rly screw'),
            PlannerSegmentDraft(segment_label='retract', right_path=lin(e, f), left_path=[],
                                end_sync_label='rly retract'),
        ]
        return PlannerActionDraft(segments=segments)

    def test_screw_draft_keeps_all_segments_and_marks_every_prescrew(self):
        from one_assembly.ScrewOperation.approach_plan import (
            screw_draft_to_sync_plan, plan_to_bridge_dict_with_indices,
        )
        draft = self._make_screw_draft()
        plan, idx = screw_draft_to_sync_plan(draft, initial_state=_make_dual_state())
        self.assertEqual(len(plan.sync_segments), 5)
        # Both 'pick_prescrew' (seg 1) and 'prescrew' (seg 2) match — and so does the latter's seg_label
        self.assertEqual(idx, {1, 2})
        d_out = plan_to_bridge_dict_with_indices(plan, idx, plan_id='unit')
        flags = [seg['policy_after'] for seg in d_out['planned_segments']]
        self.assertEqual(flags, [False, True, True, False, False])

    def test_screw_draft_truncate_after_last_match(self):
        from one_assembly.ScrewOperation.approach_plan import screw_draft_to_sync_plan
        draft = self._make_screw_draft()
        plan, idx = screw_draft_to_sync_plan(
            draft, initial_state=_make_dual_state(), truncate_after_last_match=True,
        )
        # last match is seg 2 → keep 0..2
        self.assertEqual(len(plan.sync_segments), 3)
        self.assertEqual(idx, {1, 2})

    def test_screw_draft_custom_predicate(self):
        from one_assembly.ScrewOperation.approach_plan import screw_draft_to_sync_plan
        draft = self._make_screw_draft()
        plan, idx = screw_draft_to_sync_plan(
            draft, initial_state=_make_dual_state(),
            policy_after_predicate=lambda seg, i: 'screw' in (seg.end_sync_label or '').lower()
                                                   and 'prescrew' not in (seg.end_sync_label or '').lower(),
        )
        # Only the 'rly screw' segment matches (seg 3)
        self.assertEqual(idx, {3})
        self.assertEqual(len(plan.sync_segments), 5)

    def test_screw_draft_no_match_raises(self):
        from one_assembly.ScrewOperation.approach_plan import screw_draft_to_sync_plan
        draft = self._make_screw_draft()
        with self.assertRaises(ValueError):
            screw_draft_to_sync_plan(draft, initial_state=_make_dual_state(),
                                      policy_after_substring='nothing_matches')

    def test_screw_draft_to_sync_plan_empty_raises(self):
        from one_assembly.assembly_data import PlannerActionDraft
        from one_assembly.ScrewOperation.approach_plan import screw_draft_to_sync_plan
        with self.assertRaises(ValueError):
            screw_draft_to_sync_plan(PlannerActionDraft(segments=[]),
                                      initial_state=_make_dual_state())

    def test_build_multi_phase_correction_plan_marks_all_segments(self):
        from one_assembly.ScrewOperation.approach_plan import (
            build_multi_phase_correction_plan, PrescrewPhase, plan_to_bridge_dict_with_indices,
        )
        plan, idx = build_multi_phase_correction_plan(
            initial_state=_make_dual_state(),
            phases=[
                PrescrewPhase(rgt_qs=np.full(6, 0.1, dtype=np.float32),
                              label='pick_prescrew',
                              rgt_ee_qs=np.array([-0.005], dtype=np.float32)),
                PrescrewPhase(rgt_qs=np.full(6, 0.2, dtype=np.float32),
                              label='place_prescrew',
                              rgt_ee_qs=np.array([-0.02], dtype=np.float32)),
            ],
        )
        self.assertEqual(len(plan.sync_segments), 2)
        self.assertEqual([s.label for s in plan.sync_segments],
                         ['pick_prescrew', 'place_prescrew'])
        self.assertEqual(idx, {0, 1})
        d = plan_to_bridge_dict_with_indices(plan, idx, plan_id='mp')
        self.assertEqual([s['policy_after'] for s in d['planned_segments']], [True, True])
        # Each segment ends at its prescrew_qs
        seg0_last = d['planned_segments'][0]['state_list'][-1]
        seg1_last = d['planned_segments'][1]['state_list'][-1]
        self.assertTrue(np.allclose(seg0_last['rgt_qs'], [0.1] * 6, atol=1e-6))
        self.assertTrue(np.allclose(seg1_last['rgt_qs'], [0.2] * 6, atol=1e-6))
        # EE values carry through
        self.assertAlmostEqual(float(seg0_last['rgt_ee_qs'][0]), -0.005, places=6)
        self.assertAlmostEqual(float(seg1_last['rgt_ee_qs'][0]), -0.02, places=6)

    def test_build_multi_phase_correction_plan_empty_raises(self):
        from one_assembly.ScrewOperation.approach_plan import build_multi_phase_correction_plan
        with self.assertRaises(ValueError):
            build_multi_phase_correction_plan(initial_state=_make_dual_state(), phases=[])


# ---------------------------------------------------------------------------
# correction_loop
# ---------------------------------------------------------------------------

class _LoopMockBridge:
    def __init__(self):
        self.latest_rgt_qs = np.zeros(6, dtype=np.float32)
        self.actions = []
        self.done = False

    def pump(self, t=0.0):
        pass

    def send_action(self, **kw):
        self.actions.append(kw)

    def send_done(self, side='right'):
        self.done = True


class _LoopMockArm:
    def __init__(self):
        self._tf = np.eye(4, dtype=np.float32)
        self._tf[:3, 3] = [0.4, 0.0, 0.3]
        self.qs = np.zeros(6, dtype=np.float32)

    @property
    def gl_tcp_tf(self):
        return self._tf

    def fk(self, qs=None):
        if qs is not None:
            self.qs = np.asarray(qs, dtype=np.float32)

    def ik_tcp_nearest(self, tgt_rotmat, tgt_pos, ref_qs):
        return np.asarray(ref_qs, dtype=np.float32) + np.asarray(
            [tgt_pos[0], 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        )


class _LoopMockCamera:
    def __init__(self, d):
        self.d = d
        self.taken = []

    def take_photo_with_num(self, n):
        for tag in ('cam1', 'cam2'):
            with open(os.path.join(self.d, f'{n:06d}_{tag}.png'), 'wb') as f:
                f.write(b'\x89PNG' + b'\x00' * 2000)
        self.taken.append(n)


class CorrectionLoopTest(unittest.TestCase):
    def test_default_pos_step_clip_covers_outer_ring(self):
        from one_assembly.ScrewOperation.correction_loop import CorrectionLoopConfig
        from one_assembly.ScrewOperation.spiral_metry import hex_ring_abs
        cfg = CorrectionLoopConfig(img_dir='/tmp', csv_path='/tmp/x.csv')
        coords = hex_ring_abs(cfg.num_classes, step=cfg.spiral_step)
        max_r = float(np.linalg.norm(coords, axis=1).max())
        self.assertGreater(cfg.pos_step_clip, max_r,
                           msg=f'pos_step_clip {cfg.pos_step_clip} must cover outer ring {max_r}')

    def test_state_machine_writes_initial_then_steps_then_terminates(self):
        from one_assembly.ScrewOperation.correction_loop import (
            CorrectionLoop, CorrectionLoopConfig,
        )
        with tempfile.TemporaryDirectory() as tmp:
            img_dir = os.path.join(tmp, 'images')
            os.makedirs(img_dir)
            csv_path = os.path.join(tmp, 's.csv')
            cfg = CorrectionLoopConfig(
                img_dir=img_dir, csv_path=csv_path, log_key='idx',
                num_classes=7, spiral_step=0.001, latency=0.0,
                end_on_zero_class=True, max_iterations=10, sign=-1.0,
            )
            bridge = _LoopMockBridge()
            labels = iter([(3, 0.001, 0.0005), (5, -0.0005, 0.001), (0, 0.0, 0.0)])
            loop = CorrectionLoop(
                bridge=bridge,
                camera=_LoopMockCamera(img_dir),
                robot_rgt_arm=_LoopMockArm(),
                label_source=lambda i, d: next(labels),
                config=cfg,
            )
            loop.log_initial()
            iters = 0
            while loop.tick() and iters < 100:
                iters += 1
            loop.close()
            with open(csv_path) as f:
                rows = f.read().splitlines()
        # header + initial row + 2 step rows + final terminate row
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0].split(',')[:5], ['idx', 'time', 'label', 'dx', 'dy'])
        # initial row: idx=0, label=0
        cells = rows[1].split(',')
        self.assertEqual(cells[0], '0')
        self.assertEqual(cells[2], '0')
        # last row: terminate (label=0)
        cells = rows[-1].split(',')
        self.assertEqual(cells[2], '0')
        # bridge received 2 actions + 1 done
        self.assertEqual(len(bridge.actions), 2)
        self.assertTrue(bridge.done)


# ---------------------------------------------------------------------------
# camera (yaml-only, no real device)
# ---------------------------------------------------------------------------

class CameraYamlTest(unittest.TestCase):
    def test_load_cameras_yaml_returns_specs(self):
        from one_assembly.ScrewOperation.camera import load_cameras_yaml, DEFAULT_CAMERAS_YAML, CameraSpec
        s0, s1 = load_cameras_yaml(DEFAULT_CAMERAS_YAML)
        self.assertIsInstance(s0, CameraSpec)
        self.assertIsInstance(s1, CameraSpec)
        for s in (s0, s1):
            self.assertTrue(s.device.startswith('/dev/'))
            self.assertGreater(s.width, 0)
            self.assertGreater(s.height, 0)
            self.assertGreater(s.fps, 0)

    def test_load_cameras_yaml_with_crop(self):
        from one_assembly.ScrewOperation.camera import load_cameras_yaml
        body = (
            "cameras:\n"
            "  cam0:\n"
            "    device: /dev/foo\n"
            "    width: 640\n"
            "    height: 480\n"
            "    fps: 30\n"
            "    rotate180: true\n"
            "    crop: [10, 20, 100, 80]\n"
            "  cam1:\n"
            "    device: /dev/bar\n"
            "    width: 320\n"
            "    height: 240\n"
            "    fps: 15\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'cams.yaml')
            with open(p, 'w') as f:
                f.write(body)
            s0, s1 = load_cameras_yaml(p)
        self.assertTrue(s0.rotate180)
        self.assertEqual(s0.crop, (10, 20, 100, 80))
        self.assertFalse(s1.rotate180)
        self.assertIsNone(s1.crop)


# ---------------------------------------------------------------------------
# bridge_io (rclpy-optional)
# ---------------------------------------------------------------------------

class BridgeIOTest(unittest.TestCase):
    def test_default_topic_constants(self):
        from one_assembly.ScrewOperation import bridge_io
        self.assertEqual(bridge_io._DEFAULT_PLAN_TOPIC, '/one_planner_bridge/plan')
        self.assertEqual(bridge_io._DEFAULT_ACTION_TOPIC, '/one_planner_bridge/action')
        self.assertEqual(bridge_io._DEFAULT_STATUS_TOPIC, '/one_planner_bridge/status')

    def test_default_joint_names(self):
        from one_assembly.ScrewOperation.bridge_io import (
            DEFAULT_LEFT_JOINT_NAMES, DEFAULT_RIGHT_JOINT_NAMES,
        )
        self.assertEqual(DEFAULT_LEFT_JOINT_NAMES,
                         [f'left_joint{i}' for i in range(1, 7)])
        self.assertEqual(DEFAULT_RIGHT_JOINT_NAMES,
                         [f'right_joint{i}' for i in range(1, 7)])


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------

class SessionTest(unittest.TestCase):
    def test_split_target_phase(self):
        from one_assembly.ScrewOperation.session import _split_target_phase, SCREW_PHASES
        self.assertEqual(_split_target_phase('rly_scrw_pick'), ('rly_scrw', 'pick'))
        self.assertEqual(_split_target_phase('rly_scrw_place'), ('rly_scrw', 'place'))
        self.assertEqual(_split_target_phase('blt_fld_scrw_pick'), ('blt_fld_scrw', 'pick'))
        with self.assertRaises(ValueError):
            _split_target_phase('rly_scrw')
        with self.assertRaises(ValueError):
            _split_target_phase('rly_scrw_unknown')
        self.assertEqual(SCREW_PHASES, ('pick', 'place'))

    def test_parse_session_no_history(self):
        from one_assembly.worklist import WorkList
        from one_assembly.ScrewOperation.session import parse_screw_session_string
        wl = WorkList()
        wl.init_pose('home')
        spec = parse_screw_session_string(wl, 'rly_scrw_pick')
        self.assertEqual(spec.phase, 'pick')
        self.assertEqual(spec.target_action.action_type, 'screw')
        self.assertEqual(len(spec.history_actions), 0)
        self.assertEqual(spec.target_token, 'rly_scrw_pick')
        self.assertEqual(spec.history_string, '')

    def test_parse_session_with_history(self):
        from one_assembly.worklist import WorkList
        from one_assembly.ScrewOperation.session import parse_screw_session_string
        wl = WorkList()
        wl.init_pose('home')
        spec = parse_screw_session_string(
            wl, 'rly_scrw_place:wrkbnch-brckt-cpctr-rly')
        self.assertEqual(spec.phase, 'place')
        self.assertTrue(len(spec.history_actions) >= 1)
        # Worklist state must NOT change during parsing
        self.assertEqual(int(wl.screw_counter), 0)

    def test_parse_session_unknown_token_raises(self):
        from one_assembly.worklist import WorkList
        from one_assembly.ScrewOperation.session import parse_screw_session_string
        wl = WorkList()
        wl.init_pose('home')
        with self.assertRaises(ValueError):
            parse_screw_session_string(wl, 'nonsense_pick')

    def test_parse_session_empty_string_raises(self):
        from one_assembly.worklist import WorkList
        from one_assembly.ScrewOperation.session import parse_screw_session_string
        wl = WorkList()
        wl.init_pose('home')
        with self.assertRaises(ValueError):
            parse_screw_session_string(wl, '')

    def test_screw_target_pose_pick_uses_get_screw_pose(self):
        # phase='pick' returns the rack slot pose from layout.screw.origin/pitch
        from one_assembly.worklist import WorkList
        from one_assembly.ScrewOperation.session import (
            parse_screw_session_string, screw_target_pose,
        )
        wl = WorkList()
        wl.init_pose('home')
        spec = parse_screw_session_string(wl, 'rly_scrw_pick')
        before = int(wl.screw_counter)
        pos, rotmat = screw_target_pose(wl, spec)
        # screw_counter must be preserved by the snapshot/restore
        self.assertEqual(int(wl.screw_counter), before)
        # Pose matches what get_screw_pose() returns at the same counter
        wl2 = WorkList()
        wl2.init_pose('home')
        wl2.screw_counter = before
        expected_pos, expected_rotmat = wl2.get_screw_pose()
        self.assertTrue(np.allclose(pos, expected_pos, atol=1e-6))
        self.assertTrue(np.allclose(rotmat, expected_rotmat, atol=1e-6))

    def test_screw_target_pose_place_uses_pose_after_action(self):
        from one_assembly.worklist import WorkList
        from one_assembly.ScrewOperation.session import (
            parse_screw_session_string, screw_target_pose, apply_history,
        )
        wl = WorkList()
        wl.init_pose('home')
        # Place uses Work.pose_after_action: apply the history first so the
        # underlying Work is in the correct pose.
        spec = parse_screw_session_string(wl, 'rly_scrw_place:wrkbnch-brckt-cpctr-rly')
        apply_history(wl, spec)
        before = int(wl.screw_counter)
        pos, rotmat = screw_target_pose(wl, spec)
        # screw_counter must NOT advance (place path doesn't touch it)
        self.assertEqual(int(wl.screw_counter), before)
        # Returned pose must match Work.pose_after_action for the target action
        work = wl[spec.target_action.work_idx]
        expected = work.pose_after_action(
            spec.target_action.action_idx, start_pose=work.current_pose,
        )
        self.assertIsNotNone(expected)
        self.assertTrue(np.allclose(pos, expected[0], atol=1e-6))
        self.assertTrue(np.allclose(rotmat, expected[1], atol=1e-6))

    def test_session_log_key(self):
        from one_assembly.worklist import WorkList
        from one_assembly.ScrewOperation.session import (
            parse_screw_session_string, session_log_key,
        )
        wl = WorkList()
        wl.init_pose('home')
        spec_no_hist = parse_screw_session_string(wl, 'rly_scrw_pick')
        spec_with_hist = parse_screw_session_string(
            wl, 'rly_scrw_place:wrkbnch-brckt-cpctr-rly')
        self.assertEqual(session_log_key(spec_no_hist), 'rly_scrw_pick__home')
        self.assertEqual(
            session_log_key(spec_with_hist),
            'rly_scrw_place__wrkbnch_brckt_cpctr_rly',
        )

    def test_parse_specs_single_phase(self):
        from one_assembly.worklist import WorkList
        from one_assembly.ScrewOperation.session import parse_screw_session_specs
        wl = WorkList(); wl.init_pose('home')
        specs = parse_screw_session_specs(wl, 'rly_scrw_pick')
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].phase, 'pick')
        self.assertEqual(specs[0].target_token, 'rly_scrw_pick')

    def test_parse_specs_full_screw_yields_pick_then_place(self):
        from one_assembly.worklist import WorkList
        from one_assembly.ScrewOperation.session import parse_screw_session_specs
        wl = WorkList(); wl.init_pose('home')
        specs = parse_screw_session_specs(wl, 'rly_scrw:wrkbnch-brckt-cpctr-rly')
        self.assertEqual([s.phase for s in specs], ['pick', 'place'])
        self.assertEqual(specs[0].target_token, 'rly_scrw_pick')
        self.assertEqual(specs[1].target_token, 'rly_scrw_place')
        # Same history is shared across phases
        self.assertEqual(specs[0].history_string, specs[1].history_string)
        self.assertEqual(specs[0].target_action.label, specs[1].target_action.label)


if __name__ == '__main__':
    unittest.main(verbosity=2)
