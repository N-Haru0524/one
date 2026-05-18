"""End-to-end screw-correction runner (inference variant).

Three modes:

1. ``--prescrew prescrew.yaml`` (single phase, yaml-driven)
   Reads rgt_qs / rgt_ee_qs from a yaml; uses ``--model_dir`` as the ViT.

2. ``--session 'rly_scrw_pick[:hist]'`` or ``'rly_scrw_place[:hist]'`` (single phase)
   Resolves prescrew via :func:`session.parse_screw_session_specs`; uses
   ``--model_dir`` (or ``--model_dir_<phase>``) as the ViT.

3. ``--session 'rly_scrw[:hist]'`` (full-screw: pick + place in one run)
   Builds a 2-segment plan with policy_after on both segments, runs the
   pick correction loop, waits for the bridge to enter the second policy
   phase, then runs the place correction loop. Requires
   ``--model_dir_pick`` and ``--model_dir_place``.

Examples:
  uv run python -m one_assembly.ScrewOperation.screw_correction_run \\
      --prescrew config/prescrew.yaml --model_dir model/place/001

  uv run python -m one_assembly.ScrewOperation.screw_correction_run \\
      --session 'rly_scrw_pick' --model_dir model/pick/001

  uv run python -m one_assembly.ScrewOperation.screw_correction_run \\
      --session 'rly_scrw:wrkbnch-brckt-cpctr-rly' \\
      --model_dir_pick model/pick/001 --model_dir_place model/place/001
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

import one.utils.constant as ouc
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

from one_assembly.assembly_data import DualRobotState
from one_assembly.robots.khi_bunri.khi_bunri import KHIBunri

from one_assembly.ScrewOperation.approach_plan import (
    PrescrewPhase,
    build_correction_approach_plan,
    build_multi_phase_correction_plan,
    plan_to_bridge_dict,
    plan_to_bridge_dict_with_indices,
)
from one_assembly.ScrewOperation.bridge_io import CorrectionBridgeClient
from one_assembly.ScrewOperation.camera import DualCameraRecorder
from one_assembly.ScrewOperation.config import ScrewConfig, load_config, save_config
from one_assembly.ScrewOperation.correction_loop import (
    CorrectionLoop,
    CorrectionLoopConfig,
    LabelSource,
)
from one_assembly.ScrewOperation.dataset import load_and_preprocess_pair
from one_assembly.ScrewOperation.model_builder import build_vit
from one_assembly.ScrewOperation.prescrew import resolve_prescrew
from one_assembly.ScrewOperation.spiral_metry import hex_ring_abs
from one_assembly.ScrewOperation.utils import make_mode_dir


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Per-phase setup
# ---------------------------------------------------------------------------

@dataclass
class PhaseRunner:
    """Bundles everything one correction phase needs."""

    label: str                         # 'pick_prescrew' or 'place_prescrew' (also segment label)
    phase: str                         # 'pick' / 'place'
    config: ScrewConfig
    model: torch.nn.Module
    spiral_list: np.ndarray
    img_dir: str
    csv_path: str
    prescrew_rgt_qs: np.ndarray
    prescrew_ee_qs: Optional[np.ndarray]


def _build_inference_label_source(config: ScrewConfig, model, device, spiral_list) -> LabelSource:
    def _src(sample_idx: int, img_dir: str) -> tuple[int, float, float]:
        cam1 = os.path.join(img_dir, f"{sample_idx:06d}_cam1.png")
        cam2 = os.path.join(img_dir, f"{sample_idx:06d}_cam2.png")
        x = load_and_preprocess_pair(cam1, cam2, config).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            cls = int(logits.argmax(dim=1).item())
        if cls == 0:
            return 0, 0.0, 0.0
        dx, dy = spiral_list[cls]
        return cls, float(dx), float(dy)
    return _src


def _load_model_dir(model_dir: str, device: str) -> tuple[ScrewConfig, torch.nn.Module, np.ndarray]:
    config = load_config(os.path.join(model_dir, "config.yaml"))
    model = build_vit(config).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=device))
    model.eval()
    spiral_list = hex_ring_abs(config.num_classes, step=config.spiral_step)
    return config, model, spiral_list


def _make_phase_dirs(ep_dir: str, phase_label: str) -> tuple[str, str]:
    """Returns (img_dir, csv_path) under ep_dir/<phase_label>/."""
    sub = os.path.join(ep_dir, phase_label)
    os.makedirs(sub, exist_ok=True)
    img_dir = os.path.join(sub, "images")
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(sub, "samples.csv")
    return img_dir, csv_path


# ---------------------------------------------------------------------------
# Camera redirect (one camera, multiple per-phase output dirs)
# ---------------------------------------------------------------------------

class _RedirectCamera:
    """Wraps DualCameraRecorder so each phase writes to its own dir.

    The underlying recorder keeps a single pair of cv2.VideoCapture handles;
    only the photo-saver threads' save_dir is mutated between phases.
    """

    def __init__(self, real: DualCameraRecorder):
        self._real = real
        self._active_dir: Optional[str] = None

    def set_output(self, img_dir: str) -> None:
        os.makedirs(img_dir, exist_ok=True)
        self._real.video_path = img_dir
        self._real.photo_left.save_dir = img_dir
        self._real.photo_right.save_dir = img_dir
        self._active_dir = img_dir

    def take_photo_with_num(self, n: int):
        return self._real.take_photo_with_num(n)

    def release(self) -> None:
        self._real.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


# ---------------------------------------------------------------------------
# Phase sequencer
# ---------------------------------------------------------------------------

class _PhaseSequencer:
    """Drives N correction phases sequentially against a single shared bridge.

    The bridge handles each phase as one policy phase; this sequencer waits
    for ``waiting_for_policy`` between phases, swaps the active CorrectionLoop,
    and progresses until either all phases are done or the bridge reports
    ``completed`` / a terminal error status.
    """

    def __init__(
        self,
        bridge: CorrectionBridgeClient,
        camera: _RedirectCamera,
        robot_rgt_arm,
        phase_runners: List[PhaseRunner],
        device: str,
    ):
        self.bridge = bridge
        self.camera = camera
        self.robot_rgt_arm = robot_rgt_arm
        self.phase_runners = phase_runners
        self.device = device
        self.current_idx = -1
        self.current_loop: Optional[CorrectionLoop] = None
        self.state = "AWAIT_POLICY"  # or "RUNNING_LOOP", "BETWEEN_LOOPS", "DONE"

    def _start_next_phase(self) -> bool:
        self.current_idx += 1
        if self.current_idx >= len(self.phase_runners):
            self.state = "DONE"
            return False
        runner = self.phase_runners[self.current_idx]
        self.camera.set_output(runner.img_dir)
        # Initial reference photo for this phase
        self.camera.take_photo_with_num(0)
        # Sync sim mirror to current robot joint state
        cur_rgt = (
            self.bridge.latest_rgt_qs
            if self.bridge.latest_rgt_qs is not None
            else runner.prescrew_rgt_qs
        )
        self.robot_rgt_arm.fk(qs=np.asarray(cur_rgt, dtype=np.float32))
        label_source = _build_inference_label_source(
            runner.config, runner.model, self.device, runner.spiral_list,
        )
        loop_cfg = CorrectionLoopConfig(
            img_dir=runner.img_dir,
            csv_path=runner.csv_path,
            log_key="sequence",
            spiral_step=runner.config.spiral_step,
            num_classes=runner.config.num_classes,
            latency=runner.config.latency,
            end_on_zero_class=True,
        )
        self.current_loop = CorrectionLoop(
            bridge=self.bridge,
            camera=self.camera,
            robot_rgt_arm=self.robot_rgt_arm,
            label_source=label_source,
            config=loop_cfg,
        )
        self.current_loop.log_initial()
        print(f"=== phase {self.current_idx + 1}/{len(self.phase_runners)}: "
              f"{runner.label} (phase={runner.phase}) ===")
        self.state = "RUNNING_LOOP"
        return True

    def tick(self) -> bool:
        self.bridge.pump(0.0)
        status = self.bridge.latest_status or {}
        status_str = status.get('status') if isinstance(status, dict) else None

        if status_str in ('completed', 'aborted', 'failed'):
            self.state = "DONE"
            return False

        if self.state == "AWAIT_POLICY":
            if status_str == 'waiting_for_policy':
                return self._start_next_phase()
            return True

        if self.state == "RUNNING_LOOP":
            assert self.current_loop is not None
            if not self.current_loop.tick():
                self.current_loop.close()
                self.current_loop = None
                self.state = "AWAIT_POLICY"
            return True

        # state == DONE
        return False

    def close(self) -> None:
        if self.current_loop is not None:
            self.current_loop.close()
            self.current_loop = None


# ---------------------------------------------------------------------------
# Argument resolution helpers
# ---------------------------------------------------------------------------

def _resolve_session_specs(args) -> Tuple[list, object]:
    """Returns (specs_list, worklist). Imports session lazily so the
    --prescrew yaml mode doesn't pay for worklist deps it doesn't need.
    """
    from one_assembly.worklist import WorkList
    from one_assembly.ScrewOperation.session import parse_screw_session_specs

    wl_kwargs = {}
    if args.worklist_root is not None:
        wl_kwargs['yaml_path'] = os.path.join(args.worklist_root, 'yamls')
        wl_kwargs['mesh_path'] = os.path.join(args.worklist_root, 'meshes')
        wl_kwargs['grasp_path'] = os.path.join(args.worklist_root, 'grasps')
    worklist = WorkList(**wl_kwargs)
    worklist.init_pose(args.layout)
    specs = parse_screw_session_specs(worklist, args.session, layout_name=args.layout)
    return specs, worklist


def _phase_model_dir(args, phase: str) -> str:
    """Pick the right --model_dir for this phase."""
    pick = getattr(args, 'model_dir_pick', None)
    place = getattr(args, 'model_dir_place', None)
    fallback = getattr(args, 'model_dir', None)
    if phase == 'pick' and pick is not None:
        return pick
    if phase == 'place' and place is not None:
        return place
    if fallback is not None:
        return fallback
    raise SystemExit(
        f"Need --model_dir_{phase} or --model_dir for the {phase!r} phase."
    )


def _resolve_prescrew_for_phase(
    rgt_arm,
    worklist,
    spec,
    *,
    prescrew_offset: float,
    flip_axis: bool,
    rgt_ee_override: Optional[float],
):
    from one_assembly.ScrewOperation.session import prescrew_qs_for_session
    sol = prescrew_qs_for_session(
        rgt_arm, worklist, spec,
        prescrew_offset=prescrew_offset,
        flip_axis=flip_axis,
    )
    ee = np.asarray([rgt_ee_override], dtype=np.float32) if rgt_ee_override is not None else None
    return sol.rgt_qs, ee


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--prescrew",
                           help="YAML with rgt_qs (and optional rgt_ee_qs); pairs with --model_dir")
    src_group.add_argument("--session",
                           help="ScrewSession string: phase-tagged (rly_scrw_pick / rly_scrw_place) "
                                "for single phase, bare (rly_scrw) for full-screw (pick+place).")

    ap.add_argument("--model_dir", default=None,
                    help="ViT model dir for single-phase modes (--prescrew or --session phase-tagged).")
    ap.add_argument("--model_dir_pick", default=None,
                    help="ViT model dir for the pick phase (full-screw mode).")
    ap.add_argument("--model_dir_place", default=None,
                    help="ViT model dir for the place phase (full-screw mode).")

    ap.add_argument("--worklist_root", default=None)
    ap.add_argument("--layout", default="home")
    ap.add_argument("--prescrew_offset", type=float, default=0.005)
    ap.add_argument("--rgt_ee", type=float, default=None,
                    help="Override SD shank position [m] applied at the prescrew waypoint (single-phase mode)")
    ap.add_argument("--rgt_ee_pick", type=float, default=None,
                    help="SD shank position [m] at the pick prescrew (full-screw mode)")
    ap.add_argument("--rgt_ee_place", type=float, default=None,
                    help="SD shank position [m] at the place prescrew (full-screw mode)")
    ap.add_argument("--flip_axis", action="store_true")
    ap.add_argument("--cameras_yaml", default=None)
    ap.add_argument("--episode_dir", default=None)
    ap.add_argument("--latency", type=float, default=None)
    ap.add_argument("--skip_plan", action="store_true",
                    help="Skip sending the approach plan (assume bridge already in policy phase)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Viewer + robot mirror
    base = ovw.World(cam_pos=(1.6, 0.0, 1.0), cam_lookat_pos=(0.4, 0.0, 0.3),
                     toggle_auto_cam_orbit=False)
    scene = base.scene
    ossop.frame(length_scale=0.15).attach_to(scene)
    robot = KHIBunri()
    robot.attach_to(scene)
    rgt_arm = robot.rgt_arm

    # ---- Episode dir
    ep_dir = args.episode_dir
    if ep_dir is None:
        ep_dir = make_mode_dir(BASE_DIR, "infer", sequence="auto", mode="auto")
    os.makedirs(ep_dir, exist_ok=True)
    print(f"episode dir: {ep_dir}")

    # ---- Resolve phases
    phase_runners: List[PhaseRunner] = []
    if args.prescrew is not None:
        # Single phase, yaml-driven
        model_dir = args.model_dir
        if model_dir is None:
            raise SystemExit("--prescrew requires --model_dir.")
        sol = resolve_prescrew(yaml_path=args.prescrew)
        prescrew_rgt_qs, prescrew_ee_qs = sol.rgt_qs, sol.rgt_ee_qs
        if args.latency is not None and prescrew_ee_qs is None and args.rgt_ee is not None:
            prescrew_ee_qs = np.asarray([args.rgt_ee], dtype=np.float32)
        config, model, spiral_list = _load_model_dir(model_dir, device)
        if args.latency is not None:
            config = config.model_copy(update={"latency": float(args.latency)})
        save_config(config, os.path.join(ep_dir, "config.yaml"))
        img_dir, csv_path = _make_phase_dirs(ep_dir, "prescrew")
        phase_runners.append(PhaseRunner(
            label="prescrew", phase="place",  # historical default
            config=config, model=model, spiral_list=spiral_list,
            img_dir=img_dir, csv_path=csv_path,
            prescrew_rgt_qs=prescrew_rgt_qs, prescrew_ee_qs=prescrew_ee_qs,
        ))
    else:
        # Session-driven: 1 or 2 phases
        specs, worklist = _resolve_session_specs(args)
        rgt_ee_overrides = {'pick': args.rgt_ee_pick, 'place': args.rgt_ee_place}
        if len(specs) == 1 and args.rgt_ee is not None:
            rgt_ee_overrides[specs[0].phase] = args.rgt_ee
        for spec in specs:
            model_dir = _phase_model_dir(args, spec.phase)
            cfg, model, spiral_list = _load_model_dir(model_dir, device)
            if args.latency is not None:
                cfg = cfg.model_copy(update={"latency": float(args.latency)})
            label = f"{spec.phase}_prescrew"
            img_dir, csv_path = _make_phase_dirs(ep_dir, label)
            save_config(cfg, os.path.join(ep_dir, label, "config.yaml"))
            prescrew_rgt_qs, prescrew_ee_qs = _resolve_prescrew_for_phase(
                rgt_arm, worklist, spec,
                prescrew_offset=args.prescrew_offset,
                flip_axis=args.flip_axis,
                rgt_ee_override=rgt_ee_overrides.get(spec.phase),
            )
            phase_runners.append(PhaseRunner(
                label=label, phase=spec.phase,
                config=cfg, model=model, spiral_list=spiral_list,
                img_dir=img_dir, csv_path=csv_path,
                prescrew_rgt_qs=prescrew_rgt_qs, prescrew_ee_qs=prescrew_ee_qs,
            ))
            print(f"phase {spec.phase}: target={spec.target_token!r} "
                  f"history={spec.history_string!r} model_dir={model_dir!r}")

    # ---- Bridge
    cameras_yaml = args.cameras_yaml or os.path.join(BASE_DIR, "config", "cameras.yaml")
    with CorrectionBridgeClient() as bridge, DualCameraRecorder(
        video_path=phase_runners[0].img_dir, cameras_yaml=cameras_yaml, toggle_dbg=False,
    ) as raw_camera:
        camera = _RedirectCamera(raw_camera)
        if not bridge.wait_for_joint_state("right", timeout=5.0):
            print("WARNING: no /right/joint_states received within 5s; proceeding open-loop.")

        if not args.skip_plan:
            seed_rgt = bridge.latest_rgt_qs
            if seed_rgt is None:
                seed_rgt = rgt_arm.home_qs
            initial_state = DualRobotState(
                lft_qs=np.asarray(robot.lft_arm.home_qs, dtype=np.float32),
                lft_ee_qs=np.zeros(2, dtype=np.float32),
                rgt_qs=np.asarray(seed_rgt, dtype=np.float32),
                rgt_ee_qs=np.asarray(phase_runners[0].prescrew_ee_qs if phase_runners[0].prescrew_ee_qs is not None else [0.0],
                                      dtype=np.float32),
            )
            if len(phase_runners) == 1:
                plan = build_correction_approach_plan(
                    initial_state=initial_state,
                    prescrew_rgt_qs=phase_runners[0].prescrew_rgt_qs,
                    rgt_ee_qs_at_end=phase_runners[0].prescrew_ee_qs,
                    label=phase_runners[0].label,
                )
                plan_dict = plan_to_bridge_dict(plan, plan_id="screw_correction_approach")
            else:
                plan, indices = build_multi_phase_correction_plan(
                    initial_state=initial_state,
                    phases=[PrescrewPhase(
                        rgt_qs=pr.prescrew_rgt_qs,
                        label=pr.label,
                        rgt_ee_qs=pr.prescrew_ee_qs,
                    ) for pr in phase_runners],
                )
                plan_dict = plan_to_bridge_dict_with_indices(
                    plan, indices, plan_id="screw_correction_full",
                )
            bridge.send_plan(plan_dict)
            print("approach plan published; waiting for first policy phase...")
            if not bridge.wait_for_status("waiting_for_policy", timeout=60.0):
                print("ERROR: bridge did not enter waiting_for_policy within 60s.")
                sys.exit(2)

        # Sequencer drives all phases
        sequencer = _PhaseSequencer(
            bridge=bridge, camera=camera, robot_rgt_arm=rgt_arm,
            phase_runners=phase_runners, device=device,
        )

        ghost = ossop.frame(pos=np.zeros(3), rotmat=np.eye(3), length_scale=0.05,
                            color_mat=ouc.CoordColor.MYC, alpha=0.6)
        ghost.attach_to(scene)

        def tick(_dt):
            if not sequencer.tick():
                sequencer.close()
                base.close()
            tf = rgt_arm.gl_tcp_tf
            ghost.pos = tf[:3, 3]
            ghost.rotmat = tf[:3, :3]

        base.schedule_interval(tick, interval=0.05)
        try:
            base.run()
        finally:
            sequencer.close()


if __name__ == "__main__":
    main()
