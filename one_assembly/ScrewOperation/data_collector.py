"""Training data collector: drives the right arm through a precomputed
hex_ring_abs spiral around a prescrew pose, snapping a dual-camera pair at
each step.

Same bridge/policy-phase mechanism as `screw_correction_run.py`. The only
difference is the label source (deterministic spiral indexer) and the sign
(we APPLY the offset, not invert it).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import one.utils.constant as ouc
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

from one_assembly.assembly_data import DualRobotState
from one_assembly.robots.khi_bunri.khi_bunri import KHIBunri

from one_assembly.ScrewOperation.approach_plan import (
    build_correction_approach_plan,
    plan_to_bridge_dict,
)
from one_assembly.ScrewOperation.bridge_io import CorrectionBridgeClient
from one_assembly.ScrewOperation.camera import DualCameraRecorder
from one_assembly.ScrewOperation.config import (
    ScrewConfig,
    load_config,
    merge_cli_args,
    save_config,
)
from one_assembly.ScrewOperation.correction_loop import (
    CorrectionLoop,
    CorrectionLoopConfig,
    LabelSource,
)
from one_assembly.ScrewOperation.prescrew import resolve_prescrew
from one_assembly.ScrewOperation.spiral_metry import hex_ring_abs
from one_assembly.ScrewOperation.utils import make_mode_dir


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _build_spiral_label_source(num_classes: int, spiral_list) -> tuple[LabelSource, dict]:
    state = {"cls": 1}  # idx=0 is the centre, written by log_initial; first commanded class is 1

    def _src(sample_idx: int, img_dir: str) -> tuple[int, float, float]:
        cls = state["cls"] % num_classes
        if cls == 0:
            # we wrap around — treat as "centre" again, no offset
            dx, dy = 0.0, 0.0
        else:
            dx, dy = spiral_list[cls]
        state["cls"] = (state["cls"] + 1) % num_classes
        return int(cls), float(dx), float(dy)

    return _src, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    prescrew_group = ap.add_mutually_exclusive_group(required=True)
    prescrew_group.add_argument("--prescrew",
                                help="YAML with rgt_qs (and optional rgt_ee_qs) of the prescrew pose")
    prescrew_group.add_argument("--session",
                                help="ScrewSession string (e.g. 'rly_scrw_pick' or "
                                     "'rly_scrw_place:wrkbnch-brckt-cpctr-rly')")
    ap.add_argument("--worklist_root", default=None,
                    help="WorkList asset root (defaults to one_assembly/worklists/electric_assembly)")
    ap.add_argument("--layout", default="home",
                    help="WorkList layout name (default: home)")
    ap.add_argument("--prescrew_offset", type=float, default=0.005,
                    help="TCP offset along screw axis [m] (default 5 mm)")
    ap.add_argument("--rgt_ee", type=float, default=None,
                    help="Optional SD shank position [m] at the prescrew waypoint")
    ap.add_argument("--flip_axis", action="store_true")
    ap.add_argument("--cameras_yaml", default=None)
    ap.add_argument("--episode_dir", default=None)
    ap.add_argument("--num_classes", type=int, default=None)
    ap.add_argument("--max_num_samples", type=int, default=None)
    ap.add_argument("--latency", type=float, default=None)
    ap.add_argument("--spiral_step", type=float, default=None)
    ap.add_argument("--description", type=str, default=None)
    ap.add_argument("--sequence", type=str, default=None,
                    help="ScrewConfig.sequence label used to build the output dir path")
    ap.add_argument("--mode", type=str, default=None)
    ap.add_argument("--data_source", type=str, default=None,
                    choices=("sim", "real"),
                    help="Provenance of the captured images (recorded in config.yaml)")
    ap.add_argument("--skip_plan", action="store_true")
    args = ap.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = ScrewConfig()
    config = merge_cli_args(config, args)

    if args.episode_dir is None:
        # Layout: datasets/train/{NNN}/. config.sequence / config.mode are
        # recorded only inside config.yaml — they no longer affect the path
        # (kept consistent with gen_pose_csv.py).
        ep_dir = make_mode_dir(BASE_DIR, "train")
    else:
        ep_dir = args.episode_dir
        os.makedirs(os.path.join(ep_dir, "images"), exist_ok=True)
    img_dir = os.path.join(ep_dir, "images")
    csv_path = os.path.join(ep_dir, "samples.csv")
    save_config(config, os.path.join(ep_dir, "config.yaml"))
    print(f"episode dir: {ep_dir}")

    spiral_list = hex_ring_abs(config.num_classes, step=config.spiral_step)

    base = ovw.World(cam_pos=(1.6, 0.0, 1.0), cam_lookat_pos=(0.4, 0.0, 0.3),
                     toggle_auto_cam_orbit=False)
    scene = base.scene
    ossop.frame(length_scale=0.15).attach_to(scene)
    robot = KHIBunri()
    robot.attach_to(scene)
    rgt_arm = robot.rgt_arm

    if args.session is not None:
        from one_assembly.worklist import WorkList
        from one_assembly.ScrewOperation.session import (
            parse_screw_session_string, prescrew_qs_for_session,
        )
        wl_kwargs = {}
        if args.worklist_root is not None:
            wl_kwargs['yaml_path'] = os.path.join(args.worklist_root, 'yamls')
            wl_kwargs['mesh_path'] = os.path.join(args.worklist_root, 'meshes')
            wl_kwargs['grasp_path'] = os.path.join(args.worklist_root, 'grasps')
        worklist = WorkList(**wl_kwargs)
        worklist.init_pose(args.layout)
        spec = parse_screw_session_string(worklist, args.session, layout_name=args.layout)
        _sol = prescrew_qs_for_session(
            rgt_arm, worklist, spec,
            prescrew_offset=args.prescrew_offset,
            flip_axis=args.flip_axis,
        )
        prescrew_rgt_qs = _sol.rgt_qs
        prescrew_ee_qs = (
            np.asarray([args.rgt_ee], dtype=np.float32) if args.rgt_ee is not None else None
        )
        print(f"session: target={spec.target_token!r} phase={spec.phase!r} "
              f"history={spec.history_string!r}")
    else:
        _sol = resolve_prescrew(yaml_path=args.prescrew)
        prescrew_rgt_qs, prescrew_ee_qs = _sol.rgt_qs, _sol.rgt_ee_qs

    cameras_yaml = args.cameras_yaml or os.path.join(BASE_DIR, "config", "cameras.yaml")

    loop_cfg = CorrectionLoopConfig(
        img_dir=img_dir,
        csv_path=csv_path,
        log_key="idx",
        spiral_step=config.spiral_step,
        num_classes=config.num_classes,
        latency=config.latency,
        end_on_zero_class=False,
        max_iterations=config.max_num_samples,
        sign=+1.0,  # APPLY the offset (collector mode)
    )
    label_source, _label_state = _build_spiral_label_source(config.num_classes, spiral_list)

    with CorrectionBridgeClient() as bridge, DualCameraRecorder(
        video_path=img_dir, cameras_yaml=cameras_yaml, toggle_dbg=False,
    ) as camera:
        bridge.wait_for_joint_state("right", timeout=5.0)

        if not args.skip_plan:
            seed_rgt = bridge.latest_rgt_qs
            if seed_rgt is None:
                seed_rgt = rgt_arm.home_qs
            initial_state = DualRobotState(
                lft_qs=np.asarray(robot.lft_arm.home_qs, dtype=np.float32),
                lft_ee_qs=np.zeros(2, dtype=np.float32),
                rgt_qs=np.asarray(seed_rgt, dtype=np.float32),
                rgt_ee_qs=np.asarray(prescrew_ee_qs if prescrew_ee_qs is not None else [0.0], dtype=np.float32),
            )
            plan = build_correction_approach_plan(
                initial_state=initial_state,
                prescrew_rgt_qs=prescrew_rgt_qs,
                rgt_ee_qs_at_end=prescrew_ee_qs,
            )
            plan_dict = plan_to_bridge_dict(plan, plan_id="screw_collector_approach")
            bridge.send_plan(plan_dict)
            print("approach plan published; waiting for policy phase...")
            if not bridge.wait_for_status("waiting_for_policy", timeout=60.0):
                print("ERROR: bridge did not enter waiting_for_policy within 60s.")
                sys.exit(2)

        cur_rgt = bridge.latest_rgt_qs if bridge.latest_rgt_qs is not None else prescrew_rgt_qs
        rgt_arm.fk(qs=np.asarray(cur_rgt, dtype=np.float32))

        camera.take_photo_with_num(0)

        collector = CorrectionLoop(
            bridge=bridge,
            camera=camera,
            robot_rgt_arm=rgt_arm,
            label_source=label_source,
            config=loop_cfg,
        )
        collector.log_initial()

        def tick(_dt):
            if not collector.tick():
                collector.close()
                base.close()

        base.schedule_interval(tick, interval=0.05)
        try:
            base.run()
        finally:
            collector.close()


if __name__ == "__main__":
    main()
