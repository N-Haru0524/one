"""End-to-end screw-correction runner (inference variant).

Pipeline:
  1. Boot one viewer + KHIBunri model mirror
  2. Connect to one_planner_bridge via ROS2 (rclpy)
  3. Publish a single-segment SyncPlan with policy_after=True to drive the
     right arm to a prescrew pose
  4. Wait for status == 'waiting_for_policy'
  5. Capture → ViT infer → IK → send_action loop until class 0 → send_done

The prescrew joint configuration is loaded from a YAML; it can be produced
once with `data_collector.py --record-prescrew` or hand-edited.

Usage:
  ./codex_python.sh -m one_assembly.ScrewOperation.screw_correction_run \\
      --model_dir <model_dir> --prescrew prescrew.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

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
from one_assembly.ScrewOperation.config import load_config, save_config
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


def _build_inference_label_source(config, model, device, spiral_list) -> LabelSource:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="Directory containing model.pt and config.yaml")
    ap.add_argument("--prescrew", required=True,
                    help="YAML with rgt_qs (and optional rgt_ee_qs) of the prescrew pose")
    ap.add_argument("--cameras_yaml", default=None,
                    help="Override path to cameras.yaml")
    ap.add_argument("--episode_dir", default=None,
                    help="Output directory. Default: datasets/<seq>/<mode>/infer/<NNN>")
    ap.add_argument("--latency", type=float, default=None,
                    help="Override config.latency [s]")
    ap.add_argument("--skip_plan", action="store_true",
                    help="Skip sending the approach plan (assume bridge already in policy phase)")
    args = ap.parse_args()

    config = load_config(os.path.join(args.model_dir, "config.yaml"))
    if args.latency is not None:
        config = config.model_copy(update={"latency": float(args.latency)})
    model_path = os.path.join(args.model_dir, "model.pt")
    spiral_list = hex_ring_abs(config.num_classes, step=config.spiral_step)

    if args.episode_dir is None:
        ep_dir = make_mode_dir(BASE_DIR, "infer", sequence=config.sequence, mode=config.mode)
    else:
        ep_dir = args.episode_dir
        os.makedirs(ep_dir, exist_ok=True)
        os.makedirs(os.path.join(ep_dir, "images"), exist_ok=True)
    img_dir = os.path.join(ep_dir, "images")
    save_config(config, os.path.join(ep_dir, "config.yaml"))
    csv_path = os.path.join(ep_dir, "samples.csv")
    print(f"episode dir: {ep_dir}")

    # ---- Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_vit(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ---- Viewer + robot mirror
    base = ovw.World(cam_pos=(1.6, 0.0, 1.0), cam_lookat_pos=(0.4, 0.0, 0.3),
                     toggle_auto_cam_orbit=False)
    scene = base.scene
    ossop.frame(length_scale=0.15).attach_to(scene)
    robot = KHIBunri()
    robot.attach_to(scene)

    rgt_arm = robot.rgt_arm
    _sol = resolve_prescrew(yaml_path=args.prescrew)
    prescrew_rgt_qs, prescrew_ee_qs = _sol.rgt_qs, _sol.rgt_ee_qs

    # ---- Bridge
    cameras_yaml = args.cameras_yaml or os.path.join(BASE_DIR, "config", "cameras.yaml")

    loop_cfg = CorrectionLoopConfig(
        img_dir=img_dir,
        csv_path=csv_path,
        log_key="sequence",
        spiral_step=config.spiral_step,
        num_classes=config.num_classes,
        latency=config.latency,
        end_on_zero_class=True,
    )
    label_source = _build_inference_label_source(config, model, device, spiral_list)

    with CorrectionBridgeClient() as bridge, DualCameraRecorder(
        video_path=img_dir, cameras_yaml=cameras_yaml, toggle_dbg=False,
    ) as camera:
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
                rgt_ee_qs=np.asarray(prescrew_ee_qs if prescrew_ee_qs is not None else [0.0], dtype=np.float32),
            )
            plan = build_correction_approach_plan(
                initial_state=initial_state,
                prescrew_rgt_qs=prescrew_rgt_qs,
                rgt_ee_qs_at_end=prescrew_ee_qs,
            )
            plan_dict = plan_to_bridge_dict(plan, plan_id="screw_correction_approach")
            bridge.send_plan(plan_dict)
            print("approach plan published; waiting for policy phase...")
            if not bridge.wait_for_status("waiting_for_policy", timeout=60.0):
                print("ERROR: bridge did not enter waiting_for_policy within 60s.")
                sys.exit(2)

        # Sync sim mirror to the actual right-arm joint state if available
        cur_rgt = bridge.latest_rgt_qs if bridge.latest_rgt_qs is not None else prescrew_rgt_qs
        rgt_arm.fk(qs=np.asarray(cur_rgt, dtype=np.float32))

        # Initial photo (idx=0) as the "before any correction" reference
        camera.take_photo_with_num(0)

        correction = CorrectionLoop(
            bridge=bridge,
            camera=camera,
            robot_rgt_arm=rgt_arm,
            label_source=label_source,
            config=loop_cfg,
        )
        correction.log_initial()

        ghost = ossop.frame(pos=np.zeros(3), rotmat=np.eye(3), length_scale=0.05,
                            color_mat=ouc.CoordColor.MYC, alpha=0.6)
        ghost.attach_to(scene)

        def tick(_dt):
            if not correction.tick():
                correction.close()
                base.close()
            tf = rgt_arm.gl_tcp_tf
            ghost.pos = tf[:3, 3]
            ghost.rotmat = tf[:3, :3]

        base.schedule_interval(tick, interval=0.05)
        try:
            base.run()
        finally:
            correction.close()


if __name__ == "__main__":
    main()
