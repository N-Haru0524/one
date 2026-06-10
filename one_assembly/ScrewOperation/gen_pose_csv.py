"""Offline pose-CSV generator for the Isaac sim dataset pipeline.

Runs the same prescrew resolver + ``hex_ring_abs`` spiral that
``data_collector.py`` uses, but writes the result to a CSV that
``Isaac_sim/scripts/dataset.py`` can consume directly. No ROS, no camera, no
real robot — purely ``one`` IK on KHIBunri's right arm.

Output CSV columns (DOF names match the Isaac articulation):

    right_joint1, ..., right_joint6,                       # arm DOFs (rad)
    right_sd_shank_joint                                   # only if --rgt_ee_extension
    label,                                                 # spiral class id
    dx, dy,                                                # absolute TCP offset (m)
    tcp_x, tcp_y, tcp_z,                                   # FK-derived TCP position (m)
    r6d_0..r6d_5                                           # Zhou-6D TCP rotation

DOF values are SI: revolute joints in radians, ``right_sd_shank_joint`` in
meters expressed as "extension from home" (0..0.033164), matching dataset.py's
``csv_to_raw`` convention. The trailing meta columns mirror the schema written
by the real-robot ``data_collector.py`` so training scripts can read either
source uniformly.

Each row k corresponds to spiral class k. Row 0 is the prescrew centre
(dx=dy=0); rows 1..num_classes-1 are absolute TCP offsets
``prescrew_pos + R[:,0]*spiral_list[k][0] + R[:,1]*spiral_list[k][1]`` solved
with IK seeded on prescrew_qs. The ``tcp_*`` / ``r6d_*`` columns are recorded
from FK of the IK solution (not from the target), so they reflect any
sub-millimetre IK residual. IK failures are dropped from the CSV and listed in
the companion ``.gen.yaml`` so the missing class indices are auditable.

The companion ``<out>.gen.yaml`` records the prescrew config + spiral params so
``(dx, dy)`` for each label can be reconstructed at training time.

Default output layout (matches data_collector + train_vit_spiral conventions):

    one_assembly/ScrewOperation/datasets/{stage}/{NNN}/
        poses.csv
        poses.csv.gen.yaml
        config.yaml
        images/                   ← filled by Isaac dataset.py
        samples.csv               ← filled by Isaac dataset.py
        meta.jsonl                ← filled by Isaac dataset.py

``--stage`` is ``train`` by default. ``NNN`` is auto-incremented.

Examples:

    # Default: lands under datasets/train/NNN/
    uv run python -m one_assembly.ScrewOperation.gen_pose_csv \\
        --session rly_scrw_pick \\
        --worklist_root one_assembly/worklists/electric_assembly \\
        --prescrew_offset 0.002 --rgt_ee_extension 0.005

    # Explicit path (no auto layout)
    uv run python -m one_assembly.ScrewOperation.gen_pose_csv \\
        --prescrew /path/to/prescrew.yaml --out /tmp/poses.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from one_assembly.ScrewOperation.config import ScrewConfig, save_config
from one_assembly.ScrewOperation.prescrew import resolve_prescrew
from one_assembly.ScrewOperation.spiral_metry import hex_ring_abs
from one_assembly.ScrewOperation.utils import make_mode_dir, rotmat_to_rot6d


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARM_JOINT_NAMES = [f"right_joint{i}" for i in range(1, 7)]
SHANK_DOF = "right_sd_shank_joint"
SHANK_EXT_MAX = 0.033164  # matches dataset.py SHANK_HOME_RAW magnitude

# Container-side path where /workspace_screw is mounted (see Isaac_sim/scripts/isaac.sh
# SRC_MOUNTS). Used only for the convenience hint printed at the end so the user
# can copy-paste the dataset.py invocation.
CONTAINER_SCREW_MOUNT = "/workspace_screw"


def _build_robot():
    from one_assembly.robots.khi_bunri.khi_bunri import KHIBunri
    return KHIBunri()


def _build_worklist(args):
    """Construct WorkList with the same world-frame placement that
    ``nagai/test_assembly_sequence_visual.py`` uses, so the prescrew TCP we
    solve here matches what the preview viewer (and the real assembly cell)
    sees. Returns ``None`` when no worklist context is needed (rare — kept for
    completeness)."""
    from one_assembly.worklist import WorkList
    import one.utils.constant as ouc
    import one.utils.math as oum
    wl_kwargs: dict = {
        "pos": oum.vec(0.2 + 0.09 + 0.035, 0, 0.11 + 0.018 + 0.0902),
        "collision_type": ouc.CollisionType.MESH,
    }
    if args.worklist_root is not None:
        wl_kwargs["yaml_path"] = os.path.join(args.worklist_root, "yamls")
        wl_kwargs["mesh_path"] = os.path.join(args.worklist_root, "meshes")
        wl_kwargs["grasp_path"] = os.path.join(args.worklist_root, "grasps")
    worklist = WorkList(**wl_kwargs)
    worklist.init_pose(seed=args.layout)
    return worklist


def _resolve_prescrew(args, robot, worklist) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (prescrew_qs, prescrew_pos, prescrew_rotmat) in float32."""
    if args.session is not None:
        from one_assembly.ScrewOperation.session import (
            parse_screw_session_string,
            prescrew_qs_for_session,
        )
        if worklist is None:
            raise RuntimeError("--session requires a worklist (got None).")
        spec = parse_screw_session_string(worklist, args.session, layout_name=args.layout)
        sol = prescrew_qs_for_session(
            robot.rgt_arm, worklist, spec,
            prescrew_offset=args.prescrew_offset,
            flip_axis=args.flip_axis,
        )
        print(f"session: target={spec.target_token!r} phase={spec.phase!r} "
              f"history={spec.history_string!r}", file=sys.stderr)
    else:
        sol = resolve_prescrew(yaml_path=args.prescrew)

    qs = np.asarray(sol.rgt_qs, dtype=np.float32)
    # YAML path returns zero pos / identity rotmat; recompute via FK either way.
    robot.rgt_arm.fk(qs=qs)
    tf = robot.rgt_arm.gl_tcp_tf
    pos = np.asarray(tf[:3, 3], dtype=np.float32)
    rotmat = np.asarray(tf[:3, :3], dtype=np.float32)
    return qs, pos, rotmat


def _generate_rows(
    rgt_arm,
    prescrew_qs: np.ndarray,
    prescrew_pos: np.ndarray,
    prescrew_rotmat: np.ndarray,
    spiral_list: np.ndarray,
) -> tuple[list[dict], list[int]]:
    """For each spiral class, solve IK at the absolute spiral position.

    Returns (rows, failed_classes). Each row is a dict with keys:
    ``label``, ``qs``, ``dx``, ``dy``, ``tcp_pos``, ``tcp_rotmat``. ``tcp_*``
    are computed via FK of the IK solution (i.e. the TCP actually reached
    after IK, not the IK target).
    """
    rows: list[dict] = []
    failed: list[int] = []
    num_classes = len(spiral_list)
    for cls in range(num_classes):
        dx = float(spiral_list[cls][0])
        dy = float(spiral_list[cls][1])
        if cls == 0:
            qs = prescrew_qs.copy()
        else:
            tgt_pos = (prescrew_pos
                       + prescrew_rotmat[:, 0] * dx
                       + prescrew_rotmat[:, 1] * dy).astype(np.float32)
            qs = rgt_arm.ik_tcp_nearest(
                tgt_rotmat=prescrew_rotmat,
                tgt_pos=tgt_pos,
                ref_qs=prescrew_qs,
            )
            if qs is None:
                failed.append(cls)
                continue
            qs = np.asarray(qs, dtype=np.float32)
        rgt_arm.fk(qs=qs)
        tf = rgt_arm.gl_tcp_tf
        rows.append({
            "label": cls,
            "qs": qs,
            "dx": dx,
            "dy": dy,
            "tcp_pos": np.asarray(tf[:3, 3], dtype=np.float32),
            "tcp_rotmat": np.asarray(tf[:3, :3], dtype=np.float32),
        })
    return rows, failed


def _write_csv(
    out_path: Path,
    rows: list[dict],
    *,
    rgt_ee_extension: float | None,
) -> None:
    columns = list(ARM_JOINT_NAMES)
    if rgt_ee_extension is not None:
        columns.append(SHANK_DOF)
    columns += [
        "label", "dx", "dy",
        "tcp_x", "tcp_y", "tcp_z",
        "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            qs = r["qs"]
            tcp_pos = r["tcp_pos"]
            r6d = rotmat_to_rot6d(r["tcp_rotmat"])
            out_row: dict = {name: f"{float(q):.6f}" for name, q in zip(ARM_JOINT_NAMES, qs)}
            if rgt_ee_extension is not None:
                out_row[SHANK_DOF] = f"{float(rgt_ee_extension):.6f}"
            out_row["label"] = r["label"]
            out_row["dx"] = f"{r['dx']:.6f}"
            out_row["dy"] = f"{r['dy']:.6f}"
            out_row["tcp_x"] = f"{float(tcp_pos[0]):.6f}"
            out_row["tcp_y"] = f"{float(tcp_pos[1]):.6f}"
            out_row["tcp_z"] = f"{float(tcp_pos[2]):.6f}"
            for i in range(6):
                out_row[f"r6d_{i}"] = f"{float(r6d[i]):.6f}"
            writer.writerow(out_row)


def _write_companion_config(
    out_path: Path,
    *,
    num_classes: int,
    spiral_step: float,
    prescrew_offset: float,
    flip_axis: bool,
    rgt_ee_extension: float | None,
    prescrew_pos: np.ndarray,
    prescrew_qs: np.ndarray,
    n_rows: int,
    failed: list[int],
    source_label: str,
) -> Path:
    config_path = out_path.with_suffix(out_path.suffix + ".gen.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump({
            "source": source_label,
            "num_classes": int(num_classes),
            "spiral_step": float(spiral_step),
            "prescrew_offset": float(prescrew_offset),
            "flip_axis": bool(flip_axis),
            "rgt_ee_extension": (None if rgt_ee_extension is None
                                  else float(rgt_ee_extension)),
            "prescrew_pos": [float(v) for v in prescrew_pos],
            "prescrew_qs": [float(v) for v in prescrew_qs],
            "n_rows_written": int(n_rows),
            "ik_failed_classes": [int(c) for c in failed],
        }, f, sort_keys=False)
    return config_path


def _preview(robot, rows: list[dict], *, worklist=None) -> None:
    import one.scene.scene_object_primitive as ossop
    import one.viewer.world as ovw
    # Camera params mirror test_assembly_sequence_visual.py for a known-good
    # framing of the bunri cell.
    base = ovw.World(
        cam_pos=(3.1, 1.9, 2.0),
        cam_lookat_pos=(0.18, 0.0, 0.55),
        toggle_auto_cam_orbit=False,
    )
    scene = base.scene
    ossop.frame(length_scale=0.25, radius_scale=1.2).attach_to(scene)
    if worklist is not None:
        worklist.attach_to(scene)
    robot.attach_to(scene)
    # Anchor the prescrew TCP frame so it's easy to eyeball offset alignment.
    tcp_frame = ossop.frame(
        pos=robot.rgt_tcp_tf[:3, 3], rotmat=robot.rgt_tcp_tf[:3, :3],
        length_scale=0.12, radius_scale=0.8,
    )
    tcp_frame.attach_to(scene)
    state = {"i": 0}

    def tick(_dt):
        i = state["i"]
        if i >= len(rows):
            return
        robot.rgt_arm.fk(qs=rows[i]["qs"])
        tcp_frame.set_rotmat_pos(
            rotmat=robot.rgt_tcp_tf[:3, :3], pos=robot.rgt_tcp_tf[:3, 3],
        )
        state["i"] = i + 1

    base.schedule_interval(tick, interval=0.15)
    base.run()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prescrew", help="YAML with rgt_qs (and optional rgt_ee_qs)")
    src.add_argument("--session",
                     help="ScrewSession string (e.g. 'rly_scrw_pick' or "
                          "'rly_scrw_place:wrkbnch-brckt-cpctr-rly')")
    ap.add_argument("--worklist_root", default=None,
                    help="WorkList asset root (defaults to one_assembly/worklists/"
                         "electric_assembly)")
    ap.add_argument("--layout", default="home", help="WorkList layout name")
    ap.add_argument("--prescrew_offset", type=float, default=0.005)
    ap.add_argument("--flip_axis", action="store_true")
    ap.add_argument("--num_classes", type=int, default=91)
    ap.add_argument("--spiral_step", type=float, default=0.0008)
    ap.add_argument("--rgt_ee_extension", default=None,
                    help="SD shank extension from home in meters (0..0.033164). "
                         "Pass 'max' as a shortcut for fully-extended (0.033164, "
                         "= raw 0, typical prescrew state). Omit to leave the "
                         "shank column out of the CSV (DOF then stays at stage "
                         "default).")
    # Output layout. Default: datasets/{stage}/{NNN}/ (current convention,
    # matches train_vit_spiral / eval_vit_spiral_hist directory expectations).
    # --sequence / --mode are optional metadata that, when given, insert
    # additional path segments (datasets/{sequence}/{mode}/{stage}/{NNN}/) AND
    # populate the same fields in config.yaml.
    ap.add_argument("--stage", default="train",
                    help="Dataset stage subdir (train | infer | model). Default train.")
    ap.add_argument("--sequence", default=None,
                    help="Task sequence label (e.g. rly_scrw). Recorded in "
                         "config.yaml only — does NOT affect the directory path. "
                         "When omitted and --session is given, defaults to the "
                         "full --session string so the screw index (count of "
                         "scrw in the history) is recoverable from config.yaml.")
    ap.add_argument("--mode", default=None,
                    help="Operation mode (e.g. pick, place). Recorded in "
                         "config.yaml only — does NOT affect the directory path.")
    ap.add_argument("--description", default=None,
                    help="Free-text note saved to config.yaml.description.")
    ap.add_argument("--out", default=None,
                    help="Explicit output CSV path. Overrides the make_mode_dir "
                         "layout. Useful for one-off generation outside the dataset tree.")
    ap.add_argument("--preview", action="store_true",
                    help="After writing CSV, open the pyglet viewer and step "
                         "through every IK solution.")
    args = ap.parse_args()

    if args.rgt_ee_extension is not None:
        if isinstance(args.rgt_ee_extension, str):
            if args.rgt_ee_extension.lower() == "max":
                args.rgt_ee_extension = SHANK_EXT_MAX
            else:
                try:
                    args.rgt_ee_extension = float(args.rgt_ee_extension)
                except ValueError:
                    raise SystemExit(f"--rgt_ee_extension: expected a number or "
                                     f"'max', got {args.rgt_ee_extension!r}")
        if not (0.0 <= args.rgt_ee_extension <= SHANK_EXT_MAX + 1e-6):
            raise SystemExit(f"--rgt_ee_extension {args.rgt_ee_extension} out of range "
                             f"[0.0, {SHANK_EXT_MAX}]")

    if args.out is not None:
        out_path = Path(args.out)
        ep_dir: Optional[Path] = None
    else:
        # Directory layout is always datasets/{stage}/{NNN}/ — flat under stage.
        # --sequence / --mode are recorded only inside config.yaml (not in path).
        ep_dir = Path(make_mode_dir(BASE_DIR, args.stage))
        out_path = ep_dir / "poses.csv"

    robot = _build_robot()
    # The IK chain ends at the SD bit tip; bit tip world position depends on
    # the shank extension. If we leave the screwdriver at its retracted default
    # but Isaac (or the real robot) later drives the shank to --rgt_ee_extension,
    # the bit ends up ``rgt_ee_extension`` metres deeper than the prescrew
    # target, i.e. it crashes into the screw head. Match the shank now so the
    # IK target reflects the same kinematics used during execution.
    if args.rgt_ee_extension is not None:
        sd = robot.rgt_screwdriver
        shank_raw = float(args.rgt_ee_extension) - SHANK_EXT_MAX
        lo, hi = float(sd.shank_range[0]), float(sd.shank_range[1])
        if not (lo - 1e-6 <= shank_raw <= hi + 1e-6):
            raise SystemExit(
                f"--rgt_ee_extension {args.rgt_ee_extension} → raw shank "
                f"{shank_raw} out of range {sd.shank_range.tolist()}"
            )
        sd.set_shank_len(shank_raw)
        print(f"shank set: csv_ext={args.rgt_ee_extension} m → raw={shank_raw:.6f} m",
              file=sys.stderr)

    worklist = _build_worklist(args) if args.session is not None else None
    prescrew_qs, prescrew_pos, prescrew_rotmat = _resolve_prescrew(args, robot, worklist)
    print(f"prescrew_pos = {prescrew_pos.tolist()}", file=sys.stderr)
    print(f"prescrew_qs  = {prescrew_qs.tolist()}", file=sys.stderr)

    spiral_list = hex_ring_abs(args.num_classes, step=args.spiral_step)
    rows, failed = _generate_rows(
        robot.rgt_arm, prescrew_qs, prescrew_pos, prescrew_rotmat, spiral_list,
    )
    if failed:
        print(f"WARNING: IK failed for {len(failed)} classes: {failed}", file=sys.stderr)

    _write_csv(out_path, rows, rgt_ee_extension=args.rgt_ee_extension)
    source_label = f"session:{args.session}" if args.session else f"prescrew:{args.prescrew}"
    cfg_path = _write_companion_config(
        out_path,
        num_classes=args.num_classes,
        spiral_step=args.spiral_step,
        prescrew_offset=args.prescrew_offset,
        flip_axis=args.flip_axis,
        rgt_ee_extension=args.rgt_ee_extension,
        prescrew_pos=prescrew_pos,
        prescrew_qs=prescrew_qs,
        n_rows=len(rows),
        failed=failed,
        source_label=source_label,
    )
    print(f"wrote {len(rows)} rows -> {out_path}", file=sys.stderr)
    print(f"wrote companion config -> {cfg_path}", file=sys.stderr)

    if ep_dir is not None:
        # Mirror data_collector.py: drop a ScrewConfig snapshot in the ep dir.
        screw_cfg = ScrewConfig(
            description=args.description or "",
            # Fall back to the full session string so the screw index (number
            # of scrw tokens in the history) stays recoverable from config.yaml.
            sequence=args.sequence or args.session or "",
            mode=args.mode or "",
            num_classes=args.num_classes,
            spiral_step=args.spiral_step,
        )
        cfg_yaml = ep_dir / "config.yaml"
        save_config(screw_cfg, str(cfg_yaml))
        print(f"wrote ScrewConfig snapshot -> {cfg_yaml}", file=sys.stderr)
        # Print the path dataset.py should be invoked with (container view).
        try:
            rel = ep_dir.resolve().relative_to(Path(BASE_DIR).resolve())
            container_ep = f"{CONTAINER_SCREW_MOUNT}/{rel.as_posix()}"
            print("", file=sys.stderr)
            print("Next step (Isaac sim):", file=sys.stderr)
            print(f"  ./scripts/isaac.sh run /workspace/scripts/dataset.py "
                  f"--ep_dir {container_ep}", file=sys.stderr)
        except ValueError:
            pass

    if args.preview:
        _preview(robot, rows, worklist=worklist)

    return 0


if __name__ == "__main__":
    sys.exit(main())
