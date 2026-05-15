"""Capture the current robot pose into a prescrew.yaml.

Subscribes to /right/joint_states (and optionally /left/joint_states),
waits for a stable reading, and writes a yaml file in the schema expected
by screw_correction_run.py / data_collector.py:

    rgt_qs: [j1..j6]
    rgt_ee_qs: [shank_pos_m]    # optional, only when --rgt_ee provided

Typical usage on the real machine:
  1. Jog the right arm to the prescrew pose (TCP above the screw hole,
     screwdriver shank extended to the approach length).
  2. Run:
       source <ROS workspace>/install/setup.bash
       ./codex_python.sh -m one_assembly.ScrewOperation.capture_prescrew \\
           --out config/prescrew.yaml --rgt_ee -0.020
  3. The yaml can then be passed as --prescrew to data_collector or
     screw_correction_run.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import yaml

from one_assembly.ScrewOperation.bridge_io import (
    CorrectionBridgeClient,
    DEFAULT_LEFT_JOINT_NAMES,
    DEFAULT_RIGHT_JOINT_NAMES,
)


def capture(
    out_path: str,
    *,
    include_left: bool = False,
    rgt_ee_value: Optional[float] = None,
    lft_ee_values: Optional[list[float]] = None,
    timeout: float = 5.0,
    left_topic: str = "/left/joint_states",
    right_topic: str = "/right/joint_states",
) -> dict:
    with CorrectionBridgeClient(
        node_name="capture_prescrew",
        left_joint_state_topic=left_topic,
        right_joint_state_topic=right_topic,
        left_joint_names=DEFAULT_LEFT_JOINT_NAMES,
        right_joint_names=DEFAULT_RIGHT_JOINT_NAMES,
    ) as bridge:
        if not bridge.wait_for_joint_state("right", timeout=timeout):
            raise TimeoutError(
                f"No /right/joint_states within {timeout:.1f}s. Is ROS sourced "
                "and is the controller publishing?"
            )
        rgt_qs = bridge.latest_rgt_qs
        if rgt_qs is None:
            raise RuntimeError("Right joint state still None after wait — internal error.")
        payload: dict = {"rgt_qs": [float(x) for x in rgt_qs]}
        if rgt_ee_value is not None:
            payload["rgt_ee_qs"] = [float(rgt_ee_value)]
        if include_left:
            # bridge_io subscribes both already, but we only block on right.
            # Give the left topic a brief moment if not yet seen.
            if bridge.latest_lft_qs is None:
                bridge.wait_for_joint_state("left", timeout=timeout)
            if bridge.latest_lft_qs is not None:
                payload["lft_qs"] = [float(x) for x in bridge.latest_lft_qs]
                if lft_ee_values is not None:
                    payload["lft_ee_qs"] = [float(v) for v in lft_ee_values]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, default_flow_style=None, sort_keys=False)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Destination yaml path")
    ap.add_argument("--rgt_ee", type=float, default=None,
                    help="Right SD shank pose [m] in [-0.033164, 0.0]. Optional.")
    ap.add_argument("--include_left", action="store_true",
                    help="Also capture lft_qs (and lft_ee_qs if --lft_ee_values supplied)")
    ap.add_argument("--lft_ee_values", type=float, nargs="+", default=None,
                    help="Left 2FG7 ee_qs values (typically two finger positions)")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--left_topic", default="/left/joint_states")
    ap.add_argument("--right_topic", default="/right/joint_states")
    args = ap.parse_args()

    if args.rgt_ee is not None:
        if not -0.033164 <= args.rgt_ee <= 0.0:
            raise SystemExit(
                f"--rgt_ee {args.rgt_ee} is outside the bridge SD range "
                "[-0.033164, 0.0] (params.yaml right_sd_q_min_m / q_max_m)."
            )

    payload = capture(
        args.out,
        include_left=args.include_left,
        rgt_ee_value=args.rgt_ee,
        lft_ee_values=args.lft_ee_values,
        timeout=args.timeout,
        left_topic=args.left_topic,
        right_topic=args.right_topic,
    )
    print(f"wrote {args.out}:")
    print(yaml.dump(payload, default_flow_style=None, sort_keys=False))


if __name__ == "__main__":
    main()
