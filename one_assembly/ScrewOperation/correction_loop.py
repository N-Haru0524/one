"""Shared state machine for screw-correction policy phase.

The loop drives a single right-arm trajectory by:
  capture → label (inference OR scripted spiral) → IK → send_action → wait.

Used by both `screw_correction_run.py` (label = ViT prediction) and
`data_collector.py` (label = next spiral index). The two scripts plug in
different `LabelSource` strategies.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, Protocol

import numpy as np

import one.utils.math as oum

from one_assembly.ScrewOperation.bridge_io import CorrectionBridgeClient
from one_assembly.ScrewOperation.camera import DualCameraRecorder
from one_assembly.ScrewOperation.spiral_metry import hex_ring_abs
from one_assembly.ScrewOperation.utils import csv_writer, rotmat_to_rot6d


class LabelSource(Protocol):
    """Returns the current class id and (dx, dy) for the present iteration.

    Implementations:
      - InferenceLabelSource: ViT classification of the latest cam1/cam2 PNGs
      - SpiralLabelSource:    next index in a precomputed hex_ring_abs sequence
    """

    def __call__(self, sample_idx: int, img_dir: str) -> tuple[int, float, float]: ...


@dataclass
class CorrectionLogger:
    """Append-only CSV log keyed on either 'idx' (collector) or 'sequence' (infer)."""

    path: str
    key_name: str = "idx"

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._writer = csv_writer(
            self.path,
            fieldnames=[
                self.key_name, "time", "label", "dx", "dy",
                "tcp_x", "tcp_y", "tcp_z",
                "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5",
            ],
        )

    def write(self, key: int, label: int, dx: float, dy: float, tcp_pos: np.ndarray, tcp_rotmat: np.ndarray):
        r6d = rotmat_to_rot6d(tcp_rotmat)
        self._writer.write({
            self.key_name: key,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            "label": int(label),
            "dx": float(dx),
            "dy": float(dy),
            "tcp_x": float(tcp_pos[0]),
            "tcp_y": float(tcp_pos[1]),
            "tcp_z": float(tcp_pos[2]),
            "r6d_0": float(r6d[0]),
            "r6d_1": float(r6d[1]),
            "r6d_2": float(r6d[2]),
            "r6d_3": float(r6d[3]),
            "r6d_4": float(r6d[4]),
            "r6d_5": float(r6d[5]),
        })

    def close(self):
        self._writer.close()


@dataclass
class CorrectionLoopConfig:
    img_dir: str
    csv_path: str
    log_key: str = "sequence"  # collector uses 'idx', inference uses 'sequence'
    spiral_step: float = 0.0008
    num_classes: int = 91
    latency: float = 2.0
    # max |dx|, |dy| per step in meters. hex_ring_abs(91, step=0.0008) reaches
    # 4 mm at the outer ring; 5 mm leaves a small headroom while still acting
    # as a safety net against pathological label outputs.
    pos_step_clip: float = 0.005
    end_on_zero_class: bool = True
    max_iterations: int = 10_000
    sign: float = -1.0   # set to -1 to MOVE OPPOSITE to predicted offset (closes the loop);
                          # set to +1 in collector to APPLY the offset.

    def __post_init__(self):
        self.spiral_list = hex_ring_abs(self.num_classes, step=self.spiral_step)


@dataclass
class CorrectionLoop:
    """Drives bridge actions during the policy phase using a label source.

    Caller must provide:
    - bridge: CorrectionBridgeClient already in context (in policy phase)
    - camera: DualCameraRecorder (already constructed)
    - robot_rgt_arm: KHIBunriRS007L (or compatible). Used only for IK via
                     ik_tcp_nearest(tgt_rotmat, tgt_pos, ref_qs) and fk(qs).
    - label_source: LabelSource callable
    - config: CorrectionLoopConfig
    """

    bridge: CorrectionBridgeClient
    camera: DualCameraRecorder
    robot_rgt_arm: object
    label_source: LabelSource
    config: CorrectionLoopConfig
    on_step: Optional[Callable[[dict], None]] = None

    state: str = "IDLE"
    sample_idx: int = 1
    motion_start_ns: int = 0
    last_dx: float = 0.0
    last_dy: float = 0.0
    last_class_id: int = 0
    pending_target_qs: Optional[np.ndarray] = None
    _stop: bool = False
    _logger: Optional[CorrectionLogger] = field(default=None, init=False)

    def __post_init__(self):
        self._logger = CorrectionLogger(self.config.csv_path, key_name=self.config.log_key)

    def _current_tcp(self) -> tuple[np.ndarray, np.ndarray]:
        tf = self.robot_rgt_arm.gl_tcp_tf
        return tf[:3, 3].astype(np.float32), tf[:3, :3].astype(np.float32)

    def _log_initial(self):
        pos, rotmat = self._current_tcp()
        self._logger.write(0, label=0, dx=0.0, dy=0.0, tcp_pos=pos, tcp_rotmat=rotmat)

    def log_initial(self):
        """Public: call once after the right arm reaches prescrew and a photo
        has been taken with idx=0."""
        self._log_initial()

    def _capture(self) -> None:
        self.camera.take_photo_with_num(self.sample_idx)

    def _solve_target(self, dx: float, dy: float) -> Optional[np.ndarray]:
        pos, rotmat = self._current_tcp()
        # clip to avoid wild IK jumps
        s = float(self.config.sign)
        dx_c = float(np.clip(dx, -self.config.pos_step_clip, self.config.pos_step_clip))
        dy_c = float(np.clip(dy, -self.config.pos_step_clip, self.config.pos_step_clip))
        tgt_pos = pos + s * (rotmat[:, 0] * dx_c + rotmat[:, 1] * dy_c)
        tgt_rotmat = rotmat  # keep orientation
        seed = self.bridge.latest_rgt_qs
        if seed is None:
            seed = np.asarray(self.robot_rgt_arm.qs, dtype=np.float32)
        tgt_qs = self.robot_rgt_arm.ik_tcp_nearest(
            tgt_rotmat=tgt_rotmat, tgt_pos=tgt_pos, ref_qs=seed
        )
        return tgt_qs

    def _publish(self, tgt_qs: np.ndarray):
        self.bridge.send_action(side="right", rgt_qs=tgt_qs.tolist())

    def stop(self):
        self._stop = True

    def close(self):
        if self._logger is not None:
            self._logger.close()
            self._logger = None

    def tick(self) -> bool:
        """Advance the state machine by one step. Returns False when done."""
        if self._stop:
            return False
        if self.sample_idx > self.config.max_iterations:
            print("max_iterations reached, stopping correction loop")
            return False
        # Drain bridge ROS callbacks
        self.bridge.pump(0.0)

        if self.state == "IDLE":
            self._capture()
            self.state = "WAIT_IMAGE"
            return True

        if self.state == "WAIT_IMAGE":
            cam1 = os.path.join(self.config.img_dir, f"{self.sample_idx:06d}_cam1.png")
            cam2 = os.path.join(self.config.img_dir, f"{self.sample_idx:06d}_cam2.png")
            if not (os.path.exists(cam1) and os.path.exists(cam2)):
                return True  # try again next tick
            if os.path.getsize(cam1) < 1024 or os.path.getsize(cam2) < 1024:
                return True
            self.state = "INFER"
            return True

        if self.state == "INFER":
            class_id, dx, dy = self.label_source(self.sample_idx, self.config.img_dir)
            self.last_class_id = int(class_id)
            self.last_dx = float(dx)
            self.last_dy = float(dy)

            if self.config.end_on_zero_class and class_id == 0:
                pos, rotmat = self._current_tcp()
                self._logger.write(
                    self.sample_idx,
                    label=0,
                    dx=0.0, dy=0.0,
                    tcp_pos=pos, tcp_rotmat=rotmat,
                )
                self.bridge.send_done(side="right")
                if self.on_step is not None:
                    self.on_step({"event": "done", "sample_idx": self.sample_idx})
                return False

            tgt_qs = self._solve_target(dx, dy)
            if tgt_qs is None:
                print(f"IK failed at sample {self.sample_idx} (cls={class_id}, dx={dx}, dy={dy}); skipping")
                self.sample_idx += 1
                self.state = "IDLE"
                return True
            self.pending_target_qs = np.asarray(tgt_qs, dtype=np.float32)
            self._publish(self.pending_target_qs)
            self.motion_start_ns = time.monotonic_ns()
            self.state = "LOGGING"
            return True

        if self.state == "LOGGING":
            # advance the sim model to the just-commanded pose so subsequent IK
            # uses the next "current" TCP (the bridge will catch up in latency)
            if self.pending_target_qs is not None:
                self.robot_rgt_arm.fk(qs=self.pending_target_qs)
            pos, rotmat = self._current_tcp()
            self._logger.write(
                self.sample_idx,
                label=self.last_class_id,
                dx=self.last_dx, dy=self.last_dy,
                tcp_pos=pos, tcp_rotmat=rotmat,
            )
            if self.on_step is not None:
                self.on_step({
                    "event": "step",
                    "sample_idx": self.sample_idx,
                    "class_id": self.last_class_id,
                    "dx": self.last_dx,
                    "dy": self.last_dy,
                })
            self.sample_idx += 1
            self.state = "WAIT_MOTION"
            return True

        if self.state == "WAIT_MOTION":
            if (time.monotonic_ns() - self.motion_start_ns) / 1e9 > self.config.latency:
                self.state = "IDLE"
            return True

        return True
