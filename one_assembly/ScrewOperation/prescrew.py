"""Resolve / write prescrew poses for the correction loop.

Three sources are supported:

1. **YAML** — flat file with ``rgt_qs`` and optional ``rgt_ee_qs`` keys.
   See ``config/prescrew.example.yaml`` for the schema.
2. **WorkList** — call ``WorkList.get_screw_pose()`` to obtain the current
   screw target (pos, rotmat), apply an axial offset to back the bit off
   the workpiece, then solve IK on the right arm to obtain ``rgt_qs``.
3. **Joint-state snapshot** — captured at runtime via
   :mod:`capture_prescrew`; reads ``/right/joint_states`` while the arm is
   jogged to the desired pose.

Module is pure-python + numpy and stays usable without ROS/Pyglet
(provided the caller passes a robot arm that exposes
``ik_tcp_nearest`` and ``qs``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import yaml


@dataclass
class PrescrewSolution:
    """Result of resolving a prescrew pose."""

    rgt_qs: np.ndarray
    prescrew_pos: np.ndarray
    prescrew_rotmat: np.ndarray
    rgt_ee_qs: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------

def load_prescrew_yaml(path: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Read ``rgt_qs`` and optional ``rgt_ee_qs`` from a yaml file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "rgt_qs" not in data:
        raise KeyError(f"{path}: missing required key 'rgt_qs'")
    qs = np.asarray(data["rgt_qs"], dtype=np.float32)
    if qs.shape != (6,):
        raise ValueError(f"{path}: rgt_qs must have shape (6,), got {qs.shape}")
    ee = data.get("rgt_ee_qs")
    ee_arr = np.asarray(ee, dtype=np.float32) if ee is not None else None
    return qs, ee_arr


def save_prescrew_yaml(
    out_path: str,
    rgt_qs: Sequence[float],
    *,
    rgt_ee_qs: Optional[Sequence[float]] = None,
    extra: Optional[dict] = None,
) -> dict:
    payload: dict = {"rgt_qs": [float(x) for x in rgt_qs]}
    if rgt_ee_qs is not None:
        payload["rgt_ee_qs"] = [float(v) for v in rgt_ee_qs]
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, default_flow_style=None, sort_keys=False)
    return payload


# ---------------------------------------------------------------------------
# Screw-pose -> prescrew TCP -> rgt_qs
# ---------------------------------------------------------------------------

def prescrew_pose_from_screw_pose(
    screw_pos: np.ndarray,
    screw_rotmat: np.ndarray,
    *,
    prescrew_offset: float = 0.005,
    flip_axis: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the TCP target for the prescrew pose.

    The screw's z-axis (``screw_rotmat[:, 2]``) is treated as the **advance
    direction** of the screw (the way the head moves while being driven in).
    The TCP is positioned ``prescrew_offset`` metres back along that axis so
    the bit tip sits just above the screw head. TCP orientation matches the
    screw frame so the bit aligns with the hole.

    Set ``flip_axis=True`` if the workplace convention has the screw rotmat
    z pointing OUT of the workpiece instead.
    """
    screw_pos = np.asarray(screw_pos, dtype=np.float32).reshape(3)
    screw_rotmat = np.asarray(screw_rotmat, dtype=np.float32).reshape(3, 3)
    axis = screw_rotmat[:, 2]
    if flip_axis:
        axis = -axis
    prescrew_pos = (screw_pos - axis * float(prescrew_offset)).astype(np.float32)
    return prescrew_pos, screw_rotmat


def prescrew_qs_from_screw_pose(
    rgt_arm,
    screw_pos: np.ndarray,
    screw_rotmat: np.ndarray,
    *,
    prescrew_offset: float = 0.005,
    ref_qs: Optional[np.ndarray] = None,
    flip_axis: bool = False,
) -> Optional[PrescrewSolution]:
    """Solve IK to obtain the right-arm joints at the prescrew TCP pose.

    Returns ``None`` if IK fails. The caller decides whether to retry with a
    different seed or to abort.
    """
    prescrew_pos, prescrew_rotmat = prescrew_pose_from_screw_pose(
        screw_pos, screw_rotmat, prescrew_offset=prescrew_offset, flip_axis=flip_axis,
    )
    if ref_qs is None:
        ref_qs = np.asarray(rgt_arm.qs, dtype=np.float32)
    else:
        ref_qs = np.asarray(ref_qs, dtype=np.float32)
    rgt_qs = rgt_arm.ik_tcp_nearest(
        tgt_rotmat=prescrew_rotmat,
        tgt_pos=prescrew_pos,
        ref_qs=ref_qs,
    )
    if rgt_qs is None:
        return None
    return PrescrewSolution(
        rgt_qs=np.asarray(rgt_qs, dtype=np.float32),
        prescrew_pos=prescrew_pos,
        prescrew_rotmat=prescrew_rotmat,
    )


def prescrew_qs_from_worklist(
    rgt_arm,
    worklist,
    *,
    prescrew_offset: float = 0.005,
    ref_qs: Optional[np.ndarray] = None,
    flip_axis: bool = False,
    advance_screw_counter: bool = False,
) -> Optional[PrescrewSolution]:
    """Use the worklist's *current* screw target as the prescrew pose.

    By default the worklist's ``screw_counter`` is preserved (snapshot mode),
    so the same screw can be queried repeatedly. Pass
    ``advance_screw_counter=True`` to let ``WorkList.get_screw_pose`` consume
    a slot — typical when planning a sequence.
    """
    screw_counter = int(worklist.screw_counter)
    screw_pos, screw_rotmat = worklist.get_screw_pose()
    if not advance_screw_counter:
        worklist.screw_counter = screw_counter
    return prescrew_qs_from_screw_pose(
        rgt_arm, screw_pos, screw_rotmat,
        prescrew_offset=prescrew_offset, ref_qs=ref_qs, flip_axis=flip_axis,
    )


# ---------------------------------------------------------------------------
# Unified resolver
# ---------------------------------------------------------------------------

def resolve_prescrew(
    *,
    rgt_arm=None,
    yaml_path: Optional[str] = None,
    worklist=None,
    prescrew_offset: float = 0.005,
    flip_axis: bool = False,
    rgt_ee_value: Optional[float] = None,
    ref_qs: Optional[np.ndarray] = None,
    advance_screw_counter: bool = False,
) -> PrescrewSolution:
    """Single entry point. Picks the resolution source in this priority:

    1. ``yaml_path`` — load qs directly (no IK needed).
    2. ``worklist`` + ``rgt_arm`` — compute via ``WorkList.get_screw_pose()``.

    Raises if neither source is usable or IK fails.
    """
    if yaml_path is not None:
        rgt_qs, rgt_ee = load_prescrew_yaml(yaml_path)
        return PrescrewSolution(
            rgt_qs=rgt_qs,
            prescrew_pos=np.zeros(3, dtype=np.float32),
            prescrew_rotmat=np.eye(3, dtype=np.float32),
            rgt_ee_qs=rgt_ee,
        )
    if worklist is not None and rgt_arm is not None:
        sol = prescrew_qs_from_worklist(
            rgt_arm, worklist,
            prescrew_offset=prescrew_offset,
            ref_qs=ref_qs,
            flip_axis=flip_axis,
            advance_screw_counter=advance_screw_counter,
        )
        if sol is None:
            raise RuntimeError("IK failed for worklist-derived prescrew pose.")
        if rgt_ee_value is not None:
            sol.rgt_ee_qs = np.asarray([rgt_ee_value], dtype=np.float32)
        return sol
    raise ValueError(
        "resolve_prescrew: provide either yaml_path or (worklist + rgt_arm)."
    )
