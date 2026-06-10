import os
import re
import time
import numpy as np
import csv
from pathlib import Path

import one.utils.math as oum
import one.utils.constant as ouc


def hexagon_vertex_3d(
    center: np.ndarray = np.array([0, 0, 0], dtype=np.float32),
    radius: float = 0.02,
    normal: np.ndarray = -ouc.StandardAxis.Z,
    idx: int = 0,
    angle_offset: float = 0.0,
) -> np.ndarray:
    if not (0 <= idx < 6):
        raise ValueError("idx must be in 0..5")
    c = np.asarray(center, dtype=float)
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)

    world_up = np.asarray(ouc.StandardAxis.Z, dtype=float)
    if abs(np.dot(n, world_up)) > 0.9:
        world_up = np.array([1.0, 0.0, 0.0])
    u = np.cross(n, world_up)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)

    theta = angle_offset + idx * (2 * np.pi / 6)

    vertex = c + radius * (np.cos(theta) * u + np.sin(theta) * v)
    return vertex


def compute_cam_rotmat(cam_pos, lookat_pos, fallback_up=np.array([0, -1, 0], dtype=np.float32)):
    forward = np.asarray(lookat_pos) - np.asarray(cam_pos)
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.99:
        up_vec = np.asarray(fallback_up, dtype=float)
    else:
        up_vec = world_up
    right = np.cross(up_vec, forward)
    right = right / np.linalg.norm(right)
    true_up = np.cross(forward, right)
    return np.column_stack((right, true_up, forward))


def is_success(pos, rotmat, tgt_pos, tgt_rotmat, pos_tol: float = 0.003) -> bool:
    pos_err, rot_err, _ = oum.diff_between_poses(pos, rotmat, tgt_pos, tgt_rotmat)
    print(f"pos_err: {pos_err}, rot_err: {rot_err}")
    return pos_err < pos_tol


def precise_sleep(dt: float, slack_time: float = 0.001, time_func=time.monotonic):
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass


def atomic_rename(tmp_path: Path, final_path: Path):
    Path(tmp_path).replace(Path(final_path))


def make_mode_dir(base_dir: Path, stage: str) -> str:
    """
      base_dir: Path to ScrewOperation root
      stage: 'train' | 'infer' | 'model'
      Returns next available numbered directory:
        datasets/{stage}/{NNN}/
      Flat layout: task identifiers (sequence / mode) live in config.yaml,
      never in the path. See docs/dataset_layout.md.
    """
    root = Path(os.path.join(str(base_dir), "datasets", stage))
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        int(d.name) for d in root.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    next_num = (existing[-1] + 1) if existing else 1
    ep_dir = root / f"{next_num:03d}"
    ep_dir.mkdir()
    if stage != "model":
        (ep_dir / "images").mkdir()
    return str(ep_dir)


def get_next_episode_dir(base_dir: Path, prefix: str = "ep_", width: int = 4) -> int:
    """
    Find the next unused episode index under base_dir.
    e.g., ep_0000, ep_0001 -> returns 2
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return 0

    pattern = re.compile(rf"^{re.escape(prefix)}(\d{{{width}}})$")
    indices = []
    for d in base_dir.iterdir():
        if d.is_dir():
            m = pattern.match(d.name)
            if m:
                indices.append(int(m.group(1)))

    return max(indices) + 1 if indices else 0


def rotmat_to_rot6d(rotmat: np.ndarray) -> np.ndarray:
    return rotmat[:, :2].reshape(-1, order="F").astype(np.float32)


def rot6d_to_rotmat(r6d: np.ndarray) -> np.ndarray:
    a1 = r6d[:3]
    a2 = r6d[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


class csv_writer:
    def __init__(self, path, fieldnames):
        self.f = open(path, "a", newline="", buffering=1)
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        if self.f.tell() == 0:
            self.writer.writeheader()

    def write(self, row: dict):
        self.writer.writerow(row)

    def close(self):
        self.f.close()
