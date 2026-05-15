"""Dual-camera recorder driven by config/cameras.yaml.

cam0 (yaml)  -> {idx:06d}_cam1.png  (preserves dataset naming)
cam1 (yaml)  -> {idx:06d}_cam2.png
"""
from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

DEFAULT_CAMERAS_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "cameras.yaml"
)


@dataclass
class CameraSpec:
    device: str
    width: int
    height: int
    fps: int
    rotate180: bool = False
    crop: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) in pixels


def load_cameras_yaml(path: str = DEFAULT_CAMERAS_YAML) -> Tuple[CameraSpec, CameraSpec]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cams = data.get("cameras") or {}

    def _spec(key: str) -> CameraSpec:
        if key not in cams:
            raise KeyError(f"cameras.yaml missing '{key}' under 'cameras'")
        c = cams[key]
        crop = c.get("crop")
        if crop is not None:
            crop = tuple(int(v) for v in crop)
            if len(crop) != 4:
                raise ValueError(f"cameras.{key}.crop must be [x, y, w, h]")
        return CameraSpec(
            device=str(c["device"]),
            width=int(c.get("width", 640)),
            height=int(c.get("height", 480)),
            fps=int(c.get("fps", 30)),
            rotate180=bool(c.get("rotate180", False)),
            crop=crop,
        )

    return _spec("cam0"), _spec("cam1")


class _PhotoSaverThread(threading.Thread):
    def __init__(self, save_dir: str, prefix: str, max_queue: int = 100):
        super().__init__(daemon=True)
        self.save_dir = save_dir
        self.prefix = prefix
        self.q: "queue.Queue[tuple[str, np.ndarray]]" = queue.Queue(maxsize=max_queue)
        self.running = True
        os.makedirs(self.save_dir, exist_ok=True)
        self.start()

    def run(self):
        while self.running or not self.q.empty():
            try:
                tag, frame = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            path = os.path.join(self.save_dir, f"{tag}_{self.prefix}.png")
            cv2.imwrite(path, frame)

    def push_with_num(self, frame: np.ndarray, num: int):
        if not self.running or self.q.full():
            return
        self.q.put_nowait((f"{num:06}", frame.copy()))

    def stop(self):
        self.running = False
        self.join()


def _open_capture(spec: CameraSpec) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(spec.device)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {spec.device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
    cap.set(cv2.CAP_PROP_FPS, spec.fps)
    return cap


def _apply_crop(frame: np.ndarray, crop: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if crop is None:
        return frame
    x, y, w, h = crop
    h_img, w_img = frame.shape[:2]
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(w_img, x + w); y1 = min(h_img, y + h)
    return frame[y0:y1, x0:x1]


class DualCameraRecorder:
    """take_photo_with_num(n) writes {n:06d}_cam1.png and {n:06d}_cam2.png."""

    def __init__(
        self,
        video_path: str,
        cameras_yaml: str = DEFAULT_CAMERAS_YAML,
        warmup_grabs: int = 15,
        toggle_dbg: bool = False,
    ):
        self.video_path = str(video_path)
        os.makedirs(self.video_path, exist_ok=True)
        self.toggle_dbg = toggle_dbg
        self.warmup_grabs = int(warmup_grabs)

        self.spec0, self.spec1 = load_cameras_yaml(cameras_yaml)
        self.cap0 = _open_capture(self.spec0)
        self.cap1 = _open_capture(self.spec1)

        if self.toggle_dbg:
            for tag, cap in (("cam0", self.cap0), ("cam1", self.cap1)):
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"{tag}: {w}x{h} @ {fps} fps")

        self.photo_left = _PhotoSaverThread(save_dir=self.video_path, prefix="cam1")
        self.photo_right = _PhotoSaverThread(save_dir=self.video_path, prefix="cam2")
        self.latest_frame0: Optional[np.ndarray] = None
        self.latest_frame1: Optional[np.ndarray] = None

    def _process(self, frame: np.ndarray, spec: CameraSpec) -> np.ndarray:
        if spec.rotate180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return _apply_crop(frame, spec.crop)

    def take_photo_with_num(self, num: int) -> Tuple[np.ndarray, np.ndarray]:
        for _ in range(self.warmup_grabs):
            self.cap0.grab()
            self.cap1.grab()
        ret0, frame0 = self.cap0.read()
        ret1, frame1 = self.cap1.read()
        if not ret0 or not ret1:
            raise RuntimeError("Camera read failed (cam0 ok=%s, cam1 ok=%s)" % (ret0, ret1))

        out0 = self._process(frame0, self.spec0)
        out1 = self._process(frame1, self.spec1)
        self.latest_frame0 = frame0.copy()
        self.latest_frame1 = frame1.copy()

        self.photo_left.push_with_num(out0, num)
        self.photo_right.push_with_num(out1, num)
        return out0, out1

    def release(self):
        if self.cap0 is not None:
            self.cap0.release()
            self.cap0 = None
        if self.cap1 is not None:
            self.cap1.release()
            self.cap1 = None
        self.photo_left.stop()
        self.photo_right.stop()

    def __enter__(self) -> "DualCameraRecorder":
        return self

    def __exit__(self, *exc):
        self.release()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/screw_camera_test")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--yaml", default=DEFAULT_CAMERAS_YAML)
    args = ap.parse_args()
    rec = DualCameraRecorder(video_path=args.out, cameras_yaml=args.yaml, toggle_dbg=True)
    try:
        for i in range(args.n):
            rec.take_photo_with_num(i)
            print(f"wrote frame {i}")
    finally:
        rec.release()
        print(f"output in {Path(args.out).resolve()}")
