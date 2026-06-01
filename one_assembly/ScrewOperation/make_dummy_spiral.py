"""Generate a dummy ScrewOperation dataset for smoke-testing the training
and evaluation pipelines without real camera captures.

The output layout matches what `data_collector.py` would produce, so the
dummy dirs can be fed directly to `train_vit_spiral.py` and
`eval_vit_spiral_hist.py`:

  <out_dir>/
    train/
      samples.csv             # idx, time, label, dx, dy, tcp_*, r6d_*
      config.yaml             # ScrewConfig snapshot
      images/
        {idx:06d}_cam1.png
        {idx:06d}_cam2.png
    val/
      ...
"""
from __future__ import annotations

import argparse
import os
import random
from datetime import datetime

import cv2
import numpy as np

from one_assembly.ScrewOperation.config import ScrewConfig, save_config
from one_assembly.ScrewOperation.spiral_metry import hex_ring_abs
from one_assembly.ScrewOperation.utils import csv_writer, rotmat_to_rot6d


DEFAULT_OUT_DIR = "dummy_data"
DEFAULT_NUM_TRAIN = 200
DEFAULT_NUM_VAL = 40
DEFAULT_IMG_H = 240
DEFAULT_IMG_W = 320


def _random_image(h: int, w: int, rng: random.Random) -> np.ndarray:
    # Random RGB image. Use np.random for speed; seed via rng for determinism.
    state = np.random.RandomState(rng.randint(0, 2**32 - 1))
    return state.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


def _identity_rot6d() -> np.ndarray:
    return rotmat_to_rot6d(np.eye(3, dtype=np.float32))


def make_split(
    split_dir: str,
    start_idx: int,
    num_samples: int,
    num_classes: int,
    spiral_step: float,
    img_h: int,
    img_w: int,
    seed: int,
) -> None:
    img_dir = os.path.join(split_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(split_dir, "samples.csv")
    spiral_list = hex_ring_abs(num_classes, step=spiral_step)
    rot6d = _identity_rot6d()

    rng = random.Random(seed)
    writer = csv_writer(csv_path, fieldnames=[
        "idx", "time", "label", "dx", "dy",
        "tcp_x", "tcp_y", "tcp_z",
        "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5",
    ])
    try:
        for i in range(num_samples):
            idx = start_idx + i
            cv2.imwrite(os.path.join(img_dir, f"{idx:06d}_cam1.png"), _random_image(img_h, img_w, rng))
            cv2.imwrite(os.path.join(img_dir, f"{idx:06d}_cam2.png"), _random_image(img_h, img_w, rng))
            label = rng.randint(0, num_classes - 1)
            dx, dy = spiral_list[label]
            writer.write({
                "idx": idx,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                "label": label,
                "dx": float(dx),
                "dy": float(dy),
                "tcp_x": 0.0, "tcp_y": 0.0, "tcp_z": 0.0,
                "r6d_0": float(rot6d[0]), "r6d_1": float(rot6d[1]),
                "r6d_2": float(rot6d[2]), "r6d_3": float(rot6d[3]),
                "r6d_4": float(rot6d[4]), "r6d_5": float(rot6d[5]),
            })
    finally:
        writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--num_train", type=int, default=DEFAULT_NUM_TRAIN)
    ap.add_argument("--num_val", type=int, default=DEFAULT_NUM_VAL)
    ap.add_argument("--num_classes", type=int, default=None,
                    help="Overrides ScrewConfig.num_classes")
    ap.add_argument("--spiral_step", type=float, default=None,
                    help="Overrides ScrewConfig.spiral_step")
    ap.add_argument("--img_h", type=int, default=DEFAULT_IMG_H)
    ap.add_argument("--img_w", type=int, default=DEFAULT_IMG_W)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_source", default="sim", choices=("sim", "real"),
                    help="Tag the generated dataset's provenance (default: sim)")
    args = ap.parse_args()

    config = ScrewConfig(data_source=args.data_source)
    overrides = {}
    if args.num_classes is not None:
        overrides["num_classes"] = args.num_classes
    if args.spiral_step is not None:
        overrides["spiral_step"] = args.spiral_step
    if overrides:
        config = config.model_copy(update=overrides)

    train_dir = os.path.join(args.out_dir, "train")
    val_dir = os.path.join(args.out_dir, "val")
    os.makedirs(args.out_dir, exist_ok=True)
    save_config(config, os.path.join(train_dir, "config.yaml"))

    make_split(train_dir, 0, args.num_train, config.num_classes,
               config.spiral_step, args.img_h, args.img_w, seed=args.seed)
    make_split(val_dir, args.num_train, args.num_val, config.num_classes,
               config.spiral_step, args.img_h, args.img_w, seed=args.seed + 1)

    print(f"Dummy dataset generated under {os.path.abspath(args.out_dir)}")
    print(f"  train: {args.num_train} samples -> {train_dir}")
    print(f"  val  : {args.num_val} samples -> {val_dir}")


if __name__ == "__main__":
    main()
