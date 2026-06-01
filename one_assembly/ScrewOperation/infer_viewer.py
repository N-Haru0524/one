"""Offline viewer: evaluate a screw-correction run against a trained model
and replay the captured cam1/cam2 images with predicted/true labels and a
position-error overlay.

Loads:
  <infer_dir>/config.yaml      -- ScrewConfig used during the run
  <infer_dir>/samples.csv      -- per-step log produced by screw_correction_run
  <infer_dir>/images/*.png

Outputs:
  <infer_dir>/eval_results.csv -- per-frame true/pred labels + position error
  <infer_dir>/hist_pos_err_mm.png
  <infer_dir>/scatter_with_vectors.png
  -- plus an interactive cv2 replay window with scrub/play controls.

Usage:
  uv run python -m one_assembly.ScrewOperation.infer_viewer \\
      --infer_dir <ep_dir> [--model_dir <model_dir>]

Controls: SPACE = play/pause, q = quit.
Requires an X11 display.
"""
from __future__ import annotations

import argparse
import glob
import math
import os

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Circle
from PIL import Image

from one_assembly.ScrewOperation.config import load_config
from one_assembly.ScrewOperation.dataset import load_and_preprocess_pair
from one_assembly.ScrewOperation.model_builder import build_model
from one_assembly.ScrewOperation.preprocess import apply_roi, rotate_image
from one_assembly.ScrewOperation.spiral_metry import hex_ring_abs
from one_assembly.ScrewOperation.utils import rot6d_to_rotmat


def evaluate(infer_dir: str, model_dir_override: str | None = None):
    infer_config = load_config(os.path.join(infer_dir, "config.yaml"))
    model_dir = model_dir_override or infer_config.model_dir
    if not model_dir:
        raise ValueError("model_dir not found in infer config. Specify --model_dir.")

    config = load_config(os.path.join(model_dir, "config.yaml"))
    model_path = os.path.join(model_dir, "model.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    spiral_list = hex_ring_abs(config.num_classes, step=config.spiral_step)
    image_dir = os.path.join(infer_dir, "images")
    df = pd.read_csv(os.path.join(infer_dir, "samples.csv"))

    df0 = df[df["label"] == 0]
    gt_x = float(df0["tcp_x"].mean())
    gt_y = float(df0["tcp_y"].mean())

    rows = []
    for _, row in df.iterrows():
        idx = int(row["sequence"]) if "sequence" in row else int(row["idx"])
        y_true = int(row["label"])
        if idx == 0:
            continue

        cam1 = os.path.join(image_dir, f"{idx:06d}_cam1.png")
        cam2 = os.path.join(image_dir, f"{idx:06d}_cam2.png")
        if not os.path.exists(cam1):
            continue

        x = load_and_preprocess_pair(cam1, cam2, config)
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            y_pred = int(logits.argmax(dim=1).item())

        r6d = [float(row[f"r6d_{i}"]) for i in range(6)]
        rotmat = rot6d_to_rotmat(np.array(r6d))

        dx_pred, dy_pred = (0.0, 0.0) if y_pred == 0 else spiral_list[y_pred]
        dx_true, dy_true = float(row["dx"]), float(row["dy"])

        dvec = rotmat[:, 0] * dx_true + rotmat[:, 1] * dy_true
        vec_err = math.sqrt((dx_pred - dx_true) ** 2 + (dy_pred - dy_true) ** 2)

        tcp_x = float(row["tcp_x"])
        tcp_y = float(row["tcp_y"])
        tcp_z = float(row["tcp_z"])
        tcp_pos = np.array([tcp_x, tcp_y, tcp_z], dtype=np.float32)
        online_corr_pos = tcp_pos - dvec
        x_corr = float(online_corr_pos[0])
        y_corr = float(online_corr_pos[1])

        pos_err = math.sqrt((x_corr - gt_x) ** 2 + (y_corr - gt_y) ** 2)

        rows.append({
            "idx": idx,
            "label_true": y_true,
            "label_pred": y_pred,
            "dx_true": dx_true, "dy_true": dy_true,
            "dx_pred": float(dx_pred), "dy_pred": float(dy_pred),
            "vec_err_m": vec_err,
            "tcp_x": tcp_x, "tcp_y": tcp_y,
            "x_corr": x_corr, "y_corr": y_corr,
            "pos_err_m": pos_err,
        })

    out_df = pd.DataFrame(rows)
    return out_df, config, gt_x, gt_y


def save_and_show_plots(out_df, gt_x, gt_y, spiral_step, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # histogram: position error
    plt.figure()
    plt.hist(out_df["pos_err_m"].values * 1000.0, bins=30)
    plt.title("Position error after correction")
    plt.xlabel("pos_err [mm]")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_pos_err_mm.png"), dpi=200)
    print("saved:", os.path.join(out_dir, "hist_pos_err_mm.png"))

    # scatter: correction vectors
    x0 = out_df["tcp_x"].values
    y0 = out_df["tcp_y"].values
    x1 = out_df["x_corr"].values
    y1 = out_df["y_corr"].values
    plt.figure(figsize=(6, 6))
    plt.scatter(x0[0], y0[0], c="orange", s=100, label="Start TCP", zorder=5)
    plt.scatter(x0, y0, c="black", s=50, label="Before correction")
    plt.scatter(x1, y1, c="tab:blue", s=30, label="After correction")
    for i in range(len(x0)):
        plt.arrow(
            x0[i], y0[i], x1[i] - x0[i], y1[i] - y0[i],
            head_width=0.00005, width=0.00002,
            length_includes_head=True, color="gray", alpha=0.6,
        )
    plt.scatter([gt_x], [gt_y], c="red", marker="x", s=100, linewidths=3, label="GT", zorder=5)
    r = 1.0 / 1000.0
    circle = Circle((gt_x, gt_y), r, color="red", fill=False, linestyle="--",
                     linewidth=2, label="Success (1mm)")
    plt.gca().add_patch(circle)
    sq = spiral_step * 20
    plt.xlim(gt_x - sq, gt_x + sq)
    plt.ylim(gt_y - sq, gt_y + sq)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("TCP correction vectors")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_with_vectors.png"), dpi=200)
    print("saved:", os.path.join(out_dir, "scatter_with_vectors.png"))

    plt.show(block=False)


def show_replay(infer_dir, config, out_df, fps):
    image_dir = os.path.join(infer_dir, "images")
    cam1_files = sorted(glob.glob(os.path.join(image_dir, "*_cam1.png")))
    cam2_files = sorted(glob.glob(os.path.join(image_dir, "*_cam2.png")))

    n = min(len(cam1_files), len(cam2_files))
    if n == 0:
        print("No images found")
        return

    eval_map = {}
    for _, row in out_df.iterrows():
        eval_map[int(row["idx"])] = row

    window_name = "Infer Viewer (SPACE: play/pause, q: quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("frame", window_name, 0, n - 1, lambda x: None)

    playing = False
    prev_idx = -1
    wait_ms = int(1000 / fps)

    while True:
        idx = cv2.getTrackbarPos("frame", window_name)

        if playing:
            idx = (idx + 1) % n
            cv2.setTrackbarPos("frame", window_name, idx)

        if idx != prev_idx:
            prev_idx = idx

            # Open raw -> rotate (CW) -> ROI crop, mirroring dataset.SpiralDataset.
            img1 = rotate_image(Image.open(cam1_files[idx]).convert("RGB"), int(config.rotate1))
            img2 = rotate_image(Image.open(cam2_files[idx]).convert("RGB"), int(config.rotate2))

            img1_raw = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
            img2_raw = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

            x1, y1, x2, y2 = config.roi1
            cv2.rectangle(img1_raw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            x1, y1, x2, y2 = config.roi2
            cv2.rectangle(img2_raw, (x1, y1), (x2, y2), (0, 255, 0), 2)

            info = eval_map.get(idx)
            if info is not None:
                label = (
                    f"[{idx:04d}] true:{int(info['label_true'])} "
                    f"pred:{int(info['label_pred'])} err:{info['pos_err_m'] * 1000:.2f}mm"
                )
                color = (0, 255, 0) if int(info["label_true"]) == int(info["label_pred"]) else (0, 0, 255)
            else:
                label = f"[{idx:04d}] (reference)"
                color = (255, 255, 255)

            status = "PLAY" if playing else "PAUSE"
            # Display cam2 on the LEFT and cam1 on the RIGHT to match the rig's
            # physical layout.
            raw = np.hstack((img2_raw, img1_raw))
            cv2.putText(raw, f"{label}  {status}", (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            img1_crop = np.array(apply_roi(img1, config.roi1))
            img2_crop = np.array(apply_roi(img2, config.roi2))
            img1_crop = cv2.cvtColor(img1_crop, cv2.COLOR_RGB2BGR)
            img2_crop = cv2.cvtColor(img2_crop, cv2.COLOR_RGB2BGR)
            crop = np.hstack((img2_crop, img1_crop))

            cv2.imshow(window_name, raw)
            cv2.imshow("Cropped ROI", crop)

        key = cv2.waitKey(wait_ms if playing else 30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            playing = not playing

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    print("Evaluating...")
    out_df, config, gt_x, gt_y = evaluate(args.infer_dir, args.model_dir)

    out_df.to_csv(os.path.join(args.infer_dir, "eval_results.csv"), index=False)
    print("saved:", os.path.join(args.infer_dir, "eval_results.csv"))

    cls_acc = (out_df["label_true"] == out_df["label_pred"]).sum() / max(len(out_df), 1)
    pos_err_mm = out_df["pos_err_m"].values * 1000.0
    print(f"class_acc = {cls_acc:.4f}")
    print(f"pos_err [mm]: mean={pos_err_mm.mean():.4f}, std={pos_err_mm.std(ddof=0):.4f}")

    save_and_show_plots(out_df, gt_x, gt_y, config.spiral_step, args.infer_dir)
    show_replay(args.infer_dir, config, out_df, args.fps)
    plt.close("all")


if __name__ == "__main__":
    main()
