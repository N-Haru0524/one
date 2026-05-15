import os
import math
import argparse
import numpy as np
import pandas as pd

import torch

import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from one_assembly.ScrewOperation.utils import rot6d_to_rotmat
from one_assembly.ScrewOperation.spiral_metry import hex_ring_abs
from one_assembly.ScrewOperation.config import ScrewConfig, load_config, merge_cli_args
from one_assembly.ScrewOperation.model_builder import build_vit
from one_assembly.ScrewOperation.dataset import load_and_preprocess_pair

toggle_label = False
if toggle_label:
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.size"] = 12
    matplotlib.rcParams["axes.labelsize"] = 12
    matplotlib.rcParams["legend.fontsize"] = 11
    matplotlib.rcParams["xtick.labelsize"] = 11
    matplotlib.rcParams["ytick.labelsize"] = 11


def load_pair(image_dir, idx, suffix1, suffix2):
    p1 = os.path.join(image_dir, f"{idx:06}{suffix1}")
    p2 = os.path.join(image_dir, f"{idx:06}{suffix2}")
    if not os.path.exists(p1):
        raise FileNotFoundError(p1)
    if not os.path.exists(p2):
        raise FileNotFoundError(p2)
    return p1, p2


def main(args, config: ScrewConfig):
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = build_vit(config).to(device)
    sd = torch.load(args.model, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    spiral_list = hex_ring_abs(config.num_classes, step=args.num_accu)

    df = pd.read_csv(args.csv)

    if args.gt_mode == "label0":
        df0 = df[df["label"] == 0]
        if len(df0) == 0:
            raise RuntimeError("gt_mode=label0 なのに label==0 行が CSV にありません。")
        gt_x = float(df0["tcp_x"].mean())
        gt_y = float(df0["tcp_y"].mean())
    else:
        if args.gt_x is None or args.gt_y is None:
            raise RuntimeError("gt_mode=manual の場合は --gt_x と --gt_y が必要です。")
        gt_x = float(args.gt_x)
        gt_y = float(args.gt_y)

    sgn = float(args.correction_sign)

    rows = []
    correct_cls = 0
    total = 0

    for _, row in df.iterrows():
        idx = int(row["idx"])
        y_true = int(row["label"])

        p1, p2 = load_pair(args.image_dir, idx, args.cam1_suffix, args.cam2_suffix)

        x = load_and_preprocess_pair(p1, p2, config)
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            y_pred = int(logits.argmax(dim=1).item())

        total += 1
        correct_cls += int(y_pred == y_true)

        r6d = [float(row[f"r6d_{i}"]) for i in range(6)]
        rotmat = rot6d_to_rotmat(np.array(r6d))

        if y_pred == 0:
            dx_pred, dy_pred = 0.0, 0.0
        else:
            dx_pred, dy_pred = spiral_list[y_pred]
        dx_true, dy_true = float(row["dx"]), float(row["dy"])

        dvec = rotmat[:, 0] * dx_pred + rotmat[:, 1] * dy_pred

        vec_err = math.sqrt((dx_pred - dx_true)**2 + (dy_pred - dy_true)**2)

        tcp_x = float(row["tcp_x"])
        tcp_y = float(row["tcp_y"])
        tcp_z = float(row["tcp_z"])
        tcp_pos = np.array([tcp_x, tcp_y, tcp_z], dtype=np.float32)
        online_corr_pos = tcp_pos - dvec * sgn
        x_corr = float(online_corr_pos[0])
        y_corr = float(online_corr_pos[1])

        pos_err = math.sqrt((x_corr - gt_x)**2 + (y_corr - gt_y)**2)

        rows.append({
            "idx": idx,
            "label_true": y_true,
            "label_pred": y_pred,
            "dx_true": dx_true,
            "dy_true": dy_true,
            "dx_pred": float(dx_pred),
            "dy_pred": float(dy_pred),
            "vec_err_m": vec_err,
            "tcp_x": tcp_x,
            "tcp_y": tcp_y,
            "x_corr": x_corr,
            "y_corr": y_corr,
            "pos_err_m": pos_err
        })

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "eval_results.csv")
    out_df.to_csv(out_csv, index=False)
    print("saved:", out_csv)

    cls_acc = correct_cls / max(total, 1)
    print(f"class_acc = {cls_acc:.4f}  ({correct_cls}/{total})")

    for th_mm in args.success_mm:
        th_m = th_mm / 1000.0
        succ = float((out_df["pos_err_m"] <= th_m).mean())
        print(f"success(pos_err <= {th_mm:.2f}mm) = {succ:.4f}")

    pos_err_mm = out_df["pos_err_m"].values * 1000.0
    print(f"Position error [mm]:")
    print(f"  mean = {pos_err_mm.mean():.4f}")
    print(f"  std  = {pos_err_mm.std(ddof=0):.4f}")

    def save_hist(values, title, xlabel, fname, bins=50):
        plt.figure()
        plt.hist(values, bins=bins)
        if toggle_label:
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel("count")
        path = os.path.join(args.out_dir, fname)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        print("saved:", path)

    save_hist(out_df["pos_err_m"].values * 1000.0,
              title="Position error after correction",
              xlabel="pos_err [mm]",
              fname="hist_pos_err_mm.png",
              bins=args.bins)

    save_hist(out_df["vec_err_m"].values * 1000.0,
              title="Vector error (pred - true)",
              xlabel="vec_err [mm]",
              fname="hist_vec_err_mm.png",
              bins=args.bins)

    plt.figure()
    plt.scatter(
        out_df["x_corr"].values,
        out_df["y_corr"].values,
        s=6,
        alpha=0.7,
        label="Corrected TCP"
    )
    plt.scatter(
        [gt_x], [gt_y],
        marker="x",
        s=80,
        c="red",
        label="GT"
    )
    success_radius = 1.0 / 1000.0
    circle = Circle(
        (gt_x, gt_y),
        success_radius,
        color="red",
        fill=False,
        linestyle="--",
        linewidth=2,
        label="Success region (1 mm)"
    )
    plt.gca().add_patch(circle)
    if toggle_label:
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Corrected XY scatter with success region")
        plt.axis("equal")
        plt.legend()
    path = os.path.join(args.out_dir, "scatter_corrected_xy.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print("saved:", path)

    x0 = out_df["tcp_x"].values
    y0 = out_df["tcp_y"].values
    x1 = out_df["x_corr"].values
    y1 = out_df["y_corr"].values
    plt.figure(figsize=(6, 6))

    plt.scatter(x0, y0, c="tab:blue", s=30, label="Before correction")
    plt.scatter(x1, y1, c="orange", s=30, label="After correction")

    for i in range(len(x0)):
        plt.arrow(
            x0[i], y0[i],
            x1[i] - x0[i], y1[i] - y0[i],
            head_width=0.00005,
            width=0.00002,
            length_includes_head=True,
            color="gray",
            alpha=0.6
        )

    plt.scatter([gt_x], [gt_y], c="red", marker="x", s=100, linewidths=3, label="GT")

    r = 1.0 / 1000.0
    circle = plt.Circle((gt_x, gt_y), r, color="red", fill=False, linestyle="--", linewidth=2)
    plt.gca().add_patch(circle)
    if toggle_label:
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("TCP correction vectors")
        plt.axis("equal")
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "scatter_with_vectors.png"), dpi=200)
    plt.close()

    print("done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--image_dir", required=True)

    ap.add_argument("--model", required=True)
    ap.add_argument("--num_classes", type=int, default=None)
    ap.add_argument("--num_accu", type=float, default=0.002)

    ap.add_argument("--roi1", type=int, nargs=4, default=None)
    ap.add_argument("--roi2", type=int, nargs=4, default=None)

    ap.add_argument("--out_dir", type=str, default="eval_out")
    ap.add_argument("--bins", type=int, default=50)

    ap.add_argument("--gt_mode", choices=["label0", "manual"], default="label0")
    ap.add_argument("--gt_x", type=float, default=None)
    ap.add_argument("--gt_y", type=float, default=None)

    ap.add_argument("--cam1_suffix", type=str, default="_cam1.png")
    ap.add_argument("--cam2_suffix", type=str, default="_cam2.png")

    ap.add_argument("--correction_sign", type=int, choices=[-1, 1], default=1)

    ap.add_argument("--success_mm", type=float, nargs="*", default=[1.0])

    args = ap.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        auto_config = os.path.splitext(args.model)[0] + "_config.yaml"
        if os.path.exists(auto_config):
            config = load_config(auto_config)
            print("auto-loaded config:", auto_config)
        else:
            config = ScrewConfig()

    config = merge_cli_args(config, args)
    main(args, config)
