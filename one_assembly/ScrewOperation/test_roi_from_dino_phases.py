"""Test script: visualize each processing phase of build_roi_from_dino.py.

Produces one PNG per phase under --out_dir for documentation purposes.
This is a one-off test script; reuse is not considered.
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label as cc_label

from one_assembly.ScrewOperation.build_roi_from_dino import (
    MODEL_ID_DEFAULT,
    cluster_patches,
    collect_image_paths,
    extract_patch_features,
    load_dinov3,
    orig_to_patch_rc,
    parse_xy,
    patch_bbox_to_orig,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def upsample_nearest(arr: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(arr.astype(np.uint8)).resize(size_wh, Image.NEAREST))


def upsample_bilinear(arr: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(arr.astype(np.uint8)).resize(size_wh, Image.BILINEAR))


def draw_bit_tip(ax, xy, label=None):
    ax.plot(xy[0], xy[1], "o",
            markerfacecolor="white", markeredgecolor="red",
            markersize=12, markeredgewidth=2, label=label)


# ---- Phase 0 ----
def phase0_bit_tip(image_path, bit_tip_xy, out_path):
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    draw_bit_tip(ax, bit_tip_xy, label="bit tip (manual)")
    ax.legend(loc="upper right")
    ax.set_title(f"Phase 0: Bit tip manual annotation  (x={bit_tip_xy[0]}, y={bit_tip_xy[1]})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---- Phase 1 ----
def phase1_attention(image_path, attention_map, bit_tip_xy, out_path):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    attn = attention_map.astype(np.float32)
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    attn_up = upsample_bilinear((attn * 255), (W, H)) / 255.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img)
    axes[0].set_title("Input image (360x360)")
    axes[0].axis("off")
    axes[1].imshow(img)
    axes[1].imshow(attn_up, cmap="jet", alpha=0.5)
    draw_bit_tip(axes[1], bit_tip_xy)
    axes[1].set_title("Phase 1: CLS attention (last layer, mean over heads)")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---- Phase 2 ----
def phase2_clusters(image_path, sample_labels, k, out_path):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    cmap_name = "tab20" if k > 10 else "tab10"
    cmap = plt.get_cmap(cmap_name, k)
    lbl_up = upsample_nearest(sample_labels, (W, H))
    color = cmap(lbl_up)[:, :, :3]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img)
    axes[0].set_title("Input image")
    axes[0].axis("off")
    axes[1].imshow(img)
    axes[1].imshow(color, alpha=0.55)
    axes[1].set_title(f"Phase 2: K-means clustering (K={k}, L2-normalized features)")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---- Phase 3 ----
def phase3_bit_class(image_path, sample_labels, bit_class, bit_tip_xy, out_path):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    mask = (sample_labels == bit_class).astype(np.uint8) * 255
    mask_up = upsample_nearest(mask, (W, H))
    overlay = np.zeros((H, W, 4))
    overlay[:, :, 1] = 1.0  # green channel
    overlay[:, :, 3] = (mask_up / 255.0) * 0.55

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    ax.imshow(overlay)
    draw_bit_tip(ax, bit_tip_xy, label="bit tip anchor")
    ax.legend(loc="upper right")
    ax.set_title(f"Phase 3: Bit class identified (cls={bit_class})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---- Phase 4 ----
def phase4_background_removal(
    cluster_attn, threshold, bit_class, foreground_clusters,
    image_path, sample_labels, out_path,
):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    k = len(cluster_attn)

    bar_colors = []
    for c in range(k):
        if c == bit_class:
            bar_colors.append("red")
        elif c in foreground_clusters:
            bar_colors.append("tab:green")
        else:
            bar_colors.append("lightgray")

    fg_mask = np.isin(sample_labels, list(foreground_clusters)).astype(np.uint8) * 255
    fg_mask_up = upsample_nearest(fg_mask, (W, H))
    dim_overlay = np.zeros((H, W, 4))
    dim_overlay[:, :, 3] = (1.0 - fg_mask_up / 255.0) * 0.65  # black dim on bg

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    axes[0].bar(range(k), np.nan_to_num(cluster_attn, nan=0), color=bar_colors)
    axes[0].axhline(threshold, color="black", linestyle="--",
                    label=f"threshold={threshold:.4f}")
    axes[0].set_xlabel("Cluster ID")
    axes[0].set_ylabel("Mean CLS attention")
    axes[0].set_title(
        f"Phase 4: Cluster attention\n"
        f"red=bit({bit_class}), green=foreground, gray=background"
    )
    axes[0].set_xticks(range(k))
    axes[0].legend()

    axes[1].imshow(img)
    axes[1].imshow(dim_overlay)
    axes[1].set_title("Foreground only (background dimmed)")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---- Phase 5 ----
def phase5_voting(
    votes, bit_class, target_class,
    image_path, sample_labels, foreground_clusters,
    bit_tip_xy, grid, out_path,
):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    Hp, Wp = grid
    k = len(votes)

    br, bc = orig_to_patch_rc(bit_tip_xy, (H, W), grid)
    fg_mask = np.isin(sample_labels, list(foreground_clusters))
    coords = np.argwhere(fg_mask)
    if len(coords) == 0:
        return
    d = np.linalg.norm(coords.astype(np.float32) - np.array([br, bc]), axis=1)
    j = int(np.argmin(d))
    nr, nc = coords[j]
    nearest_x = (nc + 0.5) / Wp * W
    nearest_y = (nr + 0.5) / Hp * H
    nearest_cls = int(sample_labels[nr, nc])

    bar_colors = []
    for c in range(k):
        if c == target_class:
            bar_colors.append("tab:orange")
        elif c == bit_class:
            bar_colors.append("red")
        else:
            bar_colors.append("lightgray")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    axes[0].bar(range(k), votes, color=bar_colors)
    axes[0].set_xlabel("Cluster ID")
    axes[0].set_ylabel("Votes (sum across all dataset images)")
    axes[0].set_title(
        f"Phase 5: Per-image nearest non-bit voting\n"
        f"target={target_class} ({int(votes[target_class])}/{int(votes.sum())})"
    )
    axes[0].set_xticks(range(k))

    axes[1].imshow(img)
    draw_bit_tip(axes[1], bit_tip_xy, label="bit tip")
    axes[1].plot(nearest_x, nearest_y, "^",
                 markerfacecolor="yellow", markeredgecolor="black",
                 markersize=14, markeredgewidth=2,
                 label=f"nearest non-bit patch (cls={nearest_cls})")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Per-image: nearest foreground non-bit patch → vote")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---- Phase 6 ----
def phase6_connected_components(
    image_path, sample_labels, target_class,
    bit_tip_xy, grid, out_path,
):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    Hp, Wp = grid

    br, bc = orig_to_patch_rc(bit_tip_xy, (H, W), grid)
    bit_anchor = np.array([br, bc], dtype=np.float32)

    mask = (sample_labels == target_class)
    cc, n_cc = cc_label(mask, structure=np.ones((3, 3)))

    best_cc = -1
    best_dist = np.inf
    per_cc_dist = {}
    for cc_id in range(1, n_cc + 1):
        coords = np.argwhere(cc == cc_id)
        d = np.linalg.norm(coords.astype(np.float32) - bit_anchor, axis=1).min()
        per_cc_dist[cc_id] = d
        if d < best_dist:
            best_dist = d
            best_cc = cc_id

    colors_rgba = np.zeros((Hp, Wp, 4), dtype=np.float32)
    cmap = plt.get_cmap("Set1", max(n_cc, 3))
    for cc_id in range(1, n_cc + 1):
        c_rgb = cmap(cc_id - 1)[:3]
        alpha = 0.8 if cc_id == best_cc else 0.25
        sel = (cc == cc_id)
        for ch in range(3):
            colors_rgba[..., ch] = np.where(sel, c_rgb[ch], colors_rgba[..., ch])
        colors_rgba[..., 3] = np.where(sel, alpha, colors_rgba[..., 3])

    colors_up = np.array(
        Image.fromarray((colors_rgba * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
    ) / 255.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img)
    axes[0].set_title(f"Input (target cls={target_class} has {n_cc} components)")
    axes[0].axis("off")

    axes[1].imshow(img)
    axes[1].imshow(colors_up)
    draw_bit_tip(axes[1], bit_tip_xy)
    text_lines = [f"cc{c}: d={per_cc_dist[c]:.1f}"
                  + (" (kept)" if c == best_cc else " (discarded)")
                  for c in sorted(per_cc_dist)]
    axes[1].text(0.02, 0.98, "\n".join(text_lines),
                 transform=axes[1].transAxes, va="top", fontsize=10,
                 bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    axes[1].set_title(f"Phase 6: Keep closest-to-bit component (cc={best_cc})")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---- Phase 7 ----
def phase7_bbox(image_path, all_rows, all_cols, roi, grid, bit_tip_xy, percentile, out_path):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    Hp, Wp = grid

    density = np.zeros((Hp, Wp), dtype=np.float32)
    for r, c in zip(all_rows.astype(int), all_cols.astype(int)):
        density[r, c] += 1
    density = density / density.max()
    density_up = upsample_bilinear((density * 255), (W, H)) / 255.0

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(img)
    ax.imshow(density_up, cmap="hot", alpha=0.6)
    x1, y1, x2, y2 = roi
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=3, edgecolor="lime", facecolor="none",
                         label=f"ROI ({x2-x1}x{y2-y1})")
    ax.add_patch(rect)
    draw_bit_tip(ax, bit_tip_xy, label="bit tip")
    ax.legend(loc="upper right")
    ax.set_title(
        f"Phase 7: Percentile={percentile}  ROI=({x1},{y1},{x2},{y2})\n"
        f"Heatmap = density of kept target-class patches across all images"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--bit_tip", type=parse_xy, required=True,
                        help='bit tip pixel "x,y" for the selected camera')
    parser.add_argument("--camera", type=str, default="cam1", choices=["cam1", "cam2"])
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="image index used for per-image visualizations")
    parser.add_argument("--model_id", type=str, default=MODEL_ID_DEFAULT)
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--min_attn_quantile", type=float, default=0.7)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(BASE_DIR, "tmp", "roi_dino_test"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, model={args.model_id}, camera={args.camera}")

    processor, model = load_dinov3(args.model_id, device)

    image_paths = collect_image_paths(args.train_dirs, args.camera)
    print(f"collected {len(image_paths)} images")

    print("extracting features...")
    features, attentions, grid, orig_size = extract_patch_features(
        image_paths, processor, model, device, args.resolution,
    )
    Hp, Wp = grid
    H, W = orig_size
    print(f"grid={Hp}x{Wp}, orig={H}x{W}")

    print(f"clustering (K={args.k})...")
    labels = cluster_patches(features, k=args.k)

    sample_img_path = image_paths[args.sample_idx]
    sample_labels = labels[args.sample_idx]
    sample_attn = attentions[args.sample_idx]
    print(f"sample image: {sample_img_path}")

    # Phase 0
    phase0_bit_tip(
        sample_img_path, args.bit_tip,
        os.path.join(args.out_dir, "phase0_bit_tip.png"),
    )

    # Phase 1
    phase1_attention(
        sample_img_path, sample_attn, args.bit_tip,
        os.path.join(args.out_dir, "phase1_attention.png"),
    )

    # Phase 2
    phase2_clusters(
        sample_img_path, sample_labels, args.k,
        os.path.join(args.out_dir, "phase2_clusters.png"),
    )

    # Phase 3 — bit class
    br, bc = orig_to_patch_rc(args.bit_tip, orig_size, grid)
    bit_labels_all = labels[:, br, bc]
    bit_class = int(np.bincount(bit_labels_all, minlength=args.k).argmax())
    print(f"bit_class={bit_class} at patch ({br},{bc})")
    phase3_bit_class(
        sample_img_path, sample_labels, bit_class, args.bit_tip,
        os.path.join(args.out_dir, "phase3_bit_class.png"),
    )

    # Phase 4 — background removal
    cluster_attn = np.full(args.k, np.nan, dtype=np.float32)
    for c in range(args.k):
        mask_c = labels == c
        if mask_c.any():
            cluster_attn[c] = attentions[mask_c].mean()
    attn_thr = float(np.nanquantile(cluster_attn, args.min_attn_quantile))
    foreground_clusters = set(
        int(c) for c in range(args.k)
        if not np.isnan(cluster_attn[c]) and cluster_attn[c] >= attn_thr
    )
    foreground_clusters.discard(bit_class)
    print(f"foreground non-bit clusters: {sorted(foreground_clusters)}")
    phase4_background_removal(
        cluster_attn, attn_thr, bit_class, foreground_clusters,
        sample_img_path, sample_labels,
        os.path.join(args.out_dir, "phase4_background_removal.png"),
    )

    # Phase 5 — voting
    bit_anchor = np.array([br, bc], dtype=np.float32)
    fg_set = np.array(sorted(foreground_clusters))
    votes = np.zeros(args.k, dtype=np.int64)
    for i in range(labels.shape[0]):
        fg_mask = np.isin(labels[i], fg_set)
        coords = np.argwhere(fg_mask)
        if len(coords) == 0:
            continue
        d = np.linalg.norm(coords.astype(np.float32) - bit_anchor, axis=1)
        j = int(np.argmin(d))
        r_i, c_i = coords[j]
        votes[int(labels[i, r_i, c_i])] += 1
    target_class = int(votes.argmax())
    print(f"target_class={target_class}, votes={int(votes[target_class])}/{int(votes.sum())}")
    phase5_voting(
        votes, bit_class, target_class,
        sample_img_path, sample_labels, foreground_clusters,
        args.bit_tip, grid,
        os.path.join(args.out_dir, "phase5_voting.png"),
    )

    # Phase 6 — connected components
    phase6_connected_components(
        sample_img_path, sample_labels, target_class,
        args.bit_tip, grid,
        os.path.join(args.out_dir, "phase6_connected_components.png"),
    )

    # Phase 7 — bbox
    struct = np.ones((3, 3), dtype=np.int32)
    rows_list, cols_list = [], []
    for i in range(labels.shape[0]):
        mask = labels[i] == target_class
        if not mask.any():
            continue
        cc, n_cc = cc_label(mask, structure=struct)
        best_coords = None
        best_dist = np.inf
        for cc_id in range(1, n_cc + 1):
            coords = np.argwhere(cc == cc_id)
            d = np.linalg.norm(coords.astype(np.float32) - bit_anchor, axis=1).min()
            if d < best_dist:
                best_dist = d
                best_coords = coords
        rows_list.append(best_coords[:, 0])
        cols_list.append(best_coords[:, 1])
    rows = np.concatenate(rows_list).astype(np.float32)
    cols = np.concatenate(cols_list).astype(np.float32)

    p_low = (100.0 - args.percentile) / 2.0
    p_high = 100.0 - p_low
    r_min = float(np.percentile(rows, p_low))
    r_max = float(np.percentile(rows, p_high))
    c_min = float(np.percentile(cols, p_low))
    c_max = float(np.percentile(cols, p_high))
    roi = patch_bbox_to_orig((r_min, r_max), (c_min, c_max), grid, orig_size)
    print(f"ROI = {roi}")
    phase7_bbox(
        sample_img_path, rows, cols, roi, grid, args.bit_tip, args.percentile,
        os.path.join(args.out_dir, "phase7_bbox.png"),
    )

    print(f"\nAll phase images saved under {args.out_dir}")


if __name__ == "__main__":
    main()
