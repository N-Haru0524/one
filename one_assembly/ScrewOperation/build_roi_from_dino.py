from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from scipy.ndimage import label as cc_label
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoImageProcessor, AutoModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID_DEFAULT = "facebook/dinov3-vits16-pretrain-lvd1689m"


@dataclass
class CameraResult:
    roi: tuple[int, int, int, int]
    target_class: int
    bit_class: int
    labels: np.ndarray               # (N, Hp, Wp)
    image_paths: list[str]
    patch_grid: tuple[int, int]      # (Hp, Wp)
    image_size: tuple[int, int]      # (H, W) original
    bit_patch_rc: tuple[int, int]    # bit tip in patch coords
    centroids_rc: np.ndarray         # (K, 2)


def collect_image_paths(train_dirs: list[str], cam_tag: str) -> list[str]:
    paths = []
    for d in train_dirs:
        d = d if os.path.isabs(d) else os.path.join(BASE_DIR, d)
        img_dir = os.path.join(d, "images")
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if fname.endswith(f"_{cam_tag}.png"):
                paths.append(os.path.join(img_dir, fname))
    return paths


def load_dinov3(model_id: str, device: str):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        attn_implementation="eager",
    ).to(device).eval()
    return processor, model


@torch.inference_mode()
def extract_patch_features(
    image_paths: list[str],
    processor,
    model,
    device: str,
    resolution: int,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    """Returns (N, Hp, Wp, D) features, (N, Hp, Wp) CLS attention, (Hp, Wp), (H0, W0)."""
    patch_size = model.config.patch_size
    num_register = getattr(model.config, "num_register_tokens", 0)
    Hp = Wp = resolution // patch_size

    first = Image.open(image_paths[0]).convert("RGB")
    orig_size = first.size[::-1]  # (H, W)

    proc_kwargs = {"size": {"height": resolution, "width": resolution}}

    features = np.zeros((len(image_paths), Hp, Wp, model.config.hidden_size), dtype=np.float32)
    attentions = np.zeros((len(image_paths), Hp, Wp), dtype=np.float32)
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=imgs, return_tensors="pt", **proc_kwargs).to(device)
        outputs = model(**inputs, output_attentions=True)
        hidden = outputs.last_hidden_state  # (B, 1 + R + Hp*Wp, D)
        patch_tokens = hidden[:, 1 + num_register:, :]
        patch_tokens = patch_tokens.reshape(len(batch_paths), Hp, Wp, -1)
        features[i:i + len(batch_paths)] = patch_tokens.float().cpu().numpy()

        attn = outputs.attentions[-1]          # (B, heads, seq, seq)
        attn = attn.mean(dim=1)                # (B, seq, seq)
        cls_to_patch = attn[:, 0, 1 + num_register:]  # (B, Hp*Wp)
        cls_to_patch = cls_to_patch.reshape(len(batch_paths), Hp, Wp)
        attentions[i:i + len(batch_paths)] = cls_to_patch.float().cpu().numpy()

        if (i // batch_size) % 10 == 0:
            print(f"  extracted {i + len(batch_paths)}/{len(image_paths)}")
    return features, attentions, (Hp, Wp), orig_size


def cluster_patches(features: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    N, Hp, Wp, D = features.shape
    flat = features.reshape(-1, D)
    # Normalize for cosine-like clustering
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    flat = flat / np.maximum(norms, 1e-8)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10, batch_size=4096)
    labels = km.fit_predict(flat)
    return labels.reshape(N, Hp, Wp)


def orig_to_patch_rc(xy_orig: tuple[int, int], orig_size: tuple[int, int], grid: tuple[int, int]) -> tuple[int, int]:
    x, y = xy_orig
    H, W = orig_size
    Hp, Wp = grid
    r = int(np.clip(y * Hp / H, 0, Hp - 1))
    c = int(np.clip(x * Wp / W, 0, Wp - 1))
    return r, c


def patch_bbox_to_orig(
    row_range: tuple[float, float],
    col_range: tuple[float, float],
    grid: tuple[int, int],
    orig_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    Hp, Wp = grid
    H, W = orig_size
    r_min, r_max = row_range
    c_min, c_max = col_range
    # A patch at row r covers [r/Hp, (r+1)/Hp] of the image.
    y_min = int(np.floor(r_min / Hp * H))
    y_max = int(np.ceil((r_max + 1) / Hp * H))
    x_min = int(np.floor(c_min / Wp * W))
    x_max = int(np.ceil((c_max + 1) / Wp * W))
    y_min = max(0, y_min); x_min = max(0, x_min)
    y_max = min(H, y_max); x_max = min(W, x_max)
    return x_min, y_min, x_max, y_max


def build_roi_for_camera(
    image_paths: list[str],
    bit_tip_xy: tuple[int, int],
    processor,
    model,
    device: str,
    resolution: int,
    k: int,
    percentile: float,
    min_attn_quantile: float,
) -> CameraResult:
    print(f"[{len(image_paths)} images] extracting DINO patch features...")
    features, attentions, grid, orig_size = extract_patch_features(
        image_paths, processor, model, device, resolution,
    )
    Hp, Wp = grid

    print(f"  clustering patches (K={k}) ...")
    labels = cluster_patches(features, k=k)  # (N, Hp, Wp)

    # Bit class: most frequent label at bit tip patch across all images
    br, bc = orig_to_patch_rc(bit_tip_xy, orig_size, grid)
    bit_labels = labels[:, br, bc]
    bit_class = int(np.bincount(bit_labels, minlength=k).argmax())
    print(f"  bit tip patch ({br},{bc}); bit_class={bit_class}")

    # Cluster spatial centroids (diagnostic only; not used for selection)
    centroids = np.full((k, 2), np.nan, dtype=np.float32)
    for c in range(k):
        mask = labels == c
        if not mask.any():
            continue
        coords = np.argwhere(mask)  # (M, 3): image_idx, row, col
        centroids[c, 0] = coords[:, 1].mean()
        centroids[c, 1] = coords[:, 2].mean()

    # Cluster mean attention: low-attention clusters are "background" and excluded.
    cluster_attn = np.full(k, np.nan, dtype=np.float32)
    for c in range(k):
        mask = labels == c
        if mask.any():
            cluster_attn[c] = attentions[mask].mean()
    attn_thr = float(np.nanquantile(cluster_attn, min_attn_quantile))
    foreground_clusters = set(
        int(c) for c in range(k)
        if not np.isnan(cluster_attn[c]) and cluster_attn[c] >= attn_thr
    )
    foreground_clusters.discard(bit_class)
    print(f"  cluster attention (sorted): "
          + ", ".join(
              f"cls{c}={cluster_attn[c]:.4f}"
              for c in np.argsort(-np.nan_to_num(cluster_attn, nan=-1))[:min(k, 10)]
          ))
    print(f"  attn threshold ({min_attn_quantile:.2f} quantile) = {attn_thr:.4f}")
    print(f"  foreground non-bit clusters: {sorted(foreground_clusters)}")
    if not foreground_clusters:
        raise RuntimeError("no foreground non-bit clusters after attention filtering")

    # Per-image majority vote: for each image, find the closest foreground non-bit
    # patch to bit tip and cast a vote for its cluster.
    bit_anchor = np.array([br, bc], dtype=np.float32)
    fg_set = np.array(sorted(foreground_clusters))
    votes = np.zeros(k, dtype=np.int64)
    for i in range(labels.shape[0]):
        fg_mask = np.isin(labels[i], fg_set)
        coords = np.argwhere(fg_mask)
        if len(coords) == 0:
            continue
        d = np.linalg.norm(coords.astype(np.float32) - bit_anchor, axis=1)
        j = int(np.argmin(d))
        r_i, c_i = coords[j]
        votes[int(labels[i, r_i, c_i])] += 1
    if votes.sum() == 0:
        raise RuntimeError("no votes cast; check bit_class / foreground mask")
    target_class = int(votes.argmax())
    top5 = np.argsort(votes)[::-1][:5]
    print(f"  vote top5 (bit={bit_class}): "
          + ", ".join(f"cls{int(c)}={int(votes[c])}" for c in top5))
    print(f"  target_class = {target_class} (votes={int(votes[target_class])}/{int(votes.sum())})")

    # For each image, keep only the connected component of the target class that is
    # closest to the bit tip (there may be multiple screws / disjoint regions).
    struct = np.ones((3, 3), dtype=np.int32)  # 8-connectivity
    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    n_multi = 0
    for i in range(labels.shape[0]):
        mask = labels[i] == target_class
        if not mask.any():
            continue
        cc, n_cc = cc_label(mask, structure=struct)
        if n_cc > 1:
            n_multi += 1
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
    if not rows_list:
        raise RuntimeError("target class has no patches across dataset")
    rows = np.concatenate(rows_list).astype(np.float32)
    cols = np.concatenate(cols_list).astype(np.float32)
    print(f"  {n_multi}/{labels.shape[0]} images had multiple target components; "
          "kept closest-to-bit per image")

    p_low = (100.0 - percentile) / 2.0
    p_high = 100.0 - p_low
    r_min = float(np.percentile(rows, p_low))
    r_max = float(np.percentile(rows, p_high))
    c_min = float(np.percentile(cols, p_low))
    c_max = float(np.percentile(cols, p_high))

    roi = patch_bbox_to_orig((r_min, r_max), (c_min, c_max), grid, orig_size)
    print(f"  percentile patch bbox: rows=[{r_min:.1f},{r_max:.1f}] cols=[{c_min:.1f},{c_max:.1f}]")
    print(f"  ROI (orig px): {roi}")

    return CameraResult(
        roi=roi,
        target_class=target_class,
        bit_class=bit_class,
        labels=labels,
        image_paths=image_paths,
        patch_grid=grid,
        image_size=orig_size,
        bit_patch_rc=(br, bc),
        centroids_rc=centroids,
    )


def save_visualization(
    result: CameraResult,
    sample_indices: list[int],
    bit_tip_xy: tuple[int, int],
    out_path: str,
    cmap_name: str | None = None,
):
    if cmap_name is None:
        k_est = int(result.labels.max()) + 1
        cmap_name = "tab20" if k_est > 10 else "tab10"
    Hp, Wp = result.patch_grid
    H, W = result.image_size
    k = int(result.labels.max()) + 1

    cmap = plt.get_cmap(cmap_name, k)
    n = len(sample_indices)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10), squeeze=False)

    for i, idx in enumerate(sample_indices):
        img = Image.open(result.image_paths[idx]).convert("RGB")
        img_arr = np.array(img)
        axes[0, i].imshow(img_arr)
        axes[0, i].set_title(os.path.basename(result.image_paths[idx]))
        axes[0, i].axis("off")

        # Cluster label overlay (upsampled to image size)
        lbl = result.labels[idx]
        lbl_img = np.array(
            Image.fromarray(lbl.astype(np.uint8)).resize((W, H), Image.NEAREST)
        )
        color = cmap(lbl_img)[:, :, :3]
        axes[1, i].imshow(img_arr)
        axes[1, i].imshow(color, alpha=0.5)

        # Bit tip marker
        axes[1, i].plot(bit_tip_xy[0], bit_tip_xy[1], "wo", markersize=10, markeredgecolor="k")

        # Target class bbox
        x1, y1, x2, y2 = result.roi
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=2, edgecolor="red", facecolor="none")
        axes[1, i].add_patch(rect)

        axes[1, i].set_title(
            f"bit_cls={result.bit_class} target_cls={result.target_class}"
        )
        axes[1, i].axis("off")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved viz: {out_path}")


def parse_xy(s: str) -> tuple[int, int]:
    x, y = s.split(",")
    return int(x), int(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dirs", type=str, nargs="+", required=True,
                        help="dataset dirs (each has images/ with *_cam1.png / *_cam2.png)")
    parser.add_argument("--bit_tip_cam1", type=parse_xy, required=True,
                        help='bit tip pixel in cam1 image as "x,y"')
    parser.add_argument("--bit_tip_cam2", type=parse_xy, required=True,
                        help='bit tip pixel in cam2 image as "x,y"')
    parser.add_argument("--model_id", type=str, default=MODEL_ID_DEFAULT)
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--min_attn_quantile", type=float, default=0.5,
                        help="exclude clusters whose mean CLS attention is below this quantile")
    parser.add_argument("--sample_viz", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default=os.path.join(BASE_DIR, "tmp", "roi_dino"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "model:", args.model_id, "resolution:", args.resolution)

    processor, model = load_dinov3(args.model_id, device)

    results = {}
    for cam_tag, bit_xy in [("cam1", args.bit_tip_cam1), ("cam2", args.bit_tip_cam2)]:
        print(f"\n=== {cam_tag} ===")
        paths = collect_image_paths(args.train_dirs, cam_tag)
        if not paths:
            raise RuntimeError(f"no images found for {cam_tag}")

        res = build_roi_for_camera(
            paths, bit_xy, processor, model, device,
            resolution=args.resolution, k=args.k, percentile=args.percentile,
            min_attn_quantile=args.min_attn_quantile,
        )
        results[cam_tag] = res

        # Sample visualization
        n_imgs = len(res.image_paths)
        sample_indices = np.linspace(0, n_imgs - 1, args.sample_viz, dtype=int).tolist()
        viz_path = os.path.join(args.out_dir, f"{cam_tag}_cluster_viz.png")
        save_visualization(res, sample_indices, bit_xy, viz_path)

    # Output ROIs
    out_yaml = os.path.join(args.out_dir, "rois.yaml")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.dump({
            "roi1": list(results["cam1"].roi),
            "roi2": list(results["cam2"].roi),
            "meta": {
                "model_id": args.model_id,
                "resolution": args.resolution,
                "k": args.k,
                "percentile": args.percentile,
                "bit_tip_cam1": list(args.bit_tip_cam1),
                "bit_tip_cam2": list(args.bit_tip_cam2),
                "num_cam1_images": len(results["cam1"].image_paths),
                "num_cam2_images": len(results["cam2"].image_paths),
            },
        }, f, default_flow_style=None, sort_keys=False)
    print(f"\nwrote: {out_yaml}")
    print(f"roi1 = {results['cam1'].roi}")
    print(f"roi2 = {results['cam2'].roi}")


if __name__ == "__main__":
    main()
