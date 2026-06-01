"""Interactive feature viewer for the DINOv3 twin encoder.

Counterpart to :mod:`dino_dataset_test` but for the twin encoder
(:class:`DINOv3TwinClassifier`): one DINOv3 backbone runs over each cam crop,
then bidirectional cross-attention fusion blocks mix the two token streams.

Three display modes (cycle with ``m``):
  pre     per-patch top-1 PCA score *before* fusion (raw backbone features)
  post    per-patch top-1 PCA score *after* the fusion stack
  cross   CLS cross-attention from one cam onto the other cam's patches;
          shows where each cam looks in its partner's image
          (left panel = cam1 attended by cam2's CLS; right = cam2 by cam1's CLS)

Without ``--model_dir`` the fusion weights are random — only ``pre`` is
meaningful in that case.

Usage::

    uv run python -m one_assembly.ScrewOperation.dino_twin_dataset_test \\
        --data_dir one_assembly/ScrewOperation/datasets/train/001 \\
        [--model_dir path/to/trained_twin_model]
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from one_assembly.ScrewOperation.config import ScrewConfig, load_config, merge_cli_args
from one_assembly.ScrewOperation.model_builder import (
    DINOv3TwinClassifier,
    build_model,
)
from one_assembly.ScrewOperation.preprocess import apply_rotation_and_roi, rotate_image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(
    model_dir: str | None, config: ScrewConfig, device: str
) -> DINOv3TwinClassifier:
    model = build_model(config)
    if not isinstance(model, DINOv3TwinClassifier):
        raise RuntimeError(
            f"expected encoder='dinov3_twin' in config, got '{config.encoder}'"
        )
    if model_dir is None:
        print("no model_dir given; using frozen backbone + RANDOM fusion "
              "(only 'pre' mode is meaningful)")
        return model.to(device).eval()
    ckpt = os.path.join(model_dir, "model.pt")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
        print(f"loaded weights: {ckpt}")
    else:
        print(f"no checkpoint at {ckpt}; using frozen backbone + RANDOM fusion "
              "(only 'pre' mode is meaningful)")
    return model.to(device).eval()


def build_input(
    img1_raw: Image.Image,
    img2_raw: Image.Image,
    config: ScrewConfig,
) -> tuple[torch.Tensor, Image.Image, Image.Image]:
    """Apply rotate→ROI→resize→ToTensor, hstack, then return (tensor, crop1, crop2).

    The crops returned are *pre-resize* (i.e. raw cropped pixels) so overlays
    aren't blurry. The tensor matches what SpiralDataset feeds the model.
    """
    crop1 = apply_rotation_and_roi(img1_raw, int(config.rotate1), config.roi1)
    crop2 = apply_rotation_and_roi(img2_raw, int(config.rotate2), config.roi2)
    tf = transforms.Compose([
        transforms.Resize(config.resize_per_cam),
        transforms.ToTensor(),
    ])
    a = tf(crop1)
    b = tf(crop2)
    x = torch.cat([a, b], dim=2).unsqueeze(0)  # (1, 3, H, 2W)
    return x, crop1, crop2


def pca_top1_heatmap(patches: torch.Tensor) -> np.ndarray:
    """(Hp, Wp, D) -> (Hp, Wp) normalized to [0, 1]."""
    Hp, Wp, D = patches.shape
    feats = patches.reshape(-1, D).numpy().astype(np.float32)
    feats = feats - feats.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(feats, full_matrices=False)
    score = feats @ vt[0]
    grid = score.reshape(Hp, Wp)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    return grid


def cls_cross_attn_heatmap(
    weights: torch.Tensor, patch_grid: tuple[int, int]
) -> np.ndarray:
    """(B, N+1, N+1) cross-attn -> (Hp, Wp) normalized to [0, 1].

    Takes the CLS row (index 0) of batch item 0, drops the CLS column, and
    reshapes to the patch grid.
    """
    Hp, Wp = patch_grid
    w = weights[0, 0, 1:]
    grid = w.reshape(Hp, Wp).numpy()
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    return grid


def make_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    out_size: int | None = None,
) -> np.ndarray:
    """RGB PIL + (Hp, Wp) heatmap -> BGR overlay (uint8) at the larger of
    out_size and the image's native side.
    """
    if out_size is None:
        out_size = max(image.size[0], image.size[1], 256)
    target_w = max(out_size, image.size[0])
    target_h = max(out_size, image.size[1])
    img_resized = np.array(image.resize((target_w, target_h)))
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

    heat = torch.from_numpy(heatmap)[None, None].float()
    heat_up = F.interpolate(heat, size=(target_h, target_w),
                            mode="bilinear", align_corners=False)
    heat_u8 = (heat_up[0, 0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heat_color, alpha, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default=None,
                        help="dir containing config.yaml and model.pt for the twin model. "
                             "If omitted, falls back to <data_dir>/config.yaml with "
                             "encoder forced to 'dinov3_twin' (random fusion weights — "
                             "only 'pre' mode is meaningful)")
    parser.add_argument("--roi1", type=int, nargs=4, default=None)
    parser.add_argument("--roi2", type=int, nargs=4, default=None)
    parser.add_argument("--rotate1", type=int, choices=(0, 90, 180, 270), default=None)
    parser.add_argument("--rotate2", type=int, choices=(0, 90, 180, 270), default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--display_size", type=int, default=320,
                        help="per-panel display size (each cam panel ~ this)")
    parser.add_argument("--fusion_block", type=int, default=-1,
                        help="which fusion block to read cross-attn from (-1 = last)")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    if args.model_dir is not None:
        config_path = os.path.join(args.model_dir, "config.yaml")
    else:
        config_path = os.path.join(args.data_dir, "config.yaml")
    config = load_config(config_path)
    if config.encoder != "dinov3_twin":
        print(f"overriding encoder '{config.encoder}' -> 'dinov3_twin'")
        config = config.model_copy(update={"encoder": "dinov3_twin"})
    config = merge_cli_args(config, args)

    image_dir = os.path.join(args.data_dir, "images")
    cam1_files = sorted(glob.glob(os.path.join(image_dir, "*_cam1.png")))
    cam2_files = sorted(glob.glob(os.path.join(image_dir, "*_cam2.png")))
    if not cam1_files:
        print(f"No images found in {image_dir}")
        return
    n = min(len(cam1_files), len(cam2_files))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"encoder: {config.encoder}  dinov3_resolution: {config.dinov3_resolution}  "
          f"fusion_depth: {config.dinov3_twin_fusion_depth}")
    print(f"Found {n} image pairs")
    print(f"rotate1: {config.rotate1}° CW   ROI1: {config.roi1}")
    print(f"rotate2: {config.rotate2}° CW   ROI2: {config.roi2}")
    print("Controls: SPACE=play/pause, m=cycle mode (pre/post/cross), q=quit")

    model = load_model(args.model_dir, config, device)

    modes = ["pre", "post", "cross"]
    mode_idx = 0
    fusion_idx = args.fusion_block

    window_name = "DINOv3 Twin Feature Test (SPACE play/pause, m mode, q quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("frame", window_name, 0, max(n - 1, 1), lambda x: None)

    playing = False
    prev_idx = -1
    prev_mode = None
    wait_ms = int(1000 / args.fps)

    while True:
        idx = cv2.getTrackbarPos("frame", window_name)
        if playing:
            idx = (idx + 1) % n
            cv2.setTrackbarPos("frame", window_name, idx)
        mode = modes[mode_idx]

        if idx != prev_idx or mode != prev_mode:
            prev_idx = idx
            prev_mode = mode

            img1_raw = Image.open(cam1_files[idx]).convert("RGB")
            img2_raw = Image.open(cam2_files[idx]).convert("RGB")
            x, crop1, crop2 = build_input(img1_raw, img2_raw, config)
            x = x.to(device)

            feats = model.extract_features(x)
            patch_grid = feats["patch_grid"]

            if mode == "pre":
                h1 = pca_top1_heatmap(feats["pre_patches"][0][0])
                h2 = pca_top1_heatmap(feats["pre_patches"][1][0])
                label = "pre-fusion patch PCA"
            elif mode == "post":
                h1 = pca_top1_heatmap(feats["post_patches"][0][0])
                h2 = pca_top1_heatmap(feats["post_patches"][1][0])
                label = f"post-fusion patch PCA (depth={len(feats['cross_attn'])})"
            else:  # cross
                w_1to2, w_2to1 = feats["cross_attn"][fusion_idx]
                # cam1's CLS attending to cam2 -> overlay on cam2 (right)
                h2 = cls_cross_attn_heatmap(w_1to2, patch_grid)
                # cam2's CLS attending to cam1 -> overlay on cam1 (left)
                h1 = cls_cross_attn_heatmap(w_2to1, patch_grid)
                block_label = (
                    fusion_idx if fusion_idx >= 0
                    else len(feats["cross_attn"]) + fusion_idx
                )
                label = (
                    f"cross-attn CLS  cam1<-cam2 (left) / cam2<-cam1 (right)  "
                    f"block={block_label}"
                )

            over1 = make_overlay(crop1, h1, alpha=args.alpha, out_size=args.display_size)
            over2 = make_overlay(crop2, h2, alpha=args.alpha, out_size=args.display_size)
            h, w = over1.shape[:2]
            crop1_bgr = cv2.cvtColor(np.array(crop1.resize((w, h))), cv2.COLOR_RGB2BGR)
            crop2_bgr = cv2.cvtColor(np.array(crop2.resize((w, h))), cv2.COLOR_RGB2BGR)

            top = np.hstack((crop1_bgr, crop2_bgr))
            bottom = np.hstack((over1, over2))
            view = np.vstack((top, bottom))

            status = "PLAY" if playing else "PAUSE"
            header = f"[{idx:04d}/{n - 1}] {status}  mode={mode}  {label}"
            cv2.putText(view, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, view)

            # Also show raw rotated frames with ROI rectangles, for context.
            img1_rot = np.array(rotate_image(img1_raw, int(config.rotate1)))
            img2_rot = np.array(rotate_image(img2_raw, int(config.rotate2)))
            img1_bgr = cv2.cvtColor(img1_rot, cv2.COLOR_RGB2BGR)
            img2_bgr = cv2.cvtColor(img2_rot, cv2.COLOR_RGB2BGR)
            x1, y1, x2, y2 = config.roi1
            cv2.rectangle(img1_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            x1, y1, x2, y2 = config.roi2
            cv2.rectangle(img2_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            raw = np.hstack((img1_bgr, img2_bgr))
            cv2.imshow("Raw (ROI overlay)", raw)

        key = cv2.waitKey(wait_ms if playing else 30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("m"):
            mode_idx = (mode_idx + 1) % len(modes)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
