from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from one_assembly.ScrewOperation.config import ScrewConfig, load_config, merge_cli_args


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID_DEFAULT = "facebook/dinov3-vits16-pretrain-lvd1689m"


def load_dinov3(model_id: str, device: str):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        attn_implementation="eager",
    ).to(device).eval()
    return processor, model


@torch.inference_mode()
def compute_attention_map(
    image: Image.Image,
    processor,
    model,
    device: str,
    resolution: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    proc_kwargs = {"size": {"height": resolution, "width": resolution}}
    inputs = processor(images=image, return_tensors="pt", **proc_kwargs).to(device)
    pixel_values = inputs["pixel_values"]
    _, _, h_in, w_in = pixel_values.shape

    outputs = model(pixel_values=pixel_values, output_attentions=True)
    attn = outputs.attentions[-1][0].mean(dim=0)  # [seq, seq]

    patch_size = model.config.patch_size
    num_patches_h = h_in // patch_size
    num_patches_w = w_in // patch_size
    num_patches = num_patches_h * num_patches_w
    num_prefix = attn.shape[0] - num_patches  # CLS + register tokens

    cls_to_patch = attn[0, num_prefix:]
    grid = cls_to_patch.reshape(num_patches_h, num_patches_w)

    up = F.interpolate(
        grid[None, None].float(),
        size=(h_in, w_in),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()

    up = (up - up.min()) / (up.max() - up.min() + 1e-8)
    return up, (h_in, w_in)


@torch.inference_mode()
def compute_feature_pca_map(
    image: Image.Image,
    processor,
    model,
    device: str,
    resolution: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return per-patch PCA(top-1) magnitude as a normalized heatmap."""
    proc_kwargs = {"size": {"height": resolution, "width": resolution}}
    inputs = processor(images=image, return_tensors="pt", **proc_kwargs).to(device)
    pixel_values = inputs["pixel_values"]
    _, _, h_in, w_in = pixel_values.shape

    outputs = model(pixel_values=pixel_values)
    hidden = outputs.last_hidden_state[0]  # (1+R+N, D)

    patch_size = model.config.patch_size
    num_patches_h = h_in // patch_size
    num_patches_w = w_in // patch_size
    num_register = getattr(model.config, "num_register_tokens", 0)

    patch_tokens = hidden[1 + num_register:, :]  # (N, D)
    feats = patch_tokens.float().cpu().numpy()
    feats = feats - feats.mean(axis=0, keepdims=True)
    # PCA via SVD, take top-1 component score per patch
    _, _, vt = np.linalg.svd(feats, full_matrices=False)
    score = feats @ vt[0]
    grid = score.reshape(num_patches_h, num_patches_w)

    up = torch.from_numpy(grid).float()[None, None]
    up = F.interpolate(up, size=(h_in, w_in), mode="bilinear", align_corners=False)
    up = up[0, 0].numpy()
    up = (up - up.min()) / (up.max() - up.min() + 1e-8)
    return up, (h_in, w_in)


def make_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """RGB PIL + normalized heatmap -> BGR overlay (uint8)."""
    h, w = heatmap.shape
    img_resized = np.array(image.resize((w, h)))  # RGB
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    heat_u8 = (heatmap * 255.0).clip(0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heat_color, alpha, 0)


def crop_image(img: Image.Image, roi: tuple[int, int, int, int]) -> Image.Image:
    return img.crop(roi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--roi1", type=int, nargs=4, default=None)
    parser.add_argument("--roi2", type=int, nargs=4, default=None)
    parser.add_argument("--model_id", type=str, default=MODEL_ID_DEFAULT)
    parser.add_argument("--resolution", type=int, default=224,
                        help="DINOv3 input resolution (multiple of patch size)")
    parser.add_argument("--mode", type=str, default="attn", choices=["attn", "pca"],
                        help="attn: CLS attention. pca: top-1 PCA of patch tokens")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--display_scale", type=float, default=2.0,
                        help="scale factor for display windows")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    config_path = os.path.join(args.data_dir, "config.yaml")
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = ScrewConfig()
    config = merge_cli_args(config, args)

    image_dir = os.path.join(args.data_dir, "images")
    cam1_files = sorted(glob.glob(os.path.join(image_dir, "*_cam1.png")))
    cam2_files = sorted(glob.glob(os.path.join(image_dir, "*_cam2.png")))
    if not cam1_files:
        print(f"No images found in {image_dir}")
        return
    n = min(len(cam1_files), len(cam2_files))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  model: {args.model_id}  resolution: {args.resolution}  mode: {args.mode}")
    print(f"Found {n} image pairs")
    print(f"ROI1: {config.roi1}")
    print(f"ROI2: {config.roi2}")
    print("Controls: SPACE=play/pause, m=toggle mode (attn/pca), q=quit")

    processor, model = load_dinov3(args.model_id, device)

    compute_fn = compute_attention_map if args.mode == "attn" else compute_feature_pca_map
    mode = args.mode

    window_name = "DINOv3 Dataset Test (SPACE: play/pause, m: mode, q: quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("frame", window_name, 0, max(n - 1, 1), lambda x: None)

    playing = False
    prev_idx = -1
    prev_mode = mode
    wait_ms = int(1000 / args.fps)

    while True:
        idx = cv2.getTrackbarPos("frame", window_name)
        if playing:
            idx = (idx + 1) % n
            cv2.setTrackbarPos("frame", window_name, idx)

        if idx != prev_idx or mode != prev_mode:
            prev_idx = idx
            prev_mode = mode

            img1 = Image.open(cam1_files[idx]).convert("RGB")
            img2 = Image.open(cam2_files[idx]).convert("RGB")

            crop1 = crop_image(img1, config.roi1)
            crop2 = crop_image(img2, config.roi2)

            fn = compute_attention_map if mode == "attn" else compute_feature_pca_map
            heat1, _ = fn(crop1, processor, model, device, args.resolution)
            heat2, _ = fn(crop2, processor, model, device, args.resolution)

            over1 = make_overlay(crop1, heat1, alpha=args.alpha)
            over2 = make_overlay(crop2, heat2, alpha=args.alpha)

            # Raw crops at the same display size as the overlays
            h, w = over1.shape[:2]
            crop1_bgr = cv2.cvtColor(np.array(crop1.resize((w, h))), cv2.COLOR_RGB2BGR)
            crop2_bgr = cv2.cvtColor(np.array(crop2.resize((w, h))), cv2.COLOR_RGB2BGR)

            top = np.hstack((crop1_bgr, crop2_bgr))
            bottom = np.hstack((over1, over2))
            view = np.vstack((top, bottom))

            if args.display_scale != 1.0:
                view = cv2.resize(
                    view, None,
                    fx=args.display_scale, fy=args.display_scale,
                    interpolation=cv2.INTER_NEAREST,
                )

            status = "PLAY" if playing else "PAUSE"
            label = f"[{idx:04d}/{n - 1}] {status}  mode={mode}"
            cv2.putText(view, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            # Also show the raw frame with the ROI rectangle for context
            img1_bgr = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
            img2_bgr = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
            x1, y1, x2, y2 = config.roi1
            cv2.rectangle(img1_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            x1, y1, x2, y2 = config.roi2
            cv2.rectangle(img2_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            raw = np.hstack((img1_bgr, img2_bgr))

            cv2.imshow(window_name, view)
            cv2.imshow("Raw (ROI overlay)", raw)

        key = cv2.waitKey(wait_ms if playing else 30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("m"):
            mode = "pca" if mode == "attn" else "attn"

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
