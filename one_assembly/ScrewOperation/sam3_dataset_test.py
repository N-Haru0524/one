"""Interactive SAM 3 mask viewer for ScrewOperation datasets.

SAM 3 counterpart to :mod:`dino_dataset_test`. Loads a dataset dir, applies
ScrewConfig ``rotateN`` + ``roiN`` to each cam frame (matching the training
pipeline), runs SAM 3 text-prompted segmentation on each crop, and renders
the result as either a red mask overlay or a "background painted gray"
preview.

Controls:
  SPACE   play / pause
  m       toggle overlay (red mask) / paint (gray background)
  q       quit

Usage::

    uv run python -m one_assembly.ScrewOperation.sam3_dataset_test \
        --data_dir one_assembly/ScrewOperation/datasets/train/001 \
        --text "screw head"
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from one_assembly.ScrewOperation.config import ScrewConfig, load_config, merge_cli_args
from one_assembly.ScrewOperation.preprocess import apply_rotation_and_roi, rotate_image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_ID = "facebook/sam3"


def load_sam3(model_id: str, device: str):
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id).to(device).eval()
    return processor, model


@torch.inference_mode()
def segment_one(
    image: Image.Image,
    text: str,
    processor: Sam3Processor,
    model: Sam3Model,
    device: str,
    box: tuple[int, int, int, int] | None,
    threshold: float,
    mask_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (best_mask_bool HxW, scores 1d)."""
    proc_kwargs = {
        "images": image,
        "text": text,
        "return_tensors": "pt",
    }
    if box is not None:
        proc_kwargs["input_boxes"] = [[list(box)]]

    inputs = processor(**proc_kwargs).to(device)
    outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    masks = results["masks"]
    scores = results["scores"]
    if masks.numel() == 0:
        h, w = image.size[1], image.size[0]
        return np.zeros((h, w), dtype=bool), np.zeros((0,), dtype=np.float32)

    best = int(torch.argmax(scores).item())
    return masks[best].cpu().numpy().astype(bool), scores.cpu().numpy()


def make_overlay(image_rgb: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.5) -> np.ndarray:
    """RGB image + bool mask -> BGR overlay with red mask blended."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if mask.shape != bgr.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    red = np.zeros_like(bgr)
    red[..., 2] = 255
    out = bgr.copy()
    out[mask] = cv2.addWeighted(bgr, 1.0 - alpha, red, alpha, 0)[mask]
    return out


def paint_background_gray(image_rgb: np.ndarray, mask: np.ndarray,
                          fill: int = 128) -> np.ndarray:
    """RGB image + bool mask -> BGR image with background painted gray."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if mask.shape != bgr.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    out = bgr.copy()
    out[~mask] = fill
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--roi1", type=int, nargs=4, default=None)
    parser.add_argument("--roi2", type=int, nargs=4, default=None)
    parser.add_argument("--rotate1", type=int, choices=(0, 90, 180, 270), default=None,
                        help="CW rotation (deg) applied to cam1 before ROI; overrides config")
    parser.add_argument("--rotate2", type=int, choices=(0, 90, 180, 270), default=None,
                        help="CW rotation (deg) applied to cam2 before ROI; overrides config")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--text", type=str, default="screw head",
                        help="SAM 3 text prompt")
    parser.add_argument("--mode", type=str, default="overlay",
                        choices=["overlay", "paint"],
                        help="overlay: red mask on image. paint: gray background")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="instance score threshold")
    parser.add_argument("--mask_threshold", type=float, default=0.5,
                        help="mask binarization threshold")
    parser.add_argument("--resolution", type=int, default=224,
                        help="resize each display panel to this side length "
                             "(matches dino_dataset_test panel sizing)")
    parser.add_argument("--display_scale", type=float, default=2.0,
                        help="scale factor for display windows")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="playback fps (SAM 3 inference is slow)")
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
    print(f"device: {device}  model: {args.model_id}  text: '{args.text}'")
    print(f"Found {n} image pairs")
    print(f"rotate1: {config.rotate1}° CW   ROI1: {config.roi1}")
    print(f"rotate2: {config.rotate2}° CW   ROI2: {config.roi2}")
    print("Controls: SPACE=play/pause, m=toggle mode (overlay/paint), q=quit")

    processor, model = load_sam3(args.model_id, device)
    mode = args.mode

    window_name = "SAM3 Dataset Test (SPACE: play/pause, m: mode, q: quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("frame", window_name, 0, max(n - 1, 1), lambda x: None)

    playing = False
    prev_idx = -1
    prev_mode = mode
    wait_ms = int(1000 / args.fps)

    # Cache the latest inference so a mode toggle doesn't re-run SAM 3.
    cached_idx = -1
    cached = None

    while True:
        idx = cv2.getTrackbarPos("frame", window_name)
        if playing:
            idx = (idx + 1) % n
            cv2.setTrackbarPos("frame", window_name, idx)

        need_render = idx != prev_idx or mode != prev_mode
        if need_render:
            prev_idx = idx
            prev_mode = mode

            if cached_idx != idx:
                img1_raw = Image.open(cam1_files[idx]).convert("RGB")
                img2_raw = Image.open(cam2_files[idx]).convert("RGB")
                # Apply rotation then ROI crop (matches SpiralDataset pipeline).
                img1_rot = rotate_image(img1_raw, int(config.rotate1))
                img2_rot = rotate_image(img2_raw, int(config.rotate2))
                crop1 = apply_rotation_and_roi(img1_raw, int(config.rotate1), config.roi1)
                crop2 = apply_rotation_and_roi(img2_raw, int(config.rotate2), config.roi2)

                mask1, scores1 = segment_one(
                    crop1, args.text, processor, model, device,
                    box=None,
                    threshold=args.threshold,
                    mask_threshold=args.mask_threshold,
                )
                mask2, scores2 = segment_one(
                    crop2, args.text, processor, model, device,
                    box=None,
                    threshold=args.threshold,
                    mask_threshold=args.mask_threshold,
                )

                top1 = float(scores1.max()) if len(scores1) else 0.0
                top2 = float(scores2.max()) if len(scores2) else 0.0
                print(f"[{idx:04d}/{n - 1}]  "
                      f"cam1 n={len(scores1):2d} top={top1:.3f} px={int(mask1.sum())}  |  "
                      f"cam2 n={len(scores2):2d} top={top2:.3f} px={int(mask2.sum())}")

                cached = {
                    "img1_rot": img1_rot, "img2_rot": img2_rot,
                    "crop1": np.array(crop1), "crop2": np.array(crop2),
                    "mask1": mask1, "mask2": mask2,
                    "scores1": scores1, "scores2": scores2,
                }
                cached_idx = idx

            crop1_rgb = cached["crop1"]
            crop2_rgb = cached["crop2"]
            mask1 = cached["mask1"]
            mask2 = cached["mask2"]

            if mode == "overlay":
                view1 = make_overlay(crop1_rgb, mask1, alpha=args.alpha)
                view2 = make_overlay(crop2_rgb, mask2, alpha=args.alpha)
            else:
                view1 = paint_background_gray(crop1_rgb, mask1)
                view2 = paint_background_gray(crop2_rgb, mask2)

            r = args.resolution
            view1 = cv2.resize(view1, (r, r), interpolation=cv2.INTER_NEAREST)
            view2 = cv2.resize(view2, (r, r), interpolation=cv2.INTER_NEAREST)
            crop1_bgr = cv2.resize(
                cv2.cvtColor(crop1_rgb, cv2.COLOR_RGB2BGR), (r, r))
            crop2_bgr = cv2.resize(
                cv2.cvtColor(crop2_rgb, cv2.COLOR_RGB2BGR), (r, r))

            top = np.hstack((crop1_bgr, crop2_bgr))
            bottom = np.hstack((view1, view2))
            view = np.vstack((top, bottom))

            if args.display_scale != 1.0:
                view = cv2.resize(
                    view, None,
                    fx=args.display_scale, fy=args.display_scale,
                    interpolation=cv2.INTER_NEAREST,
                )

            status = "PLAY" if playing else "PAUSE"
            top1 = float(cached["scores1"].max()) if len(cached["scores1"]) else 0.0
            top2 = float(cached["scores2"].max()) if len(cached["scores2"]) else 0.0
            label = (
                f"[{idx:04d}/{n - 1}] {status}  mode={mode}  "
                f"text='{args.text}'  top1={top1:.2f} top2={top2:.2f}"
            )
            cv2.putText(view, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            # Raw rotated frames with ROI rectangle for context.
            img1_bgr = cv2.cvtColor(np.array(cached["img1_rot"]), cv2.COLOR_RGB2BGR)
            img2_bgr = cv2.cvtColor(np.array(cached["img2_rot"]), cv2.COLOR_RGB2BGR)
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
            mode = "paint" if mode == "overlay" else "overlay"

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
