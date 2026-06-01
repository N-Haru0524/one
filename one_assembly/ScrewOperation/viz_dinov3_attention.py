from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


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
    resolution: int | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    if resolution is not None:
        proc_kwargs = {"size": {"height": resolution, "width": resolution}}
        inputs = processor(images=image, return_tensors="pt", **proc_kwargs).to(device)
    else:
        inputs = processor(images=image, return_tensors="pt").to(device)

    pixel_values = inputs["pixel_values"]
    _, _, h_in, w_in = pixel_values.shape

    outputs = model(pixel_values=pixel_values, output_attentions=True)
    attn = outputs.attentions[-1]  # [1, heads, seq, seq]
    attn = attn[0].mean(dim=0)     # [seq, seq]

    patch_size = model.config.patch_size
    num_patches_h = h_in // patch_size
    num_patches_w = w_in // patch_size
    num_patches = num_patches_h * num_patches_w

    seq_len = attn.shape[0]
    num_prefix = seq_len - num_patches  # CLS + register tokens

    cls_to_patch = attn[0, num_prefix:]                          # [num_patches]
    grid = cls_to_patch.reshape(num_patches_h, num_patches_w)    # [H', W']

    up = F.interpolate(
        grid[None, None].float(),
        size=(h_in, w_in),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()

    up = (up - up.min()) / (up.max() - up.min() + 1e-8)
    return up, (h_in, w_in)


def overlay(ax, image: Image.Image, heatmap: np.ndarray, title: str):
    img_resized = image.resize((heatmap.shape[1], heatmap.shape[0]))
    ax.imshow(img_resized)
    ax.imshow(heatmap, cmap="jet", alpha=0.5)
    ax.set_title(title)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam1", type=str, required=True)
    parser.add_argument("--cam2", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=MODEL_ID_DEFAULT)
    parser.add_argument("--resolution", type=int, default=448,
                        help="input resolution fed to DINOv3 (must be multiple of patch size)")
    parser.add_argument("--out", type=str, default=os.path.join(BASE_DIR, "tmp", "dinov3_attention.png"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "model:", args.model_id, "resolution:", args.resolution)

    processor, model = load_dinov3(args.model_id, device)

    paths = [args.cam1] + ([args.cam2] if args.cam2 else [])
    n = len(paths)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9), squeeze=False)

    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        heatmap, (h, w) = compute_attention_map(
            img, processor, model, device, resolution=args.resolution,
        )

        img_resized = img.resize((w, h))
        axes[0, i].imshow(img_resized)
        axes[0, i].set_title(os.path.basename(p))
        axes[0, i].axis("off")
        overlay(axes[1, i], img, heatmap, "CLS attention (mean over heads, last layer)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
