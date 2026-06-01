"""Offline single-pair inference. Loads cam1/cam2 PNGs from disk, runs the
ViT classifier, and prints the predicted class and correction offset.

Useful for smoke-testing a trained model without robot/camera/ROS.
"""
from __future__ import annotations

import argparse
import os

import torch

from one_assembly.ScrewOperation.config import ScrewConfig, load_config
from one_assembly.ScrewOperation.model_builder import build_model
from one_assembly.ScrewOperation.dataset import load_and_preprocess_pair
from one_assembly.ScrewOperation.spiral_metry import hex_ring_abs


def infer(model_path: str, config: ScrewConfig, cam1: str, cam2: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    spiral_list = hex_ring_abs(config.num_classes, step=config.spiral_step)

    x = load_and_preprocess_pair(cam1, cam2, config)
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        class_id = int(logits.argmax(dim=1).item())

    dx, dy = spiral_list[class_id]

    print("predicted class:", class_id)
    print("correction dx, dy [m]:", float(dx), float(dy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing model.pt and config.yaml")
    parser.add_argument("--cam1", required=True, help="Path to cam1 PNG")
    parser.add_argument("--cam2", required=True, help="Path to cam2 PNG")

    args = parser.parse_args()

    config_path = os.path.join(args.model_dir, "config.yaml")
    model_path = os.path.join(args.model_dir, "model.pt")
    config = load_config(config_path)

    infer(model_path, config, args.cam1, args.cam2)
