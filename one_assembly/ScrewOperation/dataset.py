from __future__ import annotations

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from one_assembly.ScrewOperation.config import ScrewConfig
from one_assembly.ScrewOperation.preprocess import apply_rotation_and_roi


class SpiralDataset(Dataset):
    def __init__(
        self,
        csv_paths: str | list[str],
        image_dirs: str | list[str],
        config: ScrewConfig,
    ):
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        if isinstance(image_dirs, str):
            image_dirs = [image_dirs]

        dfs = []
        for csv_path, image_dir in zip(csv_paths, image_dirs):
            df = pd.read_csv(csv_path)
            df["_image_dir"] = image_dir
            dfs.append(df)
        self.df = pd.concat(dfs, ignore_index=True)

        self.roi1 = config.roi1
        self.roi2 = config.roi2
        self.rotate1 = int(config.rotate1)
        self.rotate2 = int(config.rotate2)

        self.tf = transforms.Compose([
            transforms.Resize(config.resize_per_cam),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        sample_idx = int(row.idx)
        image_dir = row._image_dir

        img1_path = os.path.join(image_dir, f"{sample_idx:06}_cam1.png")
        img2_path = os.path.join(image_dir, f"{sample_idx:06}_cam2.png")

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        img1 = self.tf(apply_rotation_and_roi(img1, self.rotate1, self.roi1))
        img2 = self.tf(apply_rotation_and_roi(img2, self.rotate2, self.roi2))

        x = torch.cat([img1, img2], dim=2)
        y = int(row.label)

        return x, y


def load_and_preprocess_pair(
    cam1_path: str,
    cam2_path: str,
    config: ScrewConfig,
) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize(config.resize_per_cam),
        transforms.ToTensor(),
    ])
    img1 = apply_rotation_and_roi(
        Image.open(cam1_path).convert("RGB"), int(config.rotate1), config.roi1,
    )
    img2 = apply_rotation_and_roi(
        Image.open(cam2_path).convert("RGB"), int(config.rotate2), config.roi2,
    )
    img1 = tf(img1)
    img2 = tf(img2)
    return torch.cat([img1, img2], dim=2)
