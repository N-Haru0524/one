from __future__ import annotations

import argparse
import os
from typing import Any

import yaml
from pydantic import BaseModel


class ScrewConfig(BaseModel):
    description: str = ""

    sequence: str = ""
    mode: str = ""

    roi1: tuple[int, int, int, int] = (0, 0, 320, 240)
    roi2: tuple[int, int, int, int] = (0, 0, 320, 240)

    num_classes: int = 91
    patch_size: int = 5
    dim: int = 128
    depth: int = 12
    heads: int = 8
    k: int = 64
    channels: int = 3

    resize_per_cam: tuple[int, int] = (45, 40)

    epochs: int = 50
    batch_size: int = 256
    lr: float = 3e-4
    num_workers: int = 8

    spiral_step: float = 0.0008

    train_dirs: list[str] = []
    model_dir: str = ""

    max_num_samples: int = 91
    latency: float = 2.0

    @property
    def image_size(self) -> tuple[int, int]:
        h, w = self.resize_per_cam
        return (h, w * 2)


def load_config(path: str) -> ScrewConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return ScrewConfig(**data)


class _FlowList(list):
    pass


def _flow_list_representer(dumper: yaml.Dumper, data: _FlowList):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(_FlowList, _flow_list_representer)


def save_config(config: ScrewConfig, path: str) -> None:
    data = config.model_dump()
    for key in data:
        if isinstance(data[key], tuple):
            data[key] = _FlowList(data[key])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def merge_cli_args(config: ScrewConfig, args: argparse.Namespace) -> ScrewConfig:
    overrides: dict[str, Any] = {}
    for field in ScrewConfig.model_fields:
        val = getattr(args, field, None)
        if val is not None:
            field_type = ScrewConfig.model_fields[field].annotation
            if isinstance(val, list) and field_type is not list[str]:
                val = tuple(val)
            overrides[field] = val
    if overrides:
        return config.model_copy(update=overrides)
    return config
