from __future__ import annotations

import argparse
import os
from typing import Any

import yaml
from pydantic import BaseModel, field_validator


_VALID_ROTATIONS = (0, 90, 180, 270)
# Provenance of the captured images. Empty string means "unspecified" — kept
# as the default so existing config.yaml files (which lack this field) load
# without error. New captures should populate one of "sim" / "real".
_VALID_DATA_SOURCES = ("", "sim", "real")


class ScrewConfig(BaseModel):
    description: str = ""

    sequence: str = ""
    mode: str = ""

    # Where the dataset was captured. "sim" = Isaac Sim, "real" = physical
    # cameras. Empty string for legacy / unspecified data.
    data_source: str = ""

    @field_validator("data_source")
    @classmethod
    def _validate_data_source(cls, v: str) -> str:
        if v not in _VALID_DATA_SOURCES:
            raise ValueError(
                f"data_source must be one of {_VALID_DATA_SOURCES}, got {v!r}"
            )
        return v

    # ROI applied AFTER rotation, in the rotated image's pixel coordinates.
    # Order: (left, upper, right, lower) — PIL .crop() convention.
    # Defaults are sim-tuned 60×60 crops centred on the SD bit tip for each
    # wrist camera at the rly_scrw pickup prescrew pose; revisit if camera
    # mount or shank length changes.
    roi1: tuple[int, int, int, int] = (495, 175, 555, 235)
    roi2: tuple[int, int, int, int] = (80, 198, 140, 258)
    # Clockwise rotation in degrees applied to each cam image BEFORE the ROI
    # crop. Restricted to 0/90/180/270 so rotation is lossless. The raw PNG on
    # disk is unrotated (one upgrade vs. the old wrs pipeline which baked
    # rotation + square crop into the saved frame).
    rotate1: int = 0
    rotate2: int = 0

    @field_validator("rotate1", "rotate2")
    @classmethod
    def _validate_rotation(cls, v: int) -> int:
        if v not in _VALID_ROTATIONS:
            raise ValueError(
                f"rotate must be one of {_VALID_ROTATIONS} (clockwise degrees), got {v}"
            )
        return int(v)

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
