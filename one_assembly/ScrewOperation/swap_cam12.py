"""In-place swap of cam1 ↔ cam2 in a ScrewOperation dataset directory.

Useful when an existing dataset was captured with the opposite cam1/cam2
file-naming convention (e.g. an older Isaac Sim ``CAM_TAG`` mapping) and you
want to bring it in line with the current convention without re-capturing.

The script:
  - renames every ``<idx:06d>_cam1.png`` ↔ ``<idx:06d>_cam2.png`` under
    ``<ep_dir>/images/`` (using a temporary suffix so the renames don't
    collide)
  - swaps ``roi1`` ↔ ``roi2`` and ``rotate1`` ↔ ``rotate2`` in
    ``<ep_dir>/config.yaml`` (if present)

Idempotency: running twice returns the directory to its original state.

Usage:
    uv run python -m one_assembly.ScrewOperation.swap_cam12 \\
        --ep_dir one_assembly/ScrewOperation/datasets/train/001 [--dry_run]
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import yaml


_PNG_PATTERN = re.compile(r"^(\d{6})_(cam[12])\.png$")
_TEMP_SUFFIX = ".__swap_tmp__"


def _list_image_pairs(images_dir: Path) -> list[tuple[Path, Path]]:
    """Return matched (cam1_path, cam2_path) pairs sorted by sample index."""
    by_idx: dict[str, dict[str, Path]] = {}
    for p in sorted(images_dir.iterdir()):
        m = _PNG_PATTERN.match(p.name)
        if not m:
            continue
        idx, tag = m.group(1), m.group(2)
        by_idx.setdefault(idx, {})[tag] = p
    pairs: list[tuple[Path, Path]] = []
    for idx in sorted(by_idx):
        cams = by_idx[idx]
        if "cam1" in cams and "cam2" in cams:
            pairs.append((cams["cam1"], cams["cam2"]))
    return pairs


def swap_images(images_dir: Path, dry_run: bool = False) -> int:
    """Swap cam1.png ↔ cam2.png for every matched pair. Returns pair count."""
    pairs = _list_image_pairs(images_dir)
    if not pairs:
        return 0
    if dry_run:
        return len(pairs)
    # Two-phase rename to avoid collisions.
    for cam1, cam2 in pairs:
        cam1.rename(cam1.with_suffix(cam1.suffix + _TEMP_SUFFIX))
    for cam1, cam2 in pairs:
        cam2.rename(cam1)  # cam2 → cam1.png
    for cam1, cam2 in pairs:
        cam1.with_suffix(cam1.suffix + _TEMP_SUFFIX).rename(cam2)  # cam1.tmp → cam2.png
    return len(pairs)


def swap_config(config_path: Path, dry_run: bool = False) -> dict | None:
    """Swap roi1↔roi2 and rotate1↔rotate2 in config.yaml. Returns the new
    dict written (or that would be written when dry_run)."""
    if not config_path.exists():
        return None
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    swapped = False
    if "roi1" in data and "roi2" in data:
        data["roi1"], data["roi2"] = data["roi2"], data["roi1"]
        swapped = True
    if "rotate1" in data and "rotate2" in data:
        data["rotate1"], data["rotate2"] = data["rotate2"], data["rotate1"]
        swapped = True
    if not swapped:
        return None
    if not dry_run:
        # Use the project's flow-style list representer when present so the
        # written file matches what save_config would have produced.
        try:
            from one_assembly.ScrewOperation.config import _FlowList  # type: ignore
            for k in ("roi1", "roi2"):
                if isinstance(data.get(k), (list, tuple)):
                    data[k] = _FlowList(data[k])
        except Exception:
            pass
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return data


def swap_dataset(ep_dir: Path, *, dry_run: bool = False) -> dict:
    """Swap a dataset directory's cam1/cam2 files and config keys.

    Returns a small summary dict for logging.
    """
    ep_dir = Path(ep_dir)
    images_dir = ep_dir / "images"
    config_path = ep_dir / "config.yaml"

    n_pairs = 0
    if images_dir.is_dir():
        n_pairs = swap_images(images_dir, dry_run=dry_run)

    config_swapped = swap_config(config_path, dry_run=dry_run) is not None

    return {
        "ep_dir": str(ep_dir),
        "image_pairs_swapped": n_pairs,
        "config_swapped": config_swapped,
        "dry_run": dry_run,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep_dir", required=True,
                    help="Dataset directory containing images/ and config.yaml")
    ap.add_argument("--dry_run", action="store_true",
                    help="Report what would change without modifying files")
    args = ap.parse_args()
    summary = swap_dataset(Path(args.ep_dir), dry_run=args.dry_run)
    print("swap summary:", summary)


if __name__ == "__main__":
    main()
