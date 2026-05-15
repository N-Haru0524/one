from __future__ import annotations

from linformer import Linformer
from vit_pytorch.efficient import ViT

from one_assembly.ScrewOperation.config import ScrewConfig


def build_vit(config: ScrewConfig | None = None, *, num_classes: int | None = None) -> ViT:
    if config is None:
        config = ScrewConfig()
    if num_classes is not None:
        config = config.model_copy(update={"num_classes": num_classes})

    h, w = config.image_size
    seq_len = (h // config.patch_size) * (w // config.patch_size) + 1

    transformer = Linformer(
        dim=config.dim,
        seq_len=seq_len,
        depth=config.depth,
        heads=config.heads,
        k=config.k,
    )
    model = ViT(
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        dim=config.dim,
        transformer=transformer,
        channels=config.channels,
    )
    return model
