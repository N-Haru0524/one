from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class DINOv3Classifier(nn.Module):
    """Frozen DINOv3 backbone + trainable classification head.

    Pooling: CLS token concatenated with mean of patch tokens (2 * hidden_size).
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        num_classes: int = 91,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        freeze_backbone: bool = True,
        head_hidden: int | None = None,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        num_register_tokens: int = 4,
        target_size: int = 224,
    ):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_name)
        self.freeze_backbone = freeze_backbone
        self.num_register_tokens = num_register_tokens
        self.target_size = target_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        hidden_size = self.backbone.config.hidden_size
        feat_dim = hidden_size * 2

        self.norm = nn.LayerNorm(feat_dim) if use_layernorm else nn.Identity()

        if head_hidden is None:
            self.head = nn.Linear(feat_dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(feat_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, num_classes),
            )

        self.register_buffer(
            "imagenet_mean",
            torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.target_size, self.target_size):
            x = F.interpolate(
                x,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
        x = (x - self.imagenet_mean) / self.imagenet_std
        return x

    def _pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cls_token = hidden_states[:, 0, :]
        patch_start = 1 + self.num_register_tokens
        patch_mean = hidden_states[:, patch_start:, :].mean(dim=1)
        return torch.cat([cls_token, patch_mean], dim=-1)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        return self._pool(out.last_hidden_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self._extract_features(x)
        else:
            feat = self._extract_features(x)
        feat = self.norm(feat)
        return self.head(feat)

    @torch.no_grad()
    def count_trainable_params(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total, "frozen": total - trainable}


def build_dinov3(
    config: ScrewConfig | None = None, *, num_classes: int | None = None
) -> DINOv3Classifier:
    if config is None:
        config = ScrewConfig()
    if num_classes is not None:
        config = config.model_copy(update={"num_classes": num_classes})

    head_hidden = config.dinov3_head_hidden if config.dinov3_head_hidden > 0 else None

    return DINOv3Classifier(
        num_classes=config.num_classes,
        model_name=config.dinov3_model_id,
        freeze_backbone=config.dinov3_freeze,
        head_hidden=head_hidden,
        dropout=config.dinov3_dropout,
        use_layernorm=config.dinov3_use_layernorm,
        num_register_tokens=config.dinov3_num_register_tokens,
        target_size=config.dinov3_resolution,
    )


class CrossAttnFusionBlock(nn.Module):
    """Symmetric cross-attention between two token streams (pre-norm + FFN)."""

    def __init__(self, dim: int, heads: int = 8, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_kv1 = nn.LayerNorm(dim)
        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_kv2 = nn.LayerNorm(dim)
        self.attn1 = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn2 = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        return_weights: bool = False,
    ):
        q1, kv1 = self.norm_q1(t1), self.norm_kv1(t2)
        a1, w_1to2 = self.attn1(
            q1, kv1, kv1, need_weights=return_weights, average_attn_weights=True,
        )
        t1 = t1 + a1
        t1 = t1 + self.ff1(t1)

        q2, kv2 = self.norm_q2(t2), self.norm_kv2(t1)
        a2, w_2to1 = self.attn2(
            q2, kv2, kv2, need_weights=return_weights, average_attn_weights=True,
        )
        t2 = t2 + a2
        t2 = t2 + self.ff2(t2)
        if return_weights:
            return t1, t2, w_1to2, w_2to1
        return t1, t2


class DINOv3TwinClassifier(nn.Module):
    """Twin DINOv3 backbones (shared weights) + cross-attention fusion.

    Input is the same ``(B, 3, H, 2W)`` hstacked tensor produced by
    :class:`SpiralDataset`; it is split at the width center into cam1 / cam2
    internally so callers do not need a separate dataset variant.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        num_classes: int = 91,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        freeze_backbone: bool = True,
        head_hidden: int | None = None,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        num_register_tokens: int = 4,
        target_size: int = 224,
        fusion_depth: int = 1,
        fusion_heads: int = 8,
    ):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_name)
        self.freeze_backbone = freeze_backbone
        self.num_register_tokens = num_register_tokens
        self.target_size = target_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        hidden = self.backbone.config.hidden_size

        self.fusion = nn.ModuleList([
            CrossAttnFusionBlock(hidden, heads=fusion_heads, dropout=dropout)
            for _ in range(fusion_depth)
        ])

        feat_dim = hidden * 4  # [cls1, cls2, mean_patch1, mean_patch2]
        self.norm = nn.LayerNorm(feat_dim) if use_layernorm else nn.Identity()
        if head_hidden is None:
            self.head = nn.Linear(feat_dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(feat_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, num_classes),
            )

        self.register_buffer(
            "imagenet_mean",
            torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.target_size, self.target_size):
            x = F.interpolate(
                x,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
        return (x - self.imagenet_mean) / self.imagenet_std

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = x.shape[-1] // 2
        return x[..., :w], x[..., w:]

    def _encode_pair(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x1.shape[0]
        x = torch.cat([x1, x2], dim=0)  # (2B, 3, H, W)
        x = self._preprocess(x)
        h = self.backbone(pixel_values=x).last_hidden_state
        patch_start = 1 + self.num_register_tokens
        cls = h[:, 0, :]
        patches = h[:, patch_start:, :]
        return cls[:B], cls[B:], patches[:B], patches[B:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self._split(x)
        if self.freeze_backbone:
            with torch.no_grad():
                cls1, cls2, p1, p2 = self._encode_pair(x1, x2)
        else:
            cls1, cls2, p1, p2 = self._encode_pair(x1, x2)

        t1 = torch.cat([cls1.unsqueeze(1), p1], dim=1)
        t2 = torch.cat([cls2.unsqueeze(1), p2], dim=1)
        for blk in self.fusion:
            t1, t2 = blk(t1, t2)

        fused = torch.cat([
            t1[:, 0, :],
            t2[:, 0, :],
            t1[:, 1:, :].mean(dim=1),
            t2[:, 1:, :].mean(dim=1),
        ], dim=-1)
        fused = self.norm(fused)
        return self.head(fused)

    def patch_grid(self) -> tuple[int, int]:
        ps = self.backbone.config.patch_size
        n = self.target_size // ps
        return n, n

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> dict:
        """Intermediates for visualization. All tensors are CPU-detached.

        Patch tokens are reshaped to ``(B, Hp, Wp, D)``. ``cross_attn`` is a
        list with one ``(w_1to2, w_2to1)`` pair per fusion block; weights have
        shape ``(B, N+1, N+1)`` where index 0 is the CLS token.
        """
        x1, x2 = self._split(x)
        cls1, cls2, p1, p2 = self._encode_pair(x1, x2)

        Hp, Wp = self.patch_grid()
        B, _, D = p1.shape

        t1 = torch.cat([cls1.unsqueeze(1), p1], dim=1)
        t2 = torch.cat([cls2.unsqueeze(1), p2], dim=1)

        cross_attn: list[tuple[torch.Tensor, torch.Tensor]] = []
        for blk in self.fusion:
            t1, t2, w_1to2, w_2to1 = blk(t1, t2, return_weights=True)
            cross_attn.append((w_1to2.detach().cpu(), w_2to1.detach().cpu()))

        return {
            "pre_patches": (
                p1.reshape(B, Hp, Wp, D).detach().cpu(),
                p2.reshape(B, Hp, Wp, D).detach().cpu(),
            ),
            "post_patches": (
                t1[:, 1:, :].reshape(B, Hp, Wp, D).detach().cpu(),
                t2[:, 1:, :].reshape(B, Hp, Wp, D).detach().cpu(),
            ),
            "cross_attn": cross_attn,
            "patch_grid": (Hp, Wp),
        }

    @torch.no_grad()
    def count_trainable_params(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total, "frozen": total - trainable}


def build_dinov3_twin(
    config: ScrewConfig | None = None, *, num_classes: int | None = None
) -> DINOv3TwinClassifier:
    if config is None:
        config = ScrewConfig()
    if num_classes is not None:
        config = config.model_copy(update={"num_classes": num_classes})

    head_hidden = config.dinov3_head_hidden if config.dinov3_head_hidden > 0 else None

    return DINOv3TwinClassifier(
        num_classes=config.num_classes,
        model_name=config.dinov3_model_id,
        freeze_backbone=config.dinov3_freeze,
        head_hidden=head_hidden,
        dropout=config.dinov3_dropout,
        use_layernorm=config.dinov3_use_layernorm,
        num_register_tokens=config.dinov3_num_register_tokens,
        target_size=config.dinov3_resolution,
        fusion_depth=config.dinov3_twin_fusion_depth,
        fusion_heads=config.dinov3_twin_fusion_heads,
    )


def build_model(
    config: ScrewConfig | None = None, *, num_classes: int | None = None
) -> nn.Module:
    if config is None:
        config = ScrewConfig()
    encoder = config.encoder
    if encoder == "dinov3":
        return build_dinov3(config, num_classes=num_classes)
    if encoder == "dinov3_twin":
        return build_dinov3_twin(config, num_classes=num_classes)
    if encoder == "vit_efficient":
        return build_vit(config, num_classes=num_classes)
    raise ValueError(f"unknown encoder: {encoder}")
