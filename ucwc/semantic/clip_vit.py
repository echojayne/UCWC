"""Local CLIP ViT visual encoder with depth-controlled forward.

Adapted from `/home/users/dky/RKDSC/CLIP-main/clip/model.py`. This file keeps
only the visual ViT path needed by UCWC and exposes `encoder_depth` so the
semantic encoder can stop after a chosen number of transformer blocks.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import urllib.request

import torch
from torch import nn


CLIP_MODEL_URLS = {
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/"
    "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return super().forward(x.float()).to(dtype)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x, x, x, need_weights=False, attn_mask=None)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor, encoder_depth: int | None = None) -> torch.Tensor:
        depth = self.layers if encoder_depth is None else int(encoder_depth)
        if depth < 1 or depth > self.layers:
            raise ValueError(f"encoder_depth must be in [1, {self.layers}], got {depth}")
        for block in self.resblocks[:depth]:
            x = block(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.layers = layers
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    def forward(self, x: torch.Tensor, encoder_depth: int | None = None) -> torch.Tensor:
        x = x.type(self.dtype)
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, encoder_depth=encoder_depth)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        return x @ self.proj

    def forward_layers(
        self,
        x: torch.Tensor,
        encoder_depths: list[int] | tuple[int, ...],
    ) -> dict[int, torch.Tensor]:
        depths = sorted({int(depth) for depth in encoder_depths})
        if not depths:
            raise ValueError("encoder_depths must not be empty.")
        if depths[0] < 1 or depths[-1] > self.layers:
            raise ValueError(f"encoder_depths must be in [1, {self.layers}], got {depths}")

        x = x.type(self.dtype)
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)

        outputs: dict[int, torch.Tensor] = {}
        requested = set(depths)
        for index, block in enumerate(self.transformer.resblocks, start=1):
            x = block(x)
            if index in requested:
                cls_feature = self.ln_post(x[0])
                outputs[index] = cls_feature @ self.proj
            if index == depths[-1]:
                break
        return outputs

    def encode_image(self, image: torch.Tensor, encoder_depth: int | None = None) -> torch.Tensor:
        return self.forward(image, encoder_depth=encoder_depth)

    def encode_image_layers(
        self,
        image: torch.Tensor,
        encoder_depths: list[int] | tuple[int, ...],
    ) -> dict[int, torch.Tensor]:
        return self.forward_layers(image, encoder_depths=encoder_depths)


def load_vit_b16_encoder(
    *,
    model_name: str = "ViT-B/16",
    download_root: str | Path = "/home/users/dky/.cache/clip",
    device: str | torch.device = "cpu",
) -> VisionTransformer:
    checkpoint_path = resolve_clip_checkpoint(model_name, Path(download_root))
    state_dict = _load_checkpoint_state_dict(checkpoint_path)
    model = build_vit_from_state_dict(state_dict).to(device)
    if str(device) == "cpu":
        model.float()
    model.eval()
    return model


def build_vit_from_state_dict(state_dict: dict[str, torch.Tensor]) -> VisionTransformer:
    visual = {
        key.removeprefix("visual."): value
        for key, value in state_dict.items()
        if key.startswith("visual.")
    }
    if not visual:
        raise ValueError("Checkpoint does not contain visual.* CLIP weights.")
    width = int(visual["conv1.weight"].shape[0])
    patch_size = int(visual["conv1.weight"].shape[-1])
    grid = round((int(visual["positional_embedding"].shape[0]) - 1) ** 0.5)
    image_resolution = patch_size * grid
    output_dim = int(visual["proj"].shape[1])
    layers = len(
        {
            key.split(".")[2]
            for key in visual
            if key.startswith("transformer.resblocks.")
        }
    )
    heads = width // 64
    model = VisionTransformer(
        input_resolution=image_resolution,
        patch_size=patch_size,
        width=width,
        layers=layers,
        heads=heads,
        output_dim=output_dim,
    )
    model.load_state_dict(visual, strict=True)
    return model


def resolve_clip_checkpoint(model_name: str, download_root: Path) -> Path:
    if model_name not in CLIP_MODEL_URLS:
        candidate = Path(model_name).expanduser()
        if candidate.exists():
            return candidate
        raise ValueError(f"Unsupported CLIP model name: {model_name}")
    url = CLIP_MODEL_URLS[model_name]
    target = download_root.expanduser() / Path(url).name
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        target.write_bytes(response.read())
    return target


def _load_checkpoint_state_dict(path: Path) -> dict[str, torch.Tensor]:
    with path.open("rb") as handle:
        try:
            jit_model = torch.jit.load(handle, map_location="cpu").eval()
            return dict(jit_model.state_dict())
        except RuntimeError:
            handle.seek(0)
            payload = torch.load(handle, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported CLIP checkpoint payload: {type(payload)}")
    return payload
