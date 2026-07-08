#!/usr/bin/env python
"""Cache CIFAR10 CLIP semantic features for every encoder depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, InterpolationMode, Resize, ToTensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ucwc.semantic.models import (  # noqa: E402
    ClipRidasCodecConfig,
    ClipRidasSemanticCodec,
    decoder_linear_shape,
    layer_decoder_checkpoint_path,
    load_semantic_decoder,
)


DEFAULT_LAYERS = list(range(1, 13))


class HFCifar10TorchDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        split: str,
        *,
        cache_dir: Path,
        image_size: int,
        limit: int = 0,
    ) -> None:
        from datasets import load_dataset

        self.dataset = load_dataset(
            "cifar10",
            split=split,
            cache_dir=str(cache_dir),
            download_mode="reuse_cache_if_exists",
        )
        self.transform = Compose(
            [
                Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                ToTensor(),
            ]
        )
        self.limit = int(limit)

    def __len__(self) -> int:
        if self.limit > 0:
            return min(self.limit, len(self.dataset))
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.dataset[int(index)]
        return self.transform(row["img"]), int(row["label"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/semantic_features/cifar10_clip_vit_b16_layers")
    parser.add_argument("--dataset-root", default="/home/users/dky/.cache")
    parser.add_argument("--clip-download-root", default="/home/users/dky/.cache/clip")
    parser.add_argument("--ckpt-dir", default=str(PROJECT_ROOT / "ucwc" / "semantic" / "ckpt"))
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--no-normalize-features", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    dataset_root = Path(args.dataset_root).expanduser()
    hf_cache_dir = dataset_root / "huggingface" / "datasets"
    if not hf_cache_dir.exists():
        hf_cache_dir = dataset_root
    ckpt_dir = Path(args.ckpt_dir).expanduser()
    layers = _normalize_layers(args.layers)
    device = _resolve_device(args.device)
    normalize_features = not bool(args.no_normalize_features)

    decoders = {
        layer: load_semantic_decoder(layer_decoder_checkpoint_path(layer, ckpt_dir), device=device)
        for layer in layers
    }
    shapes = {layer: decoder_linear_shape(decoder) for layer, decoder in decoders.items()}
    feature_dims = {shape[0] for shape in shapes.values()}
    class_counts = {shape[1] for shape in shapes.values()}
    if len(feature_dims) != 1 or len(class_counts) != 1:
        raise ValueError(f"Layer decoder shapes are inconsistent: {shapes}")
    feature_dim = int(next(iter(feature_dims)))
    num_classes = int(next(iter(class_counts)))

    config = ClipRidasCodecConfig(
        clip_download_root=str(Path(args.clip_download_root).expanduser()),
        decoder_path=str(layer_decoder_checkpoint_path(layers[0], ckpt_dir)),
        image_size=int(args.image_size),
        feature_dim=feature_dim,
        max_depth=max(layers),
        num_classes=num_classes,
        normalize_features=normalize_features,
    )
    codec = ClipRidasSemanticCodec(config, device=device)
    codec.eval()
    for parameter in codec.parameters():
        parameter.requires_grad_(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "dataset": "cifar10",
        "clip_model_name": config.clip_model_name,
        "clip_download_root": str(config.clip_download_root),
        "ckpt_dir": str(ckpt_dir),
        "layers": layers,
        "image_size": int(args.image_size),
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "normalize_features": normalize_features,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "limit_per_split": int(args.limit_per_split),
        "splits": {},
        "created_unix_time": time.time(),
    }

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    for split in args.splits:
        split_info = cache_split(
            split,
            output_dir=output_dir,
            hf_cache_dir=hf_cache_dir,
            codec=codec,
            decoders=decoders,
            layers=layers,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            image_size=int(args.image_size),
            feature_dim=feature_dim,
            num_classes=num_classes,
            device=device,
            overwrite=bool(args.overwrite),
            limit_per_split=int(args.limit_per_split),
        )
        manifest["splits"][split] = split_info
        _write_manifest(output_dir / "manifest.json", manifest)

    _write_manifest(output_dir / "manifest.json", manifest)
    print(json.dumps({"feature_dir": str(output_dir), "manifest": str(output_dir / "manifest.json")}, indent=2))
    return 0


def cache_split(
    split: str,
    *,
    output_dir: Path,
    hf_cache_dir: Path,
    codec: ClipRidasSemanticCodec,
    decoders: dict[int, torch.nn.Module],
    layers: list[int],
    batch_size: int,
    num_workers: int,
    image_size: int,
    feature_dim: int,
    num_classes: int,
    device: torch.device,
    overwrite: bool,
    limit_per_split: int,
) -> dict[str, object]:
    dataset = HFCifar10TorchDataset(
        split,
        cache_dir=hf_cache_dir,
        image_size=image_size,
        limit=limit_per_split,
    )
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    paths = _target_paths(split_dir, layers)
    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"{split_dir} already contains cached files; rerun with --overwrite.")

    labels = np.empty(len(dataset), dtype=np.int64)
    feature_maps = {
        layer: np.lib.format.open_memmap(
            split_dir / f"features_layer{layer:02d}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(dataset), feature_dim),
        )
        for layer in layers
    }
    logit_maps = {
        layer: np.lib.format.open_memmap(
            split_dir / f"logits_layer{layer:02d}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(dataset), num_classes),
        )
        for layer in layers
    }
    pred_maps = {
        layer: np.lib.format.open_memmap(
            split_dir / f"preds_layer{layer:02d}.npy",
            mode="w+",
            dtype=np.int64,
            shape=(len(dataset),),
        )
        for layer in layers
    }

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
    )
    offset = 0
    start_time = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (images, batch_labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            batch_labels_np = batch_labels.numpy().astype(np.int64)
            stop = offset + int(images.shape[0])
            labels[offset:stop] = batch_labels_np
            features_by_layer = codec.encode_layers(images, layers)
            for layer in layers:
                features = features_by_layer[layer].float()
                logits = decoders[layer](features).float()
                feature_maps[layer][offset:stop] = features.detach().cpu().numpy().astype(np.float32)
                logits_np = logits.detach().cpu().numpy().astype(np.float32)
                logit_maps[layer][offset:stop] = logits_np
                pred_maps[layer][offset:stop] = np.argmax(logits_np, axis=1).astype(np.int64)
            offset = stop
            if batch_index == 1 or batch_index % 25 == 0 or offset == len(dataset):
                elapsed = time.perf_counter() - start_time
                print(f"{split}: cached {offset}/{len(dataset)} samples in {elapsed:.1f}s", flush=True)

    np.save(split_dir / "labels.npy", labels)
    for mmap in [*feature_maps.values(), *logit_maps.values(), *pred_maps.values()]:
        mmap.flush()

    return {
        "count": len(dataset),
        "label_file": str(split_dir / "labels.npy"),
        "feature_files": {
            str(layer): str(split_dir / f"features_layer{layer:02d}.npy") for layer in layers
        },
        "logit_files": {
            str(layer): str(split_dir / f"logits_layer{layer:02d}.npy") for layer in layers
        },
        "pred_files": {
            str(layer): str(split_dir / f"preds_layer{layer:02d}.npy") for layer in layers
        },
    }


def _target_paths(split_dir: Path, layers: Iterable[int]) -> dict[str, Path]:
    paths = {"labels": split_dir / "labels.npy"}
    for layer in layers:
        paths[f"features_{layer}"] = split_dir / f"features_layer{layer:02d}.npy"
        paths[f"logits_{layer}"] = split_dir / f"logits_layer{layer:02d}.npy"
        paths[f"preds_{layer}"] = split_dir / f"preds_layer{layer:02d}.npy"
    return paths


def _normalize_layers(layers: list[int]) -> list[int]:
    normalized = sorted({int(layer) for layer in layers})
    if not normalized or normalized[0] < 1 or normalized[-1] > 12:
        raise ValueError(f"layers must be in [1, 12], got {layers}")
    return normalized


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested {device}, but CUDA is not available in this process; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device)


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
