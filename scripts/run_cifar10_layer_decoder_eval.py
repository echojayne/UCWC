#!/usr/bin/env python
"""Directly evaluate per-layer semantic decoders on CIFAR10 test images.

This script intentionally bypasses the semantic communication link. It only
runs CLIP ViT-B/16 image encoding at each requested depth and applies the
matching decoder checkpoint from ucwc/semantic/ckpt.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Compose, Resize, ToTensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ucwc.semantic.clip_vit import load_vit_b16_encoder
from ucwc.semantic.models import (
    LAYER_DECODER_DIR,
    decoder_linear_shape,
    layer_decoder_checkpoint_path,
    load_semantic_decoder,
    preprocess_clip_tensor,
)


class HFCIFAR10Test(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, *, dataset_root: str | Path, image_size: int) -> None:
        from datasets import load_dataset

        root = Path(dataset_root).expanduser()
        cache_dir = root / "huggingface" / "datasets"
        if not cache_dir.exists():
            cache_dir = root
        self.dataset = load_dataset(
            "cifar10",
            split="test",
            cache_dir=str(cache_dir),
            download_mode="reuse_cache_if_exists",
        )
        self.transform = Compose([Resize((image_size, image_size)), ToTensor()])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.dataset[int(index)]
        return self.transform(row["img"]), int(row["label"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", default=str(LAYER_DECODER_DIR))
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--clip-model-name", default="ViT-B/16")
    parser.add_argument("--clip-download-root", default="/home/users/dky/.cache/clip")
    parser.add_argument("--dataset-root", default="/home/users/dky/.cache")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 means all CIFAR10 test samples.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="outputs/semantic_decoder_direct_cifar10")
    parser.add_argument("--no-normalize-features", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layers = sorted({int(layer) for layer in args.layers})
    if not layers:
        raise ValueError("--layers must not be empty.")

    device_name = args.device
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset: Dataset[tuple[torch.Tensor, int]] = HFCIFAR10Test(
        dataset_root=args.dataset_root,
        image_size=args.image_size,
    )
    if args.limit and args.limit > 0:
        dataset = Subset(dataset, list(range(min(int(args.limit), len(dataset)))))

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )

    encoder = load_vit_b16_encoder(
        model_name=str(args.clip_model_name),
        download_root=str(args.clip_download_root),
        device=device,
    )
    encoder.eval()

    decoders: dict[int, torch.nn.Module] = {}
    rows: dict[int, dict[str, Any]] = {}
    for layer in layers:
        decoder_path = layer_decoder_checkpoint_path(layer, args.ckpt_dir)
        decoder = load_semantic_decoder(decoder_path, device=device)
        input_dim, output_dim = decoder_linear_shape(decoder)
        decoders[layer] = decoder
        rows[layer] = {
            "layer": layer,
            "decoder_path": str(decoder_path),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "samples": 0,
            "correct": 0,
            "accuracy": "",
            "finite_logits": True,
            "can_use_for_cifar10": input_dim == 512 and output_dim == 10,
            "normalized_features": not bool(args.no_normalize_features),
            "feature_norm_mean": 0.0,
            "logit_abs_mean": 0.0,
            "_feature_norm_sum": 0.0,
            "_logit_abs_sum": 0.0,
            "_logit_count": 0,
        }

    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = preprocess_clip_tensor(images, int(args.image_size))
            features_by_layer = encoder.encode_image_layers(images, layers)

            for layer in layers:
                features = features_by_layer[layer].float()
                if not args.no_normalize_features:
                    features = F.normalize(features, dim=1, eps=1e-12)
                logits = decoders[layer](features)
                row = rows[layer]
                row["samples"] += int(labels.numel())
                row["finite_logits"] = bool(row["finite_logits"] and torch.isfinite(logits).all().item())
                row["_feature_norm_sum"] += float(features.norm(dim=1).sum().item())
                row["_logit_abs_sum"] += float(logits.abs().sum().item())
                row["_logit_count"] += int(logits.numel())
                if int(logits.shape[1]) == 10:
                    row["correct"] += int((logits.argmax(dim=1) == labels).sum().item())

            if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(loader):
                processed = min(batch_index * int(args.batch_size), len(dataset))
                print(f"processed={processed}/{len(dataset)} batches={batch_index}/{len(loader)}")

    elapsed_seconds = time.perf_counter() - start_time
    final_rows = []
    for layer in layers:
        row = rows[layer]
        if row["samples"] and row["output_dim"] == 10:
            row["accuracy"] = row["correct"] / row["samples"]
        if row["samples"]:
            row["feature_norm_mean"] = row["_feature_norm_sum"] / row["samples"]
        if row["_logit_count"]:
            row["logit_abs_mean"] = row["_logit_abs_sum"] / row["_logit_count"]
        for private_key in ["_feature_norm_sum", "_logit_abs_sum", "_logit_count"]:
            del row[private_key]
        final_rows.append(row)

    csv_path = output_dir / "cifar10_layer_decoder_eval.csv"
    json_path = output_dir / "cifar10_layer_decoder_eval.json"
    fieldnames = [
        "layer",
        "decoder_path",
        "input_dim",
        "output_dim",
        "samples",
        "correct",
        "accuracy",
        "finite_logits",
        "can_use_for_cifar10",
        "normalized_features",
        "feature_norm_mean",
        "logit_abs_mean",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)
    json_path.write_text(
        json.dumps(
            {
                "dataset": "cifar10-test",
                "sample_count": len(dataset),
                "elapsed_seconds": elapsed_seconds,
                "rows": final_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    for row in final_rows:
        accuracy = row["accuracy"]
        accuracy_text = "" if accuracy == "" else f"{float(accuracy):.4f}"
        print(
            f"layer={row['layer']:02d} shape={row['output_dim']}x{row['input_dim']} "
            f"finite={row['finite_logits']} accuracy={accuracy_text}"
        )
    print(f"elapsed_seconds={elapsed_seconds:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
