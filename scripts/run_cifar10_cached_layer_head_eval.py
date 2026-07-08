#!/usr/bin/env python
"""Evaluate CIFAR10 per-layer linear heads on cached ViT-B/16 features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ucwc.semantic.models import decoder_linear_shape, load_semantic_decoder  # noqa: E402


DEFAULT_FEATURE_DIR = PROJECT_ROOT / "data" / "cifar10_vitb16_layer_features"
DEFAULT_HEAD_DIR = PROJECT_ROOT / "ucwc" / "semantic" / "cifar10_vitb16_layer_heads"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default=str(DEFAULT_FEATURE_DIR))
    parser.add_argument("--head-dir", default=str(DEFAULT_HEAD_DIR))
    parser.add_argument("--output-dir", default="outputs/cifar10_vitb16_layer_head_eval")
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_dir = _resolve_path(args.feature_dir)
    head_dir = _resolve_path(args.head_dir)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = sorted({int(layer) for layer in args.layers})
    if not layers or layers[0] < 1 or layers[-1] > 12:
        raise ValueError(f"layers must be in [1, 12], got {args.layers}")

    device = _resolve_device(str(args.device))
    payload_path = feature_dir / f"cifar10_{args.split}_vitb16_all_layers_features.pt"
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    features_all = payload["features"].float()
    labels = payload["labels"].long()
    normalize = not bool(args.no_normalize)

    rows: list[dict[str, int | float | str | bool]] = []
    start_time = time.perf_counter()
    for layer in layers:
        head_path = head_dir / f"cifar10_vitb16_layer{layer:02d}_linear_head.pt"
        head = load_semantic_decoder(head_path, device=device)
        input_dim, output_dim = decoder_linear_shape(head)
        features = features_all[:, layer - 1, :].float()
        if normalize:
            features = F.normalize(features, dim=-1, eps=1e-12)
        row = _evaluate_layer(
            head=head,
            features=features,
            labels=labels,
            layer=layer,
            head_path=head_path,
            input_dim=input_dim,
            output_dim=output_dim,
            normalize=normalize,
            device=device,
            batch_size=int(args.batch_size),
        )
        rows.append(row)
        print(
            f"layer={layer:02d} acc={row['accuracy']:.4f} "
            f"top5={row['top5_accuracy']:.4f} loss={row['cross_entropy']:.4f}",
            flush=True,
        )

    elapsed_seconds = time.perf_counter() - start_time
    suffix = "norm" if normalize else "raw"
    csv_path = output_dir / f"cifar10_{args.split}_layer_head_eval_{suffix}.csv"
    json_path = output_dir / f"cifar10_{args.split}_layer_head_eval_{suffix}.json"
    _write_csv(rows, csv_path)
    json_path.write_text(
        json.dumps(
            {
                "feature_path": str(payload_path),
                "head_dir": str(head_dir),
                "split": str(args.split),
                "sample_count": int(labels.numel()),
                "normalized_features": normalize,
                "normalize_call": "torch.nn.functional.normalize(features, dim=-1)" if normalize else "",
                "elapsed_seconds": elapsed_seconds,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"elapsed_seconds={elapsed_seconds:.2f}")
    return 0


def _evaluate_layer(
    *,
    head: torch.nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    layer: int,
    head_path: Path,
    input_dim: int,
    output_dim: int,
    normalize: bool,
    device: torch.device,
    batch_size: int,
) -> dict[str, int | float | str | bool]:
    correct = 0
    top5_correct = 0
    total_loss = 0.0
    finite_logits = True
    logit_abs_sum = 0.0
    logit_count = 0
    norm_sum = 0.0
    samples = int(labels.numel())
    head.eval()
    with torch.inference_mode():
        for start in range(0, samples, batch_size):
            stop = min(start + batch_size, samples)
            batch = features[start:stop].to(device)
            target = labels[start:stop].to(device)
            logits = head(batch).float()
            finite_logits = finite_logits and bool(torch.isfinite(logits).all().item())
            total_loss += float(F.cross_entropy(logits, target, reduction="sum").item())
            pred = logits.argmax(dim=1)
            correct += int((pred == target).sum().item())
            k = min(5, int(logits.shape[1]))
            topk = logits.topk(k=k, dim=1).indices
            top5_correct += int((topk == target[:, None]).any(dim=1).sum().item())
            norm_sum += float(batch.norm(dim=1).sum().item())
            logit_abs_sum += float(logits.abs().sum().item())
            logit_count += int(logits.numel())
    return {
        "layer": int(layer),
        "head_path": str(head_path),
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "samples": samples,
        "correct": int(correct),
        "accuracy": float(correct / samples),
        "top5_correct": int(top5_correct),
        "top5_accuracy": float(top5_correct / samples),
        "cross_entropy": float(total_loss / samples),
        "finite_logits": bool(finite_logits),
        "normalized_features": bool(normalize),
        "feature_norm_mean": float(norm_sum / samples),
        "logit_abs_mean": float(logit_abs_sum / max(1, logit_count)),
    }


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _write_csv(rows: list[dict[str, int | float | str | bool]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
