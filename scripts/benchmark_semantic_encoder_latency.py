#!/usr/bin/env python
"""Benchmark bs=1 semantic encoder latency by CLIP ViT-B/16 depth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ucwc.semantic.clip_vit import load_vit_b16_encoder
from ucwc.semantic.models import preprocess_clip_tensor


def main() -> None:
    args = _parse_args()
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError(f"Requested {args.device}, but CUDA is not available.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = load_vit_b16_encoder(
        model_name=args.clip_model_name,
        download_root=args.clip_download_root,
        device=device,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    depths = args.depths or list(range(1, int(model.layers) + 1))
    image = torch.rand(args.batch_size, 3, args.image_size, args.image_size, device=device)
    rows = []

    with torch.inference_mode():
        for depth in depths:
            row = _benchmark_depth(
                model=model,
                image=image,
                depth=int(depth),
                image_size=args.image_size,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            rows.append(row)
            print(
                f"depth={depth:02d} "
                f"wall_p50_ms={row['wall_p50_ms']:.4f} "
                f"cuda_p50_ms={row['cuda_event_p50_ms']:.4f}"
            )

    metadata = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "clip_model_name": args.clip_model_name,
        "clip_download_root": args.clip_download_root,
        "model_dtype": str(model.conv1.weight.dtype),
        "depths": depths,
    }
    _write_csv(output_dir / "encoder_latency_by_depth.csv", rows)
    (output_dir / "encoder_latency_summary.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote_csv={output_dir / 'encoder_latency_by_depth.csv'}")
    print(f"wrote_summary={output_dir / 'encoder_latency_summary.json'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--depths", nargs="*", type=int)
    parser.add_argument("--clip-model-name", default="ViT-B/16")
    parser.add_argument("--clip-download-root", default="/home/users/dky/.cache/clip")
    parser.add_argument("--output-dir", default="outputs/benchmarks/encoder_latency_bs1_gpu1")
    return parser.parse_args()


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _benchmark_depth(
    *,
    model: torch.nn.Module,
    image: torch.Tensor,
    depth: int,
    image_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        _encode(model, image, depth, image_size)
    _sync(image.device)

    wall_times_ms: list[float] = []
    cuda_event_times_ms: list[float] = []
    for _ in range(iterations):
        start_event = torch.cuda.Event(enable_timing=True) if image.device.type == "cuda" else None
        end_event = torch.cuda.Event(enable_timing=True) if image.device.type == "cuda" else None
        _sync(image.device)
        wall_start = time.perf_counter()
        if start_event is not None and end_event is not None:
            start_event.record()
        _encode(model, image, depth, image_size)
        if start_event is not None and end_event is not None:
            end_event.record()
        _sync(image.device)
        wall_times_ms.append((time.perf_counter() - wall_start) * 1000.0)
        if start_event is not None and end_event is not None:
            cuda_event_times_ms.append(float(start_event.elapsed_time(end_event)))

    cuda_source = cuda_event_times_ms if cuda_event_times_ms else wall_times_ms
    return {
        "encoder_depth": depth,
        "batch_size": int(image.shape[0]),
        "wall_mean_ms": statistics.fmean(wall_times_ms),
        "wall_p50_ms": statistics.median(wall_times_ms),
        "wall_p95_ms": _quantile(wall_times_ms, 0.95),
        "cuda_event_mean_ms": statistics.fmean(cuda_source),
        "cuda_event_p50_ms": statistics.median(cuda_source),
        "cuda_event_p95_ms": _quantile(cuda_source, 0.95),
    }


def _encode(
    model: torch.nn.Module,
    image: torch.Tensor,
    depth: int,
    image_size: int,
) -> torch.Tensor:
    normalized = preprocess_clip_tensor(image, image_size)
    features = model.encode_image(normalized, encoder_depth=depth).float()
    return features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
