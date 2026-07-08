#!/usr/bin/env python
"""Run a CLIP/RIDAS semantic LDPC/QAM/channel grid and persist results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ucwc.semantic import RIDAS_DECODER_PATH, ensure_ridas_weights, evaluate_grid, load_clip_ridas_codec
from ucwc.semantic.storage import write_experience_rows


DEFAULT_PHY_MODES = [
    {"phy_mode_id": "ldpc12_qam4", "ldpc_code_rate": 0.5, "qam_order": 4},
]
DEFAULTS = {
    "clip_model_name": "ViT-B/16",
    "clip_download_root": "/home/users/dky/.cache/clip",
    "decoder_path": str(RIDAS_DECODER_PATH),
    "normalize_features": True,
    "dataset": "synthetic",
    "dataset_root": "/home/users/dky/.cache",
    "download_dataset": False,
    "output_dir": "outputs/semantic_link_smoke",
    "encoder_depths": [3, 6, 9, 12],
    "quantization_bits": [4],
    "snr_db": [20.0],
    "channel_model": "awgn",
    "batch_size": 2,
    "seed": 11,
    "ldpc_iterations": 25,
    "phy_modes": DEFAULT_PHY_MODES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/semantic_link_smoke.yaml")
    parser.add_argument("--clip-model-name")
    parser.add_argument("--clip-download-root")
    parser.add_argument("--decoder-path")
    parser.add_argument("--dataset", choices=["synthetic", "cifar100"])
    parser.add_argument("--dataset-root")
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--encoder-depths", nargs="+", type=int)
    parser.add_argument("--quantization-bits", nargs="+", type=int)
    parser.add_argument("--snr-db", nargs="+", type=float)
    parser.add_argument("--channel-model", choices=["awgn", "rayleigh"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ldpc-iterations", type=int)
    parser.add_argument("--overwrite-weights", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolved = _resolve_config(args)
    ensure_ridas_weights(decoder_path=resolved["decoder_path"], overwrite=args.overwrite_weights)
    codec = load_clip_ridas_codec(
        clip_model_name=str(resolved["clip_model_name"]),
        clip_download_root=str(resolved["clip_download_root"]),
        decoder_path=str(resolved["decoder_path"]),
        normalize_features=bool(resolved["normalize_features"]),
    )
    rows = evaluate_grid(
        codec,
        encoder_depths=[int(item) for item in resolved["encoder_depths"]],
        quantization_bits=[int(item) for item in resolved["quantization_bits"]],
        phy_modes=list(resolved["phy_modes"]),
        snr_values_db=[float(item) for item in resolved["snr_db"]],
        channel_model=str(resolved["channel_model"]),
        batch_size=int(resolved["batch_size"]),
        seed=int(resolved["seed"]),
        ldpc_iterations=int(resolved["ldpc_iterations"]),
        dataset=str(resolved["dataset"]),
        dataset_root=str(resolved["dataset_root"]),
        download_dataset=bool(resolved["download_dataset"]),
    )
    output_dir = Path(str(resolved["output_dir"]))
    result = write_experience_rows(
        rows,
        sqlite_path=output_dir / "semantic_results.sqlite",
        csv_path=output_dir / "semantic_experience.csv",
        replace=True,
    )
    print(result)
    return 0


def _resolve_config(args: argparse.Namespace) -> dict[str, object]:
    payload: dict[str, object] = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    resolved = dict(DEFAULTS)
    resolved.update({key: value for key, value in payload.items() if key in resolved})
    cli_values = {
        "clip_model_name": args.clip_model_name,
        "clip_download_root": args.clip_download_root,
        "decoder_path": args.decoder_path,
        "dataset": args.dataset,
        "dataset_root": args.dataset_root,
        "download_dataset": True if args.download_dataset else None,
        "output_dir": args.output_dir,
        "encoder_depths": args.encoder_depths,
        "quantization_bits": args.quantization_bits,
        "snr_db": args.snr_db,
        "channel_model": args.channel_model,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "ldpc_iterations": args.ldpc_iterations,
    }
    resolved.update({key: value for key, value in cli_values.items() if value is not None})
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
