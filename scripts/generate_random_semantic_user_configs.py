#!/usr/bin/env python
"""Generate random UE basic configs and the semantic-link configuration grid."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ucwc.semantic.evaluation import estimate_transmission_volume


DEFAULT_SNR_DB = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
DEFAULT_QAM_ORDERS = [4, 16, 64, 256]
DEFAULT_CODE_RATES = [0.5, 2.0 / 3.0, 0.75]
DEFAULT_QUANTIZATION_BITS = [2, 4, 8]
DEFAULT_LAYERS = list(range(1, 13))
DEFAULT_TASK_TYPES = ["image_classification", "semantic_retrieval", "intent_recognition"]
DEFAULT_ENCODER_LATENCY_CSV = (
    "outputs/benchmarks/encoder_latency_bs1_gpu1/encoder_latency_by_depth.csv"
)


def main() -> int:
    args = _parse_args()
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_latency_by_depth = _load_encoder_latency_by_depth(
        _resolve_path(args.encoder_latency_csv),
        args.encoder_latency_stat,
    )
    grid_rows = _build_grid_rows(args, encoder_latency_by_depth)
    user_rows = _build_user_rows(args, grid_rows, encoder_latency_by_depth)

    grid_csv = output_dir / "semantic_link_config_grid.csv"
    user_csv = output_dir / "random_user_basic_configs.csv"
    manifest_path = output_dir / "manifest.json"
    _write_csv(grid_csv, grid_rows)
    _write_csv(user_csv, user_rows)
    manifest = {
        "output_dir": str(output_dir),
        "grid_csv": str(grid_csv),
        "user_csv": str(user_csv),
        "num_users": int(args.num_users),
        "seed": int(args.seed),
        "snr_db": [float(value) for value in args.snr_db],
        "qam_orders": [int(value) for value in args.qam_orders],
        "code_rates": [float(value) for value in args.code_rates],
        "quantization_bits": [int(value) for value in args.quantization_bits],
        "layers": [int(value) for value in args.layers],
        "feature_dim": int(args.feature_dim),
        "header_bits": int(args.header_bits),
        "fixed_latency_ms": float(args.fixed_latency_ms),
        "encoder_latency_csv": str(_resolve_path(args.encoder_latency_csv)),
        "encoder_latency_stat": str(args.encoder_latency_stat),
        "grid_row_count": len(grid_rows),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote_grid={grid_csv}")
    print(f"wrote_users={user_csv}")
    print(f"grid_rows={len(grid_rows)} users={len(user_rows)}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/configs/semantic_random_user_configs")
    parser.add_argument("--num-users", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--snr-db", nargs="+", type=float, default=DEFAULT_SNR_DB)
    parser.add_argument("--qam-orders", nargs="+", type=int, default=DEFAULT_QAM_ORDERS)
    parser.add_argument("--code-rates", nargs="+", type=float, default=DEFAULT_CODE_RATES)
    parser.add_argument(
        "--quantization-bits",
        nargs="+",
        type=int,
        default=DEFAULT_QUANTIZATION_BITS,
    )
    parser.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--header-bits", type=int, default=8)
    parser.add_argument("--fixed-latency-ms", type=float, default=1.0)
    parser.add_argument("--encoder-latency-csv", default=DEFAULT_ENCODER_LATENCY_CSV)
    parser.add_argument("--encoder-latency-stat", default="wall_p50_ms")
    parser.add_argument("--task-types", nargs="+", default=DEFAULT_TASK_TYPES)
    parser.add_argument("--min-task-score-low", type=float, default=0.10)
    parser.add_argument("--min-task-score-high", type=float, default=0.75)
    parser.add_argument("--latency-ms-low", type=float, default=5.0)
    parser.add_argument("--latency-ms-high", type=float, default=25.0)
    return parser.parse_args()


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _build_grid_rows(
    args: argparse.Namespace,
    encoder_latency_by_depth: dict[int, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in sorted({int(value) for value in args.layers}):
        for q_bits in sorted({int(value) for value in args.quantization_bits}):
            feature_payload_bits = int(args.feature_dim) * q_bits
            for rate in [float(value) for value in args.code_rates]:
                for qam_order in [int(value) for value in args.qam_orders]:
                    volume = estimate_transmission_volume(
                        feature_payload_bits=feature_payload_bits,
                        header_bits_per_sample=int(args.header_bits),
                        sample_count=1,
                        coded_bits=_coded_bits_without_header(feature_payload_bits, rate),
                        ldpc_padding_bits=(-feature_payload_bits) % _ldpc_k(rate),
                        ldpc_code_rate=rate,
                        qam_order=qam_order,
                    )
                    for snr_db in [float(value) for value in args.snr_db]:
                        rows.append(
                            {
                                "config_id": _config_id(layer, q_bits, rate, qam_order, snr_db),
                                "encoder_depth": layer,
                                "quantization_bits": q_bits,
                                "ldpc_code_rate": rate,
                                "qam_order": qam_order,
                                "snr_db": snr_db,
                                "feature_dim": int(args.feature_dim),
                                "header_bits": int(args.header_bits),
                                "feature_payload_bits": feature_payload_bits,
                                "payload_bits_with_header": int(volume["payload_bits_with_header"]),
                                "coded_bits": int(volume["coded_bits_with_header"]),
                                "qam_bits_per_symbol": int(volume["qam_bits_per_symbol"]),
                                "qam_symbols": int(volume["qam_symbols_with_header"]),
                                "payload_bytes_with_header": float(
                                    volume["payload_bytes_with_header"]
                                ),
                                "coded_bytes": float(volume["coded_bytes_with_header"]),
                                "encoding_latency_ms": float(
                                    encoder_latency_by_depth.get(layer, 0.0)
                                ),
                                "fixed_latency_ms": float(args.fixed_latency_ms),
                            }
                        )
    return rows


def _build_user_rows(
    args: argparse.Namespace,
    grid_rows: list[dict[str, Any]],
    encoder_latency_by_depth: dict[int, float],
) -> list[dict[str, Any]]:
    rng = random.Random(int(args.seed))
    rows: list[dict[str, Any]] = []
    for order in range(1, int(args.num_users) + 1):
        config = rng.choice(grid_rows)
        layer = int(config["encoder_depth"])
        rows.append(
            {
                "request_id": f"req_{order:04d}",
                "ue_id": f"ue_{order:04d}",
                "arrival_order": order,
                "task_type": rng.choice(list(args.task_types)),
                "snr_db": float(config["snr_db"]),
                "min_task_score": round(
                    rng.uniform(float(args.min_task_score_low), float(args.min_task_score_high)),
                    6,
                ),
                "max_total_latency_ms": round(
                    rng.uniform(float(args.latency_ms_low), float(args.latency_ms_high)),
                    6,
                ),
                "initial_config_id": str(config["config_id"]),
                "initial_encoder_depth": layer,
                "initial_quantization_bits": int(config["quantization_bits"]),
                "initial_ldpc_code_rate": float(config["ldpc_code_rate"]),
                "initial_qam_order": int(config["qam_order"]),
                "initial_payload_bits_with_header": int(config["payload_bits_with_header"]),
                "initial_coded_bits": int(config["coded_bits"]),
                "initial_qam_symbols": int(config["qam_symbols"]),
                "encoding_latency_ms": float(encoder_latency_by_depth.get(layer, 0.0)),
                "fixed_latency_ms": float(args.fixed_latency_ms),
                "source": "random_initial_config",
            }
        )
    return rows


def _load_encoder_latency_by_depth(path: Path, latency_stat: str) -> dict[int, float]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if latency_stat not in (reader.fieldnames or []):
            raise ValueError(f"Latency statistic {latency_stat!r} not found in {path}")
        return {
            int(row["encoder_depth"]): float(row[latency_stat])
            for row in reader
            if row.get("encoder_depth")
        }


def _coded_bits_without_header(feature_payload_bits: int, code_rate: float) -> int:
    k = _ldpc_k(code_rate)
    n = 648
    padding = (-int(feature_payload_bits)) % k
    return ((int(feature_payload_bits) + padding) // k) * n


def _ldpc_k(code_rate: float) -> int:
    return int(round(648 * float(code_rate)))


def _config_id(layer: int, q_bits: int, rate: float, qam_order: int, snr_db: float) -> str:
    return (
        f"d{layer:02d}_q{q_bits}_r{_rate_label(rate)}_"
        f"qam{int(qam_order)}_snr{snr_db:g}"
    )


def _rate_label(rate: float) -> str:
    if abs(rate - 0.5) < 1e-6:
        return "12"
    if abs(rate - 2.0 / 3.0) < 1e-3:
        return "23"
    if abs(rate - 0.75) < 1e-6:
        return "34"
    return str(rate).replace(".", "p")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
