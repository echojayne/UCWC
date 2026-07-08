#!/usr/bin/env python
"""Generate fixed semantic UCWC UE queue and system-state tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ucwc.state.rule_oracle import predict_task_score, required_bandwidth_mhz
from ucwc.state.semantic_state import metadata_rows, write_csv_tables, write_state_database


DEFAULT_SEMANTIC_PRESETS = [
    {
        "encoder_depth": 4,
        "quantization_bits": 4,
        "encoding_latency_ms": 0.20,
        "decoding_latency_ms": 0.18,
    },
    {
        "encoder_depth": 8,
        "quantization_bits": 6,
        "encoding_latency_ms": 0.28,
        "decoding_latency_ms": 0.26,
    },
    {
        "encoder_depth": 12,
        "quantization_bits": 8,
        "encoding_latency_ms": 0.38,
        "decoding_latency_ms": 0.34,
    },
]

DEFAULT_TASK_SCORE_MIX = [(0.66, 0.25), (0.74, 0.40), (0.82, 0.35)]
DEFAULT_TASK_TYPES = ["image_classification", "semantic_retrieval", "intent_recognition"]
DEFAULT_ENCODER_LATENCY_CSV = (
    "outputs/benchmarks/encoder_latency_bs1_gpu1/encoder_latency_by_depth.csv"
)


def main() -> None:
    args = _parse_args()
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = _load_yaml(_resolve_path(args.scenario_config))
    map_root = _resolve_path(scenario.get("radio", {}).get("map_root", args.map_root))
    bandwidth_mhz_per_bs = float(
        args.bandwidth_mhz_per_bs
        if args.bandwidth_mhz_per_bs is not None
        else scenario.get("base_stations", {}).get("bandwidth_mhz_per_bs", 100.0)
    )

    bs_maps = _load_radio_maps(map_root, args.num_bs, args.area_size_m)
    encoder_latency_by_depth = _load_encoder_latency_by_depth(
        _resolve_path(args.encoder_latency_csv),
        args.encoder_latency_stat,
    )
    semantic_rows = _build_semantic_catalog(
        args.feature_dim,
        args.header_bits,
        args.fixed_latency_ms,
        encoder_latency_by_depth,
    )
    phy_rows = _load_phy_catalog(_resolve_path(args.phy_catalog))

    bs_rows = [
        {
            "bs_id": item["bs_id"],
            "map_file": item["map_file"],
            "x_m": item["x_m"],
            "y_m": item["y_m"],
            "bandwidth_budget_mhz": bandwidth_mhz_per_bs,
            "used_bandwidth_mhz": 0.0,
        }
        for item in bs_maps
    ]

    ue_rows, radio_rows, generation_stats = _generate_ue_and_radio_rows(
        bs_maps=bs_maps,
        semantic_rows=semantic_rows,
        phy_rows=phy_rows,
        task_types=args.task_types,
        num_users=args.num_users,
        area_size_m=args.area_size_m,
        seed=args.seed,
        snr_min_db=args.snr_min_db,
        snr_span_db=args.snr_span_db,
        deadline_scale=args.deadline_scale,
        resource_efficiency=args.resource_efficiency,
        balance_best_bs=not args.no_balance_best_bs,
        best_link_only=args.best_link_only and not args.allow_multi_link,
        non_best_sinr_penalty_db=args.non_best_sinr_penalty_db,
    )

    metadata = {
        "scenario_id": "semantic_5bs_120ue_radiomap_queue",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_scenario_config": str(_resolve_path(args.scenario_config)),
        "source_phy_catalog": str(_resolve_path(args.phy_catalog)),
        "map_root": str(map_root),
        "area_size_m": args.area_size_m,
        "num_base_stations": args.num_bs,
        "num_users": args.num_users,
        "bandwidth_mhz_per_bs": bandwidth_mhz_per_bs,
        "feature_dim": args.feature_dim,
        "header_bits": args.header_bits,
        "payload_formula": "feature_dim * quantization_bits + header_bits",
        "task_score_mix": DEFAULT_TASK_SCORE_MIX,
        "semantic_link_rule": (
            "predicted_task_score=f(layer,quant_bits,rate,qam_order,snr_db,task_type)"
        ),
        "encoder_latency_csv": str(_resolve_path(args.encoder_latency_csv)),
        "encoder_latency_stat": args.encoder_latency_stat,
        "fixed_latency_ms": args.fixed_latency_ms,
        "latency_formula": "encoding_latency_ms + transmission_latency_ms + fixed_latency_ms",
        "resource_efficiency": args.resource_efficiency,
        "seed": args.seed,
        "snr_min_db": args.snr_min_db,
        "snr_span_db": args.snr_span_db,
        "deadline_scale": args.deadline_scale,
        "best_link_only": args.best_link_only and not args.allow_multi_link,
        "non_best_sinr_penalty_db": args.non_best_sinr_penalty_db,
        "balance_best_bs": not args.no_balance_best_bs,
        "generation_stats": generation_stats,
    }

    tables = {
        "scenario_metadata": metadata_rows(metadata),
        "base_station_state": bs_rows,
        "ue_request_queue": ue_rows,
        "radio_link_state": radio_rows,
        "semantic_config_catalog": semantic_rows,
        "phy_mode_catalog": phy_rows,
        "active_session": [],
        "config_history": [],
    }
    write_state_database(output_dir / "state.sqlite", tables, overwrite=not args.no_overwrite)
    write_csv_tables(output_dir / "tables", tables)
    (output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"wrote_state_sqlite={output_dir / 'state.sqlite'}")
    print(f"wrote_tables_dir={output_dir / 'tables'}")
    print(f"base_stations={len(bs_rows)} users={len(ue_rows)} radio_links={len(radio_rows)}")
    print(f"best_bs_counts={generation_stats['best_bs_counts']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-config", default="configs/scenarios/semantic_radiomapseer.yaml")
    parser.add_argument("--phy-catalog", default="configs/catalogs/phy_modes.yaml")
    parser.add_argument("--map-root", default="data/radio_maps/radiomapseer_real/map_000/DPM")
    parser.add_argument("--output-dir", default="outputs/state/semantic_5bs_120ue")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-bs", type=int, default=5)
    parser.add_argument("--num-users", type=int, default=120)
    parser.add_argument("--area-size-m", type=float, default=255.0)
    parser.add_argument("--bandwidth-mhz-per-bs", type=float)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--header-bits", type=int, default=8)
    parser.add_argument("--encoder-latency-csv", default=DEFAULT_ENCODER_LATENCY_CSV)
    parser.add_argument("--encoder-latency-stat", default="wall_p50_ms")
    parser.add_argument("--fixed-latency-ms", type=float, default=1.0)
    parser.add_argument("--deadline-scale", type=float, default=1.30)
    parser.add_argument("--resource-efficiency", type=float, default=0.85)
    parser.add_argument("--snr-min-db", type=float, default=-4.0)
    parser.add_argument("--snr-span-db", type=float, default=24.0)
    parser.add_argument("--allow-multi-link", action="store_true")
    parser.add_argument("--best-link-only", action="store_true")
    parser.add_argument("--non-best-sinr-penalty-db", type=float, default=32.0)
    parser.add_argument("--no-balance-best-bs", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--task-types", nargs="+", default=DEFAULT_TASK_TYPES)
    return parser.parse_args()


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _load_radio_maps(map_root: Path, num_bs: int, area_size_m: float) -> list[dict[str, Any]]:
    paths = sorted(map_root.glob("*DPM_gain.png"))
    if len(paths) < num_bs:
        raise FileNotFoundError(f"Need {num_bs} DPM gain maps under {map_root}, found {len(paths)}")

    radio_maps: list[dict[str, Any]] = []
    for index, path in enumerate(paths[:num_bs], start=1):
        array = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
        y_pixel, x_pixel = np.unravel_index(int(np.argmax(array)), array.shape)
        height, width = array.shape
        radio_maps.append(
            {
                "bs_id": f"bs_{index:03d}",
                "map_file": str(path),
                "array": array,
                "width": width,
                "height": height,
                "x_m": round(_pixel_to_coord(x_pixel, width, area_size_m), 6),
                "y_m": round(_pixel_to_coord(y_pixel, height, area_size_m), 6),
            }
        )
    return radio_maps


def _build_semantic_catalog(
    feature_dim: int,
    header_bits: int,
    fixed_latency_ms: float,
    encoder_latency_by_depth: dict[int, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode_id, preset in enumerate(DEFAULT_SEMANTIC_PRESETS):
        quantization_bits = int(preset["quantization_bits"])
        payload_bits = int(feature_dim * quantization_bits + header_bits)
        encoder_depth = int(preset["encoder_depth"])
        encoding_latency_ms = encoder_latency_by_depth.get(
            encoder_depth,
            float(preset["encoding_latency_ms"]),
        )
        rows.append(
            {
                "config_id": f"sem_d{encoder_depth:02d}_q{quantization_bits}",
                "mode_id": mode_id,
                "encoder_depth": encoder_depth,
                "quantization_bits": quantization_bits,
                "feature_dim": feature_dim,
                "header_bits": header_bits,
                "payload_bits": payload_bits,
                "encoding_latency_ms": float(encoding_latency_ms),
                "decoding_latency_ms": 0.0,
                "fixed_latency_ms": float(fixed_latency_ms),
            }
        )
    return rows


def _load_encoder_latency_by_depth(path: Path, latency_stat: str) -> dict[int, float]:
    if not path.exists():
        print(f"warning=encoder_latency_csv_missing path={path} using preset latencies")
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


def _load_phy_catalog(path: Path) -> list[dict[str, Any]]:
    payload = _load_yaml(path)
    modes = payload.get("modes", [])
    if not isinstance(modes, list) or not modes:
        raise ValueError(f"No PHY modes found in {path}")

    rows: list[dict[str, Any]] = []
    for mode in modes:
        qam_order = int(mode["qam_order"])
        ldpc_code_rate = float(mode["ldpc_code_rate"])
        spectral_efficiency = float(
            mode.get("spectral_efficiency_bps_hz", ldpc_code_rate * math.log2(qam_order))
        )
        rows.append(
            {
                "phy_mode_id": str(mode["phy_mode_id"]),
                "ldpc_code_rate": ldpc_code_rate,
                "qam_order": qam_order,
                "reference_snr_db": float(
                    mode.get("reference_snr_db", mode.get("min_snr_db", 0.0))
                ),
                "spectral_efficiency_bps_hz": spectral_efficiency,
            }
        )
    return rows


def _generate_ue_and_radio_rows(
    *,
    bs_maps: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    phy_rows: list[dict[str, Any]],
    task_types: list[str],
    num_users: int,
    area_size_m: float,
    seed: int,
    snr_min_db: float,
    snr_span_db: float,
    deadline_scale: float,
    resource_efficiency: float,
    balance_best_bs: bool,
    best_link_only: bool,
    non_best_sinr_penalty_db: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    target_counts = _balanced_targets(num_users, len(bs_maps))
    best_bs_counts = [0 for _ in bs_maps]
    ue_rows: list[dict[str, Any]] = []
    radio_rows: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(10000, num_users * 2000)

    while len(ue_rows) < num_users:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not sample {num_users} covered UEs after {max_attempts} attempts. "
                "Try --no-balance-best-bs or a wider SNR span."
            )

        x_m = rng.randint(0, int(round(area_size_m)))
        y_m = rng.randint(0, int(round(area_size_m)))
        link_measurements = [
            _sample_link(item, x_m, y_m, area_size_m, snr_min_db, snr_span_db)
            for item in bs_maps
        ]
        best_index = max(
            range(len(link_measurements)),
            key=lambda idx: link_measurements[idx]["snr_db"],
        )
        if balance_best_bs and best_bs_counts[best_index] >= target_counts[best_index]:
            continue

        task_type = rng.choice(task_types)
        min_task_score = _draw_min_task_score(rng)
        reference_candidate = _best_single_user_reference_candidate(
            task_type=task_type,
            min_task_score=min_task_score,
            link_measurements=link_measurements,
            semantic_rows=semantic_rows,
            phy_rows=phy_rows,
            resource_efficiency=resource_efficiency,
        )
        if reference_candidate is None:
            continue

        transmission_budget_ms = (0.42 + rng.uniform(-0.08, 0.08)) * deadline_scale
        max_total_latency_ms = (
            float(reference_candidate["non_tx_latency_ms"])
            + transmission_budget_ms
        )
        arrival_order = len(ue_rows) + 1
        ue_id = f"ue_{arrival_order:04d}"
        ue_rows.append(
            {
                "request_id": f"req_{arrival_order:04d}",
                "ue_id": ue_id,
                "arrival_order": arrival_order,
                "x_m": float(x_m),
                "y_m": float(y_m),
                "task_type": task_type,
                "min_task_score": round(min_task_score, 6),
                "max_total_latency_ms": round(max_total_latency_ms, 6),
            }
        )
        best_bs_counts[best_index] += 1

        rank_by_bs = {
            idx: rank
            for rank, idx in enumerate(
                sorted(
                    range(len(link_measurements)),
                    key=lambda item_index: link_measurements[item_index]["snr_db"],
                    reverse=True,
                ),
                start=1,
            )
        }
        for bs_index, link in enumerate(link_measurements):
            sinr_db = float(link["snr_db"])
            if best_link_only and bs_index != best_index:
                sinr_db -= non_best_sinr_penalty_db
            radio_rows.append(
                {
                    "ue_id": ue_id,
                    "bs_id": bs_maps[bs_index]["bs_id"],
                    "snr_db": round(float(link["snr_db"]), 6),
                    "sinr_db": round(sinr_db, 6),
                    "radio_rank": rank_by_bs[bs_index],
                    "radio_gain_raw": round(float(link["radio_gain_raw"]), 6),
                    "radio_gain_norm": round(float(link["radio_gain_norm"]), 9),
                    "source_map_file": bs_maps[bs_index]["map_file"],
                }
            )

    generation_stats = {
        "sampling_attempts": attempts,
        "best_bs_counts": {
            bs_maps[index]["bs_id"]: best_bs_counts[index] for index in range(len(bs_maps))
        },
        "target_best_bs_counts": {
            bs_maps[index]["bs_id"]: target_counts[index] for index in range(len(bs_maps))
        },
        "best_link_only": best_link_only,
    }
    return ue_rows, radio_rows, generation_stats


def _balanced_targets(total: int, buckets: int) -> list[int]:
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def _sample_link(
    radio_map: dict[str, Any],
    x_m: float,
    y_m: float,
    area_size_m: float,
    snr_min_db: float,
    snr_span_db: float,
) -> dict[str, float]:
    x_pixel = _coord_to_pixel(x_m, int(radio_map["width"]), area_size_m)
    y_pixel = _coord_to_pixel(y_m, int(radio_map["height"]), area_size_m)
    gain_raw = float(radio_map["array"][y_pixel, x_pixel])
    gain_norm = gain_raw / 255.0
    return {
        "radio_gain_raw": gain_raw,
        "radio_gain_norm": gain_norm,
        "snr_db": snr_min_db + gain_norm * snr_span_db,
    }


def _coord_to_pixel(coord_m: float, size: int, area_size_m: float) -> int:
    value = int(round(float(coord_m) / area_size_m * float(size - 1)))
    return min(size - 1, max(0, value))


def _pixel_to_coord(pixel: int, size: int, area_size_m: float) -> float:
    return float(pixel) / float(size - 1) * area_size_m


def _best_single_user_reference_candidate(
    *,
    task_type: str,
    min_task_score: float,
    link_measurements: list[dict[str, float]],
    semantic_rows: list[dict[str, Any]],
    phy_rows: list[dict[str, Any]],
    resource_efficiency: float,
) -> dict[str, float] | None:
    best: dict[str, float] | None = None
    tx_budget_ms = 1.0
    for semantic in semantic_rows:
        non_tx_latency_ms = (
            float(semantic["encoding_latency_ms"])
            + float(semantic["decoding_latency_ms"])
            + float(semantic["fixed_latency_ms"])
        )
        for link in link_measurements:
            for phy in phy_rows:
                predicted = predict_task_score(
                    encoder_depth=int(semantic["encoder_depth"]),
                    quantization_bits=int(semantic["quantization_bits"]),
                    ldpc_code_rate=float(phy["ldpc_code_rate"]),
                    qam_order=int(phy["qam_order"]),
                    snr_db=float(link["snr_db"]),
                    task_type=task_type,
                    reference_snr_db=float(phy["reference_snr_db"]),
                )
                if predicted < min_task_score:
                    continue
                bandwidth_mhz = required_bandwidth_mhz(
                    payload_bits=float(semantic["payload_bits"]),
                    tx_budget_ms=tx_budget_ms,
                    ldpc_code_rate=float(phy["ldpc_code_rate"]),
                    qam_order=int(phy["qam_order"]),
                    resource_efficiency=resource_efficiency,
                )
                score = bandwidth_mhz + 0.01 * non_tx_latency_ms
                if best is None or score < best["score"]:
                    best = {
                        "score": score,
                        "non_tx_latency_ms": non_tx_latency_ms,
                    }
    return best


def _draw_min_task_score(rng: random.Random) -> float:
    draw = rng.random()
    cumulative = 0.0
    for score, probability in DEFAULT_TASK_SCORE_MIX:
        cumulative += probability
        if draw <= cumulative:
            return score
    return DEFAULT_TASK_SCORE_MIX[-1][0]


if __name__ == "__main__":
    main()
