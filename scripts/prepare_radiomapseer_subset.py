"""Prepare a small UCWC config from the RadioMapSeer zip without full extraction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
import zipfile

import yaml


UCWC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UCWC_ROOT))


QOS_CYCLE = [
    "low_latency_video",
    "reliable_control",
    "high_mobility_call",
    "high_throughput_download",
    "secure_service",
    "background_sync",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", default=str(UCWC_ROOT / "data" / "RadioMapSeer.zip"))
    parser.add_argument("--map-id", type=int, default=0)
    parser.add_argument("--method", default="DPM", choices=["DPM", "IRT2", "IRT4", "carsDPM", "carsIRT2", "carsIRT4"])
    parser.add_argument("--tx-count", type=int, default=7)
    parser.add_argument("--ue-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-gain-pixel", type=int, default=5)
    parser.add_argument("--output-config-dir", default=str(UCWC_ROOT / "configs" / "radiomapseer_real"))
    parser.add_argument("--output-data-dir", default=str(UCWC_ROOT / "data" / "radio_maps" / "radiomapseer_real"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip_path).expanduser().resolve()
    output_config_dir = Path(args.output_config_dir).expanduser().resolve()
    output_data_dir = Path(args.output_data_dir).expanduser().resolve()
    map_dir = output_data_dir / f"map_{args.map_id:03d}" / args.method
    map_dir.mkdir(parents=True, exist_ok=True)
    output_config_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        antenna_positions = json.loads(archive.read(f"antenna/{args.map_id}.json"))
        tx_ids = list(range(args.tx_count))
        extracted_maps = extract_maps(
            archive=archive,
            map_id=args.map_id,
            method=args.method,
            tx_ids=tx_ids,
            output_dir=map_dir,
        )
        candidate_points = candidate_ue_points(
            archive=archive,
            map_id=args.map_id,
            method=args.method,
            tx_ids=tx_ids,
            min_gain_pixel=args.min_gain_pixel,
        )

    rng = random.Random(args.seed)
    base_stations = build_base_station_records(
        antenna_positions=antenna_positions,
        tx_ids=tx_ids,
        ue_count=args.ue_count,
    )
    ues = build_ue_records(
        candidate_points=candidate_points,
        ue_count=args.ue_count,
        rng=rng,
    )
    static_network = {"base_stations": base_stations, "ues": ues}
    service_profiles_src = UCWC_ROOT / "configs" / "service_profiles.yaml"
    shutil.copy2(service_profiles_src, output_config_dir / "service_profiles.yaml")
    write_yaml(output_config_dir / "static_network.yaml", static_network)
    write_yaml(
        output_config_dir / "simulation.yaml",
        build_simulation_config(
            args=args,
            output_config_dir=output_config_dir,
            map_dir=map_dir,
            extracted_maps=extracted_maps,
        ),
    )
    print("radiomapseer_prepare_status=ok")
    print(f"zip_path={zip_path}")
    print(f"map_id={args.map_id}")
    print(f"method={args.method}")
    print(f"tx_count={len(tx_ids)}")
    print(f"ue_count={len(ues)}")
    print(f"output_config_dir={output_config_dir}")
    print(f"output_data_dir={map_dir}")
    return 0


def extract_maps(
    *,
    archive: zipfile.ZipFile,
    map_id: int,
    method: str,
    tx_ids: list[int],
    output_dir: Path,
) -> dict[str, Path]:
    extracted = {}
    for index, tx_id in enumerate(tx_ids, start=1):
        member = f"gain/{method}/{map_id}_{tx_id}.png"
        if member not in archive.namelist():
            raise FileNotFoundError(f"Missing RadioMapSeer member: {member}")
        bs_id = f"bs_{index:03d}"
        target = output_dir / f"{bs_id}_map{map_id:03d}_tx{tx_id:03d}_{method}_gain.png"
        with archive.open(member) as source, target.open("wb") as dest:
            shutil.copyfileobj(source, dest)
        extracted[bs_id] = target
    return extracted


def candidate_ue_points(
    *,
    archive: zipfile.ZipFile,
    map_id: int,
    method: str,
    tx_ids: list[int],
    min_gain_pixel: int,
) -> list[tuple[int, int]]:
    from PIL import Image
    from io import BytesIO

    valid: set[tuple[int, int]] = set()
    for tx_id in tx_ids:
        member = f"gain/{method}/{map_id}_{tx_id}.png"
        image = Image.open(BytesIO(archive.read(member))).convert("L")
        width, height = image.size
        pixels = list(image.getdata())
        for y in range(height):
            for x in range(width):
                if pixels[y * width + x] >= min_gain_pixel:
                    valid.add((x, y))
    if not valid:
        raise ValueError("No valid UE candidate points found in selected radio maps.")
    return sorted(valid)


def build_base_station_records(
    *,
    antenna_positions: list[list[int]],
    tx_ids: list[int],
    ue_count: int,
) -> list[dict[str, object]]:
    max_connections = max(8, int((ue_count / max(1, len(tx_ids))) * 1.7))
    records = []
    for index, tx_id in enumerate(tx_ids, start=1):
        x_m, y_m = antenna_positions[tx_id]
        records.append(
            {
                "bs_id": f"bs_{index:03d}",
                "x_m": float(x_m),
                "y_m": float(y_m),
                "z_m": 25.0,
                "max_connections": max_connections,
                "bandwidth_mhz": 40.0,
                "total_prb": 106,
                "tx_power_dbm": 43.0,
                "carrier_frequency_ghz": 3.5,
                "radiomapseer_map_id": int(tx_id),
            }
        )
    return records


def build_ue_records(
    *,
    candidate_points: list[tuple[int, int]],
    ue_count: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    if ue_count > len(candidate_points):
        points = [rng.choice(candidate_points) for _ in range(ue_count)]
    else:
        points = rng.sample(candidate_points, ue_count)
    records = []
    for index, (x_m, y_m) in enumerate(points, start=1):
        qos_profile = QOS_CYCLE[(index - 1) % len(QOS_CYCLE)]
        mobility_class, speed = mobility_for_index(index, rng)
        records.append(
            {
                "ue_id": f"ue_{index:04d}",
                "x_m": float(x_m),
                "y_m": float(y_m),
                "z_m": 1.5,
                "mobility_speed_kmh": speed,
                "mobility_class": mobility_class,
                "qos_profile": qos_profile,
                "max_tx_power_dbm": 23.0,
                "battery_level_pct": round(rng.uniform(45.0, 95.0), 2),
            }
        )
    return records


def mobility_for_index(index: int, rng: random.Random) -> tuple[str, float]:
    bucket = index % 10
    if bucket in {0, 1, 2}:
        return "static", round(rng.uniform(0.0, 1.0), 2)
    if bucket in {3, 4, 5, 6}:
        return "walking", round(rng.uniform(2.0, 6.0), 2)
    if bucket in {7, 8}:
        return "vehicle", round(rng.uniform(30.0, 75.0), 2)
    return "high_speed", round(rng.uniform(90.0, 130.0), 2)


def build_simulation_config(
    *,
    args: argparse.Namespace,
    output_config_dir: Path,
    map_dir: Path,
    extracted_maps: dict[str, Path],
) -> dict[str, object]:
    maps = []
    for bs_id, path in sorted(extracted_maps.items()):
        maps.append(
            {
                "bs_id": bs_id,
                "path": str(path.relative_to(map_dir)),
            }
        )
    return {
        "seed": args.seed,
        "timestamp_s": 0,
        "channel": {
            "model": "radiomapseer",
            "pathloss_exponent": 3.2,
            "shadowing_std_db": 2.5,
            "noise_figure_db": 7.0,
            "min_distance_m": 5.0,
            "radio_map": {
                "dataset_root": os.path.relpath(map_dir, output_config_dir),
                "value_kind": "scaled_gain",
                "pathloss_min_db": 47.0,
                "pathloss_max_db": 127.0,
                "resolution_m_per_pixel": 1.0,
                "clip_coordinates": True,
                "y_axis": "down",
                "fallback_to_synthetic": False,
                "maps": maps,
            },
        },
        "association": {
            "policy": "best_sinr_with_capacity",
            "allow_overload": False,
        },
        "verifier": {
            "min_sinr_db": -3.0,
            "max_cell_utilization": 0.92,
            "max_cross_user_degradation_pct": 8.0,
        },
    }


def write_yaml(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


if __name__ == "__main__":
    raise SystemExit(main())
