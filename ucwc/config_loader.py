"""Configuration loading for the UCWC minimal prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ucwc.components import BaseStation, Position, QoSRequirement, ResourceBudget, UE


DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return payload


def load_config_bundle(config_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    root = root.expanduser().resolve()
    bundle = {
        "config_dir": str(root),
        "static_network": load_yaml(root / "static_network.yaml"),
        "service_profiles": load_yaml(root / "service_profiles.yaml"),
        "simulation": load_yaml(root / "simulation.yaml"),
    }
    validate_config_bundle(bundle)
    return bundle


def validate_config_bundle(bundle: dict[str, Any]) -> None:
    network = bundle["static_network"]
    profiles = bundle["service_profiles"].get("qos_profiles", {})
    if not network.get("base_stations"):
        raise ValueError("static_network.yaml needs at least one base station.")
    if not network.get("ues"):
        raise ValueError("static_network.yaml needs at least one UE.")
    if not profiles:
        raise ValueError("service_profiles.yaml needs qos_profiles.")
    missing_profiles = sorted(
        {
            str(ue.get("qos_profile"))
            for ue in network.get("ues", [])
            if str(ue.get("qos_profile")) not in profiles
        }
    )
    if missing_profiles:
        raise ValueError(f"Undefined QoS profiles: {missing_profiles}")


def build_base_stations(bundle: dict[str, Any]) -> list[BaseStation]:
    base_stations = []
    for item in bundle["static_network"]["base_stations"]:
        resources = ResourceBudget(
            max_connections=int(item["max_connections"]),
            bandwidth_mhz=float(item["bandwidth_mhz"]),
            total_prb=int(item["total_prb"]),
        )
        base_stations.append(
            BaseStation(
                bs_id=str(item["bs_id"]),
                position=Position(
                    x_m=float(item["x_m"]),
                    y_m=float(item["y_m"]),
                    z_m=float(item.get("z_m", 25.0)),
                ),
                resources=resources,
                tx_power_dbm=float(item.get("tx_power_dbm", 43.0)),
                carrier_frequency_ghz=float(item.get("carrier_frequency_ghz", 3.5)),
            )
        )
    return base_stations


def build_ues(bundle: dict[str, Any]) -> list[UE]:
    profiles = bundle["service_profiles"]["qos_profiles"]
    ues = []
    for item in bundle["static_network"]["ues"]:
        profile_id = str(item["qos_profile"])
        profile = profiles[profile_id]
        qos = QoSRequirement(
            profile_id=profile_id,
            traffic_direction=str(profile["traffic_direction"]),  # type: ignore[arg-type]
            min_dl_mbps=float(profile["min_dl_mbps"]),
            min_ul_mbps=float(profile["min_ul_mbps"]),
            max_latency_ms=float(profile["max_latency_ms"]),
            min_reliability=float(profile["min_reliability"]),
            max_packet_loss=float(profile["max_packet_loss"]),
            max_jitter_ms=float(profile["max_jitter_ms"]),
            security_level=str(profile.get("security_level", "standard")),
            priority=int(profile.get("priority", 5)),
        )
        ues.append(
            UE(
                ue_id=str(item["ue_id"]),
                position=Position(
                    x_m=float(item["x_m"]),
                    y_m=float(item["y_m"]),
                    z_m=float(item.get("z_m", 1.5)),
                ),
                qos=qos,
                mobility_speed_kmh=float(item.get("mobility_speed_kmh", 0.0)),
                mobility_class=str(item.get("mobility_class", "static")),
                max_tx_power_dbm=float(item.get("max_tx_power_dbm", 23.0)),
                battery_level_pct=float(item.get("battery_level_pct", 100.0)),
            )
        )
    return ues
