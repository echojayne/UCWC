"""End-to-end scenario builder for the UCWC minimal prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ucwc.association import ConnectionDecision, associate_ues
from ucwc.channel_model import RadioLink, build_channel_model
from ucwc.components import BaseStation, UE
from ucwc.config_loader import build_base_stations, build_ues, load_config_bundle
from ucwc.table_manager import build_state_tables, write_json, write_sqlite_database, write_tables


@dataclass
class Scenario:
    metadata: dict[str, Any]
    base_stations: list[BaseStation]
    ues: list[UE]
    radio_links: list[RadioLink]
    connections: list[ConnectionDecision]
    tables: dict[str, list[dict[str, Any]]]

    def to_payload(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "base_stations": [bs.to_record() for bs in self.base_stations],
            "ues": [ue.to_record() for ue in self.ues],
            "radio_links": [link.to_record() for link in self.radio_links],
            "connections": [decision.to_record() for decision in self.connections],
            "tables": self.tables,
        }


def build_scenario(config_dir: str | Path | None = None) -> Scenario:
    bundle = load_config_bundle(config_dir)
    simulation = bundle["simulation"]
    seed = int(simulation.get("seed", 7))
    timestamp_s = int(simulation.get("timestamp_s", 0))

    base_stations = build_base_stations(bundle)
    ues = build_ues(bundle)
    channel_model = build_channel_model(
        simulation.get("channel"),
        seed=seed,
        config_dir=bundle["config_dir"],
    )
    radio_links = channel_model.build_links(base_stations=base_stations, ues=ues)

    association_cfg = simulation.get("association", {}) or {}
    connections = associate_ues(
        base_stations=base_stations,
        ues=ues,
        radio_links=radio_links,
        policy=str(association_cfg.get("policy", "best_sinr_with_capacity")),
        allow_overload=bool(association_cfg.get("allow_overload", False)),
    )
    tables = build_state_tables(
        base_stations=base_stations,
        ues=ues,
        radio_links=radio_links,
        connections=connections,
        timestamp_s=timestamp_s,
    )
    metadata = {
        "prototype": "ucwc_minimal",
        "scope": "structured_state_and_verifier_scaffold",
        "channel_model": channel_model.model_name,
        "seed": seed,
        "timestamp_s": timestamp_s,
        "config_dir": bundle["config_dir"],
        "num_base_stations": len(base_stations),
        "num_ues": len(ues),
        "num_radio_links": len(radio_links),
        "num_connected_ues": sum(1 for item in connections if item.is_connected),
        "tables": {name: len(rows) for name, rows in tables.items()},
        "verifier_config": dict(simulation.get("verifier", {}) or {}),
    }
    return Scenario(
        metadata=metadata,
        base_stations=base_stations,
        ues=ues,
        radio_links=radio_links,
        connections=connections,
        tables=tables,
    )


def write_scenario(scenario: Scenario, output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    scenario_path = root / "scenario.json"
    tables_dir = root / "tables"
    db_path = root / "network_state.sqlite"
    write_json(scenario.to_payload(), scenario_path)
    write_tables(scenario.tables, tables_dir)
    write_sqlite_database(scenario.tables, db_path)
    summary_path = root / "summary.json"
    write_json(scenario.metadata, summary_path)
    return {
        "scenario_json": str(scenario_path),
        "tables_dir": str(tables_dir),
        "sqlite_db": str(db_path),
        "summary_json": str(summary_path),
    }


def main() -> int:
    scenario = build_scenario()
    print(scenario.metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
