"""State-table generation and export helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3
from typing import Any

from ucwc.association import ConnectionDecision
from ucwc.channel_model import RadioLink
from ucwc.components import BaseStation, UE
from ucwc.physics_tools import (
    bandwidth_to_prb,
    estimate_throughput_mbps,
    jitter_proxy_ms,
    latency_proxy_ms,
    packet_loss_proxy,
    reliability_proxy,
    required_bandwidth_mhz,
)


def build_state_tables(
    base_stations: list[BaseStation],
    ues: list[UE],
    radio_links: list[RadioLink],
    connections: list[ConnectionDecision],
    timestamp_s: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    allocation = allocate_bandwidth(base_stations, ues)
    allocation_errors = validate_resource_allocation(base_stations, allocation)
    if allocation_errors:
        raise ValueError(
            "Invalid resource allocation: " + "; ".join(allocation_errors)
        )
    base_station_state = [
        _base_station_record(bs, allocation, timestamp_s) for bs in base_stations
    ]
    ue_state = [_ue_record(ue, timestamp_s) for ue in ues]
    radio_link_state = [
        {**link.to_record(), "timestamp_s": timestamp_s} for link in radio_links
    ]
    connection_state = [
        {**decision.to_record(), "timestamp_s": timestamp_s} for decision in connections
    ]
    qos_state = [
        _qos_record(ue, base_stations, radio_links, allocation, timestamp_s) for ue in ues
    ]
    config_history = [
        _initial_config_record(ue, allocation, timestamp_s) for ue in ues
    ]
    return {
        "base_station_state": base_station_state,
        "ue_state": ue_state,
        "radio_link_state": radio_link_state,
        "connection_state": connection_state,
        "qos_state": qos_state,
        "config_history": config_history,
    }


def allocate_bandwidth(
    base_stations: list[BaseStation],
    ues: list[UE],
) -> dict[str, dict[str, float | int | str]]:
    ue_by_id = {ue.ue_id: ue for ue in ues}
    allocation: dict[str, dict[str, float | int | str]] = {}
    for bs in base_stations:
        connected = [ue_by_id[ue_id] for ue_id in bs.connected_ue_ids if ue_id in ue_by_id]
        if not connected:
            continue
        fair_share = bs.resources.bandwidth_mhz / float(len(connected))
        desired: list[dict[str, Any]] = []
        for ue in connected:
            required_mhz = required_bandwidth_mhz(
                ue.qos.min_dl_mbps,
                ue.qos.min_ul_mbps,
            )
            desired_mhz = round(min(fair_share, required_mhz * 1.35), 3)
            desired_prb = bandwidth_to_prb(
                desired_mhz,
                bs.resources.bandwidth_mhz,
                bs.resources.total_prb,
            )
            desired.append(
                {
                    "ue": ue,
                    "required_bandwidth_mhz": required_mhz,
                    "allocated_prb": min(desired_prb, bs.resources.total_prb),
                    "priority": ue.qos.priority,
                }
            )

        _fit_prb_budget(desired, bs.resources.total_prb)
        prb_bandwidth_mhz = bs.resources.prb_bandwidth_mhz()
        for item in desired:
            ue = item["ue"]
            allocated_prb = int(item["allocated_prb"])
            allocated_mhz = round(allocated_prb * prb_bandwidth_mhz, 3)
            allocation[ue.ue_id] = {
                "bs_id": bs.bs_id,
                "required_bandwidth_mhz": float(item["required_bandwidth_mhz"]),
                "allocated_bandwidth_mhz": allocated_mhz,
                "allocated_prb": allocated_prb,
            }
    return allocation


def validate_resource_allocation(
    base_stations: list[BaseStation],
    allocation: dict[str, dict[str, float | int | str]],
) -> list[str]:
    errors: list[str] = []
    for bs in base_stations:
        allocated_prb = sum(
            int(item["allocated_prb"])
            for item in allocation.values()
            if item["bs_id"] == bs.bs_id
        )
        if allocated_prb > bs.resources.total_prb:
            errors.append(
                f"{bs.bs_id} allocated_prb={allocated_prb} exceeds total_prb={bs.resources.total_prb}"
            )
        if len(bs.connected_ue_ids) > bs.resources.max_connections:
            errors.append(
                f"{bs.bs_id} connected_ue_count={len(bs.connected_ue_ids)} exceeds "
                f"max_connections={bs.resources.max_connections}"
            )
    return errors


def _fit_prb_budget(items: list[dict[str, Any]], total_prb: int) -> None:
    if not items:
        return
    budget = max(0, total_prb)
    min_prb = 1 if budget >= len(items) else 0
    for item in items:
        item["allocated_prb"] = max(min_prb, int(item["allocated_prb"]))
    while sum(int(item["allocated_prb"]) for item in items) > budget:
        candidates = [
            index
            for index, item in enumerate(items)
            if int(item["allocated_prb"]) > min_prb
        ]
        if not candidates:
            break
        victim = min(
            candidates,
            key=lambda index: (
                int(items[index]["priority"]),
                -int(items[index]["allocated_prb"]),
                str(items[index]["ue"].ue_id),
            ),
        )
        items[victim]["allocated_prb"] = int(items[victim]["allocated_prb"]) - 1


def write_tables(tables: dict[str, list[dict[str, Any]]], output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for table_name, rows in tables.items():
        path = root / f"{table_name}.csv"
        if not rows:
            path.write_text("", encoding="utf-8")
            continue
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def write_json(payload: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_sqlite_database(
    tables: dict[str, list[dict[str, Any]]],
    db_path: str | Path,
    *,
    overwrite: bool = True,
) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and path.exists():
        path.unlink()
    with sqlite3.connect(path) as connection:
        for table_name, rows in tables.items():
            _validate_identifier(table_name, "table")
            columns = _table_columns(rows)
            if not columns:
                continue
            column_types = {
                column: _infer_sqlite_type([row.get(column) for row in rows])
                for column in columns
            }
            column_defs = ", ".join(
                f"{quote_identifier(column)} {column_types[column]}" for column in columns
            )
            connection.execute(
                f"CREATE TABLE {quote_identifier(table_name)} ({column_defs})"
            )
            placeholders = ", ".join("?" for _ in columns)
            quoted_columns = ", ".join(quote_identifier(column) for column in columns)
            connection.executemany(
                (
                    f"INSERT INTO {quote_identifier(table_name)} "
                    f"({quoted_columns}) VALUES ({placeholders})"
                ),
                [
                    tuple(_coerce_sqlite_value(row.get(column)) for column in columns)
                    for row in rows
                ],
            )
        connection.commit()


def list_sqlite_tables(db_path: str | Path) -> list[str]:
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    return [str(row[0]) for row in rows]


def sqlite_table_columns(db_path: str | Path, table_name: str) -> list[str]:
    _validate_identifier(table_name, "table")
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            f"PRAGMA table_info({quote_identifier(table_name)})"
        ).fetchall()
    return [str(row[1]) for row in rows]


def read_sqlite_table(
    db_path: str | Path,
    table_name: str,
    *,
    limit: int = 20,
    where_column: str | None = None,
    where_value: Any | None = None,
    order_by: str | None = None,
) -> list[dict[str, Any]]:
    _validate_identifier(table_name, "table")
    columns = sqlite_table_columns(db_path, table_name)
    if not columns:
        raise ValueError(f"Unknown or empty table: {table_name}")
    params: list[Any] = []
    sql = f"SELECT * FROM {quote_identifier(table_name)}"
    if where_column:
        _validate_identifier(where_column, "column")
        if where_column not in columns:
            raise ValueError(f"Unknown column for {table_name}: {where_column}")
        sql += f" WHERE {quote_identifier(where_column)} = ?"
        params.append(_coerce_sqlite_value(where_value))
    if order_by:
        _validate_identifier(order_by, "column")
        if order_by not in columns:
            raise ValueError(f"Unknown order_by column for {table_name}: {order_by}")
        sql += f" ORDER BY {quote_identifier(order_by)}"
    sql += " LIMIT ?"
    params.append(max(1, min(int(limit), 500)))
    return execute_readonly_sql(db_path, sql, params)


def execute_readonly_sql(
    db_path: str | Path,
    sql: str,
    params: list[Any] | tuple[Any, ...] | None = None,
) -> list[dict[str, Any]]:
    cleaned = sql.strip()
    if not _is_readonly_select(cleaned):
        raise ValueError("Only read-only SELECT/WITH SQL is allowed.")
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute(f"EXPLAIN QUERY PLAN {cleaned}", params or [])
        rows = connection.execute(cleaned, params or []).fetchall()
    return [dict(row) for row in rows]


def append_sqlite_row(
    db_path: str | Path,
    table_name: str,
    row: dict[str, Any],
) -> None:
    _validate_identifier(table_name, "table")
    columns = sqlite_table_columns(db_path, table_name)
    if not columns:
        raise ValueError(f"Unknown table: {table_name}")
    unknown = sorted(set(row) - set(columns))
    if unknown:
        raise ValueError(f"Unknown columns for {table_name}: {unknown}")
    insert_columns = [column for column in columns if column in row]
    if not insert_columns:
        raise ValueError("Row has no columns to insert.")
    placeholders = ", ".join("?" for _ in insert_columns)
    quoted_columns = ", ".join(quote_identifier(column) for column in insert_columns)
    values = [_coerce_sqlite_value(row[column]) for column in insert_columns]
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            (
                f"INSERT INTO {quote_identifier(table_name)} "
                f"({quoted_columns}) VALUES ({placeholders})"
            ),
            values,
        )
        connection.commit()


def load_sqlite_tables(db_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    return {
        table_name: read_sqlite_table(db_path, table_name, limit=10000)
        for table_name in list_sqlite_tables(db_path)
    }


def quote_identifier(name: str) -> str:
    _validate_identifier(name, "identifier")
    return '"' + name.replace('"', '""') + '"'


def _table_columns(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(key) for row in rows for key in row.keys()})


def _infer_sqlite_type(values: list[Any]) -> str:
    concrete = [value for value in values if value is not None]
    if not concrete:
        return "TEXT"
    if all(isinstance(value, bool) for value in concrete):
        return "INTEGER"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in concrete):
        return "INTEGER"
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in concrete):
        return "REAL"
    return "TEXT"


def _coerce_sqlite_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _validate_identifier(name: str, label: str) -> None:
    if not name or not name.replace("_", "").isalnum() or name[0].isdigit():
        raise ValueError(f"Invalid {label} identifier: {name!r}")


def _is_readonly_select(sql: str) -> bool:
    lowered = sql.lower().strip()
    if ";" in lowered.rstrip(";"):
        return False
    lowered = lowered.rstrip(";").strip()
    if lowered.startswith(("select ", "with ")):
        blocked = (" insert ", " update ", " delete ", " drop ", " alter ", " create ", " attach ", " detach ", " pragma ")
        padded = f" {lowered} "
        return not any(token in padded for token in blocked)
    return False


def _base_station_record(
    bs: BaseStation,
    allocation: dict[str, dict[str, float | int | str]],
    timestamp_s: int,
) -> dict[str, Any]:
    allocated_prb = sum(
        int(item["allocated_prb"])
        for item in allocation.values()
        if item["bs_id"] == bs.bs_id
    )
    return {
        **bs.to_record(),
        "allocated_prb": allocated_prb,
        "available_prb": max(0, bs.resources.total_prb - allocated_prb),
        "prb_utilization": round(allocated_prb / max(1, bs.resources.total_prb), 5),
        "timestamp_s": timestamp_s,
    }


def _ue_record(ue: UE, timestamp_s: int) -> dict[str, Any]:
    return {**ue.to_record(), "timestamp_s": timestamp_s}


def _qos_record(
    ue: UE,
    base_stations: list[BaseStation],
    radio_links: list[RadioLink],
    allocation: dict[str, dict[str, float | int | str]],
    timestamp_s: int,
) -> dict[str, Any]:
    bs_by_id = {bs.bs_id: bs for bs in base_stations}
    link_by_pair = {(link.ue_id, link.bs_id): link for link in radio_links}
    alloc = allocation.get(ue.ue_id)
    if not ue.connected_bs_id or not alloc:
        return {
            "ue_id": ue.ue_id,
            "serving_bs_id": None,
            "allocated_bandwidth_mhz": 0.0,
            "allocated_prb": 0,
            "dl_throughput_mbps": 0.0,
            "ul_throughput_mbps": 0.0,
            "e2e_latency_ms": None,
            "packet_loss": None,
            "reliability": None,
            "jitter_ms": None,
            "qos_satisfied": False,
            "timestamp_s": timestamp_s,
        }

    bs = bs_by_id[ue.connected_bs_id]
    link = link_by_pair[(ue.ue_id, ue.connected_bs_id)]
    allocated_mhz = float(alloc["allocated_bandwidth_mhz"])
    load_fraction = sum(
        int(item["allocated_prb"])
        for item in allocation.values()
        if item["bs_id"] == bs.bs_id
    ) / max(1, bs.resources.total_prb)
    dl_mbps = estimate_throughput_mbps(link.sinr_db, allocated_mhz)
    ul_mbps = estimate_throughput_mbps(link.sinr_db - 3.0, allocated_mhz)
    latency_ms = latency_proxy_ms(load_fraction, link.sinr_db, ue.mobility_speed_kmh)
    packet_loss = packet_loss_proxy(link.sinr_db, load_fraction)
    reliability = reliability_proxy(packet_loss)
    jitter_ms = jitter_proxy_ms(latency_ms, ue.mobility_speed_kmh)
    qos_satisfied = (
        dl_mbps >= ue.qos.min_dl_mbps
        and ul_mbps >= ue.qos.min_ul_mbps
        and latency_ms <= ue.qos.max_latency_ms
        and reliability >= ue.qos.min_reliability
        and packet_loss <= ue.qos.max_packet_loss
        and jitter_ms <= ue.qos.max_jitter_ms
    )
    return {
        "ue_id": ue.ue_id,
        "serving_bs_id": ue.connected_bs_id,
        "qos_profile": ue.qos.profile_id,
        "traffic_direction": ue.qos.traffic_direction,
        "allocated_bandwidth_mhz": allocated_mhz,
        "required_bandwidth_mhz": float(alloc["required_bandwidth_mhz"]),
        "allocated_prb": int(alloc["allocated_prb"]),
        "dl_throughput_mbps": dl_mbps,
        "ul_throughput_mbps": ul_mbps,
        "e2e_latency_ms": latency_ms,
        "packet_loss": packet_loss,
        "reliability": reliability,
        "jitter_ms": jitter_ms,
        "qos_satisfied": qos_satisfied,
        "timestamp_s": timestamp_s,
    }


def _initial_config_record(
    ue: UE,
    allocation: dict[str, dict[str, float | int | str]],
    timestamp_s: int,
) -> dict[str, Any]:
    alloc = allocation.get(ue.ue_id, {})
    return {
        "config_id": f"initial_{ue.ue_id}",
        "ue_id": ue.ue_id,
        "serving_bs_id": ue.connected_bs_id,
        "qos_profile": ue.qos.profile_id,
        "bandwidth_quota_mhz": float(alloc.get("allocated_bandwidth_mhz", 0.0)),
        "source": "initial_association",
        "verifier_passed": None,
        "reallocation_triggered": 0,
        "reallocation_verified": 1,
        "failure_reason": None,
        "verification_summary": None,
        "timestamp_s": timestamp_s,
    }
