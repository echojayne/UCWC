"""SQLite and CSV helpers for semantic UCWC system-state tables."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


TABLE_SCHEMAS: dict[str, str] = {
    "scenario_metadata": """
        CREATE TABLE IF NOT EXISTS scenario_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """,
    "base_station_state": """
        CREATE TABLE IF NOT EXISTS base_station_state (
            bs_id TEXT PRIMARY KEY,
            map_file TEXT NOT NULL,
            x_m REAL NOT NULL,
            y_m REAL NOT NULL,
            bandwidth_budget_mhz REAL NOT NULL,
            used_bandwidth_mhz REAL NOT NULL DEFAULT 0.0
        )
    """,
    "ue_request_queue": """
        CREATE TABLE IF NOT EXISTS ue_request_queue (
            request_id TEXT PRIMARY KEY,
            ue_id TEXT NOT NULL UNIQUE,
            arrival_order INTEGER NOT NULL UNIQUE,
            x_m REAL NOT NULL,
            y_m REAL NOT NULL,
            task_type TEXT NOT NULL,
            min_task_score REAL NOT NULL,
            max_total_latency_ms REAL NOT NULL
        )
    """,
    "radio_link_state": """
        CREATE TABLE IF NOT EXISTS radio_link_state (
            ue_id TEXT NOT NULL,
            bs_id TEXT NOT NULL,
            snr_db REAL NOT NULL,
            sinr_db REAL NOT NULL,
            radio_rank INTEGER NOT NULL,
            radio_gain_raw REAL NOT NULL,
            radio_gain_norm REAL NOT NULL,
            source_map_file TEXT NOT NULL,
            PRIMARY KEY (ue_id, bs_id)
        )
    """,
    "semantic_config_catalog": """
        CREATE TABLE IF NOT EXISTS semantic_config_catalog (
            config_id TEXT PRIMARY KEY,
            mode_id INTEGER NOT NULL UNIQUE,
            encoder_depth INTEGER NOT NULL,
            quantization_bits INTEGER NOT NULL,
            feature_dim INTEGER NOT NULL,
            header_bits INTEGER NOT NULL,
            payload_bits INTEGER NOT NULL,
            encoding_latency_ms REAL NOT NULL,
            decoding_latency_ms REAL NOT NULL,
            fixed_latency_ms REAL NOT NULL DEFAULT 0.0
        )
    """,
    "phy_mode_catalog": """
        CREATE TABLE IF NOT EXISTS phy_mode_catalog (
            phy_mode_id TEXT PRIMARY KEY,
            ldpc_code_rate REAL NOT NULL,
            qam_order INTEGER NOT NULL,
            reference_snr_db REAL NOT NULL,
            spectral_efficiency_bps_hz REAL NOT NULL
        )
    """,
    "active_session": """
        CREATE TABLE IF NOT EXISTS active_session (
            session_id TEXT PRIMARY KEY,
            request_id TEXT NOT NULL,
            ue_id TEXT NOT NULL,
            serving_bs_id TEXT NOT NULL,
            semantic_config_id TEXT NOT NULL,
            phy_mode_id TEXT NOT NULL,
            bandwidth_mhz REAL NOT NULL,
            admitted_at_order INTEGER NOT NULL,
            source TEXT NOT NULL
        )
    """,
    "config_history": """
        CREATE TABLE IF NOT EXISTS config_history (
            history_id TEXT PRIMARY KEY,
            request_id TEXT NOT NULL,
            attempted_at_order INTEGER NOT NULL,
            serving_bs_id TEXT,
            semantic_config_id TEXT,
            phy_mode_id TEXT,
            bandwidth_mhz REAL,
            verifier_passed INTEGER NOT NULL,
            failure_reason TEXT,
            source TEXT NOT NULL
        )
    """,
}


TABLE_COLUMNS: dict[str, list[str]] = {
    "scenario_metadata": ["key", "value"],
    "base_station_state": [
        "bs_id",
        "map_file",
        "x_m",
        "y_m",
        "bandwidth_budget_mhz",
        "used_bandwidth_mhz",
    ],
    "ue_request_queue": [
        "request_id",
        "ue_id",
        "arrival_order",
        "x_m",
        "y_m",
        "task_type",
        "min_task_score",
        "max_total_latency_ms",
    ],
    "radio_link_state": [
        "ue_id",
        "bs_id",
        "snr_db",
        "sinr_db",
        "radio_rank",
        "radio_gain_raw",
        "radio_gain_norm",
        "source_map_file",
    ],
    "semantic_config_catalog": [
        "config_id",
        "mode_id",
        "encoder_depth",
        "quantization_bits",
        "feature_dim",
        "header_bits",
        "payload_bits",
        "encoding_latency_ms",
        "decoding_latency_ms",
        "fixed_latency_ms",
    ],
    "phy_mode_catalog": [
        "phy_mode_id",
        "ldpc_code_rate",
        "qam_order",
        "reference_snr_db",
        "spectral_efficiency_bps_hz",
    ],
    "active_session": [
        "session_id",
        "request_id",
        "ue_id",
        "serving_bs_id",
        "semantic_config_id",
        "phy_mode_id",
        "bandwidth_mhz",
        "admitted_at_order",
        "source",
    ],
    "config_history": [
        "history_id",
        "request_id",
        "attempted_at_order",
        "serving_bs_id",
        "semantic_config_id",
        "phy_mode_id",
        "bandwidth_mhz",
        "verifier_passed",
        "failure_reason",
        "source",
    ],
}


def create_schema(connection: sqlite3.Connection) -> None:
    """Create all semantic UCWC state tables."""

    for schema in TABLE_SCHEMAS.values():
        connection.execute(schema)
    connection.commit()


def reset_schema(connection: sqlite3.Connection) -> None:
    """Drop and recreate all semantic UCWC state tables."""

    for table in reversed(TABLE_COLUMNS):
        connection.execute(f"DROP TABLE IF EXISTS {table}")
    connection.commit()
    create_schema(connection)


def write_state_database(
    db_path: str | Path,
    tables: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    overwrite: bool = True,
) -> None:
    """Write a full semantic UCWC state database."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing database: {path}")
    with sqlite3.connect(path) as connection:
        reset_schema(connection)
        for table, rows in tables.items():
            insert_rows(connection, table, list(rows))
        connection.commit()


def insert_rows(
    connection: sqlite3.Connection,
    table: str,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Insert rows into one known state table."""

    columns = TABLE_COLUMNS[table]
    row_list = list(rows)
    if not row_list:
        return
    placeholders = ", ".join("?" for _ in columns)
    quoted_columns = ", ".join(columns)
    connection.executemany(
        f"INSERT INTO {table} ({quoted_columns}) VALUES ({placeholders})",
        [[_encode_value(row.get(column)) for column in columns] for row in row_list],
    )


def write_csv_tables(
    tables_dir: str | Path,
    tables: Mapping[str, Iterable[Mapping[str, Any]]],
) -> None:
    """Write every generated state table as a CSV file."""

    path = Path(tables_dir)
    path.mkdir(parents=True, exist_ok=True)
    for table, rows in tables.items():
        columns = TABLE_COLUMNS[table]
        with (path / f"{table}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: _encode_value(row.get(column)) for column in columns})


def load_table(connection: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    """Load one known state table as a list of dictionaries."""

    columns = TABLE_COLUMNS[table]
    cursor = connection.execute(f"SELECT {', '.join(columns)} FROM {table}")
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def load_state_database(db_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load all known state tables from a SQLite database."""

    with sqlite3.connect(db_path) as connection:
        return {table: load_table(connection, table) for table in TABLE_COLUMNS}


def metadata_rows(metadata: Mapping[str, Any]) -> list[dict[str, str]]:
    """Convert a metadata mapping into key/value table rows."""

    return [{"key": key, "value": _metadata_value(value)} for key, value in metadata.items()]


def _metadata_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _encode_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        return value
    return json.dumps(value, sort_keys=True)
