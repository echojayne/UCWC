"""SQLite and CSV storage for semantic link evaluation results."""

from __future__ import annotations

import csv
from pathlib import Path
import sqlite3
from typing import Iterable

from .evaluation import SemanticEvalRow


EXPERIENCE_TABLE = "semantic_link_experience"


def write_experience_rows(
    rows: Iterable[SemanticEvalRow],
    *,
    sqlite_path: str | Path,
    csv_path: str | Path | None = None,
    replace: bool = True,
) -> dict[str, str | int]:
    row_list = list(rows)
    db_path = Path(sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        if replace:
            connection.execute(f"DROP TABLE IF EXISTS {EXPERIENCE_TABLE}")
        _create_table(connection)
        for row in row_list:
            payload = row.to_dict()
            columns = list(payload.keys())
            placeholders = ", ".join("?" for _ in columns)
            quoted = ", ".join(columns)
            connection.execute(
                f"INSERT INTO {EXPERIENCE_TABLE} ({quoted}) VALUES ({placeholders})",
                [payload[column] for column in columns],
            )
        connection.commit()

    result: dict[str, str | int] = {"sqlite_path": str(db_path), "row_count": len(row_list)}
    if csv_path is not None:
        csv_file = Path(csv_path)
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(row_list, csv_file)
        result["csv_path"] = str(csv_file)
    return result


def _create_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {EXPERIENCE_TABLE} (
            config_id TEXT NOT NULL,
            encoder_depth INTEGER NOT NULL,
            quantization_bits INTEGER NOT NULL,
            ldpc_code_rate REAL NOT NULL,
            qam_order INTEGER NOT NULL,
            snr_db REAL NOT NULL,
            channel_model TEXT NOT NULL,
            dataset TEXT NOT NULL,
            batch_size INTEGER NOT NULL,
            feature_dim INTEGER NOT NULL,
            payload_bits INTEGER NOT NULL,
            header_bits INTEGER NOT NULL,
            payload_bits_with_header INTEGER NOT NULL,
            coded_bits INTEGER NOT NULL,
            coded_bits_with_header INTEGER NOT NULL,
            qam_bits_per_symbol INTEGER NOT NULL,
            qam_symbols INTEGER NOT NULL,
            qam_symbols_with_header INTEGER NOT NULL,
            payload_bytes REAL NOT NULL,
            payload_bytes_with_header REAL NOT NULL,
            coded_bytes REAL NOT NULL,
            coded_bytes_with_header REAL NOT NULL,
            ldpc_padding_bits INTEGER NOT NULL,
            bit_error_rate REAL NOT NULL,
            bit_errors INTEGER NOT NULL,
            block_errors INTEGER NOT NULL,
            block_error_rate REAL NOT NULL,
            ldpc_nonconvergence_rate REAL NOT NULL,
            ldpc_blocks INTEGER NOT NULL,
            ldpc_converged_blocks INTEGER NOT NULL,
            ldpc_max_iterations_used INTEGER NOT NULL,
            semantic_score REAL NOT NULL,
            classifier_agreement REAL NOT NULL,
            original_accuracy REAL NOT NULL,
            recovered_accuracy REAL NOT NULL,
            encoding_latency_ms REAL NOT NULL,
            decoding_latency_ms REAL NOT NULL
        )
        """
    )


def _write_csv(rows: list[SemanticEvalRow], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].to_dict().keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([row.to_dict() for row in rows])
