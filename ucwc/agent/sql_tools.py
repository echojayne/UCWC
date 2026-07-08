"""Read-only NL2SQL grounding tools for semantic UCWC state databases."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from ucwc.agent.protocol import SqlQueryResult
from ucwc.state import TABLE_COLUMNS

_MUTATING_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|pragma|attach|detach|vacuum|reindex)\b",
    re.IGNORECASE,
)


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite DB in read-only mode."""

    path = Path(db_path).resolve()
    uri = f"file:{path.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def connect_writable(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite DB for verifier-gated commits."""

    connection = sqlite3.connect(Path(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def inspect_schema(connection: sqlite3.Connection) -> dict[str, list[str]]:
    """Return the live schema for known semantic UCWC tables."""

    schema: dict[str, list[str]] = {}
    for table in TABLE_COLUMNS:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        if rows:
            schema[table] = [str(row["name"]) for row in rows]
    return schema


def schema_prompt_text(schema: dict[str, list[str]]) -> str:
    """Render a compact schema block for the LLM."""

    lines: list[str] = []
    for table, columns in schema.items():
        lines.append(f"- {table}({', '.join(columns)})")
    return "\n".join(lines)


def validate_readonly_sql(connection: sqlite3.Connection, sql: str) -> tuple[bool, str]:
    """Validate that SQL is a single bounded read-only SELECT/CTE statement."""

    stripped = sql.strip()
    if not stripped:
        return False, "empty SQL"
    if len(stripped) > 5000:
        return False, "SQL is too long"
    without_final_semicolon = stripped[:-1].strip() if stripped.endswith(";") else stripped
    if ";" in without_final_semicolon:
        return False, "multiple SQL statements are not allowed"
    first_token = without_final_semicolon.split(None, 1)[0].lower()
    if first_token not in {"select", "with"}:
        return False, "only SELECT or WITH read-only queries are allowed"
    if _MUTATING_SQL.search(without_final_semicolon):
        return False, "mutating or unsafe SQL keyword is not allowed"
    try:
        connection.execute(f"EXPLAIN QUERY PLAN {without_final_semicolon}")
    except sqlite3.Error as error:
        return False, f"SQLite validation failed: {error}"
    return True, "ok"


def execute_readonly_sql(
    connection: sqlite3.Connection,
    sql: str,
    *,
    max_rows: int = 40,
) -> SqlQueryResult:
    """Execute validated SQL and return capped rows."""

    ok, reason = validate_readonly_sql(connection, sql)
    if not ok:
        raise ValueError(reason)
    cursor = connection.execute(sql.strip().rstrip(";"))
    columns = [str(item[0]) for item in cursor.description or []]
    rows: list[dict[str, Any]] = []
    truncated = False
    for index, row in enumerate(cursor):
        if index >= max_rows:
            truncated = True
            break
        rows.append({column: row[column] for column in columns})
    return SqlQueryResult(
        sql=sql.strip(),
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
    )


def default_request_evidence_sql(request_id: str) -> str:
    """Deterministic fallback grounding query for one UE request."""

    safe_request_id = _safe_identifier(request_id)
    return f"""
SELECT
  rq.request_id,
  rq.ue_id,
  rq.arrival_order,
  rq.task_type,
  rq.min_task_score,
  rq.max_total_latency_ms,
  rl.bs_id,
  rl.snr_db,
  rl.sinr_db,
  rl.radio_rank,
  bs.bandwidth_budget_mhz,
  bs.used_bandwidth_mhz
FROM ue_request_queue AS rq
JOIN radio_link_state AS rl ON rl.ue_id = rq.ue_id
JOIN base_station_state AS bs ON bs.bs_id = rl.bs_id
WHERE rq.request_id = '{safe_request_id}'
ORDER BY rl.radio_rank ASC
LIMIT 10
""".strip()


def first_request_id(connection: sqlite3.Connection) -> str:
    row = connection.execute(
        "SELECT request_id FROM ue_request_queue ORDER BY arrival_order ASC LIMIT 1"
    ).fetchone()
    if row is None:
        raise ValueError("ue_request_queue is empty")
    return str(row["request_id"])


def _safe_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_\-:.]+", value):
        raise ValueError(f"unsafe identifier value: {value!r}")
    return value
