"""UCWC NL2SQL and state-management tools for the local agent skeleton."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Any


UCWC_ROOT = Path(__file__).resolve().parents[1]
if str(UCWC_ROOT) not in sys.path:
    sys.path.insert(0, str(UCWC_ROOT))

from ucwc.config_plan import SessionConfigPlan  # noqa: E402
from ucwc.table_manager import (  # noqa: E402
    append_sqlite_row,
    execute_readonly_sql,
    list_sqlite_tables,
    load_sqlite_tables,
    quote_identifier,
    read_sqlite_table,
    sqlite_table_columns,
)
from .tools import ToolDefinition, ToolRegistry  # noqa: E402
from ucwc.verifier import verify_config_plan  # noqa: E402


TABLE_DESCRIPTIONS = {
    "base_station_state": "Base-station resource, load, connection, bandwidth, and PRB state.",
    "ue_state": "UE location, serving BS, mobility, capability proxy, QoS profile, and QoS targets.",
    "radio_link_state": "Per UE-BS radio evidence: distance, pathloss/RSS, SINR, CQI, rank.",
    "connection_state": "Current UE-BS association decision and serving SINR.",
    "qos_state": "Current QoS proxy metrics under the existing association and allocation.",
    "config_history": "Session-level config records and commit/verifier status.",
}


@dataclass
class UcwcToolState:
    db_path: str
    verifier_config: dict[str, Any] = field(default_factory=dict)
    target_ue: str | None = None
    last_plan: dict[str, Any] | None = None
    last_feedback: dict[str, Any] | None = None
    executed_sql: list[dict[str, Any]] = field(default_factory=list)


def build_ucwc_tool_registry(state: UcwcToolState) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in build_ucwc_tools(state):
        registry.register(tool)
    return registry


def build_ucwc_tools(state: UcwcToolState) -> list[ToolDefinition]:
    return [
        create_schema_link_tool(state),
        create_read_table_tool(state),
        create_sql_generate_tool(state),
        create_sql_execute_tool(state),
        create_sql_correct_tool(state),
        create_config_verifier_tool(state),
        create_commit_config_tool(state),
        create_write_table_tool(state),
    ]


def create_schema_link_tool(state: UcwcToolState) -> ToolDefinition:
    def handler(arguments: dict[str, object]) -> str:
        target_ue = _str_arg(arguments, "target_ue", state.target_ue)
        intent = _str_arg(arguments, "intent", "")
        schema = inspect_schema(state.db_path)
        result = {
            "target_ue": target_ue,
            "intent": intent,
            "tables": schema,
            "recommended_join_keys": [
                "ue_state.ue_id = qos_state.ue_id",
                "ue_state.ue_id = connection_state.ue_id",
                "ue_state.ue_id = radio_link_state.ue_id",
                "base_station_state.bs_id = radio_link_state.bs_id",
                "base_station_state.bs_id = connection_state.serving_bs_id",
            ],
            "recommended_evidence_groups": [
                "session_state",
                "radio_candidates",
                "capacity",
                "qos",
                "config_history",
            ],
        }
        return _json(result)

    return ToolDefinition(
        name="ucwc_schema_link",
        description="Inspect UCWC SQLite schema and return table/column grounding for NL2SQL.",
        parameters={
            "type": "object",
            "properties": {
                "target_ue": {"type": "string"},
                "intent": {"type": "string"},
            },
        },
        handler=handler,
    )


def create_read_table_tool(state: UcwcToolState) -> ToolDefinition:
    def handler(arguments: dict[str, object]) -> str:
        table_name = _required_str(arguments, "table_name")
        rows = read_sqlite_table(
            state.db_path,
            table_name,
            limit=_int_arg(arguments, "limit", 20),
            where_column=_str_arg(arguments, "where_column", None),
            where_value=arguments.get("where_value"),
            order_by=_str_arg(arguments, "order_by", None),
        )
        return _json({"table_name": table_name, "row_count": len(rows), "rows": rows})

    return ToolDefinition(
        name="ucwc_read_table",
        description="Read bounded rows from a UCWC SQLite table with an optional equality filter.",
        parameters={
            "type": "object",
            "properties": {
                "table_name": {"type": "string"},
                "where_column": {"type": "string"},
                "where_value": {},
                "order_by": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            "required": ["table_name"],
        },
        handler=handler,
    )


def create_sql_generate_tool(state: UcwcToolState) -> ToolDefinition:
    def handler(arguments: dict[str, object]) -> str:
        target_ue = _str_arg(arguments, "target_ue", state.target_ue)
        query_goal = _required_str(arguments, "query_goal")
        sql = generate_sql(query_goal=query_goal, target_ue=target_ue)
        return _json({"query_goal": query_goal, "target_ue": target_ue, "sql": sql})

    return ToolDefinition(
        name="ucwc_sql_generate",
        description="Generate bounded read-only SQL for UCWC state grounding goals.",
        parameters={
            "type": "object",
            "properties": {
                "query_goal": {
                    "type": "string",
                    "description": "One of session_state, radio_candidates, capacity, qos, config_history, or a short natural-language goal.",
                },
                "target_ue": {"type": "string"},
            },
            "required": ["query_goal"],
        },
        handler=handler,
    )


def create_sql_execute_tool(state: UcwcToolState) -> ToolDefinition:
    def handler(arguments: dict[str, object]) -> str:
        sql = _required_str(arguments, "sql")
        sql = ensure_limit(sql, _int_arg(arguments, "limit", 100))
        rows = execute_readonly_sql(state.db_path, sql)
        payload = {"sql": sql, "row_count": len(rows), "rows": rows}
        state.executed_sql.append({"sql": sql, "row_count": len(rows)})
        return _json(payload)

    return ToolDefinition(
        name="ucwc_sql_execute",
        description="Validate and execute bounded read-only SELECT/WITH SQL against the UCWC database.",
        parameters={
            "type": "object",
            "properties": {
                "sql": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            "required": ["sql"],
        },
        handler=handler,
    )


def create_sql_correct_tool(state: UcwcToolState) -> ToolDefinition:
    def handler(arguments: dict[str, object]) -> str:
        target_ue = _str_arg(arguments, "target_ue", state.target_ue)
        query_goal = _str_arg(arguments, "query_goal", "session_state")
        corrected_sql = generate_sql(query_goal=query_goal or "session_state", target_ue=target_ue)
        return _json(
            {
                "failed_sql": _str_arg(arguments, "failed_sql", ""),
                "error_message": _str_arg(arguments, "error_message", ""),
                "query_goal": query_goal,
                "target_ue": target_ue,
                "corrected_sql": corrected_sql,
            }
        )

    return ToolDefinition(
        name="ucwc_sql_correct",
        description="Correct failed or ungrounded SQL by regenerating it from a UCWC query goal.",
        parameters={
            "type": "object",
            "properties": {
                "failed_sql": {"type": "string"},
                "error_message": {"type": "string"},
                "query_goal": {"type": "string"},
                "target_ue": {"type": "string"},
            },
        },
        handler=handler,
    )


def create_config_verifier_tool(state: UcwcToolState) -> ToolDefinition:
    def handler(arguments: dict[str, object]) -> str:
        plan = plan_from_arguments(arguments, state.target_ue)
        tables = load_sqlite_tables(state.db_path)
        feedback = verify_config_plan(plan, tables, state.verifier_config)
        state.last_plan = plan.to_record()
        state.last_feedback = feedback
        return _json(feedback)

    return ToolDefinition(
        name="ucwc_verify_config_plan",
        description="Run deterministic UCWC verifier checks for a session-level config plan.",
        parameters={
            "type": "object",
            "properties": {
                "ue_id": {"type": "string"},
                "serving_bs_id": {"type": "string"},
                "bandwidth_quota_mhz": {"type": "number"},
                "qos_profile": {"type": "string"},
                "backup_bs_id": {"type": "string"},
                "handover_policy": {"type": "string"},
                "security_profile": {"type": "string"},
                "service_policy": {"type": "string"},
                "source": {"type": "string"},
            },
            "required": ["serving_bs_id", "bandwidth_quota_mhz", "qos_profile"],
        },
        handler=handler,
    )


def create_commit_config_tool(state: UcwcToolState) -> ToolDefinition:
    def handler(arguments: dict[str, object]) -> str:
        if not state.last_plan or not state.last_feedback:
            raise ValueError("No verified plan is available to commit.")
        if not bool(state.last_feedback.get("passed")):
            raise ValueError("Last verifier result did not pass; refusing commit.")
        plan = dict(state.last_plan)
        config_id = _str_arg(arguments, "config_id", f"agent_commit_{plan['ue_id']}")
        row = {
            "config_id": config_id,
            "ue_id": plan["ue_id"],
            "serving_bs_id": plan["serving_bs_id"],
            "qos_profile": plan["qos_profile"],
            "bandwidth_quota_mhz": plan["bandwidth_quota_mhz"],
            "source": plan.get("source", "ucwc_llm_agent"),
            "verifier_passed": 1,
            "failure_reason": None,
            "timestamp_s": _int_arg(arguments, "timestamp_s", 0),
        }
        append_sqlite_row(state.db_path, "config_history", row)
        return _json({"committed": True, "row": row})

    return ToolDefinition(
        name="ucwc_commit_config_plan",
        description="Commit the last verifier-passed UCWC config plan into config_history.",
        parameters={
            "type": "object",
            "properties": {
                "config_id": {"type": "string"},
                "timestamp_s": {"type": "integer"},
            },
        },
        handler=handler,
    )


def create_write_table_tool(state: UcwcToolState) -> ToolDefinition:
    def handler(arguments: dict[str, object]) -> str:
        table_name = _required_str(arguments, "table_name")
        if table_name != "config_history":
            raise ValueError("ucwc_write_table is restricted to config_history in this prototype.")
        row = arguments.get("row")
        if not isinstance(row, dict):
            raise ValueError("row must be a JSON object.")
        append_sqlite_row(state.db_path, table_name, row)
        return _json({"written": True, "table_name": table_name, "row": row})

    return ToolDefinition(
        name="ucwc_write_table",
        description="Append a row to config_history. General writes are intentionally restricted.",
        parameters={
            "type": "object",
            "properties": {
                "table_name": {"type": "string"},
                "row": {"type": "object"},
            },
            "required": ["table_name", "row"],
        },
        handler=handler,
    )


def inspect_schema(db_path: str) -> list[dict[str, Any]]:
    tables = []
    with sqlite3.connect(db_path) as connection:
        for table_name in list_sqlite_tables(db_path):
            columns = sqlite_table_columns(db_path, table_name)
            row_count = connection.execute(
                f"SELECT COUNT(*) FROM {quote_identifier(table_name)}"
            ).fetchone()[0]
            tables.append(
                {
                    "table_name": table_name,
                    "description": TABLE_DESCRIPTIONS.get(table_name, ""),
                    "columns": columns,
                    "row_count": int(row_count),
                }
            )
    return tables


def generate_sql(query_goal: str, target_ue: str | None) -> str:
    goal = query_goal.lower().strip()
    radio_where = _ue_predicate("r", target_ue)
    qos_where = _ue_predicate("q", target_ue)
    history_where = _ue_predicate(None, target_ue)
    session_where = _ue_predicate("u", target_ue)
    if "radio" in goal or "candidate" in goal:
        return (
            "SELECT r.ue_id, r.bs_id, r.sinr_db, r.cqi, r.received_power_dbm, "
            "r.coverage_rank, b.available_connections, b.available_prb, b.prb_utilization "
            "FROM radio_link_state AS r "
            "JOIN base_station_state AS b ON b.bs_id = r.bs_id "
            f"WHERE {radio_where} "
            "ORDER BY r.sinr_db DESC LIMIT 10"
        )
    if "capacity" in goal or "resource" in goal or "base" in goal:
        return (
            "SELECT bs_id, max_connections, connected_ue_count, available_connections, "
            "bandwidth_mhz, total_prb, allocated_prb, available_prb, prb_utilization "
            "FROM base_station_state ORDER BY available_prb DESC LIMIT 20"
        )
    if "qos" in goal:
        return (
            "SELECT q.*, u.min_dl_mbps, u.min_ul_mbps, u.max_latency_ms, "
            "u.min_reliability, u.max_packet_loss, u.max_jitter_ms "
            "FROM qos_state AS q JOIN ue_state AS u ON u.ue_id = q.ue_id "
            f"WHERE {qos_where} LIMIT 5"
        )
    if "history" in goal or "config" in goal:
        return (
            "SELECT * FROM config_history "
            f"WHERE {history_where} ORDER BY timestamp_s DESC LIMIT 20"
        )
    return (
        "SELECT u.ue_id, u.connected_bs_id, u.mobility_speed_kmh, u.mobility_class, "
        "u.profile_id, u.traffic_direction, u.min_dl_mbps, u.min_ul_mbps, "
        "u.max_latency_ms, u.min_reliability, u.max_packet_loss, u.max_jitter_ms, "
        "u.security_level, q.dl_throughput_mbps, q.ul_throughput_mbps, "
        "q.e2e_latency_ms, q.packet_loss, q.reliability, q.jitter_ms, "
        "c.serving_bs_id, c.serving_sinr_db "
        "FROM ue_state AS u "
        "LEFT JOIN qos_state AS q ON q.ue_id = u.ue_id "
        "LEFT JOIN connection_state AS c ON c.ue_id = u.ue_id "
        f"WHERE {session_where} LIMIT 5"
    )


def ensure_limit(sql: str, limit: int) -> str:
    cleaned = sql.strip().rstrip(";")
    if " limit " in cleaned.lower():
        return cleaned
    return f"SELECT * FROM ({cleaned}) AS bounded_query LIMIT {max(1, min(limit, 500))}"


def plan_from_arguments(arguments: dict[str, object], default_ue: str | None) -> SessionConfigPlan:
    ue_id = _str_arg(arguments, "ue_id", default_ue)
    if not ue_id:
        raise ValueError("ue_id is required.")
    return SessionConfigPlan(
        ue_id=ue_id,
        serving_bs_id=_required_str(arguments, "serving_bs_id"),
        bandwidth_quota_mhz=float(arguments["bandwidth_quota_mhz"]),
        qos_profile=_required_str(arguments, "qos_profile"),
        backup_bs_id=_str_arg(arguments, "backup_bs_id", None),
        handover_policy=_str_arg(arguments, "handover_policy", "stable") or "stable",
        security_profile=_str_arg(arguments, "security_profile", "standard") or "standard",
        service_policy=_str_arg(arguments, "service_policy", "session_level_ucwc") or "session_level_ucwc",
        source=_str_arg(arguments, "source", "ucwc_llm_agent") or "ucwc_llm_agent",
    )


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _required_str(arguments: dict[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required.")
    return value.strip()


def _str_arg(arguments: dict[str, object], key: str, default: str | None) -> str | None:
    value = arguments.get(key, default)
    if value is None:
        return None
    return str(value).strip()


def _int_arg(arguments: dict[str, object], key: str, default: int) -> int:
    value = arguments.get(key, default)
    return int(value)


def _escape_sql_literal(value: str | None) -> str:
    return str(value or "").replace("'", "''")


def _ue_predicate(alias: str | None, target_ue: str | None) -> str:
    if not target_ue:
        return "1 = 1"
    column = "ue_id" if alias is None else f"{alias}.ue_id"
    return f"{column} = '{_escape_sql_literal(target_ue)}'"
