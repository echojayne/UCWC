"""UCWC agent loop entrypoint.

The main path builds a SQLite network-state database, then runs the local
tool-calling agent with UCWC/NL2SQL tools. A deterministic fallback remains for
offline smoke tests, but the default path is schema/SQL/verifier tool calling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

UCWC_ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = Path(__file__).resolve().parent
from .agent import Agent, AgentConfig
from ucwc.config_plan import SessionConfigPlan
from .context import ContextManager
from .llm import LlmClient, load_llm_config
from ucwc.scenario_builder import build_scenario
from ucwc.table_manager import write_json, write_sqlite_database
from .ucwc_prompts import build_ucwc_system_prompt
from .ucwc_tools import UcwcToolState, build_ucwc_tool_registry
from ucwc.verifier import propose_config_plan as deterministic_propose_config_plan
from ucwc.verifier import verify_config_plan


@dataclass(frozen=True)
class AgentLoopConfig:
    ue_id: str
    user_intent: str = "Improve this UE session while preserving QoS and safety."
    config_dir: str | None = None
    max_repair_steps: int = 2
    output_trace: str | None = None
    db_path: str | None = None
    agent_mode: str = "llm"
    max_steps: int = 20
    fallback_to_deterministic: bool = False


def run_minimal_agent(config: AgentLoopConfig) -> dict[str, Any]:
    if config.agent_mode == "deterministic":
        return run_deterministic_agent(config)
    if config.agent_mode != "llm":
        raise ValueError("agent_mode must be 'llm' or 'deterministic'.")

    scenario = build_scenario(config.config_dir)
    verifier_cfg = _load_verifier_cfg(scenario)
    db_path = _resolve_db_path(config)
    write_sqlite_database(scenario.tables, db_path)
    trace: dict[str, Any]
    try:
        llm_config = load_llm_config(AGENT_ROOT / "config.toml")
        llm = LlmClient(llm_config)
        context = ContextManager()
        tools = build_ucwc_tool_registry(
            UcwcToolState(
                db_path=str(db_path),
                verifier_config=verifier_cfg,
                target_ue=config.ue_id,
            )
        )
        agent = Agent(
            config=AgentConfig(
                system_prompt=build_ucwc_system_prompt(
                    db_path=str(db_path),
                    target_ue=config.ue_id,
                ),
                max_steps=config.max_steps,
            ),
            llm=llm,
            context=context,
            tools=tools,
        )
        events = []
        final_content: str | None = None
        for event in agent.run_stream(_build_user_turn(config)):
            event_payload = {
                "session_id": event.session_id,
                "turn_id": event.turn_id,
                "sequence": event.sequence,
                "timestamp": event.timestamp,
                "type": event.type,
                "payload": event.payload,
            }
            events.append(event_payload)
            if event.type == "turn.completed":
                content = event.payload.get("content")
                final_content = content if isinstance(content, str) else None
        trace = {
            "trace_type": "ucwc_llm_nl2sql_tool_agent",
            "config": asdict(config),
            "terminal_status": "completed" if final_content else "incomplete",
            "scenario": scenario.metadata,
            "sqlite_db": str(db_path),
            "events": events,
            "final_content": final_content,
        }
    except Exception as error:
        if config.fallback_to_deterministic:
            return run_deterministic_agent(config, llm_failure=error, db_path=db_path)
        trace = {
            "trace_type": "ucwc_llm_nl2sql_tool_agent",
            "config": asdict(config),
            "terminal_status": "failed",
            "scenario": scenario.metadata,
            "sqlite_db": str(db_path),
            "events": [
                {
                    "sequence": 1,
                    "type": "turn.failed",
                    "payload": {
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                }
            ],
            "final_content": None,
        }
    if config.output_trace:
        write_json(trace, config.output_trace)
    return trace


def run_deterministic_agent(
    config: AgentLoopConfig,
    *,
    llm_failure: Exception | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    scenario = build_scenario(config.config_dir)
    verifier_cfg = _load_verifier_cfg(scenario)
    if db_path is None:
        db_path = _resolve_db_path(config)
        write_sqlite_database(scenario.tables, db_path)
    events: list[dict[str, Any]] = []
    _append_event(
        events,
        "inspect_state",
        {
            "ue_id": config.ue_id,
            "user_intent": config.user_intent,
            "scenario": scenario.metadata,
        },
    )
    if llm_failure is not None:
        _append_event(
            events,
            "llm_fallback",
            {
                "error_type": type(llm_failure).__name__,
                "message": str(llm_failure),
            },
        )

    plan = deterministic_propose_config_plan(scenario.tables, config.ue_id)
    _append_event(events, "propose_config", plan.to_record())
    feedback = verify_config_plan(plan, scenario.tables, verifier_cfg)
    _append_event(events, "verify_config", feedback)

    tried_bs_ids = {plan.serving_bs_id}
    repair_step = 0
    while not feedback["passed"] and repair_step < config.max_repair_steps:
        repair_step += 1
        repaired = repair_plan(plan, feedback, scenario.tables, tried_bs_ids)
        if repaired is None:
            _append_event(
                events,
                "repair_config",
                {
                    "repair_step": repair_step,
                    "status": "no_repair_available",
                    "failure_reasons": feedback.get("failure_reasons", []),
                },
            )
            break
        plan = repaired
        tried_bs_ids.add(plan.serving_bs_id)
        _append_event(
            events,
            "repair_config",
            {"repair_step": repair_step, "plan": plan.to_record()},
        )
        feedback = verify_config_plan(plan, scenario.tables, verifier_cfg)
        _append_event(events, "verify_config", feedback)

    terminal_status = "committed" if feedback["passed"] else "not_committed"
    _append_event(
        events,
        "commit_config",
        {
            "status": terminal_status,
            "plan": plan.to_record(),
            "verifier_passed": feedback["passed"],
            "failure_reasons": feedback.get("failure_reasons", []),
        },
    )
    trace = {
        "trace_type": "ucwc_deterministic_fallback_agent_loop",
        "config": asdict(config),
        "terminal_status": terminal_status,
        "sqlite_db": str(db_path),
        "events": events,
        "final_plan": plan.to_record(),
        "final_verifier_feedback": feedback,
    }
    if config.output_trace:
        write_json(trace, config.output_trace)
    return trace


def _build_user_turn(config: AgentLoopConfig) -> str:
    return (
        "Use UCWC tools to satisfy this session-level request.\n"
        f"target_ue: {config.ue_id}\n"
        f"user_intent: {config.user_intent}\n"
        "Required evidence: session_state, radio_candidates, capacity, qos.\n"
        "Required actions: schema_link, SQL generation/execution, verifier, and commit only if verifier passes.\n"
        "Return a concise final summary with commit status."
    )


def _resolve_db_path(config: AgentLoopConfig) -> Path:
    if config.db_path:
        return Path(config.db_path).expanduser().resolve()
    if config.output_trace:
        return Path(config.output_trace).expanduser().resolve().with_name("network_state.sqlite")
    return UCWC_ROOT / "outputs" / "agent_runtime" / "network_state.sqlite"


def repair_plan(
    plan: SessionConfigPlan,
    feedback: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    tried_bs_ids: set[str],
) -> SessionConfigPlan | None:
    reasons = set(feedback.get("failure_reasons", []))
    if "radio" in reasons or "resource" in reasons:
        alternative = _next_radio_candidate(tables, plan.ue_id, tried_bs_ids)
        if alternative is not None:
            return SessionConfigPlan(
                ue_id=plan.ue_id,
                serving_bs_id=str(alternative["bs_id"]),
                backup_bs_id=plan.serving_bs_id,
                bandwidth_quota_mhz=plan.bandwidth_quota_mhz,
                qos_profile=plan.qos_profile,
                handover_policy="radio_or_resource_repair",
                security_profile=plan.security_profile,
                source="minimal_agent_repair",
            )
    if "qos" in reasons:
        bs = _one(tables["base_station_state"], "bs_id", plan.serving_bs_id)
        if not bs:
            return None
        increased = min(float(bs["bandwidth_mhz"]) * 0.6, plan.bandwidth_quota_mhz * 1.5)
        if increased > plan.bandwidth_quota_mhz + 0.001:
            return SessionConfigPlan(
                ue_id=plan.ue_id,
                serving_bs_id=plan.serving_bs_id,
                backup_bs_id=plan.backup_bs_id,
                bandwidth_quota_mhz=round(increased, 3),
                qos_profile=plan.qos_profile,
                handover_policy=plan.handover_policy,
                security_profile=plan.security_profile,
                source="minimal_agent_repair",
            )
    if "cross_user" in reasons:
        reduced = max(1.0, plan.bandwidth_quota_mhz * 0.8)
        if reduced < plan.bandwidth_quota_mhz - 0.001:
            return SessionConfigPlan(
                ue_id=plan.ue_id,
                serving_bs_id=plan.serving_bs_id,
                backup_bs_id=plan.backup_bs_id,
                bandwidth_quota_mhz=round(reduced, 3),
                qos_profile=plan.qos_profile,
                handover_policy=plan.handover_policy,
                security_profile=plan.security_profile,
                source="minimal_agent_repair",
            )
    return None


def _next_radio_candidate(
    tables: dict[str, list[dict[str, Any]]],
    ue_id: str,
    tried_bs_ids: set[str],
) -> dict[str, Any] | None:
    candidates = sorted(
        [
            row
            for row in tables["radio_link_state"]
            if row["ue_id"] == ue_id and str(row["bs_id"]) not in tried_bs_ids
        ],
        key=lambda row: float(row["sinr_db"]),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_verifier_cfg(scenario: Any) -> dict[str, Any]:
    defaults = {
        "min_sinr_db": -3.0,
        "max_cell_utilization": 0.92,
        "max_cross_user_degradation_pct": 8.0,
    }
    defaults.update(dict(scenario.metadata.get("verifier_config", {}) or {}))
    return defaults


def _append_event(events: list[dict[str, Any]], action: str, detail: dict[str, Any]) -> None:
    events.append({"sequence": len(events) + 1, "action": action, "detail": detail})


def _one(rows: list[dict[str, Any]], key: str, value: Any) -> dict[str, Any] | None:
    for row in rows:
        if row.get(key) == value:
            return row
    return None
