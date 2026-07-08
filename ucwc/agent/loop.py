"""Orchestrated UCWC admission-agent loop."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from ucwc.agent import candidates
from ucwc.agent.context import AgentContext
from ucwc.agent.llm import OpenAICompatibleChatClient
from ucwc.agent.prompts import (
    build_system_prompt,
    final_user_prompt,
    grounding_user_prompt,
    selection_user_prompt,
)
from ucwc.agent.protocol import AgentConfig, AgentRunResult, CandidateEvaluation
from ucwc.agent.sql_tools import (
    connect_writable,
    default_request_evidence_sql,
    first_request_id,
    schema_prompt_text,
    validate_readonly_sql,
)
from ucwc.agent.tools import UCWCToolbox
from ucwc.agent.trace import TraceWriter


class UCWCAdmissionAgent:
    """Single-agent controller around NL2SQL grounding and verifier-gated search."""

    def __init__(
        self,
        *,
        config: AgentConfig,
        llm: OpenAICompatibleChatClient,
    ) -> None:
        self.config = config
        self.llm = llm

    def run(self, user_request: str, *, request_id: str | None = None) -> AgentRunResult:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        working_db = self._prepare_working_db()
        trace = TraceWriter(output_dir / "trace.jsonl")
        candidate_csv_path = output_dir / "candidate_evaluations.csv"
        trace.write(
            "turn.started",
            {
                "user_request": user_request,
                "request_id_hint": request_id,
                "state_db": str(self.config.state_db),
                "working_db": str(working_db),
                "llm": self.llm.config.redacted(),
            },
        )

        with connect_writable(working_db) as connection:
            toolbox = UCWCToolbox(connection)
            schema = toolbox.inspect_schema()
            system_prompt = build_system_prompt(schema_prompt_text(schema))
            context = AgentContext(user_request=user_request, request_id=request_id, schema=schema)

            grounding, grounding_text = self._ground(system_prompt, user_request, request_id, trace)
            target_request_id = self._resolve_request_id(connection, request_id, grounding)
            context.request_id = target_request_id
            trace.write(
                "llm.grounding.completed",
                {
                    "raw_text": grounding_text,
                    "json": grounding,
                    "resolved_request_id": target_request_id,
                },
            )

            sql_results = self._run_grounding_sql(
                toolbox,
                grounding=grounding,
                request_id=target_request_id,
                trace=trace,
            )
            context.sql_results = sql_results

            bandwidth_multipliers = self._bandwidth_multipliers(grounding)
            bs_top_k = self._bs_top_k(grounding)
            selected: CandidateEvaluation | None = None
            selection: dict[str, Any] = {}

            for round_index in range(1, self.config.max_rounds + 1):
                candidate_list = toolbox.build_candidate_grid(
                    target_request_id,
                    bs_top_k=bs_top_k,
                    bandwidth_multipliers=bandwidth_multipliers,
                    max_candidates=self.config.max_candidates,
                )
                evaluations = toolbox.evaluate_candidates(candidate_list)
                summary = candidates.summarize_evaluations(evaluations)
                context.candidate_summary = summary
                candidates.write_candidate_csv(candidate_csv_path, evaluations)
                trace.write(
                    "candidates.evaluated",
                    {
                        "round": round_index,
                        "candidate_count": len(evaluations),
                        "summary": summary,
                        "candidate_csv_path": str(candidate_csv_path),
                    },
                )

                try:
                    selection, selection_text = self._select(
                        system_prompt=system_prompt,
                        user_request=user_request,
                        context=context,
                        trace=trace,
                    )
                    trace.write(
                        "llm.selection.completed",
                        {"round": round_index, "raw_text": selection_text, "json": selection},
                    )
                except Exception as error:
                    selection = _deterministic_selection(context, error)
                    trace.write(
                        "llm.selection.failed",
                        {"round": round_index, "error": str(error), "fallback": selection},
                    )
                selected = candidates.select_best_candidate(
                    evaluations,
                    _string_or_none(selection.get("selected_candidate_id")),
                )
                if selected is not None:
                    context.selected_candidate_id = selected.candidate.candidate_id
                    trace.write("candidate.selected", selected.compact())
                    break
                if round_index < self.config.max_rounds:
                    bs_top_k = min(5, max(bs_top_k, 5))
                    bandwidth_multipliers = sorted(
                        set(bandwidth_multipliers + [1.5, 2.0])
                    )
                    trace.write(
                        "search.refined",
                        {
                            "reason": "no feasible candidate selected",
                            "bs_top_k": bs_top_k,
                            "bandwidth_multipliers": bandwidth_multipliers,
                        },
                    )

            committed = False
            commit_record: dict[str, object] | None = None
            if selected is not None and self.config.commit:
                fresh = toolbox.evaluate_candidate(selected.candidate)
                trace.write("candidate.fresh_verifier", fresh.compact())
                if fresh.verifier_passed:
                    commit_record = toolbox.commit_verified_candidate(
                        fresh,
                        source=self.config.source,
                    )
                    committed = True
                    selected = fresh
                    trace.write("candidate.committed", commit_record)
                else:
                    selected = fresh
                    trace.write("candidate.commit_blocked", fresh.compact())

            failure_summary = (
                context.candidate_summary.failure_summary if context.candidate_summary else {}
            )
            final_message = self._final_message(
                system_prompt=system_prompt,
                user_request=user_request,
                selected=selected,
                candidate_state=context.candidate_state_for_llm(),
                committed=committed,
                commit_record=commit_record,
                failure_summary=failure_summary,
                trace=trace,
            )
            result = AgentRunResult(
                request_id=target_request_id,
                selected_candidate=selected,
                committed=committed,
                commit_record=commit_record,
                final_message=final_message,
                trace_path=trace.path,
                candidate_csv_path=candidate_csv_path,
                sql_results=sql_results,
                llm_grounding=grounding,
                llm_selection=selection,
            )
            trace.write("turn.completed", result.as_dict())
            (output_dir / "result.json").write_text(
                json.dumps(result.as_dict(), indent=2, ensure_ascii=True, sort_keys=True),
                encoding="utf-8",
            )
            return result

    def _prepare_working_db(self) -> Path:
        if self.config.working_db is not None:
            working_db = self.config.working_db
        else:
            working_db = self.config.output_dir / "state_working.sqlite"
        working_db.parent.mkdir(parents=True, exist_ok=True)
        if working_db.resolve() != self.config.state_db.resolve():
            shutil.copy2(self.config.state_db, working_db)
        return working_db

    def _ground(
        self,
        system_prompt: str,
        user_request: str,
        request_id: str | None,
        trace: TraceWriter,
    ) -> tuple[dict[str, Any], str]:
        prompt = grounding_user_prompt(user_request, request_id)
        trace.write("llm.grounding.started", {"prompt": prompt})
        return self.llm.complete_json(system_prompt=system_prompt, user_prompt=prompt)

    def _select(
        self,
        *,
        system_prompt: str,
        user_request: str,
        context: AgentContext,
        trace: TraceWriter,
    ) -> tuple[dict[str, Any], str]:
        prompt = selection_user_prompt(
            user_request=user_request,
            evidence=context.evidence_for_llm(),
            candidate_state=context.candidate_state_for_llm(),
        )
        trace.write("llm.selection.started", {"prompt": prompt})
        return self.llm.complete_json(system_prompt=system_prompt, user_prompt=prompt)

    def _final_message(
        self,
        *,
        system_prompt: str,
        user_request: str,
        selected: CandidateEvaluation | None,
        candidate_state: dict[str, Any],
        committed: bool,
        commit_record: dict[str, object] | None,
        failure_summary: dict[str, int],
        trace: TraceWriter,
    ) -> str:
        prompt = final_user_prompt(
            user_request=user_request,
            selected_candidate=selected.compact() if selected else None,
            candidate_state=candidate_state,
            committed=committed,
            commit_record=commit_record,
            failure_summary=failure_summary,
        )
        trace.write("llm.final.started", {"prompt": prompt})
        try:
            message = self.llm.complete(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=1200,
            )
        except Exception as error:
            trace.write("llm.final.failed", {"error": str(error)})
            return _deterministic_final(selected, committed, commit_record, failure_summary)
        trace.write("llm.final.completed", {"message": message})
        return message.strip()

    def _run_grounding_sql(
        self,
        toolbox: UCWCToolbox,
        *,
        grounding: dict[str, Any],
        request_id: str,
        trace: TraceWriter,
    ):
        sql_values = grounding.get("sql_queries")
        queries = [str(sql) for sql in sql_values] if isinstance(sql_values, list) else []
        queries.append(default_request_evidence_sql(request_id))
        results = []
        seen: set[str] = set()
        for sql in queries:
            normalized = sql.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ok, reason = validate_readonly_sql(toolbox.connection, normalized)
            if not ok:
                trace.write("sql.rejected", {"sql": normalized, "reason": reason})
                continue
            try:
                result = toolbox.execute_readonly_sql(
                    normalized,
                    max_rows=self.config.max_sql_rows,
                )
            except Exception as error:
                trace.write("sql.failed", {"sql": normalized, "error": str(error)})
                continue
            results.append(result)
            trace.write("sql.executed", result.compact(max_rows=5))
            if len(results) >= self.config.max_sql_queries:
                break
        if not results:
            raise RuntimeError("No grounding SQL query executed successfully")
        return results

    def _resolve_request_id(
        self,
        connection,
        request_id_hint: str | None,
        grounding: dict[str, Any],
    ) -> str:
        for value in (
            request_id_hint,
            _string_or_none(grounding.get("target_request_id")),
            _extract_request_id(str(grounding)),
        ):
            if value and _request_exists(connection, value):
                return value
        return first_request_id(connection)

    def _bs_top_k(self, grounding: dict[str, Any]) -> int:
        policy = grounding.get("search_policy")
        if isinstance(policy, dict):
            value = policy.get("bs_top_k")
            if isinstance(value, int):
                return min(5, max(1, value))
        return self.config.bs_top_k

    def _bandwidth_multipliers(self, grounding: dict[str, Any]) -> list[float]:
        policy = grounding.get("search_policy")
        if isinstance(policy, dict):
            value = policy.get("bandwidth_multipliers")
            if isinstance(value, list):
                parsed = [float(item) for item in value if isinstance(item, (int, float))]
                parsed = [item for item in parsed if item > 0.0]
                if parsed:
                    return sorted(set(parsed))
        return [1.0, 1.1, 1.25]


def _request_exists(connection, request_id: str) -> bool:
    if not re.fullmatch(r"req_[0-9]{4,}", request_id):
        return False
    row = connection.execute(
        "SELECT request_id FROM ue_request_queue WHERE request_id = ?",
        (request_id,),
    ).fetchone()
    return row is not None


def _extract_request_id(text: str) -> str | None:
    match = re.search(r"req_[0-9]{4,}", text)
    return match.group(0) if match else None


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value.strip() and value.strip().lower() != "null":
        return value.strip()
    return None


def _deterministic_selection(context: AgentContext, error: Exception) -> dict[str, Any]:
    summary = context.candidate_summary
    if summary is not None and summary.feasible_count > 0:
        return {
            "selected_candidate_id": None,
            "decision_summary": (
                "LLM selection failed; falling back to the highest-utility verifier-passed "
                "candidate from deterministic evaluation."
            ),
            "repair_or_refine": None,
            "selection_error": str(error),
        }
    failure_summary = summary.failure_summary if summary is not None else {}
    return {
        "selected_candidate_id": None,
        "decision_summary": "LLM selection failed and no verifier-passed candidate is available.",
        "repair_or_refine": f"Verifier failure summary: {failure_summary}",
        "selection_error": str(error),
    }


def _deterministic_final(
    selected: CandidateEvaluation | None,
    committed: bool,
    commit_record: dict[str, object] | None,
    failure_summary: dict[str, int],
) -> str:
    if selected is None:
        return f"No verifier-passed candidate was found. Failure summary: {failure_summary}"
    status = "committed" if committed else "not committed"
    return (
        f"Selected {selected.candidate.candidate_id} and {status}: "
        f"BS={selected.candidate.serving_bs_id}, "
        f"semantic={selected.candidate.semantic_config_id}, "
        f"PHY={selected.candidate.phy_mode_id}, "
        f"bandwidth={selected.candidate.bandwidth_mhz:.4f} MHz, "
        f"score={selected.predicted_task_score:.4f}, "
        f"latency={selected.total_latency_ms:.4f} ms, "
        f"verifier_passed={selected.verifier_passed}, "
        f"commit_record={commit_record}."
    )
