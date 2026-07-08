"""Bounded model context for the UCWC admission agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ucwc.agent.protocol import CandidateSummary, SqlQueryResult


@dataclass(slots=True)
class AgentContext:
    """Compact state passed between orchestration stages.

    Full SQL outputs and candidate metrics are persisted to trace/CSV. The LLM
    only receives bounded summaries so the loop remains stable as the candidate
    grid grows.
    """

    user_request: str
    request_id: str | None = None
    schema: dict[str, list[str]] = field(default_factory=dict)
    sql_results: list[SqlQueryResult] = field(default_factory=list)
    candidate_summary: CandidateSummary | None = None
    selected_candidate_id: str | None = None

    def evidence_for_llm(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "sql_results": [result.compact(max_rows=5) for result in self.sql_results],
        }

    def candidate_state_for_llm(self) -> dict[str, Any]:
        if self.candidate_summary is None:
            return {}
        return {
            "total_candidates": self.candidate_summary.total_candidates,
            "feasible_count": self.candidate_summary.feasible_count,
            "top_candidates": self.candidate_summary.top_candidates,
            "pareto_frontier": self.candidate_summary.pareto_frontier,
            "failure_summary": self.candidate_summary.failure_summary,
        }
