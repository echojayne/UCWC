"""Domain tool facade used by the UCWC admission loop."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from typing import Any

from ucwc.agent import candidates
from ucwc.agent.protocol import CandidateConfig, CandidateEvaluation, SqlQueryResult
from ucwc.agent.sql_tools import execute_readonly_sql, inspect_schema
from ucwc.verifier.core import commit_verified_candidate


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str


class UCWCToolbox:
    """Small explicit tool surface for the agent loop."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition("inspect_schema", "Inspect semantic UCWC SQLite tables."),
            ToolDefinition("execute_readonly_sql", "Run validated bounded SELECT/WITH SQL."),
            ToolDefinition("build_candidate_grid", "Enumerate BS/semantic/PHY/bandwidth candidates."),
            ToolDefinition("evaluate_candidates", "Compute deterministic candidate metrics."),
            ToolDefinition("summarize_evaluations", "Build top-k, Pareto, and failure summaries."),
            ToolDefinition("commit_verified_candidate", "Commit only a fresh verifier-passed candidate."),
        ]

    def inspect_schema(self) -> dict[str, list[str]]:
        return inspect_schema(self.connection)

    def execute_readonly_sql(self, sql: str, *, max_rows: int) -> SqlQueryResult:
        return execute_readonly_sql(self.connection, sql, max_rows=max_rows)

    def build_candidate_grid(
        self,
        request_id: str,
        *,
        bs_top_k: int,
        bandwidth_multipliers: list[float],
        max_candidates: int,
    ) -> list[CandidateConfig]:
        return candidates.build_candidate_grid(
            self.connection,
            request_id,
            bs_top_k=bs_top_k,
            bandwidth_multipliers=bandwidth_multipliers,
            max_candidates=max_candidates,
        )

    def evaluate_candidates(
        self,
        candidate_list: list[CandidateConfig],
    ) -> list[CandidateEvaluation]:
        return candidates.evaluate_candidates(self.connection, candidate_list)

    def evaluate_candidate(self, candidate: CandidateConfig) -> CandidateEvaluation:
        return candidates.evaluate_candidate(self.connection, candidate)

    def summarize_evaluations(
        self,
        evaluations: list[CandidateEvaluation],
    ) -> dict[str, Any]:
        return asdict(candidates.summarize_evaluations(evaluations))

    def commit_verified_candidate(
        self,
        evaluation: CandidateEvaluation,
        *,
        source: str,
    ) -> dict[str, object]:
        return commit_verified_candidate(self.connection, evaluation, source=source)
