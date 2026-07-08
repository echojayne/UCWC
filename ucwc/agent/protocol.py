"""Shared data contracts for the semantic UCWC admission agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]


@dataclass(frozen=True, slots=True)
class LlmConfig:
    """Connection details for an OpenAI-compatible chat-completions endpoint."""

    api_key: str
    base_url: str
    model: str
    profile: str
    source_path: Path
    timeout_s: float = 90.0

    def redacted(self) -> JsonDict:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "profile": self.profile,
            "source_path": str(self.source_path),
            "api_key": "***",
            "timeout_s": self.timeout_s,
        }


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Runtime settings for one UCWC admission-agent turn."""

    state_db: Path
    output_dir: Path
    working_db: Path | None = None
    max_rounds: int = 2
    bs_top_k: int = 5
    max_sql_queries: int = 4
    max_sql_rows: int = 40
    max_candidates: int = 250
    commit: bool = True
    source: str = "ucwc_agentic_nl2sql"


@dataclass(frozen=True, slots=True)
class SqlQueryResult:
    """Capped output from one validated read-only SQL query."""

    sql: str
    columns: list[str]
    rows: list[JsonDict]
    row_count: int
    truncated: bool = False

    def compact(self, max_rows: int = 5) -> JsonDict:
        return {
            "sql": self.sql,
            "columns": self.columns,
            "row_count": self.row_count,
            "sample_rows": self.rows[:max_rows],
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    """One concrete semantic-communication admission candidate."""

    candidate_id: str
    request_id: str
    ue_id: str
    serving_bs_id: str
    semantic_config_id: str
    phy_mode_id: str
    bandwidth_mhz: float
    bandwidth_multiplier: float
    source: str = "candidate_grid"

    def as_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Verifier-facing metrics and decision for one candidate."""

    candidate: CandidateConfig
    verifier_passed: bool
    failure_reasons: list[str]
    predicted_task_score: float
    min_task_score: float
    score_margin: float
    total_latency_ms: float
    max_total_latency_ms: float
    latency_margin_ms: float
    transmission_latency_ms: float
    required_bandwidth_mhz: float
    residual_bandwidth_mhz: float
    bandwidth_margin_mhz: float
    utility: float
    radio_rank: int
    snr_db: float
    sinr_db: float
    encoder_depth: int
    quantization_bits: int
    ldpc_code_rate: float
    qam_order: int
    task_type: str

    @property
    def primary_failure(self) -> str | None:
        return None if self.verifier_passed else self.failure_reasons[0]

    def as_dict(self) -> JsonDict:
        data = asdict(self)
        data.update(self.candidate.as_dict())
        data.pop("candidate", None)
        data["primary_failure"] = self.primary_failure
        return data

    def compact(self) -> JsonDict:
        return {
            "candidate_id": self.candidate.candidate_id,
            "serving_bs_id": self.candidate.serving_bs_id,
            "semantic_config_id": self.candidate.semantic_config_id,
            "phy_mode_id": self.candidate.phy_mode_id,
            "bandwidth_mhz": round(self.candidate.bandwidth_mhz, 6),
            "verifier_passed": self.verifier_passed,
            "failure_reasons": self.failure_reasons,
            "predicted_task_score": round(self.predicted_task_score, 6),
            "score_margin": round(self.score_margin, 6),
            "total_latency_ms": round(self.total_latency_ms, 6),
            "latency_margin_ms": round(self.latency_margin_ms, 6),
            "required_bandwidth_mhz": round(self.required_bandwidth_mhz, 6),
            "bandwidth_margin_mhz": round(self.bandwidth_margin_mhz, 6),
            "utility": round(self.utility, 6),
            "radio_rank": self.radio_rank,
            "sinr_db": round(self.sinr_db, 6),
        }


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    """Compact state handed back to the model after deterministic evaluation."""

    total_candidates: int
    feasible_count: int
    top_candidates: list[JsonDict]
    pareto_frontier: list[JsonDict]
    failure_summary: dict[str, int]


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Final result from one admission-agent run."""

    request_id: str
    selected_candidate: CandidateEvaluation | None
    committed: bool
    commit_record: JsonDict | None
    final_message: str
    trace_path: Path
    candidate_csv_path: Path
    sql_results: list[SqlQueryResult] = field(default_factory=list)
    llm_grounding: JsonDict = field(default_factory=dict)
    llm_selection: JsonDict = field(default_factory=dict)

    def as_dict(self) -> JsonDict:
        return {
            "request_id": self.request_id,
            "selected_candidate": (
                self.selected_candidate.as_dict() if self.selected_candidate else None
            ),
            "committed": self.committed,
            "commit_record": self.commit_record,
            "final_message": self.final_message,
            "trace_path": str(self.trace_path),
            "candidate_csv_path": str(self.candidate_csv_path),
            "sql_results": [result.compact() for result in self.sql_results],
            "llm_grounding": self.llm_grounding,
            "llm_selection": self.llm_selection,
        }
