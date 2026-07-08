"""Candidate mining and batch evaluation tools for semantic UCWC admission."""

from __future__ import annotations

import csv
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable

from ucwc.agent.protocol import CandidateConfig, CandidateEvaluation, CandidateSummary
from ucwc.state import required_bandwidth_mhz
from ucwc.verifier.core import evaluate_candidate_from_rows


def build_candidate_grid(
    connection: sqlite3.Connection,
    request_id: str,
    *,
    bs_top_k: int = 5,
    bandwidth_multipliers: Iterable[float] = (1.0, 1.1, 1.25),
    max_candidates: int = 250,
) -> list[CandidateConfig]:
    """Generate concrete BS/semantic/PHY/bandwidth candidates for one request."""

    request = _one(
        connection,
        "SELECT * FROM ue_request_queue WHERE request_id = ?",
        (request_id,),
        f"request not found: {request_id}",
    )
    radios = connection.execute(
        """
        SELECT * FROM radio_link_state
        WHERE ue_id = ?
        ORDER BY radio_rank ASC
        LIMIT ?
        """,
        (request["ue_id"], bs_top_k),
    ).fetchall()
    semantics = connection.execute(
        """
        SELECT * FROM semantic_config_catalog
        ORDER BY encoder_depth ASC, quantization_bits ASC
        """
    ).fetchall()
    phys = connection.execute(
        """
        SELECT * FROM phy_mode_catalog
        ORDER BY ldpc_code_rate ASC, qam_order ASC
        """
    ).fetchall()
    resource_efficiency = _resource_efficiency(connection)
    candidates: list[CandidateConfig] = []
    for radio in radios:
        for semantic_config in semantics:
            fixed_latency = float(semantic_config["fixed_latency_ms"])
            base_latency = (
                float(semantic_config["encoding_latency_ms"])
                + float(semantic_config["decoding_latency_ms"])
                + fixed_latency
            )
            tx_budget_ms = float(request["max_total_latency_ms"]) - base_latency
            for phy in phys:
                recommended = required_bandwidth_mhz(
                    payload_bits=float(semantic_config["payload_bits"]),
                    tx_budget_ms=tx_budget_ms,
                    ldpc_code_rate=float(phy["ldpc_code_rate"]),
                    qam_order=int(phy["qam_order"]),
                    resource_efficiency=resource_efficiency,
                )
                bandwidth_base = _safe_positive_bandwidth(recommended)
                for multiplier in bandwidth_multipliers:
                    bandwidth = _round_bandwidth(bandwidth_base * float(multiplier))
                    candidate_id = (
                        f"cand_{request_id}_{radio['bs_id']}_{semantic_config['config_id']}_"
                        f"{phy['phy_mode_id']}_m{str(multiplier).replace('.', 'p')}"
                    )
                    candidates.append(
                        CandidateConfig(
                            candidate_id=candidate_id,
                            request_id=str(request["request_id"]),
                            ue_id=str(request["ue_id"]),
                            serving_bs_id=str(radio["bs_id"]),
                            semantic_config_id=str(semantic_config["config_id"]),
                            phy_mode_id=str(phy["phy_mode_id"]),
                            bandwidth_mhz=bandwidth,
                            bandwidth_multiplier=float(multiplier),
                        )
                    )
                    if len(candidates) >= max_candidates:
                        return candidates
    return candidates


def evaluate_candidate(
    connection: sqlite3.Connection,
    candidate: CandidateConfig,
) -> CandidateEvaluation:
    """Load the required rows and verify one candidate."""

    request = _one(
        connection,
        "SELECT * FROM ue_request_queue WHERE request_id = ? AND ue_id = ?",
        (candidate.request_id, candidate.ue_id),
        f"request/UE not found: {candidate.request_id}/{candidate.ue_id}",
    )
    radio = _one(
        connection,
        "SELECT * FROM radio_link_state WHERE ue_id = ? AND bs_id = ?",
        (candidate.ue_id, candidate.serving_bs_id),
        f"radio link not found: {candidate.ue_id}/{candidate.serving_bs_id}",
    )
    semantic_config = _one(
        connection,
        "SELECT * FROM semantic_config_catalog WHERE config_id = ?",
        (candidate.semantic_config_id,),
        f"semantic config not found: {candidate.semantic_config_id}",
    )
    phy_mode = _one(
        connection,
        "SELECT * FROM phy_mode_catalog WHERE phy_mode_id = ?",
        (candidate.phy_mode_id,),
        f"PHY mode not found: {candidate.phy_mode_id}",
    )
    base_station = _one(
        connection,
        "SELECT * FROM base_station_state WHERE bs_id = ?",
        (candidate.serving_bs_id,),
        f"base station not found: {candidate.serving_bs_id}",
    )
    residual = _residual_bandwidth(connection, candidate.serving_bs_id, base_station)
    conflict = _has_active_conflict(connection, candidate.request_id, candidate.ue_id)
    return evaluate_candidate_from_rows(
        candidate,
        request=request,
        radio=radio,
        semantic_config=semantic_config,
        phy_mode=phy_mode,
        base_station=base_station,
        residual_bandwidth_mhz=residual,
        resource_efficiency=_resource_efficiency(connection),
        active_conflict=conflict,
    )


def evaluate_candidates(
    connection: sqlite3.Connection,
    candidates: Iterable[CandidateConfig],
) -> list[CandidateEvaluation]:
    return [evaluate_candidate(connection, candidate) for candidate in candidates]


def summarize_evaluations(
    evaluations: list[CandidateEvaluation],
    *,
    top_k: int = 8,
    frontier_k: int = 10,
) -> CandidateSummary:
    sorted_evals = sorted(
        evaluations,
        key=lambda item: (item.verifier_passed, item.utility),
        reverse=True,
    )
    failure_counter: Counter[str] = Counter()
    for evaluation in evaluations:
        if not evaluation.verifier_passed:
            for reason in evaluation.failure_reasons:
                failure_counter[reason] += 1
    return CandidateSummary(
        total_candidates=len(evaluations),
        feasible_count=sum(1 for item in evaluations if item.verifier_passed),
        top_candidates=[item.compact() for item in sorted_evals[:top_k]],
        pareto_frontier=[item.compact() for item in pareto_frontier(evaluations)[:frontier_k]],
        failure_summary=dict(failure_counter),
    )


def pareto_frontier(evaluations: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    """Return non-dominated candidates among feasible evaluations."""

    feasible = [item for item in evaluations if item.verifier_passed]
    frontier: list[CandidateEvaluation] = []
    for item in feasible:
        dominated = False
        for other in feasible:
            if other is item:
                continue
            better_or_equal = (
                other.predicted_task_score >= item.predicted_task_score
                and other.total_latency_ms <= item.total_latency_ms
                and other.candidate.bandwidth_mhz <= item.candidate.bandwidth_mhz
            )
            strictly_better = (
                other.predicted_task_score > item.predicted_task_score
                or other.total_latency_ms < item.total_latency_ms
                or other.candidate.bandwidth_mhz < item.candidate.bandwidth_mhz
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(item)
    return sorted(frontier, key=lambda item: item.utility, reverse=True)


def write_candidate_csv(path: str | Path, evaluations: list[CandidateEvaluation]) -> None:
    """Persist full candidate metrics for audit and plotting."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [evaluation.as_dict() for evaluation in evaluations]
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def select_best_candidate(
    evaluations: list[CandidateEvaluation],
    candidate_id: str | None = None,
) -> CandidateEvaluation | None:
    if candidate_id:
        for evaluation in evaluations:
            if evaluation.candidate.candidate_id == candidate_id and evaluation.verifier_passed:
                return evaluation
    feasible = [item for item in evaluations if item.verifier_passed]
    if feasible:
        return sorted(feasible, key=lambda item: item.utility, reverse=True)[0]
    return None


def _resource_efficiency(connection: sqlite3.Connection) -> float:
    row = connection.execute(
        "SELECT value FROM scenario_metadata WHERE key = 'resource_efficiency'"
    ).fetchone()
    return 0.85 if row is None else float(row["value"])


def _residual_bandwidth(
    connection: sqlite3.Connection,
    bs_id: str,
    base_station: sqlite3.Row,
) -> float:
    active = connection.execute(
        "SELECT COALESCE(SUM(bandwidth_mhz), 0.0) AS used FROM active_session WHERE serving_bs_id = ?",
        (bs_id,),
    ).fetchone()
    active_used = 0.0 if active is None else float(active["used"])
    recorded_used = float(base_station["used_bandwidth_mhz"])
    used = max(active_used, recorded_used)
    return float(base_station["bandwidth_budget_mhz"]) - used


def _has_active_conflict(connection: sqlite3.Connection, request_id: str, ue_id: str) -> bool:
    row = connection.execute(
        """
        SELECT session_id FROM active_session
        WHERE request_id = ? OR ue_id = ?
        LIMIT 1
        """,
        (request_id, ue_id),
    ).fetchone()
    return row is not None


def _one(
    connection: sqlite3.Connection,
    sql: str,
    params: tuple[object, ...],
    error: str,
) -> sqlite3.Row:
    row = connection.execute(sql, params).fetchone()
    if row is None:
        raise ValueError(error)
    return row


def _safe_positive_bandwidth(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return 0.0
    return value


def _round_bandwidth(value: float) -> float:
    if not math.isfinite(value):
        return value
    return math.ceil(value * 10000.0) / 10000.0
