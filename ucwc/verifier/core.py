"""Deterministic verifier for semantic UCWC admission candidates."""

from __future__ import annotations

import math
import sqlite3
from uuid import uuid4

from ucwc.agent.protocol import CandidateConfig, CandidateEvaluation
from ucwc.state import predict_task_score, required_bandwidth_mhz


def evaluate_candidate_from_rows(
    candidate: CandidateConfig,
    *,
    request: sqlite3.Row,
    radio: sqlite3.Row,
    semantic_config: sqlite3.Row,
    phy_mode: sqlite3.Row,
    base_station: sqlite3.Row,
    residual_bandwidth_mhz: float,
    resource_efficiency: float,
    active_conflict: bool,
) -> CandidateEvaluation:
    """Evaluate one candidate with deterministic semantic/link/resource rules."""

    task_score = predict_task_score(
        encoder_depth=int(semantic_config["encoder_depth"]),
        quantization_bits=int(semantic_config["quantization_bits"]),
        ldpc_code_rate=float(phy_mode["ldpc_code_rate"]),
        qam_order=int(phy_mode["qam_order"]),
        snr_db=float(radio["sinr_db"]),
        task_type=str(request["task_type"]),
        reference_snr_db=float(phy_mode["reference_snr_db"]),
    )
    fixed_latency = float(semantic_config["fixed_latency_ms"])
    encoding_latency = float(semantic_config["encoding_latency_ms"])
    decoding_latency = float(semantic_config["decoding_latency_ms"])
    deadline = float(request["max_total_latency_ms"])
    tx_budget_ms = deadline - encoding_latency - decoding_latency - fixed_latency
    recommended = required_bandwidth_mhz(
        payload_bits=float(semantic_config["payload_bits"]),
        tx_budget_ms=tx_budget_ms,
        ldpc_code_rate=float(phy_mode["ldpc_code_rate"]),
        qam_order=int(phy_mode["qam_order"]),
        resource_efficiency=resource_efficiency,
    )
    tx_latency_ms = _transmission_latency_ms(
        payload_bits=float(semantic_config["payload_bits"]),
        bandwidth_mhz=float(candidate.bandwidth_mhz),
        ldpc_code_rate=float(phy_mode["ldpc_code_rate"]),
        qam_order=int(phy_mode["qam_order"]),
        resource_efficiency=resource_efficiency,
    )
    total_latency_ms = encoding_latency + decoding_latency + fixed_latency + tx_latency_ms
    score_margin = task_score - float(request["min_task_score"])
    latency_margin_ms = deadline - total_latency_ms
    bandwidth_margin_mhz = residual_bandwidth_mhz - float(candidate.bandwidth_mhz)

    failures: list[str] = []
    if active_conflict:
        failures.append("duplicate_active_session")
    if not math.isfinite(candidate.bandwidth_mhz) or candidate.bandwidth_mhz <= 0.0:
        failures.append("nonpositive_bandwidth")
    if score_margin < -1e-9:
        failures.append("semantic_score_below_min")
    if latency_margin_ms < -1e-9:
        failures.append("latency_exceeded")
    if bandwidth_margin_mhz < -1e-9:
        failures.append("bandwidth_exceeded")
    passed = not failures
    utility = _utility(
        passed=passed,
        task_score=task_score,
        latency_margin_ms=latency_margin_ms,
        bandwidth_mhz=float(candidate.bandwidth_mhz),
        bandwidth_margin_mhz=bandwidth_margin_mhz,
        radio_rank=int(radio["radio_rank"]),
        failure_count=len(failures),
    )
    return CandidateEvaluation(
        candidate=candidate,
        verifier_passed=passed,
        failure_reasons=failures,
        predicted_task_score=task_score,
        min_task_score=float(request["min_task_score"]),
        score_margin=score_margin,
        total_latency_ms=total_latency_ms,
        max_total_latency_ms=deadline,
        latency_margin_ms=latency_margin_ms,
        transmission_latency_ms=tx_latency_ms,
        required_bandwidth_mhz=recommended,
        residual_bandwidth_mhz=residual_bandwidth_mhz,
        bandwidth_margin_mhz=bandwidth_margin_mhz,
        utility=utility,
        radio_rank=int(radio["radio_rank"]),
        snr_db=float(radio["snr_db"]),
        sinr_db=float(radio["sinr_db"]),
        encoder_depth=int(semantic_config["encoder_depth"]),
        quantization_bits=int(semantic_config["quantization_bits"]),
        ldpc_code_rate=float(phy_mode["ldpc_code_rate"]),
        qam_order=int(phy_mode["qam_order"]),
        task_type=str(request["task_type"]),
    )


def commit_verified_candidate(
    connection: sqlite3.Connection,
    evaluation: CandidateEvaluation,
    *,
    source: str,
) -> dict[str, object]:
    """Commit a freshly verified candidate to active_session and config_history."""

    if not evaluation.verifier_passed:
        raise ValueError(f"Refusing to commit failed candidate: {evaluation.failure_reasons}")
    candidate = evaluation.candidate
    request = connection.execute(
        "SELECT arrival_order FROM ue_request_queue WHERE request_id = ?",
        (candidate.request_id,),
    ).fetchone()
    if request is None:
        raise ValueError(f"request not found at commit time: {candidate.request_id}")
    conflict = connection.execute(
        """
        SELECT session_id FROM active_session
        WHERE request_id = ? OR ue_id = ?
        LIMIT 1
        """,
        (candidate.request_id, candidate.ue_id),
    ).fetchone()
    if conflict is not None:
        raise ValueError(f"active session already exists for request/UE: {candidate.request_id}")

    session_id = f"session_{uuid4().hex[:12]}"
    history_id = f"hist_{uuid4().hex[:12]}"
    connection.execute(
        """
        INSERT INTO active_session (
          session_id,
          request_id,
          ue_id,
          serving_bs_id,
          semantic_config_id,
          phy_mode_id,
          bandwidth_mhz,
          admitted_at_order,
          source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            candidate.request_id,
            candidate.ue_id,
            candidate.serving_bs_id,
            candidate.semantic_config_id,
            candidate.phy_mode_id,
            candidate.bandwidth_mhz,
            int(request["arrival_order"]),
            source,
        ),
    )
    connection.execute(
        """
        INSERT INTO config_history (
          history_id,
          request_id,
          attempted_at_order,
          serving_bs_id,
          semantic_config_id,
          phy_mode_id,
          bandwidth_mhz,
          verifier_passed,
          failure_reason,
          source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            history_id,
            candidate.request_id,
            int(request["arrival_order"]),
            candidate.serving_bs_id,
            candidate.semantic_config_id,
            candidate.phy_mode_id,
            candidate.bandwidth_mhz,
            1,
            None,
            source,
        ),
    )
    current_used = connection.execute(
        "SELECT used_bandwidth_mhz FROM base_station_state WHERE bs_id = ?",
        (candidate.serving_bs_id,),
    ).fetchone()
    used = 0.0 if current_used is None else float(current_used["used_bandwidth_mhz"])
    connection.execute(
        """
        UPDATE base_station_state
        SET used_bandwidth_mhz = ?
        WHERE bs_id = ?
        """,
        (used + candidate.bandwidth_mhz, candidate.serving_bs_id),
    )
    connection.commit()
    return {
        "session_id": session_id,
        "history_id": history_id,
        "request_id": candidate.request_id,
        "ue_id": candidate.ue_id,
        "serving_bs_id": candidate.serving_bs_id,
        "semantic_config_id": candidate.semantic_config_id,
        "phy_mode_id": candidate.phy_mode_id,
        "bandwidth_mhz": candidate.bandwidth_mhz,
        "source": source,
    }


def _transmission_latency_ms(
    *,
    payload_bits: float,
    bandwidth_mhz: float,
    ldpc_code_rate: float,
    qam_order: int,
    resource_efficiency: float,
) -> float:
    if bandwidth_mhz <= 0.0:
        return math.inf
    effective_bits_per_hz = ldpc_code_rate * math.log2(qam_order) * resource_efficiency
    if effective_bits_per_hz <= 0.0:
        return math.inf
    return payload_bits / (bandwidth_mhz * 1e6 * effective_bits_per_hz) * 1000.0


def _utility(
    *,
    passed: bool,
    task_score: float,
    latency_margin_ms: float,
    bandwidth_mhz: float,
    bandwidth_margin_mhz: float,
    radio_rank: int,
    failure_count: int,
) -> float:
    utility = (
        100.0 * task_score
        + 3.0 * latency_margin_ms
        + 0.10 * bandwidth_margin_mhz
        - 0.08 * bandwidth_mhz
        - 0.5 * max(0, radio_rank - 1)
    )
    if not passed:
        utility -= 100.0 + 10.0 * failure_count
    return utility
