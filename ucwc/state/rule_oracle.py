"""Rule-based semantic-link scoring and resource formulas."""

from __future__ import annotations

import math


TASK_SCORE_OFFSETS = {
    "image_classification": 0.0,
    "semantic_retrieval": -0.025,
    "intent_recognition": 0.015,
}


def predict_task_score(
    *,
    encoder_depth: int,
    quantization_bits: int,
    ldpc_code_rate: float,
    qam_order: int,
    snr_db: float,
    task_type: str,
    reference_snr_db: float | None = None,
) -> float:
    """Predict task success score from semantic config, PHY mode, and SNR.

    This is a calibrated rule, not a measured semantic-link result. SNR is a
    continuous soft factor; it is not used as a hard admission threshold.
    """

    source_quality = 0.58 + 0.018 * float(encoder_depth) + 0.022 * float(quantization_bits)
    source_quality = min(0.98, max(0.0, source_quality))
    reference = (
        soft_reference_snr_db(ldpc_code_rate, qam_order)
        if reference_snr_db is None
        else float(reference_snr_db)
    )
    channel_quality = _sigmoid((float(snr_db) - reference) / 4.0)
    qam_bits = math.log2(int(qam_order))
    code_penalty = 0.025 * max(0.0, float(ldpc_code_rate) - 0.5)
    qam_penalty = 0.008 * max(0.0, qam_bits - 2.0)
    task_offset = TASK_SCORE_OFFSETS.get(task_type, 0.0)
    score = source_quality * (0.52 + 0.48 * channel_quality)
    score = score - code_penalty - qam_penalty + task_offset
    return min(0.999, max(0.0, score))


def soft_reference_snr_db(ldpc_code_rate: float, qam_order: int) -> float:
    """Return the SNR midpoint of the rule-based link-quality curve."""

    qam_bits = math.log2(int(qam_order))
    return -2.0 + 3.0 * qam_bits + 8.0 * (float(ldpc_code_rate) - 0.5)


def required_bandwidth_mhz(
    *,
    payload_bits: float,
    tx_budget_ms: float,
    ldpc_code_rate: float,
    qam_order: int,
    resource_efficiency: float,
) -> float:
    """Estimate bandwidth needed to send one payload inside the tx budget."""

    effective_bits_per_hz = (
        float(ldpc_code_rate) * math.log2(int(qam_order)) * float(resource_efficiency)
    )
    if tx_budget_ms <= 0.0:
        return math.inf
    if effective_bits_per_hz <= 0.0:
        return math.inf
    return float(payload_bits) / ((float(tx_budget_ms) / 1000.0) * effective_bits_per_hz * 1e6)


def _sigmoid(value: float) -> float:
    if value >= 40.0:
        return 1.0
    if value <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))
