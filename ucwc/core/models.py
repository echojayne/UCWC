"""Core dataclasses for the semantic UCWC scenario."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BaseStationState:
    bs_id: str
    x_m: float
    y_m: float
    bandwidth_budget_mhz: float
    used_bandwidth_mhz: float = 0.0


@dataclass(frozen=True, slots=True)
class UeRequest:
    request_id: str
    ue_id: str
    x_m: float
    y_m: float
    task_type: str
    min_semantic_score: float
    max_total_latency_ms: float


@dataclass(frozen=True, slots=True)
class RadioLinkState:
    ue_id: str
    bs_id: str
    snr_db: float
    sinr_db: float
    radio_rank: int


@dataclass(frozen=True, slots=True)
class SemanticConfig:
    config_id: str
    encoder_depth: int
    quantization_bits: int
    payload_bits: int
    semantic_score: float
    encoding_latency_ms: float
    decoding_latency_ms: float


@dataclass(frozen=True, slots=True)
class PhyMode:
    phy_mode_id: str
    ldpc_code_rate: float
    qam_order: int
    min_snr_db: float
    spectral_efficiency_bps_hz: float


@dataclass(frozen=True, slots=True)
class SessionPlan:
    request_id: str
    ue_id: str
    serving_bs_id: str
    semantic_config_id: str
    phy_mode_id: str
    bandwidth_mhz: float
    source: str
