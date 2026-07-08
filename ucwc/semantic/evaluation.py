"""Semantic link evaluation and initial experience-table generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import time

import numpy as np
import torch
import torch.nn.functional as F

from .link import transmit_semantic_features
from .models import make_image_batch


@dataclass(frozen=True, slots=True)
class SemanticEvalConfig:
    encoder_depth: int
    quantization_bits: int
    ldpc_code_rate: float
    qam_order: int
    snr_db: float
    header_bits: int = 0
    channel_model: str = "awgn"
    batch_size: int = 8
    seed: int = 0
    ldpc_iterations: int = 30
    dataset: str = "synthetic"
    dataset_root: str = "/home/users/dky/.cache"
    download_dataset: bool = False


@dataclass(frozen=True, slots=True)
class SemanticEvalRow:
    config_id: str
    encoder_depth: int
    quantization_bits: int
    ldpc_code_rate: float
    qam_order: int
    snr_db: float
    channel_model: str
    dataset: str
    batch_size: int
    feature_dim: int
    payload_bits: int
    header_bits: int
    payload_bits_with_header: int
    coded_bits: int
    coded_bits_with_header: int
    qam_bits_per_symbol: int
    qam_symbols: int
    qam_symbols_with_header: int
    payload_bytes: float
    payload_bytes_with_header: float
    coded_bytes: float
    coded_bytes_with_header: float
    ldpc_padding_bits: int
    bit_error_rate: float
    bit_errors: int
    block_errors: int
    block_error_rate: float
    ldpc_nonconvergence_rate: float
    ldpc_blocks: int
    ldpc_converged_blocks: int
    ldpc_max_iterations_used: int
    semantic_score: float
    classifier_agreement: float
    original_accuracy: float
    recovered_accuracy: float
    encoding_latency_ms: float
    decoding_latency_ms: float

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@torch.no_grad()
def evaluate_semantic_link(
    codec: Any,
    config: SemanticEvalConfig,
) -> SemanticEvalRow:
    device = next(codec.parameters()).device
    images, labels = make_image_batch(
        batch_size=config.batch_size,
        image_size=codec.config.image_size,
        num_classes=codec.config.num_classes,
        seed=config.seed,
        dataset=config.dataset,
        dataset_root=config.dataset_root,
        download_dataset=config.download_dataset,
    )
    images = images.to(device)
    labels = labels.to(device)
    codec.eval()

    start = time.perf_counter()
    features = codec.encode(images, encoder_depth=config.encoder_depth)
    encoding_latency_ms = (time.perf_counter() - start) * 1000.0
    feature_np = features.detach().cpu().numpy().astype(np.float32)

    transmission = transmit_semantic_features(
        feature_np,
        quantization_bits=config.quantization_bits,
        ldpc_code_rate=config.ldpc_code_rate,
        qam_order=config.qam_order,
        snr_db=config.snr_db,
        channel_model=config.channel_model,
        seed=config.seed,
        ldpc_iterations=config.ldpc_iterations,
    )

    recovered_features = torch.from_numpy(transmission.recovered_features).to(device)
    start = time.perf_counter()
    original_logits = codec.decode(features)
    recovered_logits = codec.decode(recovered_features)
    decoding_latency_ms = (time.perf_counter() - start) * 1000.0

    original_pred = original_logits.argmax(dim=1)
    recovered_pred = recovered_logits.argmax(dim=1)
    semantic_score = float(
        F.cosine_similarity(features.float(), recovered_features.float(), dim=1).mean().item()
    )
    classifier_agreement = float((original_pred == recovered_pred).float().mean().item())
    original_accuracy = float((original_pred == labels).float().mean().item())
    recovered_accuracy = float((recovered_pred == labels).float().mean().item())
    bit_result = transmission.bit_result
    volume = estimate_transmission_volume(
        feature_payload_bits=bit_result.transmitted_bits,
        header_bits_per_sample=config.header_bits,
        sample_count=config.batch_size,
        coded_bits=bit_result.coded_bits,
        ldpc_padding_bits=bit_result.padding_bits,
        ldpc_code_rate=config.ldpc_code_rate,
        qam_order=config.qam_order,
    )
    config_id = (
        f"d{config.encoder_depth}_q{config.quantization_bits}_"
        f"r{_rate_label(config.ldpc_code_rate)}_qam{config.qam_order}_snr{config.snr_db:g}"
    )
    return SemanticEvalRow(
        config_id=config_id,
        encoder_depth=config.encoder_depth,
        quantization_bits=config.quantization_bits,
        ldpc_code_rate=config.ldpc_code_rate,
        qam_order=config.qam_order,
        snr_db=config.snr_db,
        channel_model=config.channel_model,
        dataset=config.dataset,
        batch_size=config.batch_size,
        feature_dim=codec.config.feature_dim,
        payload_bits=bit_result.transmitted_bits,
        header_bits=int(config.header_bits),
        payload_bits_with_header=volume["payload_bits_with_header"],
        coded_bits=bit_result.coded_bits,
        coded_bits_with_header=volume["coded_bits_with_header"],
        qam_bits_per_symbol=volume["qam_bits_per_symbol"],
        qam_symbols=volume["qam_symbols"],
        qam_symbols_with_header=volume["qam_symbols_with_header"],
        payload_bytes=volume["payload_bytes"],
        payload_bytes_with_header=volume["payload_bytes_with_header"],
        coded_bytes=volume["coded_bytes"],
        coded_bytes_with_header=volume["coded_bytes_with_header"],
        ldpc_padding_bits=bit_result.padding_bits,
        bit_error_rate=bit_result.bit_error_rate,
        bit_errors=bit_result.bit_errors,
        block_errors=bit_result.block_errors,
        block_error_rate=bit_result.block_error_rate,
        ldpc_nonconvergence_rate=bit_result.ldpc_nonconvergence_rate,
        ldpc_blocks=bit_result.ldpc_blocks,
        ldpc_converged_blocks=bit_result.ldpc_converged_blocks,
        ldpc_max_iterations_used=bit_result.ldpc_max_iterations_used,
        semantic_score=semantic_score,
        classifier_agreement=classifier_agreement,
        original_accuracy=original_accuracy,
        recovered_accuracy=recovered_accuracy,
        encoding_latency_ms=encoding_latency_ms,
        decoding_latency_ms=decoding_latency_ms,
    )


def evaluate_grid(
    codec: Any,
    *,
    encoder_depths: list[int],
    quantization_bits: list[int],
    phy_modes: list[dict[str, float | int | str]],
    snr_values_db: list[float],
    channel_model: str = "awgn",
    batch_size: int = 8,
    seed: int = 0,
    ldpc_iterations: int = 30,
    dataset: str = "synthetic",
    dataset_root: str = "/home/users/dky/.cache",
    download_dataset: bool = False,
) -> list[SemanticEvalRow]:
    rows: list[SemanticEvalRow] = []
    case_index = 0
    for encoder_depth in encoder_depths:
        for q_bits in quantization_bits:
            for mode in phy_modes:
                for snr_db in snr_values_db:
                    case_index += 1
                    rows.append(
                        evaluate_semantic_link(
                            codec,
                            SemanticEvalConfig(
                                encoder_depth=encoder_depth,
                                quantization_bits=q_bits,
                                ldpc_code_rate=float(mode["ldpc_code_rate"]),
                                qam_order=int(mode["qam_order"]),
                                snr_db=float(snr_db),
                                header_bits=int(mode.get("header_bits", 0)),
                                channel_model=channel_model,
                                batch_size=batch_size,
                                seed=seed + case_index,
                                ldpc_iterations=ldpc_iterations,
                                dataset=dataset,
                                dataset_root=dataset_root,
                                download_dataset=download_dataset,
                            ),
                        )
                    )
    return rows


def _rate_label(rate: float) -> str:
    if abs(rate - 0.5) < 1e-6:
        return "12"
    if abs(rate - 2.0 / 3.0) < 1e-3:
        return "23"
    if abs(rate - 0.75) < 1e-6:
        return "34"
    return str(rate).replace(".", "p")


def estimate_transmission_volume(
    *,
    feature_payload_bits: int,
    header_bits_per_sample: int,
    sample_count: int,
    coded_bits: int,
    ldpc_padding_bits: int,
    ldpc_code_rate: float,
    qam_order: int,
) -> dict[str, int | float]:
    """Return feature/header/coded/QAM data-volume accounting for one batch."""

    from .ldpc import LDPCCode
    from .qam import QAMModem

    header_bits_total = int(header_bits_per_sample) * int(sample_count)
    payload_bits_with_header = int(feature_payload_bits) + header_bits_total
    code = LDPCCode.for_rate(float(ldpc_code_rate))
    modem = QAMModem(int(qam_order))
    padding_with_header = (-payload_bits_with_header) % code.k
    coded_bits_with_header = (
        0
        if payload_bits_with_header == 0
        else ((payload_bits_with_header + padding_with_header) // code.k) * code.n
    )
    qam_symbols = int(coded_bits) // modem.bits_per_symbol
    qam_symbols_with_header = coded_bits_with_header // modem.bits_per_symbol
    return {
        "header_bits": header_bits_total,
        "payload_bits_with_header": payload_bits_with_header,
        "coded_bits_with_header": int(coded_bits_with_header),
        "qam_bits_per_symbol": int(modem.bits_per_symbol),
        "qam_symbols": int(qam_symbols),
        "qam_symbols_with_header": int(qam_symbols_with_header),
        "payload_bytes": int(feature_payload_bits) / 8.0,
        "payload_bytes_with_header": payload_bits_with_header / 8.0,
        "coded_bytes": int(coded_bits) / 8.0,
        "coded_bytes_with_header": int(coded_bits_with_header) / 8.0,
        "ldpc_padding_bits": int(ldpc_padding_bits),
    }
