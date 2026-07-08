"""End-to-end bit and semantic-feature transmission."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .channels import simulate_channel
from .ldpc import LDPCCode, LDPCStreamStats
from .qam import QAMModem
from .quantization import FeatureQuantizer


@dataclass(frozen=True, slots=True)
class BitTransmissionResult:
    transmitted_bits: int
    coded_bits: int
    bit_errors: int
    bit_error_rate: float
    block_errors: int
    block_error_rate: float
    ldpc_nonconvergence_rate: float
    ldpc_blocks: int
    ldpc_converged_blocks: int
    ldpc_max_iterations_used: int
    padding_bits: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SemanticTransmissionResult:
    recovered_features: np.ndarray
    payload_bits: np.ndarray
    recovered_bits: np.ndarray
    bit_result: BitTransmissionResult


def transmit_bits(
    payload_bits: np.ndarray,
    *,
    ldpc_code_rate: float,
    qam_order: int,
    snr_db: float,
    channel_model: str = "awgn",
    seed: int = 0,
    ldpc_iterations: int = 30,
) -> tuple[np.ndarray, BitTransmissionResult]:
    payload = np.asarray(payload_bits, dtype=np.uint8).reshape(-1)
    code = LDPCCode.for_rate(ldpc_code_rate)
    modem = QAMModem(qam_order)
    if code.n % modem.bits_per_symbol != 0:
        raise ValueError(
            f"LDPC block length {code.n} must be divisible by QAM bits_per_symbol "
            f"{modem.bits_per_symbol}; got qam_order={qam_order}."
        )
    if len(payload) == 0:
        return payload.copy(), _build_bit_result(
            payload_bits=payload,
            coded_bits=np.empty(0, dtype=np.uint8),
            bit_errors=0,
            stream_stats=LDPCStreamStats(
                blocks=0,
                converged_blocks=0,
                max_iterations_used=0,
                padding_bits=0,
                block_errors=0,
            ),
        )

    coded_bits, padding_bits = code.encode_stream(payload)
    symbols = modem.modulate(coded_bits)
    channel = simulate_channel(symbols, snr_db=snr_db, channel_model=channel_model, seed=seed)
    llr = modem.demodulate_llr(
        channel.received_symbols,
        noise_variance=channel.noise_variance,
        channel_gain=channel.channel_gain,
    )
    recovered, stream_stats = code.decode_stream(
        llr,
        original_length=len(payload),
        padding_bits=padding_bits,
        max_iterations=ldpc_iterations,
        reference_bits=payload,
    )
    bit_errors = int(np.sum(payload != recovered))
    return recovered, _build_bit_result(
        payload_bits=payload,
        coded_bits=coded_bits,
        bit_errors=bit_errors,
        stream_stats=stream_stats,
    )


def transmit_semantic_features(
    features: np.ndarray,
    *,
    quantization_bits: int,
    ldpc_code_rate: float,
    qam_order: int,
    snr_db: float,
    channel_model: str = "awgn",
    seed: int = 0,
    ldpc_iterations: int = 30,
    quantizer_clip_value: float = 3.0,
) -> SemanticTransmissionResult:
    quantizer = FeatureQuantizer(quantization_bits, clip_value=quantizer_clip_value)
    payload_bits, shape = quantizer.features_to_bits(np.asarray(features, dtype=np.float32))
    recovered_bits, bit_result = transmit_bits(
        payload_bits,
        ldpc_code_rate=ldpc_code_rate,
        qam_order=qam_order,
        snr_db=snr_db,
        channel_model=channel_model,
        seed=seed,
        ldpc_iterations=ldpc_iterations,
    )
    recovered_features = quantizer.bits_to_features(recovered_bits, shape)
    return SemanticTransmissionResult(
        recovered_features=recovered_features,
        payload_bits=payload_bits,
        recovered_bits=recovered_bits,
        bit_result=bit_result,
    )


def _build_bit_result(
    *,
    payload_bits: np.ndarray,
    coded_bits: np.ndarray,
    bit_errors: int,
    stream_stats: LDPCStreamStats,
) -> BitTransmissionResult:
    block_errors = stream_stats.block_errors
    if block_errors is None:
        block_errors = int(bit_errors > 0)
    if stream_stats.blocks == 0:
        block_error_rate = 0.0
        ldpc_nonconvergence_rate = 0.0
    else:
        block_error_rate = int(block_errors) / stream_stats.blocks
        ldpc_nonconvergence_rate = 1.0 - (stream_stats.converged_blocks / stream_stats.blocks)
    return BitTransmissionResult(
        transmitted_bits=int(len(payload_bits)),
        coded_bits=int(len(coded_bits)),
        bit_errors=bit_errors,
        bit_error_rate=bit_errors / max(1, int(len(payload_bits))),
        block_errors=int(block_errors),
        block_error_rate=block_error_rate,
        ldpc_nonconvergence_rate=ldpc_nonconvergence_rate,
        ldpc_blocks=stream_stats.blocks,
        ldpc_converged_blocks=stream_stats.converged_blocks,
        ldpc_max_iterations_used=stream_stats.max_iterations_used,
        padding_bits=stream_stats.padding_bits,
    )
