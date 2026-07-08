from pathlib import Path
import sqlite3

import numpy as np
import torch
from torch import nn

from ucwc.semantic import (
    LDPCCode,
    QAMModem,
    RIDAS_DECODER_PATH,
    ensure_ridas_weights,
    evaluate_semantic_link,
    transmit_bits,
)
from ucwc.semantic.evaluation import SemanticEvalConfig
from ucwc.semantic.storage import EXPERIENCE_TABLE, write_experience_rows


class DummyConfig:
    image_size = 16
    feature_dim = 8
    max_depth = 2
    num_classes = 4


class DummyCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = DummyConfig()
        self.proj = nn.Linear(3, self.config.feature_dim)
        self.decoder = nn.Linear(self.config.feature_dim, self.config.num_classes)

    @torch.no_grad()
    def encode(self, images: torch.Tensor, encoder_depth: int | None = None) -> torch.Tensor:
        scale = 1.0 if encoder_depth is None else float(encoder_depth)
        pooled = images.mean(dim=(2, 3))
        return self.proj(pooled) * scale

    @torch.no_grad()
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)


def test_ldpc_roundtrip_with_clean_llr() -> None:
    rng = np.random.default_rng(3)
    code = LDPCCode.for_rate(0.5, block_length=648)
    payload = rng.integers(0, 2, size=code.k, dtype=np.uint8)
    codeword = code.encode_block(payload)
    assert code.syndrome_weight(codeword) == 0
    llr = np.where(codeword == 0, 20.0, -20.0)
    decoded, stats = code.decode_block(llr, max_iterations=10)
    assert stats.converged
    assert np.array_equal(decoded, payload)


def test_qam_exact_soft_demod_clean_symbols() -> None:
    modem = QAMModem(16)
    bits = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1], dtype=np.uint8)
    symbols = modem.modulate(bits)
    llr = modem.demodulate_llr(symbols, noise_variance=1e-4)
    hard = (llr < 0.0).astype(np.uint8)
    assert np.array_equal(hard, bits)


def test_transmit_bits_rejects_incompatible_qam_block_length() -> None:
    bits = np.zeros(32, dtype=np.uint8)
    try:
        transmit_bits(bits, ldpc_code_rate=0.5, qam_order=1024, snr_db=20.0)
    except ValueError as exc:
        assert "LDPC block length" in str(exc)
    else:
        raise AssertionError("Expected qam_order=1024 to be rejected for LDPC n=648.")


def test_transmit_bits_handles_empty_payload() -> None:
    recovered, result = transmit_bits(
        np.empty(0, dtype=np.uint8),
        ldpc_code_rate=0.5,
        qam_order=16,
        snr_db=20.0,
    )
    assert recovered.size == 0
    assert result.bit_error_rate == 0.0
    assert result.block_error_rate == 0.0
    assert result.ldpc_nonconvergence_rate == 0.0


def test_transmit_bits_high_snr_awgn_roundtrip() -> None:
    rng = np.random.default_rng(9)
    bits = rng.integers(0, 2, size=2048, dtype=np.uint8)
    recovered, result = transmit_bits(
        bits,
        ldpc_code_rate=0.5,
        qam_order=16,
        snr_db=100.0,
        channel_model="awgn",
        seed=9,
        ldpc_iterations=10,
    )
    assert np.array_equal(recovered, bits)
    assert result.bit_errors == 0
    assert result.block_errors == 0


def test_semantic_eval_writes_sqlite(tmp_path: Path) -> None:
    codec = DummyCodec()
    row = evaluate_semantic_link(
        codec,
        SemanticEvalConfig(
            encoder_depth=2,
            quantization_bits=4,
            ldpc_code_rate=0.5,
            qam_order=4,
            snr_db=20.0,
            batch_size=2,
            seed=5,
            ldpc_iterations=10,
        ),
    )
    db_path = tmp_path / "semantic.sqlite"
    result = write_experience_rows([row], sqlite_path=db_path, csv_path=tmp_path / "semantic.csv")
    assert result["row_count"] == 1
    with sqlite3.connect(db_path) as connection:
        count = connection.execute(f"SELECT COUNT(*) FROM {EXPERIENCE_TABLE}").fetchone()[0]
    assert count == 1


def test_ridas_decoder_weight_shape() -> None:
    ensure_ridas_weights()
    state = torch.load(RIDAS_DECODER_PATH, map_location="cpu", weights_only=True)
    assert tuple(state["weight"].shape) == (100, 512)
    assert tuple(state["bias"].shape) == (100,)
