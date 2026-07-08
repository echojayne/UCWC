"""Gray-coded square QAM modulation and exact soft demodulation."""

from __future__ import annotations

import math

import numpy as np


class QAMModem:
    def __init__(self, qam_order: int) -> None:
        root = int(math.isqrt(qam_order))
        if root * root != qam_order or qam_order < 4:
            raise ValueError("qam_order must be a square QAM order such as 4, 16, or 64.")
        if root & (root - 1):
            raise ValueError("QAM square root must be a power of two.")
        self.qam_order = int(qam_order)
        self.axis_levels = root
        self.bits_per_symbol = int(math.log2(qam_order))
        self.bits_per_axis = self.bits_per_symbol // 2
        self._bits, self._symbols = self._build_constellation()
        self._bit_shifts = np.arange(self.bits_per_symbol - 1, -1, -1, dtype=np.uint16)
        self._symbol_by_bits = {
            tuple(int(bit) for bit in bits.tolist()): symbol
            for bits, symbol in zip(self._bits, self._symbols, strict=True)
        }

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        bit_array = np.asarray(bits, dtype=np.uint8).reshape(-1)
        if len(bit_array) % self.bits_per_symbol != 0:
            raise ValueError("bit length must be divisible by bits_per_symbol.")
        if len(bit_array) == 0:
            return np.empty(0, dtype=np.complex128)
        grouped = bit_array.reshape(-1, self.bits_per_symbol)
        indices = np.sum(grouped.astype(np.uint16) << self._bit_shifts[None, :], axis=1)
        return self._symbols[indices]

    def demodulate_llr(
        self,
        received: np.ndarray,
        *,
        noise_variance: float,
        channel_gain: np.ndarray | complex | None = None,
    ) -> np.ndarray:
        if noise_variance <= 0:
            raise ValueError("noise_variance must be positive.")
        y = np.asarray(received, dtype=np.complex128).reshape(-1)
        if channel_gain is None:
            gains = np.ones_like(y, dtype=np.complex128)
        else:
            gains = np.asarray(channel_gain, dtype=np.complex128)
            if gains.ndim == 0:
                gains = np.full_like(y, gains.item(), dtype=np.complex128)
            gains = gains.reshape(-1)
            if len(gains) != len(y):
                raise ValueError("channel_gain length must match received symbols.")

        llrs = np.empty((len(y), self.bits_per_symbol), dtype=np.float64)
        chunk_size = 65536
        for start in range(0, len(y), chunk_size):
            stop = min(start + chunk_size, len(y))
            expected = gains[start:stop, None] * self._symbols[None, :]
            log_likelihood = -(np.abs(y[start:stop, None] - expected) ** 2) / noise_variance
            for bit_index in range(self.bits_per_symbol):
                zero = log_likelihood[:, self._bits[:, bit_index] == 0]
                one = log_likelihood[:, self._bits[:, bit_index] == 1]
                llrs[start:stop, bit_index] = np.logaddexp.reduce(zero, axis=1) - np.logaddexp.reduce(
                    one,
                    axis=1,
                )
        return llrs.reshape(-1)

    def _build_constellation(self) -> tuple[np.ndarray, np.ndarray]:
        labels = np.arange(self.axis_levels, dtype=np.uint16)
        level_by_gray_label = np.array(
            [2 * _gray_to_binary(int(label)) - (self.axis_levels - 1) for label in labels],
            dtype=np.float64,
        )
        bit_rows: list[list[int]] = []
        symbols: list[complex] = []
        for i_label in labels:
            for q_label in labels:
                bits = _integer_bits(int(i_label), self.bits_per_axis) + _integer_bits(
                    int(q_label),
                    self.bits_per_axis,
                )
                symbol = level_by_gray_label[i_label] + 1j * level_by_gray_label[q_label]
                bit_rows.append(bits)
                symbols.append(symbol)
        raw = np.asarray(symbols, dtype=np.complex128)
        normalizer = math.sqrt(float(np.mean(np.abs(raw) ** 2)))
        return np.asarray(bit_rows, dtype=np.uint8), raw / normalizer


def _gray_to_binary(gray: int) -> int:
    value = gray
    while gray > 0:
        gray >>= 1
        value ^= gray
    return value


def _integer_bits(value: int, width: int) -> list[int]:
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]
