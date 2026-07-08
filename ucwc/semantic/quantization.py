"""Feature quantization and bit packing utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FeatureQuantizer:
    quantization_bits: int
    clip_value: float = 3.0

    def __post_init__(self) -> None:
        if self.quantization_bits < 1 or self.quantization_bits > 16:
            raise ValueError("quantization_bits must be in [1, 16].")
        if self.clip_value <= 0:
            raise ValueError("clip_value must be positive.")

    @property
    def levels(self) -> int:
        return 1 << self.quantization_bits

    def quantize(self, features: np.ndarray) -> np.ndarray:
        clipped = np.clip(features, -self.clip_value, self.clip_value)
        scaled = (clipped + self.clip_value) / (2.0 * self.clip_value)
        levels = np.rint(scaled * (self.levels - 1)).astype(np.uint16)
        return np.clip(levels, 0, self.levels - 1).astype(np.uint16)

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        scaled = indices.astype(np.float32) / float(self.levels - 1)
        return (scaled * (2.0 * self.clip_value) - self.clip_value).astype(np.float32)

    def features_to_bits(self, features: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        indices = self.quantize(features)
        return integers_to_bits(indices.reshape(-1), self.quantization_bits), tuple(indices.shape)

    def bits_to_features(self, bits: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        values = bits_to_integers(bits, self.quantization_bits)
        expected = int(np.prod(shape))
        if len(values) < expected:
            raise ValueError(f"Not enough bits for feature shape {shape}: got {len(values)} values")
        return self.dequantize(values[:expected].reshape(shape))


def integers_to_bits(values: np.ndarray, bit_width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64).reshape(-1)
    shifts = np.arange(bit_width - 1, -1, -1, dtype=np.uint64)
    bits = ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    return bits.reshape(-1)


def bits_to_integers(bits: np.ndarray, bit_width: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if len(bits) % bit_width != 0:
        raise ValueError("bit length must be divisible by bit_width.")
    grouped = bits.reshape(-1, bit_width).astype(np.uint64)
    shifts = np.arange(bit_width - 1, -1, -1, dtype=np.uint64)
    return np.sum(grouped << shifts[None, :], axis=1).astype(np.uint16)
