"""Minimal real LDPC encoder/decoder.

This module implements a genuine binary LDPC code: a sparse parity-check
matrix, systematic encoding through a dual-diagonal parity section, and
iterative sum-product decoding from channel LLRs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LDPCDecodeStats:
    iterations: int
    converged: bool
    syndrome_weight: int


@dataclass(frozen=True, slots=True)
class LDPCStreamStats:
    blocks: int
    converged_blocks: int
    max_iterations_used: int
    padding_bits: int
    block_errors: int | None = None


class LDPCCode:
    def __init__(self, h_matrix: np.ndarray, k: int) -> None:
        h_bool = np.asarray(h_matrix, dtype=bool)
        if h_bool.ndim != 2:
            raise ValueError("h_matrix must be two-dimensional.")
        if k <= 0 or k >= h_bool.shape[1]:
            raise ValueError("k must be in (0, n).")
        self.h = h_bool
        self.k = int(k)
        self.n = int(h_bool.shape[1])
        self.m = int(h_bool.shape[0])
        self.rate = self.k / self.n
        self._a = self.h[:, : self.k].astype(np.uint8)
        self._check_neighbors = [np.flatnonzero(self.h[row]).astype(np.int32) for row in range(self.m)]
        self._variable_neighbors = [
            np.flatnonzero(self.h[:, column]).astype(np.int32) for column in range(self.n)
        ]
        self._edge_checks, self._edge_vars, self._check_edge_indices, self._variable_edge_indices = (
            self._build_edge_index()
        )

    @classmethod
    def for_rate(
        cls,
        code_rate: float,
        *,
        block_length: int = 648,
        column_weight: int = 3,
        seed: int = 17,
    ) -> "LDPCCode":
        if not 0.0 < code_rate < 1.0:
            raise ValueError("code_rate must be between 0 and 1.")
        k = int(round(block_length * code_rate))
        if k <= 0 or k >= block_length:
            raise ValueError("Invalid block_length/code_rate pair.")
        m = block_length - k
        rng = np.random.default_rng(seed + int(round(code_rate * 10000)))
        a = np.zeros((m, k), dtype=bool)
        weight = max(1, min(column_weight, m))
        for column in range(k):
            rows = rng.choice(m, size=weight, replace=False)
            a[rows, column] = True

        parity = np.eye(m, dtype=bool)
        if m > 1:
            parity[np.arange(1, m), np.arange(0, m - 1)] = True
        h = np.concatenate([a, parity], axis=1)
        return cls(h, k=k)

    def encode_block(self, info_bits: np.ndarray) -> np.ndarray:
        info = np.asarray(info_bits, dtype=np.uint8).reshape(-1)
        if len(info) != self.k:
            raise ValueError(f"LDPC info block must have length {self.k}, got {len(info)}.")
        syndrome = (self._a @ info) % 2
        parity = np.zeros(self.m, dtype=np.uint8)
        if self.m:
            parity[0] = syndrome[0]
            for index in range(1, self.m):
                parity[index] = syndrome[index] ^ parity[index - 1]
        return np.concatenate([info, parity]).astype(np.uint8)

    def encode_stream(self, payload_bits: np.ndarray) -> tuple[np.ndarray, int]:
        payload = np.asarray(payload_bits, dtype=np.uint8).reshape(-1)
        if len(payload) == 0:
            return np.empty(0, dtype=np.uint8), 0
        padding = (-len(payload)) % self.k
        if padding:
            payload = np.concatenate([payload, np.zeros(padding, dtype=np.uint8)])
        blocks = payload.reshape(-1, self.k)
        encoded = [self.encode_block(block) for block in blocks]
        return np.concatenate(encoded).astype(np.uint8), padding

    def decode_block(
        self,
        channel_llr: np.ndarray,
        *,
        max_iterations: int = 30,
        llr_clip: float = 50.0,
    ) -> tuple[np.ndarray, LDPCDecodeStats]:
        llr = np.asarray(channel_llr, dtype=np.float64).reshape(-1)
        if len(llr) != self.n:
            raise ValueError(f"LDPC LLR block must have length {self.n}, got {len(llr)}.")
        llr = np.clip(llr, -llr_clip, llr_clip)

        v_to_c: dict[tuple[int, int], float] = {}
        c_to_v: dict[tuple[int, int], float] = {}
        for check_index, variables in enumerate(self._check_neighbors):
            for variable_index in variables:
                key = (check_index, int(variable_index))
                v_to_c[key] = float(llr[variable_index])
                c_to_v[key] = 0.0

        hard = np.zeros(self.n, dtype=np.uint8)
        iterations_used = 0
        for iteration in range(1, max_iterations + 1):
            iterations_used = iteration
            for check_index, variables in enumerate(self._check_neighbors):
                tanh_values = np.array(
                    [
                        np.tanh(np.clip(v_to_c[(check_index, int(variable_index))], -llr_clip, llr_clip) / 2.0)
                        for variable_index in variables
                    ],
                    dtype=np.float64,
                )
                for local_index, variable_index in enumerate(variables):
                    if len(tanh_values) == 1:
                        product = 1.0
                    else:
                        product = float(np.prod(np.delete(tanh_values, local_index)))
                    product = float(np.clip(product, -0.999999999999, 0.999999999999))
                    c_to_v[(check_index, int(variable_index))] = float(2.0 * np.arctanh(product))

            posterior = np.array(llr, copy=True)
            for variable_index, checks in enumerate(self._variable_neighbors):
                total = float(llr[variable_index])
                for check_index in checks:
                    total += c_to_v[(int(check_index), variable_index)]
                posterior[variable_index] = np.clip(total, -llr_clip, llr_clip)
                for check_index in checks:
                    v_to_c[(int(check_index), variable_index)] = float(
                        posterior[variable_index] - c_to_v[(int(check_index), variable_index)]
                    )

            hard = (posterior < 0.0).astype(np.uint8)
            syndrome_weight = self.syndrome_weight(hard)
            if syndrome_weight == 0:
                return hard[: self.k], LDPCDecodeStats(
                    iterations=iterations_used,
                    converged=True,
                    syndrome_weight=0,
                )

        return hard[: self.k], LDPCDecodeStats(
            iterations=iterations_used,
            converged=False,
            syndrome_weight=self.syndrome_weight(hard),
        )

    def decode_stream(
        self,
        channel_llr: np.ndarray,
        *,
        original_length: int,
        padding_bits: int,
        max_iterations: int = 30,
        reference_bits: np.ndarray | None = None,
    ) -> tuple[np.ndarray, LDPCStreamStats]:
        llr = np.asarray(channel_llr, dtype=np.float64).reshape(-1)
        if len(llr) % self.n != 0:
            raise ValueError("LLR stream length must be divisible by LDPC block length.")
        if len(llr) == 0:
            if original_length != 0:
                raise ValueError("Empty LLR stream cannot recover a non-empty payload.")
            return np.empty(0, dtype=np.uint8), LDPCStreamStats(
                blocks=0,
                converged_blocks=0,
                max_iterations_used=0,
                padding_bits=padding_bits,
                block_errors=0 if reference_bits is not None else None,
            )
        blocks = llr.reshape(-1, self.n)
        decoded_padded, converged_blocks, max_iterations_used = self._decode_blocks_vectorized(
            blocks,
            max_iterations=max_iterations,
        )
        block_errors = self._count_block_errors(decoded_padded, reference_bits, len(blocks))
        decoded = decoded_padded
        if padding_bits:
            decoded = decoded[:-padding_bits]
        decoded = decoded[:original_length]
        return decoded, LDPCStreamStats(
            blocks=len(blocks),
            converged_blocks=converged_blocks,
            max_iterations_used=max_iterations_used,
            padding_bits=padding_bits,
            block_errors=block_errors,
        )

    def syndrome_weight(self, codeword_bits: np.ndarray) -> int:
        bits = np.asarray(codeword_bits, dtype=np.uint8).reshape(-1)
        if len(bits) != self.n:
            raise ValueError(f"Codeword length must be {self.n}, got {len(bits)}.")
        syndrome = (self.h.astype(np.uint8) @ bits) % 2
        return int(np.sum(syndrome))

    def row_weight_summary(self) -> dict[str, float]:
        weights = np.sum(self.h, axis=1)
        return {
            "min": float(np.min(weights)),
            "mean": float(np.mean(weights)),
            "max": float(np.max(weights)),
        }

    def _count_block_errors(
        self,
        decoded_padded: np.ndarray,
        reference_bits: np.ndarray | None,
        block_count: int,
    ) -> int | None:
        if reference_bits is None:
            return None
        reference = np.asarray(reference_bits, dtype=np.uint8).reshape(-1)
        padding = (-len(reference)) % self.k
        if padding:
            reference = np.concatenate([reference, np.zeros(padding, dtype=np.uint8)])
        if len(reference) != block_count * self.k:
            raise ValueError("reference_bits length does not match the decoded LDPC block count.")
        reference_blocks = reference.reshape(block_count, self.k)
        decoded_blocks = decoded_padded.reshape(block_count, self.k)
        return int(np.any(reference_blocks != decoded_blocks, axis=1).sum())

    def _build_edge_index(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        edge_checks: list[int] = []
        edge_vars: list[int] = []
        check_edges: list[np.ndarray] = []
        variable_edges: list[list[int]] = [[] for _ in range(self.n)]
        for check_index, variables in enumerate(self._check_neighbors):
            edges: list[int] = []
            for variable_index in variables:
                edge_index = len(edge_checks)
                variable = int(variable_index)
                edge_checks.append(check_index)
                edge_vars.append(variable)
                edges.append(edge_index)
                variable_edges[variable].append(edge_index)
            check_edges.append(np.asarray(edges, dtype=np.int32))
        return (
            np.asarray(edge_checks, dtype=np.int32),
            np.asarray(edge_vars, dtype=np.int32),
            check_edges,
            [np.asarray(edges, dtype=np.int32) for edges in variable_edges],
        )

    def _decode_blocks_vectorized(
        self,
        block_llr: np.ndarray,
        *,
        max_iterations: int,
        llr_clip: float = 50.0,
    ) -> tuple[np.ndarray, int, int]:
        llr = np.clip(np.asarray(block_llr, dtype=np.float64), -llr_clip, llr_clip)
        if llr.ndim != 2 or llr.shape[1] != self.n:
            raise ValueError(f"block_llr must have shape [blocks, {self.n}].")

        block_count = int(llr.shape[0])
        if block_count == 0:
            return np.empty((0, self.k), dtype=np.uint8).reshape(-1), 0, 0

        v_to_c = llr[:, self._edge_vars].copy()
        c_to_v = np.zeros_like(v_to_c)
        hard = (llr < 0.0).astype(np.uint8)
        converged = np.zeros(block_count, dtype=bool)
        decoded_blocks = np.empty((block_count, self.k), dtype=np.uint8)
        max_iterations_used = 0

        for iteration in range(1, max_iterations + 1):
            max_iterations_used = iteration
            for edges in self._check_edge_indices:
                if len(edges) == 0:
                    continue
                values = np.tanh(np.clip(v_to_c[:, edges], -llr_clip, llr_clip) / 2.0)
                if len(edges) == 1:
                    products = np.ones_like(values, dtype=np.float64)
                else:
                    prefix = np.ones_like(values, dtype=np.float64)
                    suffix = np.ones_like(values, dtype=np.float64)
                    prefix[:, 1:] = np.cumprod(values[:, :-1], axis=1)
                    suffix[:, :-1] = np.cumprod(values[:, :0:-1], axis=1)[:, ::-1]
                    products = prefix * suffix
                products = np.clip(products, -0.999999999999, 0.999999999999)
                c_to_v[:, edges] = 2.0 * np.arctanh(products)

            posterior = llr.copy()
            for variable_index, edges in enumerate(self._variable_edge_indices):
                if len(edges) == 0:
                    continue
                posterior[:, variable_index] += np.sum(c_to_v[:, edges], axis=1)
            posterior = np.clip(posterior, -llr_clip, llr_clip)
            for variable_index, edges in enumerate(self._variable_edge_indices):
                if len(edges) == 0:
                    continue
                v_to_c[:, edges] = posterior[:, [variable_index]] - c_to_v[:, edges]

            hard = (posterior < 0.0).astype(np.uint8)
            syndrome_weight = self._syndrome_weight_batch(hard)
            newly_converged = (~converged) & (syndrome_weight == 0)
            if np.any(newly_converged):
                decoded_blocks[newly_converged] = hard[newly_converged, : self.k]
                converged[newly_converged] = True
            if bool(np.all(converged)):
                break

        if not bool(np.all(converged)):
            decoded_blocks[~converged] = hard[~converged, : self.k]
        return decoded_blocks.reshape(-1).astype(np.uint8), int(np.sum(converged)), max_iterations_used

    def _syndrome_weight_batch(self, codeword_bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(codeword_bits, dtype=np.uint8)
        if bits.ndim != 2 or bits.shape[1] != self.n:
            raise ValueError(f"codeword_bits must have shape [blocks, {self.n}].")
        weights = np.zeros(bits.shape[0], dtype=np.int32)
        for variables in self._check_neighbors:
            weights += (np.sum(bits[:, variables], axis=1) & 1).astype(np.int32)
        return weights
