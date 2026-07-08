#!/usr/bin/env python
"""Sweep SNR on a binary payload and report BER/BLER cliff behavior."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ucwc.semantic.link import transmit_bits


DEFAULT_SNR_DB = [float(value) for value in range(0, 23, 2)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="tests/test.bin")
    parser.add_argument("--output-dir", default="outputs/semantic_link_cliff_test")
    parser.add_argument("--channel-models", nargs="+", default=["awgn", "rayleigh"])
    parser.add_argument("--snr-db", nargs="+", type=float, default=DEFAULT_SNR_DB)
    parser.add_argument("--ldpc-code-rate", type=float, default=0.5)
    parser.add_argument("--qam-order", type=int, default=16)
    parser.add_argument("--ldpc-iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--repeats", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload_bits = _read_payload_bits(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, int | float | str]] = []
    summary: dict[str, object] = {
        "input": str(Path(args.input)),
        "payload_bits": int(len(payload_bits)),
        "qam_order": int(args.qam_order),
        "ldpc_code_rate": float(args.ldpc_code_rate),
        "ldpc_iterations": int(args.ldpc_iterations),
        "repeats": int(args.repeats),
        "channels": {},
    }

    for channel_model in args.channel_models:
        channel_rows = _sweep_channel(payload_bits, args, channel_model)
        all_rows.extend(channel_rows)
        summary["channels"][channel_model] = _summarize_cliff(channel_rows)

    csv_path = output_dir / "test_bin_cliff_sweep.csv"
    _write_csv(all_rows, csv_path)
    summary_path = output_dir / "test_bin_cliff_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary["channels"], indent=2, sort_keys=True))
    return 0


def _read_payload_bits(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if not data:
        raise ValueError(f"{path} is empty.")
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def _sweep_channel(
    payload_bits: np.ndarray,
    args: argparse.Namespace,
    channel_model: str,
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for snr_db in args.snr_db:
        bit_errors = 0
        block_errors = 0
        transmitted_bits = 0
        ldpc_blocks = 0
        ldpc_nonconverged_blocks = 0
        coded_bits = 0
        max_iterations = 0
        for repeat_index in range(int(args.repeats)):
            _recovered, result = transmit_bits(
                payload_bits,
                ldpc_code_rate=float(args.ldpc_code_rate),
                qam_order=int(args.qam_order),
                snr_db=float(snr_db),
                channel_model=channel_model,
                seed=int(args.seed) + repeat_index,
                ldpc_iterations=int(args.ldpc_iterations),
            )
            bit_errors += result.bit_errors
            block_errors += result.block_errors
            transmitted_bits += result.transmitted_bits
            ldpc_blocks += result.ldpc_blocks
            ldpc_nonconverged_blocks += result.ldpc_blocks - result.ldpc_converged_blocks
            coded_bits += result.coded_bits
            max_iterations = max(max_iterations, result.ldpc_max_iterations_used)

        row = {
            "channel_model": channel_model,
            "snr_db": float(snr_db),
            "repeats": int(args.repeats),
            "transmitted_bits": int(transmitted_bits),
            "coded_bits": int(coded_bits),
            "bit_errors": int(bit_errors),
            "bit_error_rate": bit_errors / max(1, transmitted_bits),
            "block_errors": int(block_errors),
            "block_error_rate": block_errors / max(1, ldpc_blocks),
            "ldpc_blocks": int(ldpc_blocks),
            "ldpc_nonconverged_blocks": int(ldpc_nonconverged_blocks),
            "ldpc_nonconvergence_rate": ldpc_nonconverged_blocks / max(1, ldpc_blocks),
            "ldpc_max_iterations_used": int(max_iterations),
        }
        rows.append(row)
        print(
            f"{channel_model:8s} snr={snr_db:5.1f} "
            f"ber={row['bit_error_rate']:.6g} bler={row['block_error_rate']:.6g} "
            f"nonconv={row['ldpc_nonconvergence_rate']:.6g}"
        )
    return rows


def _summarize_cliff(rows: list[dict[str, int | float | str]]) -> dict[str, float | bool | None]:
    if len(rows) < 2:
        return {"has_ber_cliff": False, "largest_ber_drop_ratio": None, "drop_start_snr_db": None}

    best_ratio = 0.0
    best_snr: float | None = None
    crosses_1e3 = False
    previous = float(rows[0]["bit_error_rate"])
    for row in rows[1:]:
        current = float(row["bit_error_rate"])
        if previous > 0.0 and current > 0.0:
            ratio = previous / current
        elif previous > 0.0 and current == 0.0:
            ratio = float("inf")
        else:
            ratio = 1.0
        if ratio > best_ratio:
            best_ratio = ratio
            best_snr = float(row["snr_db"])
        crosses_1e3 = crosses_1e3 or (previous > 1e-3 and current <= 1e-3)
        previous = current

    has_cliff = crosses_1e3 or best_ratio >= 10.0
    return {
        "has_ber_cliff": bool(has_cliff),
        "largest_ber_drop_ratio": float(best_ratio) if np.isfinite(best_ratio) else None,
        "drop_end_snr_db": best_snr,
    }


def _write_csv(rows: list[dict[str, int | float | str]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
