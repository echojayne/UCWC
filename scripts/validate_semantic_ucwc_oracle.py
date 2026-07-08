#!/usr/bin/env python
"""Validate growing-admission satisfaction under a global oracle."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ucwc.state.rule_oracle import predict_task_score, required_bandwidth_mhz
from ucwc.state.semantic_state import load_state_database


@dataclass(frozen=True, slots=True)
class CandidateOption:
    request_id: str
    ue_id: str
    arrival_order: int
    bs_id: str
    semantic_config_id: str
    phy_mode_id: str
    bandwidth_mhz: float
    tx_latency_ms: float
    total_latency_ms: float
    non_tx_latency_ms: float
    radio_metric_db: float
    predicted_task_score: float
    required_task_score: float
    spectral_efficiency_bps_hz: float


@dataclass(frozen=True, slots=True)
class OracleResult:
    prefix_size: int
    accepted_count: int
    selected_options: list[CandidateOption]
    solver: str
    optimal: bool


def main() -> None:
    args = _parse_args()
    state = load_state_database(_resolve_path(args.state_db))
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requests = sorted(
        state["ue_request_queue"],
        key=lambda row: int(row["arrival_order"]),
    )
    base_stations = sorted(state["base_station_state"], key=lambda row: str(row["bs_id"]))
    bs_budgets = {str(row["bs_id"]): float(row["bandwidth_budget_mhz"]) for row in base_stations}
    options_by_ue = _build_candidate_options(
        state,
        args.radio_metric,
        args.resource_efficiency,
    )
    solver_mode = _solver_mode(options_by_ue)
    prefixes = _prefixes(
        len(requests),
        args.prefix_step,
        [*args.prefixes, *args.checkpoint_prefixes],
    )

    results: list[OracleResult] = []
    for prefix_size in prefixes:
        result = _solve_prefix(
            prefix_size=prefix_size,
            requests=requests,
            options_by_ue=options_by_ue,
            bs_budgets=bs_budgets,
            solver_mode=solver_mode,
            time_limit_s=args.time_limit_s,
        )
        results.append(result)

    curve_rows = [_curve_row(result, bs_budgets) for result in results]
    _write_csv(output_dir / "oracle_prefix_satisfaction.csv", curve_rows)

    final_result = results[-1]
    assignment_rows = [_assignment_row(option) for option in final_result.selected_options]
    _write_csv(output_dir / "oracle_final_assignments.csv", assignment_rows)

    summary = _build_summary(
        results,
        len(requests),
        solver_mode,
        args.radio_metric,
        args.resource_efficiency,
    )
    (output_dir / "oracle_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"wrote_prefix_curve={output_dir / 'oracle_prefix_satisfaction.csv'}")
    print(f"wrote_final_assignments={output_dir / 'oracle_final_assignments.csv'}")
    print(f"wrote_summary={output_dir / 'oracle_summary.json'}")
    print(
        "final="
        f"{summary['final_accepted_users']}/{summary['max_users']} "
        f"ratio={summary['final_satisfaction_ratio']:.6f} solver={solver_mode}"
    )
    print(f"last_full_satisfaction_prefix={summary['last_full_satisfaction_prefix']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-db", default="outputs/state/semantic_5bs_120ue/state.sqlite")
    parser.add_argument(
        "--output-dir",
        default="outputs/state/semantic_5bs_120ue/oracle_validation",
    )
    parser.add_argument("--prefix-step", type=int, default=0)
    parser.add_argument("--prefixes", nargs="*", type=int, default=[1, 20, 40, 50, 80, 100, 120])
    parser.add_argument("--checkpoint-prefixes", nargs="*", type=int, default=[])
    parser.add_argument("--radio-metric", choices=["sinr_db", "snr_db"], default="sinr_db")
    parser.add_argument("--resource-efficiency", type=float, default=0.85)
    parser.add_argument("--time-limit-s", type=float, default=120.0)
    return parser.parse_args()


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _build_candidate_options(
    state: dict[str, list[dict[str, Any]]],
    radio_metric: str,
    resource_efficiency: float,
) -> dict[str, list[CandidateOption]]:
    semantic_rows = state["semantic_config_catalog"]
    phy_rows = state["phy_mode_catalog"]
    radio_by_ue: dict[str, list[dict[str, Any]]] = {}
    for row in state["radio_link_state"]:
        radio_by_ue.setdefault(str(row["ue_id"]), []).append(row)

    options_by_ue: dict[str, list[CandidateOption]] = {}
    for request in state["ue_request_queue"]:
        ue_id = str(request["ue_id"])
        min_task_score = float(request["min_task_score"])
        max_total_latency_ms = float(request["max_total_latency_ms"])
        best_by_bs: dict[str, CandidateOption] = {}

        for semantic in semantic_rows:
            semantic_latency_ms = float(semantic["encoding_latency_ms"]) + float(
                semantic["decoding_latency_ms"]
            ) + float(
                semantic.get("fixed_latency_ms", 0.0)
            )
            tx_budget_ms = max_total_latency_ms - semantic_latency_ms
            if tx_budget_ms <= 0.0:
                continue

            for radio in radio_by_ue.get(ue_id, []):
                metric_db = float(radio[radio_metric])
                bs_id = str(radio["bs_id"])
                for phy in phy_rows:
                    predicted_task_score = predict_task_score(
                        encoder_depth=int(semantic["encoder_depth"]),
                        quantization_bits=int(semantic["quantization_bits"]),
                        ldpc_code_rate=float(phy["ldpc_code_rate"]),
                        qam_order=int(phy["qam_order"]),
                        snr_db=metric_db,
                        task_type=str(request["task_type"]),
                        reference_snr_db=float(phy["reference_snr_db"]),
                    )
                    if predicted_task_score < min_task_score:
                        continue
                    spectral_efficiency = float(phy["spectral_efficiency_bps_hz"])
                    bandwidth_mhz = required_bandwidth_mhz(
                        payload_bits=float(semantic["payload_bits"]),
                        tx_budget_ms=tx_budget_ms,
                        ldpc_code_rate=float(phy["ldpc_code_rate"]),
                        qam_order=int(phy["qam_order"]),
                        resource_efficiency=resource_efficiency,
                    )
                    option = CandidateOption(
                        request_id=str(request["request_id"]),
                        ue_id=ue_id,
                        arrival_order=int(request["arrival_order"]),
                        bs_id=bs_id,
                        semantic_config_id=str(semantic["config_id"]),
                        phy_mode_id=str(phy["phy_mode_id"]),
                        bandwidth_mhz=bandwidth_mhz,
                        tx_latency_ms=tx_budget_ms,
                        total_latency_ms=max_total_latency_ms,
                        non_tx_latency_ms=semantic_latency_ms,
                        radio_metric_db=metric_db,
                        predicted_task_score=predicted_task_score,
                        required_task_score=min_task_score,
                        spectral_efficiency_bps_hz=spectral_efficiency,
                    )
                    previous = best_by_bs.get(bs_id)
                    if previous is None or option.bandwidth_mhz < previous.bandwidth_mhz:
                        best_by_bs[bs_id] = option

        options_by_ue[ue_id] = sorted(best_by_bs.values(), key=lambda option: option.bs_id)
    return options_by_ue


def _solver_mode(options_by_ue: dict[str, list[CandidateOption]]) -> str:
    max_candidate_bs = max(
        (len({option.bs_id for option in options}) for options in options_by_ue.values()),
        default=0,
    )
    if max_candidate_bs <= 1:
        return "unique_bs_exact"
    return "milp"


def _prefixes(total: int, prefix_step: int, checkpoints: list[int]) -> list[int]:
    if prefix_step < 0:
        raise ValueError("--prefix-step must be non-negative")
    values = set(range(prefix_step, total + 1, prefix_step)) if prefix_step else set()
    values.add(total)
    for checkpoint in checkpoints:
        if checkpoint < 1 or checkpoint > total:
            raise ValueError(f"Checkpoint prefix {checkpoint} is outside [1, {total}]")
        values.add(checkpoint)
    return sorted(values)


def _solve_prefix(
    *,
    prefix_size: int,
    requests: list[dict[str, Any]],
    options_by_ue: dict[str, list[CandidateOption]],
    bs_budgets: dict[str, float],
    solver_mode: str,
    time_limit_s: float,
) -> OracleResult:
    active_ue_ids = [str(row["ue_id"]) for row in requests[:prefix_size]]
    if solver_mode == "unique_bs_exact":
        selected = _solve_unique_bs_exact(active_ue_ids, options_by_ue, bs_budgets)
        return OracleResult(prefix_size, len(selected), selected, solver_mode, True)
    selected = _solve_milp(active_ue_ids, options_by_ue, bs_budgets, time_limit_s)
    return OracleResult(prefix_size, len(selected), selected, solver_mode, True)


def _solve_unique_bs_exact(
    active_ue_ids: list[str],
    options_by_ue: dict[str, list[CandidateOption]],
    bs_budgets: dict[str, float],
) -> list[CandidateOption]:
    selected: list[CandidateOption] = []
    for bs_id, budget_mhz in bs_budgets.items():
        candidates = [
            options_by_ue[ue_id][0]
            for ue_id in active_ue_ids
            if len(options_by_ue.get(ue_id, [])) == 1 and options_by_ue[ue_id][0].bs_id == bs_id
        ]
        used_mhz = 0.0
        for option in sorted(candidates, key=lambda item: (item.bandwidth_mhz, item.arrival_order)):
            if used_mhz + option.bandwidth_mhz <= budget_mhz + 1e-9:
                selected.append(option)
                used_mhz += option.bandwidth_mhz
    return sorted(selected, key=lambda option: option.arrival_order)


def _solve_milp(
    active_ue_ids: list[str],
    options_by_ue: dict[str, list[CandidateOption]],
    bs_budgets: dict[str, float],
    time_limit_s: float,
) -> list[CandidateOption]:
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as exc:
        raise RuntimeError("Multi-BS oracle validation requires scipy.optimize.milp.") from exc

    options = [
        option
        for ue_id in active_ue_ids
        for option in options_by_ue.get(ue_id, [])
        if option.bandwidth_mhz <= bs_budgets[option.bs_id] + 1e-9
    ]
    if not options:
        return []

    ue_index = {ue_id: index for index, ue_id in enumerate(active_ue_ids)}
    bs_ids = list(bs_budgets)
    bs_index = {bs_id: index for index, bs_id in enumerate(bs_ids)}
    row_count = len(active_ue_ids) + len(bs_ids)
    matrix = lil_matrix((row_count, len(options)), dtype=float)
    upper_bounds = [1.0] * len(active_ue_ids) + [bs_budgets[bs_id] for bs_id in bs_ids]

    for column, option in enumerate(options):
        matrix[ue_index[option.ue_id], column] = 1.0
        matrix[len(active_ue_ids) + bs_index[option.bs_id], column] = option.bandwidth_mhz

    objective = np.array([-1.0 + 1e-6 * option.bandwidth_mhz for option in options], dtype=float)
    result = milp(
        c=objective,
        constraints=LinearConstraint(matrix.tocsr(), -np.inf, np.array(upper_bounds)),
        bounds=Bounds(0.0, 1.0),
        integrality=np.ones(len(options)),
        options={"time_limit": time_limit_s, "disp": False},
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"MILP oracle failed for {len(active_ue_ids)} active UEs: {result.message}"
        )
    return sorted(
        [option for option, value in zip(options, result.x, strict=True) if value > 0.5],
        key=lambda option: option.arrival_order,
    )


def _curve_row(result: OracleResult, bs_budgets: dict[str, float]) -> dict[str, Any]:
    used_by_bs = {bs_id: 0.0 for bs_id in bs_budgets}
    for option in result.selected_options:
        used_by_bs[option.bs_id] += option.bandwidth_mhz

    row: dict[str, Any] = {
        "prefix_size": result.prefix_size,
        "accepted_users": result.accepted_count,
        "unmet_users": result.prefix_size - result.accepted_count,
        "satisfaction_ratio": result.accepted_count / max(1, result.prefix_size),
        "used_total_mhz": sum(used_by_bs.values()),
        "solver": result.solver,
        "optimal": int(result.optimal),
    }
    for bs_id in bs_budgets:
        row[f"used_{bs_id}_mhz"] = used_by_bs[bs_id]
        row[f"util_{bs_id}"] = used_by_bs[bs_id] / bs_budgets[bs_id]
    return row


def _assignment_row(option: CandidateOption) -> dict[str, Any]:
    return {
        "request_id": option.request_id,
        "ue_id": option.ue_id,
        "arrival_order": option.arrival_order,
        "serving_bs_id": option.bs_id,
        "semantic_config_id": option.semantic_config_id,
        "phy_mode_id": option.phy_mode_id,
        "bandwidth_mhz": option.bandwidth_mhz,
        "tx_latency_ms": option.tx_latency_ms,
        "total_latency_ms": option.total_latency_ms,
        "non_tx_latency_ms": option.non_tx_latency_ms,
        "radio_metric_db": option.radio_metric_db,
        "predicted_task_score": option.predicted_task_score,
        "required_task_score": option.required_task_score,
        "spectral_efficiency_bps_hz": option.spectral_efficiency_bps_hz,
    }


def _build_summary(
    results: list[OracleResult],
    max_users: int,
    solver_mode: str,
    radio_metric: str,
    resource_efficiency: float,
) -> dict[str, Any]:
    final = results[-1]
    full_prefixes = [
        result.prefix_size for result in results if result.accepted_count == result.prefix_size
    ]
    first_below_full = next(
        (result.prefix_size for result in results if result.accepted_count < result.prefix_size),
        None,
    )
    return {
        "max_users": max_users,
        "evaluated_prefix_count": len(results),
        "solver": solver_mode,
        "oracle_type": "rule_based_global_optimum",
        "radio_metric": radio_metric,
        "resource_efficiency": resource_efficiency,
        "semantic_feasibility": "predicted_task_score >= min_task_score",
        "latency_formula": "encoding_latency_ms + tx_latency_ms + fixed_latency_ms",
        "active_users_are_prefix_persistent": True,
        "last_full_satisfaction_prefix": max(full_prefixes) if full_prefixes else 0,
        "first_below_full_satisfaction_prefix": first_below_full,
        "final_accepted_users": final.accepted_count,
        "final_satisfaction_ratio": final.accepted_count / max(1, final.prefix_size),
        "final_unmet_users": final.prefix_size - final.accepted_count,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0])
    else:
        fieldnames = []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
