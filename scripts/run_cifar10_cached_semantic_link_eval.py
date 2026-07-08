#!/usr/bin/env python
"""Run CIFAR10 semantic link evaluation from cached ViT-B/16 layer features."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ucwc.semantic.evaluation import SemanticEvalRow, estimate_transmission_volume  # noqa: E402
from ucwc.semantic.ldpc import LDPCCode  # noqa: E402
from ucwc.semantic.link import transmit_semantic_features  # noqa: E402
from ucwc.semantic.models import load_semantic_decoder  # noqa: E402
from ucwc.semantic.qam import QAMModem  # noqa: E402
from ucwc.semantic.storage import write_experience_rows  # noqa: E402


DEFAULT_LAYERS = list(range(1, 13))
DEFAULT_SNR_DB = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
DEFAULT_CODE_RATES = [0.5, 2.0 / 3.0, 0.75]
DEFAULT_QAM_ORDERS = [4, 16, 64, 256]
DEFAULT_QUANTIZATION_BITS = [2, 4, 8]
DEFAULT_HEAD_DIR = PROJECT_ROOT / "ucwc" / "semantic" / "cifar10_vitb16_layer_heads"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="data/cifar10_vitb16_layer_features")
    parser.add_argument("--output-dir", default="outputs/semantic_cifar10_vitb16_full_grid")
    parser.add_argument("--ckpt-dir", default=str(DEFAULT_HEAD_DIR))
    parser.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--snr-db", nargs="+", type=float, default=DEFAULT_SNR_DB)
    parser.add_argument("--code-rates", nargs="+", type=float, default=DEFAULT_CODE_RATES)
    parser.add_argument("--qam-orders", nargs="+", type=int, default=DEFAULT_QAM_ORDERS)
    parser.add_argument("--sample-count", type=int, default=10000)
    parser.add_argument("--sample-seed", type=int, default=-1)
    parser.add_argument(
        "--quantization-bits",
        nargs="+",
        type=int,
        default=DEFAULT_QUANTIZATION_BITS,
    )
    parser.add_argument("--header-bits", type=int, default=8)
    parser.add_argument("--channel-model", choices=["awgn", "rayleigh", "rician"], default="awgn")
    parser.add_argument("--ldpc-iterations", type=int, default=25)
    parser.add_argument("--quantizer-clip-value", type=float, default=3.0)
    parser.add_argument("--decoder-batch-size", type=int, default=4096)
    parser.add_argument("--decoder-device", default="cuda:1")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0 selects a conservative automatic value.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_dir = _resolve_project_path(args.feature_dir)
    output_dir = _resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"semantic_cifar10_{int(args.sample_count)}_link_experience.csv"
    sqlite_path = output_dir / f"semantic_cifar10_{int(args.sample_count)}_link_results.sqlite"
    manifest_path = output_dir / "manifest.json"
    if csv_path.exists() and not args.overwrite:
        raise FileExistsError(f"{csv_path} already exists; rerun with --overwrite.")

    layers = _normalize_layers(args.layers)
    indices, labels, features_by_layer, feature_source = _load_feature_bundle(
        feature_dir=feature_dir,
        layers=layers,
        sample_count=int(args.sample_count),
        sample_seed=int(args.sample_seed),
    )
    valid_modes, skipped_modes = _split_valid_modes(args.code_rates, args.qam_orders)
    if not valid_modes:
        raise ValueError("No valid LDPC/QAM modes remain after compatibility checks.")

    original_preds_by_layer = _predict_original_by_layer(
        layers=layers,
        ckpt_dir=Path(args.ckpt_dir),
        features_by_layer=features_by_layer,
        batch_size=int(args.decoder_batch_size),
        device=_resolve_decoder_device(str(args.decoder_device)),
    )
    original_accuracy_by_layer = {
        layer: float(np.mean(original_preds_by_layer[layer] == labels))
        for layer in layers
    }

    tasks = _build_tasks(
        args=args,
        layers=layers,
        labels=labels,
        features_by_layer=features_by_layer,
        original_preds_by_layer=original_preds_by_layer,
        original_accuracy_by_layer=original_accuracy_by_layer,
        valid_modes=valid_modes,
    )
    workers = _resolve_worker_count(int(args.workers), len(tasks))
    manifest: dict[str, Any] = {
        "dataset": "cifar10",
        "feature_dir": str(feature_dir),
        "feature_source": str(feature_source),
        "ckpt_dir": str(Path(args.ckpt_dir).expanduser()),
        "output_dir": str(output_dir),
        "layers": layers,
        "sample_count": int(len(indices)),
        "sample_seed": int(args.sample_seed),
        "sample_indices_file": str(output_dir / "sample_indices.npy"),
        "snr_db": [float(value) for value in args.snr_db],
        "code_rates": [float(value) for value in args.code_rates],
        "qam_orders": [int(value) for value in args.qam_orders],
        "valid_modes": [{"ldpc_code_rate": rate, "qam_order": qam} for rate, qam in valid_modes],
        "skipped_modes": skipped_modes,
        "quantization_bits": [int(value) for value in args.quantization_bits],
        "header_bits_per_sample": int(args.header_bits),
        "channel_model": str(args.channel_model),
        "ldpc_iterations": int(args.ldpc_iterations),
        "quantizer_clip_value": float(args.quantizer_clip_value),
        "decoder_device": str(_resolve_decoder_device(str(args.decoder_device))),
        "normalized_features": True,
        "normalize_call": "torch.nn.functional.normalize(features, dim=-1)",
        "workers": workers,
        "started_unix_time": time.time(),
    }
    np.save(output_dir / "sample_indices.npy", indices)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    rows = _run_tasks(tasks, workers=workers, csv_path=csv_path)
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            int(row["encoder_depth"]),
            int(row["quantization_bits"]),
            float(row["ldpc_code_rate"]),
            int(row["qam_order"]),
            float(row["snr_db"]),
        ),
    )
    semantic_rows = [SemanticEvalRow(**row) for row in sorted_rows]
    result = write_experience_rows(
        semantic_rows,
        sqlite_path=sqlite_path,
        csv_path=csv_path,
        replace=True,
    )
    manifest["finished_unix_time"] = time.time()
    manifest["row_count"] = len(sorted_rows)
    manifest["csv_path"] = str(csv_path)
    manifest["sqlite_path"] = str(sqlite_path)
    manifest["original_accuracy_by_layer"] = {
        str(layer): original_accuracy_by_layer[layer] for layer in layers
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def _normalize_layers(layers: list[int]) -> list[int]:
    normalized = sorted({int(layer) for layer in layers})
    if not normalized or normalized[0] < 1 or normalized[-1] > 12:
        raise ValueError(f"layers must be in [1, 12], got {layers}")
    return normalized


def _load_feature_bundle(
    *,
    feature_dir: Path,
    layers: list[int],
    sample_count: int,
    sample_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], Path]:
    pt_path = feature_dir / "cifar10_test_vitb16_all_layers_features.pt"
    if pt_path.exists():
        payload = _torch_load_tensor_payload(pt_path)
        features = payload["features"].float()
        labels = payload["labels"].long().cpu().numpy().astype(np.int64)
        indices = _select_indices(len(labels), sample_count, sample_seed)
        selected_labels = labels[indices]
        selected_by_layer: dict[int, np.ndarray] = {}
        for layer in layers:
            layer_features = features[indices, layer - 1, :].float()
            layer_features = F.normalize(layer_features, dim=-1, eps=1e-12)
            selected_by_layer[layer] = layer_features.cpu().numpy().astype(np.float32)
        return indices, selected_labels, selected_by_layer, pt_path

    legacy_test_dir = feature_dir / "test"
    labels_path = legacy_test_dir / "labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(f"Cannot find {pt_path} or {labels_path}.")
    labels = np.load(labels_path)
    indices = _select_indices(len(labels), sample_count, sample_seed)
    selected_by_layer = {}
    for layer in layers:
        layer_path = legacy_test_dir / f"features_layer{layer:02d}.npy"
        layer_features = np.asarray(np.load(layer_path, mmap_mode="r")[indices], dtype=np.float32)
        tensor = F.normalize(torch.from_numpy(layer_features).float(), dim=-1, eps=1e-12)
        selected_by_layer[layer] = tensor.cpu().numpy().astype(np.float32)
    return indices, labels[indices].astype(np.int64), selected_by_layer, legacy_test_dir


def _torch_load_tensor_payload(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "features" not in payload or "labels" not in payload:
        raise ValueError(f"{path} must contain a dict with 'features' and 'labels'.")
    return payload


def _select_indices(total: int, sample_count: int, sample_seed: int) -> np.ndarray:
    if sample_count <= 0 or sample_count > total:
        raise ValueError(f"sample_count must be in [1, {total}], got {sample_count}")
    if sample_seed < 0:
        return np.arange(sample_count, dtype=np.int64)
    rng = np.random.default_rng(sample_seed)
    return np.sort(rng.choice(total, size=sample_count, replace=False)).astype(np.int64)


def _split_valid_modes(
    code_rates: list[float],
    qam_orders: list[int],
) -> tuple[list[tuple[float, int]], list[dict[str, Any]]]:
    valid: list[tuple[float, int]] = []
    skipped: list[dict[str, Any]] = []
    for rate in code_rates:
        code = LDPCCode.for_rate(float(rate))
        for qam_order in qam_orders:
            modem = QAMModem(int(qam_order))
            if code.n % modem.bits_per_symbol != 0:
                skipped.append(
                    {
                        "ldpc_code_rate": float(rate),
                        "qam_order": int(qam_order),
                        "bits_per_symbol": int(modem.bits_per_symbol),
                        "ldpc_block_length": int(code.n),
                        "reason": "LDPC block length not divisible by QAM bits/symbol",
                    }
                )
                continue
            valid.append((float(rate), int(qam_order)))
    return valid, skipped


def _predict_original_by_layer(
    *,
    layers: list[int],
    ckpt_dir: Path,
    features_by_layer: dict[int, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    preds_by_layer: dict[int, np.ndarray] = {}
    for layer in layers:
        decoder_path = _decoder_checkpoint_path(layer, ckpt_dir)
        decoder = load_semantic_decoder(decoder_path, device=device)
        preds_by_layer[layer] = _predict(
            decoder,
            features_by_layer[layer],
            device=device,
            batch_size=batch_size,
        )
    return preds_by_layer


def _build_tasks(
    *,
    args: argparse.Namespace,
    layers: list[int],
    labels: np.ndarray,
    features_by_layer: dict[int, np.ndarray],
    original_preds_by_layer: dict[int, np.ndarray],
    original_accuracy_by_layer: dict[int, float],
    valid_modes: list[tuple[float, int]],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for layer in layers:
        for q_bits in args.quantization_bits:
            for rate, qam_order in valid_modes:
                for snr_index, snr_db in enumerate(args.snr_db):
                    tasks.append(
                        {
                            "layer": int(layer),
                            "features": features_by_layer[layer],
                            "labels": labels,
                            "original_preds": original_preds_by_layer[layer],
                            "original_accuracy": float(original_accuracy_by_layer[layer]),
                            "decoder_path": str(
                                _decoder_checkpoint_path(layer, Path(args.ckpt_dir))
                            ),
                            "quantization_bits": int(q_bits),
                            "header_bits": int(args.header_bits),
                            "ldpc_code_rate": float(rate),
                            "qam_order": int(qam_order),
                            "snr_db": float(snr_db),
                            "channel_model": str(args.channel_model),
                            "seed": int(
                                1_000_000 * layer
                                + 100_000 * int(q_bits)
                                + 1_000 * _rate_seed(rate)
                                + 10 * int(qam_order)
                                + snr_index
                            ),
                            "ldpc_iterations": int(args.ldpc_iterations),
                            "quantizer_clip_value": float(args.quantizer_clip_value),
                            "decoder_batch_size": int(args.decoder_batch_size),
                            "decoder_device": str(
                                _resolve_decoder_device(str(args.decoder_device))
                            ),
                        }
                    )
    return tasks


def _resolve_worker_count(requested: int, task_count: int) -> int:
    if task_count <= 1:
        return 1
    if requested > 0:
        return max(1, min(requested, task_count))
    cpu_count = os.cpu_count() or 1
    return max(1, min(task_count, 12, max(1, cpu_count // 2)))


def _run_tasks(
    tasks: list[dict[str, Any]],
    *,
    workers: int,
    csv_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] | None = None
    if csv_path.exists():
        csv_path.unlink()
    started = time.perf_counter()
    completed = 0

    if workers == 1:
        for task in tasks:
            row = _evaluate_case(task)
            completed += 1
            rows.append(row)
            fieldnames = _append_row(csv_path, row, fieldnames)
            _print_progress(completed, len(tasks), row, started)
        return rows

    groups = _group_tasks_by_layer_quant(tasks)
    workers = min(workers, len(groups))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
        future_to_group = {
            executor.submit(_evaluate_task_group, group): group_key
            for group_key, group in groups.items()
        }
        for future in as_completed(future_to_group):
            group_rows = future.result()
            for row in group_rows:
                completed += 1
                rows.append(row)
                fieldnames = _append_row(csv_path, row, fieldnames)
                _print_progress(completed, len(tasks), row, started)
    return rows


def _group_tasks_by_layer_quant(
    tasks: list[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for task in tasks:
        key = (int(task["layer"]), int(task["quantization_bits"]))
        groups.setdefault(key, []).append(task)
    return groups


def _evaluate_task_group(tasks: list[dict[str, Any]]) -> list[dict[str, int | float | str]]:
    if not tasks:
        return []
    torch.set_num_threads(1)
    device = _resolve_decoder_device(str(tasks[0]["decoder_device"]))
    decoder = load_semantic_decoder(tasks[0]["decoder_path"], device=device)
    rows: list[dict[str, int | float | str]] = []
    for index, task in enumerate(tasks, start=1):
        row = _evaluate_case(task, decoder=decoder)
        rows.append(row)
        print(
            "layer-worker "
            f"layer={int(row['encoder_depth'])} local_case={index}/{len(tasks)} "
            f"q={int(row['quantization_bits'])} "
            f"rate={float(row['ldpc_code_rate']):.4g} qam={int(row['qam_order'])} "
            f"snr={float(row['snr_db']):g} ber={float(row['bit_error_rate']):.4g} "
            f"sem={float(row['semantic_score']):.4g}",
            flush=True,
        )
    return rows


def _evaluate_case(
    task: dict[str, Any],
    decoder: torch.nn.Module | None = None,
) -> dict[str, int | float | str]:
    torch.set_num_threads(1)
    features = np.asarray(task["features"], dtype=np.float32)
    labels = np.asarray(task["labels"], dtype=np.int64)
    original_preds = np.asarray(task["original_preds"], dtype=np.int64)
    if decoder is None:
        decoder = load_semantic_decoder(
            task["decoder_path"],
            device=_resolve_decoder_device(str(task["decoder_device"])),
        )

    start = time.perf_counter()
    transmission = transmit_semantic_features(
        features,
        quantization_bits=int(task["quantization_bits"]),
        ldpc_code_rate=float(task["ldpc_code_rate"]),
        qam_order=int(task["qam_order"]),
        snr_db=float(task["snr_db"]),
        channel_model=str(task["channel_model"]),
        seed=int(task["seed"]),
        ldpc_iterations=int(task["ldpc_iterations"]),
        quantizer_clip_value=float(task["quantizer_clip_value"]),
    )
    link_latency_ms = (time.perf_counter() - start) * 1000.0
    recovered_features = transmission.recovered_features.astype(np.float32)

    start = time.perf_counter()
    recovered_preds = _predict(
        decoder,
        recovered_features,
        device=_resolve_decoder_device(str(task["decoder_device"])),
        batch_size=int(task["decoder_batch_size"]),
    )
    decoder_latency_ms = (time.perf_counter() - start) * 1000.0

    bit_result = transmission.bit_result
    volume = estimate_transmission_volume(
        feature_payload_bits=bit_result.transmitted_bits,
        header_bits_per_sample=int(task["header_bits"]),
        sample_count=int(len(labels)),
        coded_bits=bit_result.coded_bits,
        ldpc_padding_bits=bit_result.padding_bits,
        ldpc_code_rate=float(task["ldpc_code_rate"]),
        qam_order=int(task["qam_order"]),
    )
    layer = int(task["layer"])
    rate = float(task["ldpc_code_rate"])
    qam_order = int(task["qam_order"])
    snr_db = float(task["snr_db"])
    row = SemanticEvalRow(
        config_id=(
            f"cifar10_vitb16_1000_d{layer}_q{int(task['quantization_bits'])}_"
            f"r{_rate_label(rate)}_qam{qam_order}_snr{snr_db:g}"
        ),
        encoder_depth=layer,
        quantization_bits=int(task["quantization_bits"]),
        ldpc_code_rate=rate,
        qam_order=qam_order,
        snr_db=snr_db,
        channel_model=str(task["channel_model"]),
        dataset="cifar10_vitb16_layer_features_test",
        batch_size=int(len(labels)),
        feature_dim=int(features.shape[1]),
        payload_bits=bit_result.transmitted_bits,
        header_bits=int(task["header_bits"]),
        payload_bits_with_header=int(volume["payload_bits_with_header"]),
        coded_bits=bit_result.coded_bits,
        coded_bits_with_header=int(volume["coded_bits_with_header"]),
        qam_bits_per_symbol=int(volume["qam_bits_per_symbol"]),
        qam_symbols=int(volume["qam_symbols"]),
        qam_symbols_with_header=int(volume["qam_symbols_with_header"]),
        payload_bytes=float(volume["payload_bytes"]),
        payload_bytes_with_header=float(volume["payload_bytes_with_header"]),
        coded_bytes=float(volume["coded_bytes"]),
        coded_bytes_with_header=float(volume["coded_bytes_with_header"]),
        ldpc_padding_bits=bit_result.padding_bits,
        bit_error_rate=bit_result.bit_error_rate,
        bit_errors=bit_result.bit_errors,
        block_errors=bit_result.block_errors,
        block_error_rate=bit_result.block_error_rate,
        ldpc_nonconvergence_rate=bit_result.ldpc_nonconvergence_rate,
        ldpc_blocks=bit_result.ldpc_blocks,
        ldpc_converged_blocks=bit_result.ldpc_converged_blocks,
        ldpc_max_iterations_used=bit_result.ldpc_max_iterations_used,
        semantic_score=_cosine_mean(features, recovered_features),
        classifier_agreement=float(np.mean(original_preds == recovered_preds)),
        original_accuracy=float(task["original_accuracy"]),
        recovered_accuracy=float(np.mean(recovered_preds == labels)),
        encoding_latency_ms=0.0,
        decoding_latency_ms=link_latency_ms + decoder_latency_ms,
    )
    return row.to_dict()


def _predict(
    decoder: torch.nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    preds: list[np.ndarray] = []
    decoder.to(device)
    decoder.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            batch = torch.from_numpy(np.asarray(features[start:stop], dtype=np.float32)).to(device)
            logits = decoder(batch).float()
            preds.append(logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64))
    return np.concatenate(preds)


def _cosine_mean(left: np.ndarray, right: np.ndarray) -> float:
    left_t = torch.from_numpy(np.asarray(left, dtype=np.float32))
    right_t = torch.from_numpy(np.asarray(right, dtype=np.float32))
    return float(F.cosine_similarity(left_t, right_t, dim=1).mean().item())


def _append_row(
    path: Path,
    row: dict[str, int | float | str],
    fieldnames: list[str] | None,
) -> list[str]:
    if fieldnames is None:
        fieldnames = list(row.keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        return fieldnames
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)
    return fieldnames


def _print_progress(
    completed: int,
    total: int,
    row: dict[str, int | float | str],
    started: float,
) -> None:
    elapsed = time.perf_counter() - started
    print(
        "case "
        f"{completed}/{total}: layer={int(row['encoder_depth'])} "
        f"q={int(row['quantization_bits'])} "
        f"rate={float(row['ldpc_code_rate']):.4g} qam={int(row['qam_order'])} "
        f"snr={float(row['snr_db']):g} ber={float(row['bit_error_rate']):.4g} "
        f"bler={float(row['block_error_rate']):.4g} sem={float(row['semantic_score']):.4g} "
        f"acc={float(row['recovered_accuracy']):.4g} elapsed={elapsed:.1f}s",
        flush=True,
    )


def _rate_seed(rate: float) -> int:
    return int(round(float(rate) * 1000))


def _rate_label(rate: float) -> str:
    if abs(rate - 0.5) < 1e-6:
        return "12"
    if abs(rate - 2.0 / 3.0) < 1e-3:
        return "23"
    if abs(rate - 0.75) < 1e-6:
        return "34"
    return str(rate).replace(".", "p")


def _decoder_checkpoint_path(layer: int, ckpt_dir: Path) -> Path:
    ckpt_dir = _resolve_project_path(ckpt_dir)
    candidates = [
        ckpt_dir / f"decoder_layer{int(layer)}.pth",
        ckpt_dir / f"decoder_layer{int(layer):02d}.pth",
        ckpt_dir / f"cifar10_vitb16_layer{int(layer):02d}_linear_head.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No decoder checkpoint for layer {layer}; tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _resolve_decoder_device(value: str) -> torch.device:
    if value == "auto":
        device = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
        return torch.device(device)
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {value}, but CUDA is not available.")
    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError(
            f"Requested {value}, but only {torch.cuda.device_count()} CUDA devices exist."
        )
    return device


if __name__ == "__main__":
    raise SystemExit(main())
