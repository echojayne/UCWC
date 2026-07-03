"""Deterministic verifier for session-level UCWC config plans."""

from __future__ import annotations

from typing import Any

from ucwc.config_plan import SessionConfigPlan
from ucwc.physics_tools import (
    bandwidth_to_prb,
    estimate_throughput_mbps,
    jitter_proxy_ms,
    latency_proxy_ms,
    packet_loss_proxy,
    reliability_proxy,
    required_bandwidth_mhz,
)


SECURITY_RANK = {"low": 0, "standard": 1, "high": 2, "critical": 3}


def propose_config_plan(
    tables: dict[str, list[dict[str, Any]]],
    ue_id: str,
) -> SessionConfigPlan:
    ue = _one(tables["ue_state"], "ue_id", ue_id)
    links = [
        row for row in tables["radio_link_state"] if row["ue_id"] == ue_id
    ]
    if not ue:
        raise ValueError(f"Unknown UE: {ue_id}")
    if not links:
        raise ValueError(f"No radio links for UE: {ue_id}")

    bs_rows = {row["bs_id"]: row for row in tables["base_station_state"]}
    sorted_links = sorted(links, key=lambda row: float(row["sinr_db"]), reverse=True)
    selected = sorted_links[0]
    for link in sorted_links:
        bs = bs_rows[str(link["bs_id"])]
        if int(bs["available_connections"]) > 0 or ue.get("connected_bs_id") == link["bs_id"]:
            selected = link
            break

    required_mhz = required_bandwidth_mhz(
        float(ue["min_dl_mbps"]),
        float(ue["min_ul_mbps"]),
    )
    serving_bs = bs_rows[str(selected["bs_id"])]
    bandwidth_quota = min(
        float(serving_bs["bandwidth_mhz"]) * 0.45,
        max(2.0, required_mhz * 1.5),
    )
    backup = next(
        (
            str(link["bs_id"])
            for link in sorted_links
            if str(link["bs_id"]) != str(selected["bs_id"])
        ),
        None,
    )
    return SessionConfigPlan(
        ue_id=ue_id,
        serving_bs_id=str(selected["bs_id"]),
        backup_bs_id=backup,
        bandwidth_quota_mhz=round(bandwidth_quota, 3),
        qos_profile=str(ue["profile_id"]),
        security_profile=str(ue.get("security_level", "standard")),
    )


def verify_config_plan(
    plan: SessionConfigPlan,
    tables: dict[str, list[dict[str, Any]]],
    verifier_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verifier_cfg = verifier_cfg or {}
    ue = _one(tables["ue_state"], "ue_id", plan.ue_id)
    bs = _one(tables["base_station_state"], "bs_id", plan.serving_bs_id)
    link = _one_pair(tables["radio_link_state"], "ue_id", plan.ue_id, "bs_id", plan.serving_bs_id)
    if not ue or not bs or not link:
        return {
            "passed": False,
            "checks": {
                "lookup": {
                    "passed": False,
                    "reason": "Plan references an unknown UE, BS, or radio link.",
                }
            },
        }

    min_sinr_db = float(verifier_cfg.get("min_sinr_db", -3.0))
    max_util = float(verifier_cfg.get("max_cell_utilization", 0.92))
    max_cross = float(verifier_cfg.get("max_cross_user_degradation_pct", 8.0))

    required_prb = bandwidth_to_prb(
        plan.bandwidth_quota_mhz,
        float(bs["bandwidth_mhz"]),
        int(bs["total_prb"]),
    )
    current_same_bs = ue.get("connected_bs_id") == plan.serving_bs_id
    available_prb = int(bs["available_prb"])
    available_connections = int(bs["available_connections"])
    projected_prb = int(bs["allocated_prb"]) + (0 if current_same_bs else required_prb)
    projected_util = projected_prb / max(1, int(bs["total_prb"]))
    load_for_qos = min(1.0, projected_util)
    dl_mbps = estimate_throughput_mbps(float(link["sinr_db"]), plan.bandwidth_quota_mhz)
    ul_mbps = estimate_throughput_mbps(float(link["sinr_db"]) - 3.0, plan.bandwidth_quota_mhz)
    latency_ms = latency_proxy_ms(
        load_for_qos,
        float(link["sinr_db"]),
        float(ue["mobility_speed_kmh"]),
    )
    packet_loss = packet_loss_proxy(float(link["sinr_db"]), load_for_qos)
    reliability = reliability_proxy(packet_loss)
    jitter_ms = jitter_proxy_ms(latency_ms, float(ue["mobility_speed_kmh"]))
    cross_user_degradation_pct = round(
        100.0 * plan.bandwidth_quota_mhz / max(float(bs["bandwidth_mhz"]), 1.0)
        / max(1, int(bs["connected_ue_count"])),
        3,
    )

    checks = {
        "resource": {
            "passed": (
                (current_same_bs or available_connections > 0)
                and (current_same_bs or available_prb >= required_prb)
                and projected_util <= max_util
            ),
            "required_prb": required_prb,
            "available_prb": available_prb,
            "projected_prb_utilization": round(projected_util, 5),
        },
        "radio": {
            "passed": float(link["sinr_db"]) >= min_sinr_db,
            "sinr_db": float(link["sinr_db"]),
            "min_sinr_db": min_sinr_db,
        },
        "qos": {
            "passed": (
                dl_mbps >= float(ue["min_dl_mbps"])
                and ul_mbps >= float(ue["min_ul_mbps"])
                and latency_ms <= float(ue["max_latency_ms"])
                and reliability >= float(ue["min_reliability"])
                and packet_loss <= float(ue["max_packet_loss"])
                and jitter_ms <= float(ue["max_jitter_ms"])
            ),
            "predicted_dl_mbps": dl_mbps,
            "required_dl_mbps": float(ue["min_dl_mbps"]),
            "predicted_ul_mbps": ul_mbps,
            "required_ul_mbps": float(ue["min_ul_mbps"]),
            "predicted_latency_ms": latency_ms,
            "max_latency_ms": float(ue["max_latency_ms"]),
            "predicted_reliability": reliability,
            "min_reliability": float(ue["min_reliability"]),
            "predicted_packet_loss": packet_loss,
            "max_packet_loss": float(ue["max_packet_loss"]),
            "predicted_jitter_ms": jitter_ms,
            "max_jitter_ms": float(ue["max_jitter_ms"]),
        },
        "security": {
            "passed": _security_rank(plan.security_profile) >= _security_rank(str(ue["security_level"])),
            "required_security": str(ue["security_level"]),
            "planned_security": plan.security_profile,
        },
        "handover": {
            "passed": float(ue["mobility_speed_kmh"]) < 80.0 or plan.backup_bs_id is not None,
            "mobility_speed_kmh": float(ue["mobility_speed_kmh"]),
            "backup_bs_id": plan.backup_bs_id,
        },
        "cross_user": {
            "passed": cross_user_degradation_pct <= max_cross,
            "estimated_cross_user_degradation_pct": cross_user_degradation_pct,
            "max_cross_user_degradation_pct": max_cross,
        },
    }
    passed = all(bool(item["passed"]) for item in checks.values())
    return {
        "passed": passed,
        "plan": plan.to_record(),
        "checks": checks,
        "failure_reasons": [
            name for name, item in checks.items() if not bool(item["passed"])
        ],
    }


def _security_rank(level: str) -> int:
    return SECURITY_RANK.get(str(level), SECURITY_RANK["standard"])


def _one(rows: list[dict[str, Any]], key: str, value: Any) -> dict[str, Any] | None:
    for row in rows:
        if row.get(key) == value:
            return row
    return None


def _one_pair(
    rows: list[dict[str, Any]],
    key_a: str,
    value_a: Any,
    key_b: str,
    value_b: Any,
) -> dict[str, Any] | None:
    for row in rows:
        if row.get(key_a) == value_a and row.get(key_b) == value_b:
            return row
    return None
