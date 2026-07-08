from __future__ import annotations

import shutil
from pathlib import Path

from ucwc.agent.candidates import build_candidate_grid, evaluate_candidates, select_best_candidate
from ucwc.agent.sql_tools import connect_writable, execute_readonly_sql, validate_readonly_sql
from ucwc.state import metadata_rows, write_state_database
from ucwc.verifier import commit_verified_candidate


def test_readonly_sql_validation_rejects_mutation(tmp_path: Path) -> None:
    state_db = _make_state_db(tmp_path)
    with connect_writable(state_db) as connection:
        ok, reason = validate_readonly_sql(connection, "DELETE FROM active_session")
        assert not ok
        assert "only SELECT" in reason or "unsafe" in reason
        result = execute_readonly_sql(
            connection,
            "SELECT request_id FROM ue_request_queue ORDER BY arrival_order LIMIT 1",
            max_rows=5,
        )
        assert result.row_count == 1
        assert result.columns == ["request_id"]


def test_candidate_grid_finds_feasible_candidate_for_req0001(tmp_path: Path) -> None:
    state_db = _make_state_db(tmp_path)
    with connect_writable(state_db) as connection:
        candidates = build_candidate_grid(connection, "req_0001", bs_top_k=5)
        evaluations = evaluate_candidates(connection, candidates)
    assert candidates
    assert any(item.verifier_passed for item in evaluations)
    selected = select_best_candidate(evaluations)
    assert selected is not None
    assert selected.verifier_passed
    assert selected.candidate.request_id == "req_0001"
    assert selected.predicted_task_score >= selected.min_task_score
    assert selected.total_latency_ms <= selected.max_total_latency_ms


def test_candidate_selection_falls_back_from_invalid_llm_id(tmp_path: Path) -> None:
    state_db = _make_state_db(tmp_path)
    with connect_writable(state_db) as connection:
        evaluations = evaluate_candidates(
            connection,
            build_candidate_grid(connection, "req_0001", bs_top_k=5),
        )
    selected = select_best_candidate(evaluations, "not_a_real_candidate")
    assert selected is not None
    assert selected.verifier_passed


def test_commit_verified_candidate_updates_working_copy(tmp_path: Path) -> None:
    state_db = _make_state_db(tmp_path)
    working_db = tmp_path / "state.sqlite"
    shutil.copy2(state_db, working_db)
    with connect_writable(working_db) as connection:
        evaluations = evaluate_candidates(
            connection,
            build_candidate_grid(connection, "req_0001", bs_top_k=5),
        )
        selected = select_best_candidate(evaluations)
        assert selected is not None
        record = commit_verified_candidate(connection, selected, source="test")
        assert record["request_id"] == "req_0001"
        active_count = connection.execute(
            "SELECT COUNT(*) AS count FROM active_session WHERE request_id = 'req_0001'"
        ).fetchone()["count"]
        history_count = connection.execute(
            "SELECT COUNT(*) AS count FROM config_history WHERE request_id = 'req_0001'"
        ).fetchone()["count"]
    assert active_count == 1
    assert history_count == 1


def _make_state_db(tmp_path: Path) -> Path:
    state_db = tmp_path / "fixture_state.sqlite"
    tables = {
        "scenario_metadata": metadata_rows(
            {
                "scenario_id": "test_semantic_ucwc_admission",
                "resource_efficiency": 0.85,
            }
        ),
        "base_station_state": [
            {
                "bs_id": "bs_001",
                "map_file": "fixture_map.png",
                "x_m": 0.0,
                "y_m": 0.0,
                "bandwidth_budget_mhz": 100.0,
                "used_bandwidth_mhz": 0.0,
            }
        ],
        "ue_request_queue": [
            {
                "request_id": "req_0001",
                "ue_id": "ue_0001",
                "arrival_order": 1,
                "x_m": 10.0,
                "y_m": 10.0,
                "task_type": "image_classification",
                "min_task_score": 0.75,
                "max_total_latency_ms": 10.0,
            }
        ],
        "radio_link_state": [
            {
                "ue_id": "ue_0001",
                "bs_id": "bs_001",
                "snr_db": 20.0,
                "sinr_db": 20.0,
                "radio_rank": 1,
                "radio_gain_raw": 220.0,
                "radio_gain_norm": 0.9,
                "source_map_file": "fixture_map.png",
            }
        ],
        "semantic_config_catalog": [
            {
                "config_id": "sem_d04_q4",
                "mode_id": 0,
                "encoder_depth": 4,
                "quantization_bits": 4,
                "feature_dim": 512,
                "header_bits": 8,
                "payload_bits": 2056,
                "encoding_latency_ms": 0.20,
                "decoding_latency_ms": 0.18,
                "fixed_latency_ms": 1.0,
            },
            {
                "config_id": "sem_d12_q8",
                "mode_id": 1,
                "encoder_depth": 12,
                "quantization_bits": 8,
                "feature_dim": 512,
                "header_bits": 8,
                "payload_bits": 4104,
                "encoding_latency_ms": 0.38,
                "decoding_latency_ms": 0.34,
                "fixed_latency_ms": 1.0,
            },
        ],
        "phy_mode_catalog": [
            {
                "phy_mode_id": "ldpc12_qam4",
                "ldpc_code_rate": 0.5,
                "qam_order": 4,
                "reference_snr_db": 4.0,
                "spectral_efficiency_bps_hz": 1.0,
            }
        ],
        "active_session": [],
        "config_history": [],
    }
    write_state_database(state_db, tables)
    return state_db
