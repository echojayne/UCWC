# Semantic UCWC Project Outline

## Claim

The project demonstrates session-level semantic UCWC configuration using
structured state access, NL2SQL evidence retrieval, deterministic verification,
and bounded repair. RadioMapSeer is used as radio-state evidence, not as a
locally executed ray-tracing solver.

## Core Tables

- `base_station_state`: bandwidth budget, used bandwidth, available bandwidth,
  utilization.
- `ue_request`: incoming UE request, task type, performance target, latency
  deadline, and position.
- `radio_link_state`: per UE-BS SNR/SINR and radio rank.
- `semantic_config_catalog`: encoder depth, quantization bits, payload size,
  semantic score, encoding latency, decoding latency.
- `phy_mode_catalog`: LDPC code rate, QAM order, minimum SNR, effective
  spectral efficiency.
- `active_session`: committed verifier-passed sessions.
- `config_history`: attempted and committed plans, verifier result, failure
  reason, and trace metadata.

## Method Boundary

The LLM/NL2SQL agent may inspect schema, generate bounded read-only SQL,
retrieve evidence, propose plans, and repair failed plans. Commits require a
fresh deterministic verifier pass.

Baselines must share the same offline evaluator, but only the main method uses
the full NL2SQL and verifier-aware repair loop.
