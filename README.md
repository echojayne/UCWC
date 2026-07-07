# UCWC

Session-level semantic user-centric wireless communication orchestration
prototype.

This repository is being rebuilt around a narrow, auditable claim:
structured wireless state access through SQLite/NL2SQL, RadioMapSeer-backed
radio evidence, deterministic verification, and repair-aware semantic/PHY
configuration. It is not a full PHY/MAC simulator and should not be described
as a global wireless optimizer.

## Project Boundary

The active scenario is semantic communication between UEs and base stations.
For each incoming UE request, the system should select:

- `serving_bs_id`
- `encoder_depth`
- `quantization_bits`
- `ldpc_code_rate`
- `qam_order`
- `bandwidth_mhz`

The verifier is responsible for checking lookup validity, SNR feasibility,
semantic performance, end-to-end latency, BS bandwidth budget, and commit
freshness before any plan is written as an active session.

## Layout

```text
UCWC/
├── configs/          # Scenario, catalog, LLM, and experiment configs
├── data/             # Radio-map evidence and semantic catalog inputs
├── docs/             # Project contracts and paper-facing notes
├── outputs/          # Generated outputs only
├── scripts/          # CLI entrypoints
├── tests/            # Unit and smoke tests
└── ucwc/             # Python package
    ├── agent/        # LLM/NL2SQL tool loop and audit traces
    ├── baselines/    # Prompt, ICL, WirelessAgent-style, greedy, oracle methods
    ├── core/         # Shared dataclasses and contracts
    ├── experiments/  # E1-E5 experiment runners
    ├── plotting/     # Figure/table generation from CSV/JSON sources
    ├── radio/        # RadioMapSeer evidence adapters
    ├── semantic/     # Semantic encoder/catalog abstractions
    ├── state/        # SQLite schema, loading, and commit helpers
    └── verifier/     # Deterministic verification and repair feedback
```

## Planned Experiments

- E1: main growing-admission performance.
- E4: request-to-commit latency breakdown.
- E2: NL2SQL robustness and repair ablation.
- E3: large-scale state/link scaling.
- E5: model ablation for the main method.

Every experiment should write CSV/JSON source data first. Figures and tables
must be generated from those source files.
