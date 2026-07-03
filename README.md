# UCWC Minimal Prototype

This directory is a lightweight UCWC prototype skeleton. It keeps the magazine
prototype scope narrow: structured wireless state, simple channel proxies,
SQLite-backed state tables, session-level configuration plans, deterministic
verification, and an LLM/NL2SQL tool-calling agent loop.

## Layout

```text
UCWC/
├── agent/                 # LLM tool-calling runtime and UCWC NL2SQL tools
├── configs/               # Static network, service profile, and simulation configs
├── data/                  # Optional input artifacts
├── outputs/               # Generated outputs
├── scripts/               # Runnable entrypoints
└── ucwc/                  # UCWC business package
    ├── components.py      # BaseStation, UE, QoS, resource data classes
    ├── config_loader.py   # YAML config loading and validation
    ├── physics_tools.py   # Bandwidth, noise, SINR, throughput proxies
    ├── channel_model.py   # Synthetic RSRP/SINR radio-link generation
    ├── association.py     # UE-to-BS association
    ├── scenario_builder.py
    ├── table_manager.py   # State-table CSV/JSON/SQLite generation and export
    ├── config_plan.py     # Session-level config plan data class
    └── verifier.py        # Deterministic UCWC verifier checks
```

## Smoke Runs

Build a scenario:

```bash
python UCWC/scripts/build_scenario.py \
  --output-dir UCWC/outputs/scenario_smoke
```

Run the UCWC LLM/NL2SQL tool-agent demo:

```bash
python UCWC/scripts/run_agent_demo.py \
  --ue-id ue_001 \
  --agent-mode llm \
  --output-dir UCWC/outputs/agent_smoke
```

The generated radio values are synthetic proxy features. They are not
ray-tracing output and should not be described as 3GPP-compliant simulation.

You can also run the underlying `UCWC/agent` CLI directly against a generated DB:

```bash
python UCWC/agent/main.py \
  --db-path UCWC/outputs/scenario_smoke/network_state.sqlite \
  --target-ue ue_001
```

For offline smoke tests without calling an LLM:

```bash
python UCWC/scripts/run_agent_demo.py \
  --ue-id ue_001 \
  --agent-mode deterministic \
  --output-dir UCWC/outputs/agent_smoke
```

## Radio-Map Environment

The default channel model remains `synthetic`. For a public ray-tracing-backed
radio-map source, use the RadioMapSeer-style adapter.

Prepare a small subset from the downloaded zip without extracting the full
archive:

```bash
python UCWC/scripts/prepare_radiomapseer_subset.py \
  --zip-path UCWC/data/RadioMapSeer.zip \
  --map-id 0 \
  --method DPM \
  --tx-count 7 \
  --ue-count 100 \
  --output-config-dir UCWC/configs/radiomapseer_real \
  --output-data-dir UCWC/data/radio_maps/radiomapseer_real
```

Build the real RadioMapSeer-backed UCWC scenario:

```bash
python UCWC/scripts/build_scenario.py \
  --config-dir UCWC/configs/radiomapseer_real \
  --output-dir UCWC/outputs/radiomapseer_real
```

For the tiny built-in fixture:

```bash
python UCWC/scripts/build_scenario.py \
  --config-dir UCWC/configs/radiomapseer_demo \
  --output-dir UCWC/outputs/radiomapseer_demo
```

The `radiomapseer_demo` config uses tiny CSV pathloss maps under
`UCWC/data/radio_maps/mock_radiomapseer/` only as a smoke fixture. For real
RadioMapSeer data, keep the full zip outside source control and use
`prepare_radiomapseer_subset.py` to extract only the maps needed by a run. The
adapter supports:

- CSV grids with pathloss values in dB.
- NPY/NPZ 2D arrays with pathloss values in dB.
- Grayscale PNG/TIFF/JPEG maps, interpreted as `scaled_gain` or
  `scaled_pathloss` using `pathloss_min_db` and `pathloss_max_db`.

Example config shape:

```yaml
channel:
  model: radiomapseer
  noise_figure_db: 7.0
  radio_map:
    dataset_root: ../../data/radio_maps/radiomapseer_real/map_000/DPM
    value_kind: scaled_gain
    pathloss_min_db: 45.0
    pathloss_max_db: 160.0
    resolution_m_per_pixel: 1.0
    maps:
      - bs_id: bs_001
        path: bs_001_map000_tx000_DPM_gain.png
      - bs_id: bs_002
        path: bs_002_map000_tx001_DPM_gain.png
```

This is a structured radio-state source, not a full physical-link simulator.
