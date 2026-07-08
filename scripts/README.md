# Scripts

Command-line entrypoints will live here. Planned entrypoints:

- build a semantic UCWC SQLite state database.
- run one admission queue with a selected method.
- run E1-E5 experiments.
- render figures and tables from CSV/JSON sources.

Implemented entrypoints:

- `generate_semantic_ucwc_state.py`: generates a fixed 5-BS/120-UE semantic
  UCWC queue from RadioMapSeer DPM maps and writes SQLite plus CSV state tables.
- `generate_random_semantic_user_configs.py`: writes the full semantic-link
  configuration grid and a random UE basic-config table.
- `benchmark_semantic_encoder_latency.py`: measures bs=1 CLIP ViT-B/16 encoder
  latency by output depth on a selected device.
- `validate_semantic_ucwc_oracle.py`: evaluates persistent-prefix admission
  snapshots under a rule-based global oracle and writes satisfaction curves plus
  final assignments. The rule oracle searches BS association, semantic depth,
  quantization bits, LDPC rate, and QAM order.
