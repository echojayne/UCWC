# Configs

This directory separates scenario construction, catalog definitions, LLM
runtime settings, and experiment runner settings.

- `scenarios/`: multi-cell UE/BS layout and RadioMapSeer evidence settings.
- `catalogs/`: semantic and PHY action-space definitions.
- `llm/`: non-secret LLM runtime templates.
- `experiments/`: E1-E5 runner configs.

Local secrets should use `*.local.*` files under `configs/llm/`; those files
are ignored by git.
