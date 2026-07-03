"""Build a UCWC minimal scenario and export JSON/CSV tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


UCWC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UCWC_ROOT))

from ucwc.scenario_builder import build_scenario, write_scenario  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(UCWC_ROOT / "configs"))
    parser.add_argument("--output-dir", default=str(UCWC_ROOT / "outputs" / "scenario_smoke"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenario = build_scenario(args.config_dir)
    outputs = write_scenario(scenario, args.output_dir)
    print("scenario_status=ok")
    print(f"num_base_stations={scenario.metadata['num_base_stations']}")
    print(f"num_ues={scenario.metadata['num_ues']}")
    print(f"num_connected_ues={scenario.metadata['num_connected_ues']}")
    print(f"scenario_json={outputs['scenario_json']}")
    print(f"tables_dir={outputs['tables_dir']}")
    print(f"sqlite_db={outputs['sqlite_db']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
