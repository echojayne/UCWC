"""Run the UCWC minimal verifier-aware agent loop."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


UCWC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UCWC_ROOT))

from agent.ucwc_loop import AgentLoopConfig, run_minimal_agent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ue-id", default="ue_001")
    parser.add_argument(
        "--intent",
        default="Improve this UE session while preserving QoS and security.",
    )
    parser.add_argument("--config-dir", default=str(UCWC_ROOT / "configs"))
    parser.add_argument("--output-dir", default=str(UCWC_ROOT / "outputs" / "agent_smoke"))
    parser.add_argument("--max-repair-steps", type=int, default=2)
    parser.add_argument("--db-path", default=None)
    parser.add_argument(
        "--agent-mode",
        choices=["llm", "deterministic"],
        default="llm",
        help="Use the UCWC LLM/NL2SQL tool agent or the offline deterministic fallback.",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument(
        "--fallback-to-deterministic",
        action="store_true",
        help="If the LLM/tool-calling path fails, run the deterministic fallback and record the failure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{args.ue_id}_trace.json"
    trace = run_minimal_agent(
        AgentLoopConfig(
            ue_id=args.ue_id,
            user_intent=args.intent,
            config_dir=args.config_dir,
            max_repair_steps=args.max_repair_steps,
            output_trace=str(trace_path),
            db_path=args.db_path,
            agent_mode=args.agent_mode,
            max_steps=args.max_steps,
            fallback_to_deterministic=args.fallback_to_deterministic,
        )
    )
    print("agent_status=ok")
    print(f"terminal_status={trace['terminal_status']}")
    if "final_verifier_feedback" in trace:
        print(f"verifier_passed={trace['final_verifier_feedback']['passed']}")
    print(f"sqlite_db={trace.get('sqlite_db')}")
    print(f"trace_json={trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
