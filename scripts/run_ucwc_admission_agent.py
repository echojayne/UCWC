#!/usr/bin/env python3
"""Run the semantic UCWC admission agent on a SQLite state database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ucwc.agent import AgentConfig, OpenAICompatibleChatClient, UCWCAdmissionAgent, load_llm_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WCM_ROOT = PROJECT_ROOT.parent


def main() -> int:
    args = _parse_args()
    state_db = _resolve_project_path(args.state_db)
    output_dir = _resolve_project_path(args.output_dir)
    working_db = _resolve_project_path(args.working_db) if args.working_db else None
    models_md = Path(args.models_md).expanduser()
    llm_config = load_llm_config(
        models_md,
        profile=args.profile,
        model=args.model,
        timeout_s=args.timeout_s,
    )
    agent = UCWCAdmissionAgent(
        config=AgentConfig(
            state_db=state_db,
            output_dir=output_dir,
            working_db=working_db,
            max_rounds=args.max_rounds,
            bs_top_k=args.bs_top_k,
            max_candidates=args.max_candidates,
            commit=not args.no_commit,
        ),
        llm=OpenAICompatibleChatClient(llm_config),
    )
    user_request = args.user_request or (
        f"Admit {args.request_id} into the semantic UCWC system. Use NL2SQL grounding first, "
        "then search semantic/PHY/radio candidates and commit only a verifier-passed plan."
    )
    result = agent.run(user_request, request_id=args.request_id)
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=True, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-db",
        default="outputs/state/semantic_5bs_120ue/state.sqlite",
        help="Semantic UCWC state SQLite database, relative to UCWC by default.",
    )
    parser.add_argument("--request-id", default="req_0001")
    parser.add_argument(
        "--user-request",
        default=None,
        help="Natural language request. Defaults to an admission request for --request-id.",
    )
    parser.add_argument(
        "--models-md",
        default=str(WCM_ROOT / "models.md"),
        help="Path to WCM models.md containing llmcfg1 and deepseek-v4-flash.",
    )
    parser.add_argument("--profile", default="llmcfg1")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--output-dir", default="outputs/agent/ucwc_admission_deepseek_smoke")
    parser.add_argument("--working-db", default=None)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--bs-top-k", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=250)
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Evaluate and select but do not write active_session/config_history.",
    )
    return parser.parse_args()


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
