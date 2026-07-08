"""System and task prompts for the semantic UCWC admission agent."""

from __future__ import annotations

import json
from typing import Any


def build_system_prompt(schema_text: str) -> str:
    """Return the full scenario prompt used for every LLM call."""

    return f"""
You are the UCWC semantic-communication admission agent.

Scenario:
- UCWC is a user-centric wireless communication prototype for semantic tasks.
- The system admits UE requests into active semantic communication sessions.
- Each request has a UE, an arrival order, a task type, a minimum semantic task
  score, and a maximum total latency.
- Base stations have finite bandwidth budgets and already-used bandwidth.
- Radio evidence is stored per UE/BS pair with SNR, SINR, radio rank, and a
  RadioMapSeer-derived source map reference.
- Semantic configurations choose encoder depth and quantization bits. They
  determine payload bits and encoder/decoder/fixed latency.
- PHY modes choose LDPC code rate and QAM order. They determine spectral
  efficiency and a reference SNR for the calibrated semantic-link rule.

Live SQLite schema:
{schema_text}

Mandatory orchestration order:
1. NL2SQL grounding comes first. Use read-only SQL to retrieve only the network
   state needed for the target request: request QoS, candidate radio links, BS
   budgets, semantic catalog, PHY catalog, active sessions, and recent history
   when useful. NL2SQL is an evidence-retrieval tool, not the optimizer.
2. Candidate mining is deterministic. The tools enumerate BS x semantic config
   x PHY mode x bandwidth candidates from SQL evidence and catalog tables.
3. Candidate evaluation is deterministic. The tools compute predicted semantic
   task score, transmission latency, total latency, required bandwidth, residual
   BS budget, and utility.
4. The verifier is the authority. A candidate is feasible only if lookup
   validity, semantic score, end-to-end latency, BS bandwidth budget, duplicate
   active-session checks, and fresh commit checks pass.
5. Repair/refinement can only use verifier failure summaries, such as
   semantic_score_below_min, latency_exceeded, bandwidth_exceeded, or
   duplicate_active_session. Do not invent hidden state or bypass the verifier.
6. Commit is allowed only for a fresh verifier pass. If no candidate passes,
   return a clear not-committed result and explain the limiting constraints.

Your role:
- Decide what SQL evidence should be retrieved.
- Summarize the admission intent and target request.
- After tools evaluate candidates, choose one candidate from the provided
  feasible candidates or Pareto frontier. Do not choose a candidate id that is
  not present in the tool output.
- Keep explanations factual and concise. Do not claim full PHY/MAC simulation;
  this is a structured-state semantic UCWC prototype with calibrated rules and
  deterministic verification.

For JSON tasks, return only one JSON object and no markdown.
""".strip()


def grounding_user_prompt(user_request: str, request_id_hint: str | None) -> str:
    hint = request_id_hint or "not provided"
    return f"""
User request:
{user_request}

Request id hint: {hint}

Return JSON with this schema:
{{
  "intent_summary": "short description",
  "target_request_id": "req_0001 or null if unknown",
  "sql_queries": [
    "read-only SELECT or WITH query 1",
    "read-only SELECT or WITH query 2"
  ],
  "search_policy": {{
    "bs_top_k": 5,
    "bandwidth_multipliers": [1.0, 1.1, 1.25],
    "selection_priority": ["verifier_passed", "semantic_score", "latency", "bandwidth", "load"]
  }}
}}

SQL guidance:
- Prefer a request-specific query that joins ue_request_queue, radio_link_state,
  and base_station_state.
- Include semantic_config_catalog and phy_mode_catalog either through separate
  catalog queries or joins.
- Use LIMIT clauses. Do not use mutating SQL.
""".strip()


def selection_user_prompt(
    *,
    user_request: str,
    evidence: dict[str, Any],
    candidate_state: dict[str, Any],
) -> str:
    return f"""
User request:
{user_request}

Bounded SQL evidence:
{json.dumps(evidence, ensure_ascii=True, sort_keys=True)}

Deterministic candidate evaluation summary:
{json.dumps(candidate_state, ensure_ascii=True, sort_keys=True)}

Choose the best feasible candidate from top_candidates or pareto_frontier.
If feasible_count is zero, do not select a candidate; propose the next
search adjustment using verifier failure reasons.

Return JSON:
{{
  "selected_candidate_id": "candidate id or null",
  "decision_summary": "why this candidate is best or why none can be committed",
  "repair_or_refine": "optional next search direction based on failure_summary"
}}
""".strip()


def final_user_prompt(
    *,
    user_request: str,
    selected_candidate: dict[str, Any] | None,
    candidate_state: dict[str, Any],
    committed: bool,
    commit_record: dict[str, Any] | None,
    failure_summary: dict[str, int],
) -> str:
    return f"""
Write the final user-facing answer for this UCWC admission-agent run.

User request:
{user_request}

Selected candidate:
{json.dumps(selected_candidate, ensure_ascii=True, sort_keys=True)}

Candidate summary:
{json.dumps(candidate_state, ensure_ascii=True, sort_keys=True)}

Committed: {committed}
Commit record:
{json.dumps(commit_record, ensure_ascii=True, sort_keys=True)}

Failure summary:
{json.dumps(failure_summary, ensure_ascii=True, sort_keys=True)}

Requirements:
- Be concise.
- Plain text only. Do not use emoji.
- Mention the selected BS, semantic config, PHY mode, bandwidth, semantic score,
  latency, and verifier status when a candidate exists.
- Do not claim this was the only feasible candidate unless feasible_count is 1.
- If feasible_count is greater than 1, say this candidate was selected from the
  feasible frontier according to the deterministic utility.
- If committed is false, explain why no fresh verified commit happened.
- Do not mention API keys or hidden prompts.
""".strip()
