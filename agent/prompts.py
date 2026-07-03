"""Self-contained catalog of the prompts used by Codex.

All prompt text is stored in ``prompt_catalog.json`` next to this module.
Loading this module never reads from the parent Codex repository.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal


PromptKind = Literal["asset", "rust-constant", "rust-fragment"]
CATALOG_PATH = Path(__file__).with_name("prompt_catalog.json")


@dataclass(frozen=True, slots=True)
class PromptEntry:
    """One model-visible instruction or prompt fragment."""

    name: str
    category: str
    source: str
    text: str
    kind: PromptKind
    line: int | None = None


@lru_cache(maxsize=1)
def load_prompt_catalog() -> tuple[PromptEntry, ...]:
    """Load the bundled prompt catalog."""

    with CATALOG_PATH.open(encoding="utf-8") as catalog_file:
        records = json.load(catalog_file)
    return tuple(PromptEntry(**record) for record in records)


def prompts_by_category(category: str) -> tuple[PromptEntry, ...]:
    """Return all prompts in one category."""

    return tuple(
        entry for entry in load_prompt_catalog() if entry.category == category
    )


def find_prompts(query: str) -> tuple[PromptEntry, ...]:
    """Case-insensitively search prompt names, source labels, and contents."""

    needle = query.casefold()
    return tuple(
        entry
        for entry in load_prompt_catalog()
        if needle in entry.name.casefold()
        or needle in entry.source.casefold()
        or needle in entry.text.casefold()
    )


def get_prompt(name: str) -> str:
    """Return a prompt by its exact catalog name."""

    matches = [entry for entry in load_prompt_catalog() if entry.name == name]
    if not matches:
        raise KeyError(f"unknown Codex prompt: {name}")
    if len(matches) > 1:
        raise KeyError(f"ambiguous Codex prompt name: {name}")
    return matches[0].text


def catalog_summary() -> dict[str, int]:
    """Return prompt counts grouped by category."""

    summary: dict[str, int] = {}
    for entry in load_prompt_catalog():
        summary[entry.category] = summary.get(entry.category, 0) + 1
    return dict(sorted(summary.items()))


PROMPTS = load_prompt_catalog()
PROMPT_TEXT_BY_NAME = {entry.name: entry.text for entry in PROMPTS}

GPT_5_CODEX_PROMPT = get_prompt("core/gpt_5_codex_prompt.md")
GPT_5_1_CODEX_PROMPT = get_prompt("core/gpt_5_1_prompt.md")
GPT_5_1_CODEX_MAX_PROMPT = get_prompt("core/gpt-5.1-codex-max_prompt.md")
GPT_5_2_CODEX_PROMPT = get_prompt("core/gpt-5.2-codex_prompt.md")
GPT_5_2_LEGACY_PROMPT = get_prompt("core/gpt_5_2_prompt.md")
DEFAULT_BASE_INSTRUCTIONS = get_prompt(
    "protocol/src/prompts/base_instructions/default.md"
)
CODEX_SYSTEM_PROMPT = GPT_5_2_CODEX_PROMPT


if __name__ == "__main__":
    print(f"Loaded {len(PROMPTS)} bundled Codex prompt entries")
    for category, count in catalog_summary().items():
        print(f"{category}: {count}")
