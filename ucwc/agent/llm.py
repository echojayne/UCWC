"""OpenAI-compatible LLM client used by the UCWC admission agent."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ucwc.agent.protocol import LlmConfig


def load_llm_config(
    models_path: str | Path,
    *,
    profile: str = "llmcfg1",
    model: str = "deepseek-v4-flash",
    timeout_s: float = 90.0,
) -> LlmConfig:
    """Load one model entry from the user's loose models.md profile block."""

    path = Path(models_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    in_profile = False
    api_key: str | None = None
    base_url: str | None = None
    models: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.fullmatch(r"\[([A-Za-z0-9_\-]+)\]", stripped)
        if match:
            if in_profile:
                break
            in_profile = match.group(1) == profile
            continue
        if not in_profile:
            continue
        key_value = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)", stripped)
        if key_value is None:
            continue
        key = key_value.group(1)
        raw_value = key_value.group(2).strip().rstrip(",")
        value = _strip_quotes(raw_value)
        if key == "api_key":
            api_key = value
        elif key == "base_url":
            base_url = value
        elif key == "model":
            models.append(value)
    if api_key is None:
        raise ValueError(f"api_key not found in [{profile}] of {path}")
    if base_url is None:
        raise ValueError(f"base_url not found in [{profile}] of {path}")
    selected_model = model
    if selected_model not in models:
        available = ", ".join(models) if models else "none"
        raise ValueError(f"model {selected_model!r} not found in [{profile}], available: {available}")
    return LlmConfig(
        api_key=api_key,
        base_url=base_url,
        model=selected_model,
        profile=profile,
        source_path=path,
        timeout_s=timeout_s,
    )


class OpenAICompatibleChatClient:
    """Small chat-completions client with no third-party dependency."""

    def __init__(self, config: LlmConfig) -> None:
        self.config = config

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {error.code}: {body[:1000]}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"LLM request failed: {error}") from error
        choices = response_data.get("choices")
        if not choices:
            raise RuntimeError(f"LLM response has no choices: {response_data}")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"LLM response has no text content: {response_data}")
        return content

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> tuple[dict[str, Any], str]:
        text = self.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _parse_json_object(text), text


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"LLM did not return a JSON object: {text[:500]}")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError(f"LLM JSON output is not an object: {value!r}")
    return value
