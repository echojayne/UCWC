"""JSONL trace writer for UCWC admission-agent turns."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TraceWriter:
    """Append-only local trace writer.

    The trace is intentionally lightweight: raw tool inputs and summarized
    outputs are written as ordered JSONL events so UI/debug code can replay the
    admission decision without depending on chat transcripts.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._sequence = 0

    def write(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        self._sequence += 1
        event = {
            "seq": self._sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "payload": _jsonable(payload or {}),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
