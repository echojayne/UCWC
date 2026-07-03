import json
import sqlite3
from pathlib import Path

from .event import AgentEvent


class SQLiteAuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._create_schema()

    def _create_schema(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_events_session_sequence
            ON agent_events (session_id, sequence)
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_events_turn_sequence
            ON agent_events (turn_id, sequence)
            """
        )
        self._connection.commit()

    def append(self, event: AgentEvent) -> None:
        self._connection.execute(
            """
            INSERT INTO agent_events (
                session_id,
                turn_id,
                sequence,
                timestamp,
                event_type,
                payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.session_id,
                event.turn_id,
                event.sequence,
                event.timestamp,
                event.type,
                json.dumps(event.payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SQLiteAuditLog":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
