from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

@dataclass(frozen=True, slots=True)
class AgentEvent:
    session_id: str
    turn_id: str
    sequence: int
    timestamp: str
    type: str
    payload: dict[str, object]

class AgentEventFactory:
    def __init__(self, session_id: str | None = None) -> None:
        if session_id is not None:
            self.session_id = session_id
        else:
            self.session_id = uuid4().hex
        self._sequence = 0

    def create(
        self,
        turn_id: str,
        event_type: str,
        payload: dict[str, object] | None = None,
    ) -> AgentEvent:

        self._sequence += 1

        event_time = datetime.now(timezone.utc).isoformat()

        return AgentEvent(
            session_id=self.session_id,
            turn_id=turn_id,
            sequence=self._sequence,
            timestamp=event_time,
            type=event_type,
            payload=payload or {},
        )