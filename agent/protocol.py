from typing import TypeAlias
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, object]

@dataclass(frozen=True, slots=True)
class UserMessage:
    content: str

@dataclass(frozen=True, slots=True)
class AssistantMessage:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()

@dataclass(frozen=True, slots=True)
class ToolMessage:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False

ConversationItem: TypeAlias = (
    UserMessage | AssistantMessage | ToolMessage
)