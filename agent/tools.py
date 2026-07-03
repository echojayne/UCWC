from typing import TypeAlias
from dataclasses import dataclass
from collections.abc import Callable
from .protocol import ToolCall, ToolMessage

JsonObject: TypeAlias = dict[str, object]
ToolHandler: TypeAlias = Callable[[JsonObject], str]

@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: JsonObject
    handler: ToolHandler

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name is None or tool.name.strip() == "":
            raise ValueError("Tool name must not be empty")
        if tool.name in self._tools:
            raise ValueError(f"Tool with name {tool.name} is already registered")   
        self._tools[tool.name] = tool

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._tools.values())

    def execute(self, call: ToolCall) -> ToolMessage:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolMessage(
                tool_call_id=call.id,
                name=call.name,
                content=f"Unknown tool: {call.name}",
                is_error=True,
            )
        try:
            result = tool.handler(call.arguments)
            return ToolMessage(
                tool_call_id=call.id,
                name=call.name,
                content=result,
                is_error=False,
            )
        except Exception as error:
            return ToolMessage(
                tool_call_id=call.id,
                name=call.name,
                content=str(error),
                is_error=True,
            )
