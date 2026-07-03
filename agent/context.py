from .protocol import (
    AssistantMessage,
    ConversationItem,
    ToolCall,
    ToolMessage,
    UserMessage,
)

class ContextManager:
    def __init__(self) -> None:
        self._items: list[ConversationItem] = []
        self._pending_calls: dict[str, ToolCall] = {}
    
    def _ensure_no_pending_calls(self) -> None:
        if self._pending_calls:
            raise RuntimeError("There are pending tool calls that have not been completed")

    def record_user(self, content: str) -> UserMessage:
        self._ensure_no_pending_calls()
        message = UserMessage(content=content)
        self._items.append(message)
        return message

    def record_assistant(self, message: AssistantMessage) -> None:
        self._ensure_no_pending_calls()

        if message.content is None and not message.tool_calls:
            raise ValueError("Assistant message has no content or tool calls")

        new_calls: dict[str, ToolCall] = {}

        for tool_call in message.tool_calls:
            if tool_call.id in new_calls:
                raise ValueError(f"Duplicate tool call id: {tool_call.id}")
            new_calls[tool_call.id] = tool_call

        self._items.append(message)
        self._pending_calls = new_calls

    def record_tool(self, message: ToolMessage) -> None:
        expected_call = self._pending_calls.get(message.tool_call_id)
        if expected_call is None:
            raise ValueError(f"No pending tool call found for tool message with id {message.tool_call_id}")
        if expected_call.name != message.name:
            raise ValueError(f"Tool message name {message.name} does not match expected tool call name {expected_call.name} for id {message.tool_call_id}")
        self._items.append(message)
        self._pending_calls.pop(expected_call.id, None) 

    def pending_tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(self._pending_calls.values())

    def snapshot(self) -> tuple[ConversationItem, ...]:
        return tuple(self._items)
