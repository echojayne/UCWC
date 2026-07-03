from dataclasses import dataclass
from collections.abc import Iterator
from uuid import uuid4

from .llm import LlmClient
from .context import ContextManager
from .event import AgentEvent, AgentEventFactory
from .protocol import AssistantMessage, ToolCall, ToolMessage
from .tools import ToolRegistry

@dataclass(frozen=True, slots=True)
class AgentConfig:
    system_prompt: str
    max_steps: int = 20

class Agent:
    def __init__(
        self,
        config: AgentConfig,
        llm: LlmClient,
        context: ContextManager,
        tools: ToolRegistry,
        *,
        event_factory: AgentEventFactory | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._context = context
        self._tools = tools
        self._events = event_factory or AgentEventFactory()

    def run(self, user_input: str) -> str:
        final_content: str | None = None
        for event in self.run_stream(user_input):
            if event.type == "turn.completed":
                content = event.payload.get("content")
                if isinstance(content, str):
                    final_content = content
        if final_content is None:
            raise RuntimeError("Agent did not produce a final answer")
        return final_content

    def run_stream(self, user_input: str) -> Iterator[AgentEvent]:
        turn_id = uuid4().hex
        current_step = 0
        try:
            yield self._event(
                turn_id,
                "turn.started",
                {"user_input": user_input},
            )
            self._context.record_user(user_input)
            yield self._event(
                turn_id,
                "user.recorded",
                {"context_items": len(self._context.snapshot())},
            )

            while current_step < self._config.max_steps:
                current_step += 1
                tools = self._tools.definitions()
                context_snapshot = self._context.snapshot()
                yield self._event(
                    turn_id,
                    "llm.started",
                    {
                        "step": current_step,
                        "context_items": len(context_snapshot),
                        "tools": [tool.name for tool in tools],
                    },
                )
                assistant_message = self._llm.complete(
                    system_prompt=self._config.system_prompt,
                    items=context_snapshot,
                    tools=tools,
                )
                yield self._event(
                    turn_id,
                    "llm.completed",
                    {
                        "step": current_step,
                        "assistant": self._assistant_payload(assistant_message),
                    },
                )

                self._context.record_assistant(assistant_message)
                yield self._event(
                    turn_id,
                    "assistant.recorded",
                    {
                        "step": current_step,
                        "context_items": len(self._context.snapshot()),
                    },
                )

                if assistant_message.tool_calls:
                    for tool_call in assistant_message.tool_calls:
                        yield self._event(
                            turn_id,
                            "tool.started",
                            {
                                "step": current_step,
                                "tool_call": self._tool_call_payload(tool_call),
                            },
                        )
                        tool_message = self._tools.execute(tool_call)
                        yield self._event(
                            turn_id,
                            "tool.completed",
                            {
                                "step": current_step,
                                "tool_result": self._tool_message_payload(
                                    tool_message
                                ),
                            },
                        )
                        self._context.record_tool(tool_message)
                        yield self._event(
                            turn_id,
                            "tool.recorded",
                            {
                                "step": current_step,
                                "tool_call_id": tool_message.tool_call_id,
                                "context_items": len(self._context.snapshot()),
                            },
                        )
                    continue

                if assistant_message.content is None:
                    raise RuntimeError("LLM response has no content or tool calls")

                yield self._event(
                    turn_id,
                    "turn.completed",
                    {
                        "steps": current_step,
                        "content": assistant_message.content,
                    },
                )
                return

            raise RuntimeError(
                "Maximum number of steps exceeded without producing a final answer"
            )
        except Exception as error:
            yield self._event(
                turn_id,
                "turn.failed",
                {
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "steps": current_step,
                },
            )
            raise

    def _event(
        self,
        turn_id: str,
        event_type: str,
        payload: dict[str, object] | None = None,
    ) -> AgentEvent:
        return self._events.create(
            turn_id=turn_id,
            event_type=event_type,
            payload=payload,
        )

    def _assistant_payload(self, message: AssistantMessage) -> dict[str, object]:
        return {
            "content": message.content,
            "tool_call_count": len(message.tool_calls),
            "tool_calls": [
                self._tool_call_payload(tool_call)
                for tool_call in message.tool_calls
            ],
        }

    def _tool_call_payload(self, tool_call: ToolCall) -> dict[str, object]:
        return {
            "id": tool_call.id,
            "name": tool_call.name,
            "arguments": tool_call.arguments,
        }

    def _tool_message_payload(self, message: ToolMessage) -> dict[str, object]:
        return {
            "tool_call_id": message.tool_call_id,
            "name": message.name,
            "content": message.content,
            "is_error": message.is_error,
        }
