import json
import tomllib
from pathlib import Path
from typing import cast, Literal
from dataclasses import dataclass
from collections.abc import Iterable

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionToolParam,
    ChatCompletionMessageParam, 
    ChatCompletionToolMessageParam,
    ChatCompletionAssistantMessageParam,
)
from openai.types.shared import ReasoningEffort

from .tools import ToolDefinition
from .protocol import (
    AssistantMessage,
    ConversationItem,
    ToolMessage,
    UserMessage,
    ToolCall
)

def _to_sdk_tools(
    definitions: Iterable[ToolDefinition],
) -> list[ChatCompletionToolParam]:
    sdk_tools: list[ChatCompletionToolParam] = []
    for tool in definitions:
        sdk_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        })
    return sdk_tools

def _to_sdk_messages(
    system_prompt: str,
    items: Iterable[ConversationItem],
) -> list[ChatCompletionMessageParam]:

    sdk_messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt}
    ]

    for item in items:

        if isinstance(item, UserMessage):
            sdk_messages.append({"role": "user", "content": item.content})

        elif isinstance(item, AssistantMessage):
            sdk_message: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": item.content,
            }
            if item.tool_calls:
                sdk_message["tool_calls"] = [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments),
                    },
                } for tool_call in item.tool_calls]
            sdk_messages.append(sdk_message)

        elif isinstance(item, ToolMessage):
            tool_message: ChatCompletionToolMessageParam = {
                "role": "tool",
                "content": item.content,
                "tool_call_id": item.tool_call_id,
            }
            sdk_messages.append(tool_message)
        else:
            raise ValueError(f"Unknown conversation item type: {type(item)}")

    return sdk_messages

def _from_sdk_message(
    message: ChatCompletionMessage,
) -> AssistantMessage:
    tool_calls: list[ToolCall] = []
    for tool_call_data in message.tool_calls or []:
        if tool_call_data.type != "function":
            raise ValueError(f"Unsupported tool call type: {tool_call_data.type}")
        arguments = _parse_tool_arguments(
            tool_call_data.function.arguments,
            tool_call_data.function.name,
        )
        tool_calls.append(ToolCall(
            id=tool_call_data.id,
            name=tool_call_data.function.name,
            arguments=arguments,
        ))

    return AssistantMessage(content=message.content, tool_calls=tuple(tool_calls))

def _parse_tool_arguments(raw_arguments: str, tool_name: str) -> dict[str, object]:
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        try:
            arguments = json.loads(raw_arguments, strict=False)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON arguments for tool {tool_name}") from error

    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object")
    return arguments

@dataclass(frozen=True, slots=True)
class LlmConfig:
    api_key: str
    base_url: str
    model: str
    stream: Literal[False]
    reasoning_effort: ReasoningEffort
    extra_body: dict[str, object]

def load_llm_config(path: Path) -> LlmConfig:
    with path.open("rb") as config_file:
        llmcfg = tomllib.load(config_file)["llmcfg"]
        request = llmcfg["request"]
    if not llmcfg["api_key"].strip():
        raise ValueError("API key must not be empty")
    if request["stream"]:
        raise ValueError("Streaming responses are not supported in this implementation")
    return LlmConfig(
        api_key=llmcfg["api_key"],
        base_url=llmcfg["base_url"],
        model=llmcfg["model"],
        # stream=request.get("stream", False),
        stream=False,
        reasoning_effort=cast(
            ReasoningEffort,
            request.get("reasoning_effort", "medium"),
        ),
        extra_body=request.get("extra_body", {})
    )

class LlmClient:
    def __init__(
        self,
        config: LlmConfig,
        *,
        sdk_client: OpenAI | None = None,
    ) -> None:
        self._config = config
        self._client = sdk_client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def complete(
        self,
        system_prompt: str,
        items: Iterable[ConversationItem],
        tools: Iterable[ToolDefinition] = (),
    ) -> AssistantMessage:

        sdk_messages = _to_sdk_messages(system_prompt, items)
        sdk_tools = _to_sdk_tools(tools)

        if sdk_tools:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=sdk_messages,
                tools=sdk_tools,
                stream=self._config.stream,
                reasoning_effort=self._config.reasoning_effort,
                extra_body=self._config.extra_body,
            )
        else:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=sdk_messages,
                stream=self._config.stream,
                reasoning_effort=self._config.reasoning_effort,
                extra_body=self._config.extra_body,
            )
        if not response.choices:
            raise RuntimeError("Model returned no choices")
        return _from_sdk_message(response.choices[0].message)
