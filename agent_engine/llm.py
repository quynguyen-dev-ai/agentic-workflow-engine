"""LLM adapters. Each returns a normalized `LLMResponse`: either final text or
a list of tool calls. The agent loop never touches provider-specific types."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class LLM(Protocol):
    def complete(self, messages: list[Message], tools: list[dict]) -> LLMResponse: ...


class ScriptedLLM:
    """Deterministic LLM for tests and evals: replays a policy function that
    decides the next response from the conversation so far."""

    def __init__(self, policy: Callable[[list[Message]], LLMResponse]):
        self.policy = policy
        self.calls = 0

    def complete(self, messages: list[Message], tools: list[dict]) -> LLMResponse:
        self.calls += 1
        return self.policy(messages)


class OpenAIChat:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, messages: list[Message], tools: list[dict]) -> LLMResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[self._to_openai(m) for m in messages],
            tools=[{"type": "function", "function": t} for t in tools] or None,
        )
        msg = resp.choices[0].message
        calls = [
            ToolCall(c.id, c.function.name, json.loads(c.function.arguments or "{}")) for c in (msg.tool_calls or [])
        ]
        usage = resp.usage
        return LLMResponse(
            msg.content, calls, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0
        )

    @staticmethod
    def _to_openai(m: Message) -> dict:
        if m.role == "tool":
            return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
        out: dict = {"role": m.role, "content": m.content}
        if m.tool_calls:
            out["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in m.tool_calls
            ]
        return out


class AnthropicChat:
    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 1024):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, messages: list[Message], tools: list[dict]) -> LLMResponse:
        system = "\n".join(m.content for m in messages if m.role == "system")
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=[
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools
            ],
            messages=self._to_anthropic([m for m in messages if m.role != "system"]),
        )
        text = "".join(b.text for b in resp.content if b.type == "text") or None
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return LLMResponse(text, calls, resp.usage.input_tokens, resp.usage.output_tokens)

    @staticmethod
    def _to_anthropic(messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m.role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
                # Consecutive tool results must share one user turn.
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m.role == "assistant" and m.tool_calls:
                blocks: list[dict] = [{"type": "text", "text": m.content}] if m.content else []
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls]
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": m.role, "content": m.content})
        return out
