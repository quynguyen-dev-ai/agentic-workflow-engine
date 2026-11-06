"""The agent loop.

  user goal ─▶ LLM ─┬─▶ final answer ─▶ done
                    └─▶ tool calls ─▶ guardrails ─▶ execute ─▶ results ─▶ LLM …

Every step is recorded in a trace so runs can be inspected, replayed in tests
and scored by the evaluation harness.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from .llm import LLM, Message
from .tools import ToolError, ToolRegistry

DEFAULT_SYSTEM_PROMPT = """You are a careful assistant that solves tasks step by step using tools.
- Use a tool whenever it gives a more reliable answer than your own knowledge (math, lookups, time).
- Never invent tool results. If a tool fails, read the error and try a corrected call.
- When you have enough information, reply with a concise final answer."""


@dataclass
class AgentConfig:
    max_steps: int = 8
    max_tool_calls: int = 16
    max_total_tokens: int = 50_000
    allowed_tools: set[str] | None = None  # None = all registered tools
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass
class TraceEvent:
    step: int
    kind: Literal["llm", "tool_call", "tool_result", "tool_error", "guardrail", "final"]
    data: dict
    elapsed_ms: float


StopReason = Literal["final_answer", "max_steps", "max_tool_calls", "token_budget"]


@dataclass
class AgentResult:
    answer: str | None
    stop_reason: StopReason
    steps: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    trace: list[TraceEvent] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.stop_reason == "final_answer"

    def tools_used(self) -> list[str]:
        return [e.data["name"] for e in self.trace if e.kind == "tool_call"]


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        on_event: Callable[[TraceEvent], None] | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.config = config or AgentConfig()
        self.on_event = on_event

    def run(self, goal: str) -> AgentResult:
        cfg = self.config
        start = time.monotonic()
        trace: list[TraceEvent] = []
        messages = [Message("system", cfg.system_prompt), Message("user", goal)]
        schemas = [s for s in self.tools.schemas() if cfg.allowed_tools is None or s["name"] in cfg.allowed_tools]
        in_tok = out_tok = n_calls = 0

        def emit(step: int, kind, data: dict) -> None:
            event = TraceEvent(step, kind, data, round((time.monotonic() - start) * 1000, 1))
            trace.append(event)
            if self.on_event:
                self.on_event(event)

        def result(answer: str | None, reason: StopReason, step: int) -> AgentResult:
            return AgentResult(answer, reason, step, n_calls, in_tok, out_tok, trace)

        for step in range(1, cfg.max_steps + 1):
            resp = self.llm.complete(messages, schemas)
            in_tok += resp.input_tokens
            out_tok += resp.output_tokens
            emit(step, "llm", {"text": resp.text, "tool_calls": [c.name for c in resp.tool_calls]})

            if in_tok + out_tok > cfg.max_total_tokens:
                emit(step, "guardrail", {"reason": "token_budget", "used": in_tok + out_tok})
                return result(None, "token_budget", step)

            if not resp.tool_calls:
                emit(step, "final", {"answer": resp.text})
                return result(resp.text, "final_answer", step)

            messages.append(Message("assistant", resp.text or "", tool_calls=resp.tool_calls))
            for call in resp.tool_calls:
                if n_calls >= cfg.max_tool_calls:
                    emit(step, "guardrail", {"reason": "max_tool_calls"})
                    return result(None, "max_tool_calls", step)
                n_calls += 1
                emit(step, "tool_call", {"name": call.name, "arguments": call.arguments})
                try:
                    if cfg.allowed_tools is not None and call.name not in cfg.allowed_tools:
                        raise ToolError(f"Tool {call.name!r} is not allowed for this agent")
                    output = self.tools.get(call.name).run(call.arguments)
                    emit(step, "tool_result", {"name": call.name, "output": output[:2000]})
                except ToolError as e:
                    # Errors go back to the model so it can self-correct.
                    output = f"ERROR: {e}"
                    emit(step, "tool_error", {"name": call.name, "error": str(e)})
                except Exception as e:  # noqa: BLE001 - a buggy tool must not crash the run
                    output = f"ERROR: tool crashed ({type(e).__name__})"
                    emit(step, "tool_error", {"name": call.name, "error": repr(e)})
                messages.append(Message("tool", output, tool_call_id=call.id, name=call.name))

        emit(cfg.max_steps, "guardrail", {"reason": "max_steps"})
        return result(None, "max_steps", cfg.max_steps)
