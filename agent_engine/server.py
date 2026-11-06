"""HTTP API: run an agent and stream its trace as Server-Sent Events."""

import json
import os
import queue
import threading
from dataclasses import asdict

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent import Agent, AgentConfig, TraceEvent
from .llm import LLM, AnthropicChat, OpenAIChat
from .tools import ToolRegistry
from .tools.builtin import calculator, current_time, make_knowledge_base

app = FastAPI(title="Agentic Workflow Engine", version="1.0.0")

KNOWLEDGE = {
    "kb-1": "Refunds are possible within 30 days of purchase if the product is unused.",
    "kb-2": "The Pro plan costs 29 EUR per user per month, billed annually.",
    "kb-3": "Support hours are Monday to Friday, 9:00 to 18:00 Central European Time.",
}


def build_llm() -> LLM:
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "anthropic":
        return AnthropicChat(os.getenv("LLM_MODEL", "claude-sonnet-4-5"))
    return OpenAIChat(os.getenv("LLM_MODEL", "gpt-4o-mini"))


def build_tools() -> ToolRegistry:
    return ToolRegistry().register(calculator, current_time, make_knowledge_base(KNOWLEDGE))


class RunRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=4000)
    max_steps: int = Field(default=8, ge=1, le=20)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/tools")
def list_tools() -> list[dict]:
    return build_tools().schemas()


@app.post("/runs")
def run(req: RunRequest) -> StreamingResponse:
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            agent = Agent(build_llm(), build_tools(), AgentConfig(max_steps=req.max_steps), on_event=events.put)
            res = agent.run(req.goal)
            events.put(("done", {"answer": res.answer, "stop_reason": res.stop_reason, "steps": res.steps}))
        except Exception as e:  # noqa: BLE001
            events.put(("error", {"message": str(e)}))
        events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while (item := events.get()) is not None:
            if isinstance(item, TraceEvent):
                name, data = item.kind, asdict(item)
            else:
                name, data = item
            yield f"event: {name}\ndata: {json.dumps(data, default=str)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
