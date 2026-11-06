from fastapi.testclient import TestClient

from agent_engine import server
from agent_engine.llm import LLMResponse, ScriptedLLM, ToolCall


def test_run_streams_trace_events(monkeypatch):
    def policy(msgs):
        if not any(m.role == "tool" for m in msgs):
            return LLMResponse(tool_calls=[ToolCall("1", "search_knowledge_base", {"query": "Pro plan cost"})])
        return LLMResponse(text="29 EUR per user per month")

    monkeypatch.setattr(server, "build_llm", lambda: ScriptedLLM(policy))
    client = TestClient(server.app)

    assert {t["name"] for t in client.get("/tools").json()} == {
        "calculator",
        "current_time",
        "search_knowledge_base",
    }
    body = client.post("/runs", json={"goal": "Pro plan price?"}).text
    for event in ("event: llm", "event: tool_call", "event: tool_result", "event: final", "event: done"):
        assert event in body
    assert "29 EUR" in body
