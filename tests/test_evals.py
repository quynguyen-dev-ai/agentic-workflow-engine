from pathlib import Path

from agent_engine import Agent, ToolRegistry
from agent_engine.evals import EvalCase, load_cases, run_suite
from agent_engine.llm import LLMResponse, ScriptedLLM, ToolCall
from agent_engine.tools.builtin import calculator


def make_agent():
    def policy(msgs):
        tool_msgs = [m for m in msgs if m.role == "tool"]
        if "math" in msgs[1].content and not tool_msgs:
            return LLMResponse(tool_calls=[ToolCall("1", "calculator", {"expression": "2 + 2"})])
        return LLMResponse(text=f"Result: {tool_msgs[-1].content}" if tool_msgs else "Hallo!")

    return Agent(ScriptedLLM(policy), ToolRegistry().register(calculator))


def test_suite_reports_passes_and_specific_failures():
    cases = [
        EvalCase("math", "do math", expect_contains=["4"], expect_tools=["calculator"]),
        EvalCase("greet", "say hi", expect_regex="hallo", forbid_tools=["calculator"]),
        EvalCase("wrong", "do math", expect_contains=["5"], max_steps=1),
    ]
    report = run_suite(make_agent, cases)
    by_id = {r.case_id: r for r in report.results}
    assert by_id["math"].passed and by_id["greet"].passed
    assert not by_id["wrong"].passed
    assert "answer missing '5'" in by_id["wrong"].failures
    assert "took 2 steps (max 1)" in by_id["wrong"].failures
    assert round(report.pass_rate, 2) == 0.67
    assert "Pass rate: 67%" in report.summary()


def test_bundled_suite_is_valid():
    cases = load_cases(Path(__file__).parent.parent / "evals" / "suite.json")
    assert len(cases) >= 5 and len({c.id for c in cases}) == len(cases)
