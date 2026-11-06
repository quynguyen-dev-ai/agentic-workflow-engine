import itertools

from agent_engine import Agent, AgentConfig, ToolRegistry, tool
from agent_engine.llm import LLMResponse, Message, ScriptedLLM, ToolCall
from agent_engine.tools.builtin import calculator

_ids = itertools.count()


def call(name: str, **args) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(f"c{next(_ids)}", name, args)], input_tokens=10, output_tokens=5)


def final(text: str) -> LLMResponse:
    return LLMResponse(text=text, input_tokens=10, output_tokens=5)


def last_tool_output(messages: list[Message]) -> str:
    return next(m.content for m in reversed(messages) if m.role == "tool")


def test_tool_then_final_answer():
    def policy(msgs):
        if not any(m.role == "tool" for m in msgs):
            return call("calculator", expression="6 * 7")
        return final(f"The answer is {last_tool_output(msgs)}")

    result = Agent(ScriptedLLM(policy), ToolRegistry().register(calculator)).run("6 times 7?")
    assert result.ok and result.answer == "The answer is 42"
    assert result.tools_used() == ["calculator"]
    assert result.steps == 2 and result.input_tokens == 20 and result.output_tokens == 10
    assert [e.kind for e in result.trace] == ["llm", "tool_call", "tool_result", "llm", "final"]


def test_tool_errors_are_fed_back_so_the_model_can_recover():
    def policy(msgs):
        tool_msgs = [m for m in msgs if m.role == "tool"]
        if not tool_msgs:
            return call("calculator", expression="6 x 7")  # invalid
        if tool_msgs[-1].content.startswith("ERROR"):
            return call("calculator", expression="6 * 7")
        return final(tool_msgs[-1].content)

    result = Agent(ScriptedLLM(policy), ToolRegistry().register(calculator)).run("6 times 7?")
    assert result.answer == "42"
    assert any(e.kind == "tool_error" for e in result.trace)


def test_max_steps_guardrail_stops_infinite_loops():
    llm = ScriptedLLM(lambda msgs: call("calculator", expression="1 + 1"))
    result = Agent(llm, ToolRegistry().register(calculator), AgentConfig(max_steps=3)).run("loop")
    assert result.stop_reason == "max_steps" and llm.calls == 3 and result.answer is None


def test_max_tool_calls_guardrail():
    many = LLMResponse(tool_calls=[ToolCall(f"x{i}", "calculator", {"expression": "1"}) for i in range(5)])
    result = Agent(ScriptedLLM(lambda m: many), ToolRegistry().register(calculator), AgentConfig(max_tool_calls=3)).run(
        "x"
    )
    assert result.stop_reason == "max_tool_calls" and result.tool_calls == 3


def test_token_budget_guardrail():
    llm = ScriptedLLM(lambda m: LLMResponse(text="hi", input_tokens=900, output_tokens=200))
    result = Agent(llm, ToolRegistry(), AgentConfig(max_total_tokens=1000)).run("x")
    assert result.stop_reason == "token_budget"


def test_disallowed_tool_is_blocked_and_hidden_from_the_model():
    @tool()
    def delete_everything(confirm: bool) -> str:
        """Dangerous."""
        raise AssertionError("must never run")

    seen_schemas = []

    class Spy(ScriptedLLM):
        def complete(self, messages, tools):
            seen_schemas.append([t["name"] for t in tools])
            return super().complete(messages, tools)

    def policy(msgs):
        if not any(m.role == "tool" for m in msgs):
            return call("delete_everything", confirm=True)
        return final(last_tool_output(msgs))

    reg = ToolRegistry().register(calculator, delete_everything)
    result = Agent(Spy(policy), reg, AgentConfig(allowed_tools={"calculator"})).run("x")
    assert "not allowed" in result.answer
    assert seen_schemas[0] == ["calculator"]


def test_crashing_tool_does_not_crash_the_run():
    @tool()
    def flaky(x: int) -> str:
        """Always crashes."""
        raise RuntimeError("boom")

    def policy(msgs):
        if not any(m.role == "tool" for m in msgs):
            return call("flaky", x=1)
        return final(last_tool_output(msgs))

    result = Agent(ScriptedLLM(policy), ToolRegistry().register(flaky)).run("x")
    assert result.ok and "tool crashed" in result.answer


def test_events_are_streamed_to_callback():
    events = []
    Agent(ScriptedLLM(lambda m: final("done")), ToolRegistry(), on_event=events.append).run("x")
    assert [e.kind for e in events] == ["llm", "final"]
