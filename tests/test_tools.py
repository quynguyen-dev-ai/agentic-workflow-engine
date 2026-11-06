import pytest

from agent_engine.tools import ToolError, ToolRegistry, tool
from agent_engine.tools.builtin import calculator, make_knowledge_base


@pytest.mark.parametrize(
    "expr,expected",
    [("2 + 3 * 4", "14"), ("(10 - 4) / 4", "1.5"), ("2 ** 10", "1024"), ("-5 + 2", "-3"), ("7 // 2", "3")],
)
def test_calculator(expr, expected):
    assert calculator.run({"expression": expr}) == expected


@pytest.mark.parametrize("expr", ["__import__('os')", "open('x')", "a + 1", "2 ** 1000", "1 / 0", "1 +"])
def test_calculator_rejects_unsafe_or_invalid(expr):
    with pytest.raises(ToolError):
        calculator.run({"expression": expr})


def test_schema_is_derived_from_signature():
    @tool()
    def greet(name: str, times: int = 1) -> str:
        """Greet someone."""
        return " ".join([f"hi {name}"] * times)

    schema = greet.json_schema
    assert schema["name"] == "greet"
    assert schema["description"] == "Greet someone."
    assert schema["parameters"]["required"] == ["name"]
    assert schema["parameters"]["properties"]["times"]["type"] == "integer"
    assert greet.run({"name": "Ana", "times": 2}) == "hi Ana hi Ana"


def test_invalid_arguments_raise_tool_error():
    with pytest.raises(ToolError, match="Invalid arguments"):
        calculator.run({"wrong": "x"})


def test_untyped_tool_is_rejected_at_definition_time():
    with pytest.raises(TypeError):

        @tool()
        def bad(x):  # noqa: ANN001
            return x


def test_registry_rejects_duplicates_and_unknown_tools():
    reg = ToolRegistry().register(calculator)
    with pytest.raises(ValueError):
        reg.register(calculator)
    with pytest.raises(ToolError, match="Unknown tool"):
        reg.get("nope")


def test_knowledge_base_search():
    kb = make_knowledge_base({"a": "Refunds within 30 days", "b": "Office in Berlin"})
    assert "[a]" in kb.run({"query": "refunds policy"})
    assert kb.run({"query": "parental leave"}) == "No matching entries."
