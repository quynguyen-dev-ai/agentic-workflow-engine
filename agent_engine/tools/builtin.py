"""A few safe, dependency-free tools useful for demos and evals."""

import ast
import operator
from datetime import UTC, datetime

from . import ToolError, tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
            raise ToolError("Exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ToolError(f"Unsupported expression element: {type(node).__name__}")


@tool()
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. "(12.5 * 4) / 3". Supports + - * / // % ** and parentheses."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"Not a valid expression: {expression!r}") from e
    try:
        value = _eval(tree.body)
    except ZeroDivisionError as e:
        raise ToolError("Division by zero") from e
    return str(round(value, 10))


@tool()
def current_time(timezone_offset_hours: float = 0.0) -> str:
    """Return the current date and time as ISO 8601 for a UTC offset in hours (e.g. 1 for Berlin in winter)."""
    from datetime import timedelta, timezone

    tz = timezone(timedelta(hours=timezone_offset_hours))
    return datetime.now(UTC).astimezone(tz).isoformat(timespec="seconds")


def make_knowledge_base(docs: dict[str, str]):
    """Build a keyword-search tool over a small in-memory knowledge base."""

    @tool(name="search_knowledge_base")
    def search(query: str, max_results: int = 3) -> str:
        """Search the internal knowledge base. Returns the most relevant entries with their ids."""
        terms = {t for t in query.lower().split() if len(t) > 2}
        scored = sorted(
            ((sum(t in text.lower() for t in terms), doc_id, text) for doc_id, text in docs.items()),
            reverse=True,
        )
        hits = [f"[{doc_id}] {text}" for score, doc_id, text in scored[:max_results] if score > 0]
        return "\n".join(hits) if hits else "No matching entries."

    return search
