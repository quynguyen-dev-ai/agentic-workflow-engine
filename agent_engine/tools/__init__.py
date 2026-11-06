"""Tool definitions.

A tool is a plain Python function with type hints. The `@tool` decorator turns
it into a `Tool` whose JSON schema is derived from a Pydantic model built from
the signature, so arguments the LLM produces are validated before the function
ever runs.
"""

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError, create_model


class ToolError(Exception):
    """Raised for failures the agent should see and recover from."""


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    args_model: type[BaseModel]
    timeout_s: float = 10.0

    @property
    def json_schema(self) -> dict:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {"name": self.name, "description": self.description, "parameters": schema}

    def run(self, raw_args: dict) -> str:
        try:
            args = self.args_model.model_validate(raw_args)
        except ValidationError as e:
            raise ToolError(f"Invalid arguments for {self.name}: {e.errors(include_url=False)}") from e
        start = time.monotonic()
        result = self.fn(**args.model_dump())
        if time.monotonic() - start > self.timeout_s:
            raise ToolError(f"{self.name} exceeded its {self.timeout_s}s budget")
        return result if isinstance(result, str) else repr(result)


def tool(name: str | None = None, timeout_s: float = 10.0) -> Callable[[Callable], Tool]:
    """Decorator: build a Tool from a typed function and its docstring."""

    def wrap(fn: Callable) -> Tool:
        sig = inspect.signature(fn)
        fields: dict[str, Any] = {}
        for pname, p in sig.parameters.items():
            if p.annotation is inspect.Parameter.empty:
                raise TypeError(f"Parameter {pname!r} of {fn.__name__} needs a type hint")
            default = ... if p.default is inspect.Parameter.empty else p.default
            fields[pname] = (p.annotation, default)
        model = create_model(f"{fn.__name__}_args", **fields)
        doc = inspect.getdoc(fn) or ""
        return Tool(name or fn.__name__, doc.strip(), fn, model, timeout_s)

    return wrap


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, *tools: Tool) -> "ToolRegistry":
        for t in tools:
            if t.name in self.tools:
                raise ValueError(f"Duplicate tool name: {t.name}")
            self.tools[t.name] = t
        return self

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise ToolError(f"Unknown tool {name!r}. Available: {sorted(self.tools)}")
        return self.tools[name]

    def schemas(self) -> list[dict]:
        return [t.json_schema for t in self.tools.values()]
