"""Evaluation harness.

Agents are non-deterministic, so "it worked when I tried it" is not a test.
An eval suite is a JSON file of cases; each case states the goal and the checks
the run must pass. The harness runs every case and reports pass rate, cost and
latency so prompt or model changes can be compared objectively.
"""

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .agent import Agent, AgentResult


@dataclass
class EvalCase:
    id: str
    goal: str
    expect_contains: list[str] = field(default_factory=list)
    expect_regex: str | None = None
    expect_tools: list[str] = field(default_factory=list)
    forbid_tools: list[str] = field(default_factory=list)
    max_steps: int | None = None


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    failures: list[str]
    steps: int
    tool_calls: int
    tokens: int
    latency_ms: float
    answer: str | None


@dataclass
class EvalReport:
    results: list[CaseResult]

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0

    def summary(self) -> str:
        lines = [f"{'case':<28} {'result':<6} {'steps':>5} {'tools':>5} {'tokens':>7} {'ms':>7}"]
        for r in self.results:
            lines.append(
                f"{r.case_id:<28} {'PASS' if r.passed else 'FAIL':<6} {r.steps:>5} {r.tool_calls:>5} "
                f"{r.tokens:>7} {r.latency_ms:>7.0f}"
            )
            lines += [f"    - {f}" for f in r.failures]
        lines.append(f"\nPass rate: {self.pass_rate:.0%} ({sum(r.passed for r in self.results)}/{len(self.results)})")
        return "\n".join(lines)


def check(case: EvalCase, run: AgentResult) -> list[str]:
    failures: list[str] = []
    answer = (run.answer or "").lower()
    if not run.ok:
        failures.append(f"did not finish: {run.stop_reason}")
    for needle in case.expect_contains:
        if needle.lower() not in answer:
            failures.append(f"answer missing {needle!r}")
    if case.expect_regex and not re.search(case.expect_regex, run.answer or "", re.IGNORECASE):
        failures.append(f"answer does not match /{case.expect_regex}/")
    used = run.tools_used()
    for t in case.expect_tools:
        if t not in used:
            failures.append(f"expected tool {t!r} to be called")
    for t in case.forbid_tools:
        if t in used:
            failures.append(f"tool {t!r} must not be called")
    if case.max_steps is not None and run.steps > case.max_steps:
        failures.append(f"took {run.steps} steps (max {case.max_steps})")
    return failures


def load_cases(path: str | Path) -> list[EvalCase]:
    return [EvalCase(**c) for c in json.loads(Path(path).read_text())]


def run_suite(make_agent: Callable[[], Agent], cases: list[EvalCase]) -> EvalReport:
    results = []
    for case in cases:
        agent = make_agent()  # fresh agent per case: no state leaks between cases
        t0 = time.monotonic()
        run = agent.run(case.goal)
        failures = check(case, run)
        results.append(
            CaseResult(
                case.id,
                not failures,
                failures,
                run.steps,
                run.tool_calls,
                run.input_tokens + run.output_tokens,
                (time.monotonic() - t0) * 1000,
                run.answer,
            )
        )
    return EvalReport(results)
