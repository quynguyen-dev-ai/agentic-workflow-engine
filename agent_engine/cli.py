"""Command line: `python -m agent_engine.cli run "goal"` or `... eval evals/suite.json`."""

import argparse
import sys

from .agent import Agent, TraceEvent
from .evals import load_cases, run_suite
from .server import build_llm, build_tools

COLORS = {"llm": "\033[36m", "tool_call": "\033[33m", "tool_result": "\033[32m", "tool_error": "\033[31m"}


def print_event(e: TraceEvent) -> None:
    color = COLORS.get(e.kind, "\033[35m")
    print(f"{color}[step {e.step} · {e.kind} · {e.elapsed_ms:.0f}ms]\033[0m {e.data}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-engine")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Run the agent on a goal and print its trace")
    r.add_argument("goal")
    e = sub.add_parser("eval", help="Run an evaluation suite")
    e.add_argument("suite")
    e.add_argument("--min-pass-rate", type=float, default=0.8)
    args = parser.parse_args(argv)

    if args.cmd == "run":
        result = Agent(build_llm(), build_tools(), on_event=print_event).run(args.goal)
        print(f"\n{result.answer}\n({result.stop_reason}, {result.steps} steps)")
        return 0 if result.ok else 1

    report = run_suite(lambda: Agent(build_llm(), build_tools()), load_cases(args.suite))
    print(report.summary())
    return 0 if report.pass_rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
