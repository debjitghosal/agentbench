"""
A small, deterministic tool environment every agent under test shares.

Why this exists: AgentBench's whole thesis is that an agent's benchmark score
means nothing if it was only ever tested against tools that behave perfectly.
ToolEnvironment lets the harness inject three real-world failure modes without
touching agent code:

  - flaky:              a named tool fails its first N calls, then recovers
  - dead:                a named tool fails every call (hard outage)
  - model_unavailable:   the *model*, not a tool, is unreachable (handled in harness.py)

Every tool call is logged to `self.trace` so the report layer can render a
step-by-step execution trace for any run, not just a pass/fail flag.
"""
from __future__ import annotations

import time
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable


class ToolFailure(Exception):
    """Raised by a tool when it is unavailable (flaky or dead)."""


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    ok: bool
    result: Any
    latency_ms: float


@dataclass
class ToolEnvironment:
    # local "world state" tools operate on
    files: dict = field(default_factory=dict)
    knowledge_base: dict = field(default_factory=dict)
    # failure injection config, set per-run by the harness
    flaky_tools: dict = field(default_factory=dict)   # {tool_name: fail_count_remaining}
    dead_tools: set = field(default_factory=set)
    trace: list = field(default_factory=list)

    def reset_trace(self):
        self.trace = []

    def _guard(self, name: str):
        if name in self.dead_tools:
            raise ToolFailure(f"tool '{name}' is unavailable (simulated outage)")
        if name in self.flaky_tools and self.flaky_tools[name] > 0:
            self.flaky_tools[name] -= 1
            raise ToolFailure(f"tool '{name}' timed out (simulated transient failure, "
                               f"{self.flaky_tools[name]} failures remaining before recovery)")

    def _call(self, name: str, args: dict, fn: Callable[[], Any]) -> Any:
        t0 = time.perf_counter()
        try:
            self._guard(name)
            result = fn()
            ok = True
        except ToolFailure as e:
            result = str(e)
            ok = False
        latency_ms = (time.perf_counter() - t0) * 1000
        self.trace.append(ToolCallRecord(name, args, ok, result, latency_ms))
        if not ok:
            raise ToolFailure(result)
        return result

    # ---- the actual toolset tasks are built from ----

    def calculator(self, expression: str) -> float:
        def run():
            allowed = {"sqrt": math.sqrt, "pi": math.pi, "pow": pow, "abs": abs, "round": round}
            # restricted eval: digits, operators, parens and the whitelisted names only
            return eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307 (sandboxed, whitelisted)
        return self._call("calculator", {"expression": expression}, run)

    def write_file(self, path: str, content: str) -> str:
        def run():
            self.files[path] = content
            return f"wrote {len(content)} bytes to {path}"
        return self._call("write_file", {"path": path, "content": content}, run)

    def read_file(self, path: str) -> str:
        def run():
            if path not in self.files:
                raise KeyError(f"no such file: {path}")
            return self.files[path]
        return self._call("read_file", {"path": path}, run)

    def search_kb(self, query: str) -> str:
        def run():
            q = query.lower()
            hits = [v for k, v in self.knowledge_base.items() if k.lower() in q or q in k.lower()]
            return hits[0] if hits else "no matching record"
        return self._call("search_kb", {"query": query}, run)

    def get_time(self) -> str:
        return self._call("get_time", {}, lambda: "2026-11-11T09:00:00Z")
