from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    success: bool
    steps: int
    tool_calls: int
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    decisions: list = field(default_factory=list)  # human-readable trace of what the agent decided


class Agent:
    """Every agent under test implements run(task, env) -> AgentResult.

    Agents must be framework-independent of AgentBench itself — the harness
    only ever calls `.run()`. This is what lets AgentBench benchmark a
    hand-written bot and a full LLM agent with the exact same measurement code.
    """
    name: str = "unnamed-agent"
    model_name: str = "n/a"
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0

    def run(self, task, env) -> AgentResult:  # noqa: D401
        raise NotImplementedError
