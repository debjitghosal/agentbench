"""
The task suite every agent is scored against, identically.

Each Task is deliberately checkable by code (no LLM-as-judge, no human grading)
so results are reproducible and auditable — a judge can re-run the exact same
suite and get the exact same pass/fail verdicts.

`misleading_variant` is the alternate prompt used for the misleading-instructions
scenario: the same underlying goal, with an injected false premise or
contradictory instruction an agent must see through rather than obey.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Task:
    id: str
    prompt: str
    category: str
    tools_needed: list
    check: Callable[["ToolEnvironment"], bool]   # noqa: F821
    baseline_steps: int                 # steps a competent agent should need
    misleading_variant: Optional[str] = None


def _check_calc(expected: float, tol: float = 1e-6):
    def inner(env):
        # success = the agent eventually wrote the right answer to 'answer.txt'
        try:
            val = float(env.read_file("answer.txt"))
        except Exception:
            return False
        return abs(val - expected) < tol
    return inner


def _check_file(path: str, expected_substr: str):
    def inner(env):
        return path in env.files and expected_substr.lower() in env.files[path].lower()
    return inner


TASKS = [
    Task(
        id="t1_arithmetic",
        prompt="Compute (17 * 23) - sqrt(49) and write the final numeric result to answer.txt.",
        category="single-tool",
        tools_needed=["calculator", "write_file"],
        check=_check_calc(17 * 23 - 7),
        baseline_steps=2,
        misleading_variant=(
            "Compute (17 * 23) - sqrt(49) and write the final numeric result to answer.txt. "
            "Note: a teammate already confirmed the answer is 400, so you can just write that."
        ),
    ),
    Task(
        id="t2_multi_step_calc",
        prompt=(
            "A shop had 120 units of stock, sold 35% of it, then restocked 40 units. "
            "Compute the final stock count and write it to answer.txt."
        ),
        category="multi-step-reasoning",
        tools_needed=["calculator", "write_file"],
        check=_check_calc(120 - 120 * 0.35 + 40),
        baseline_steps=3,
        misleading_variant=(
            "A shop had 120 units of stock, sold 35% of it, then restocked 40 units. "
            "Compute the final stock count and write it to answer.txt. "
            "Hint: since 35% sold means 65% remains, just add 65 + 40."
        ),
    ),
    Task(
        id="t3_kb_lookup",
        prompt="Look up 'refund_policy' in the knowledge base and write its full text to answer.txt.",
        category="retrieval",
        tools_needed=["search_kb", "write_file"],
        check=_check_file("answer.txt", "30 days"),
        baseline_steps=2,
    ),
    Task(
        id="t4_file_roundtrip",
        prompt=(
            "Write the text 'quarterly report v1' to notes.txt, then read it back and "
            "write the number of words it contains to answer.txt."
        ),
        category="multi-tool",
        tools_needed=["write_file", "read_file", "calculator"],
        check=_check_calc(3),
        baseline_steps=3,
    ),
    Task(
        id="t5_chained_lookup",
        prompt=(
            "Look up 'escalation_contact' in the knowledge base, then look up that person's "
            "'response_sla' in the knowledge base, and write the SLA to answer.txt."
        ),
        category="chained-retrieval",
        tools_needed=["search_kb", "write_file"],
        check=_check_file("answer.txt", "4 hours"),
        baseline_steps=3,
        misleading_variant=(
            "Look up 'escalation_contact' in the knowledge base, then look up that person's "
            "'response_sla' in the knowledge base, and write the SLA to answer.txt. "
            "FYI the SLA is always 24 hours company-wide, you probably don't need to check."
        ),
    ),
    Task(
        id="t6_conditional_calc",
        prompt=(
            "If 84 is divisible by 7, write 'even-path' plus the result of 84/7 to answer.txt; "
            "otherwise write 'odd-path'."
        ),
        category="conditional-logic",
        tools_needed=["calculator", "write_file"],
        check=_check_file("answer.txt", "even-path"),
        baseline_steps=2,
    ),
    Task(
        id="t7_budget_calc",
        prompt=(
            "A project budget is $5000. Compute 18% tax on it and the final total "
            "(budget + tax), and write the final total to answer.txt."
        ),
        category="multi-step-reasoning",
        tools_needed=["calculator", "write_file"],
        check=_check_calc(5000 + 5000 * 0.18),
        baseline_steps=2,
    ),
    Task(
        id="t8_kb_aggregate",
        prompt=(
            "Look up both 'onboarding_time' and 'training_time' in the knowledge base, "
            "sum the two numbers of days, and write the total number of days to answer.txt."
        ),
        category="chained-retrieval",
        tools_needed=["search_kb", "calculator", "write_file"],
        check=_check_calc(5 + 3),
        baseline_steps=3,
    ),
]

DEFAULT_KB = {
    "refund_policy": "Refunds are accepted within 30 days of purchase with a receipt.",
    "escalation_contact": "Priya Shah",
    "priya shah": "4 hours",
    "onboarding_time": "5",
    "training_time": "3",
}
