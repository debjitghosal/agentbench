from __future__ import annotations
from agentbench.tools import ToolFailure
from agentbench.agents.base import Agent, AgentResult


class GreedyBot(Agent):
    """A deterministic, hand-written solver.

    It is deliberately task-aware (it knows what t1..t8 are asking) but tool-
    calling is real: every number it writes comes from an actual `calculator`
    or `search_kb` call against the shared ToolEnvironment, not from literals.
    It is "competent" in two specific ways the benchmark is designed to catch:

      1. It retries a tool up to 3 times on ToolFailure before giving up
         (resilience to flaky tools / outages).
      2. It never trusts a hint embedded in the task prompt — it always
         re-derives the answer itself (resistance to misleading instructions).
    """
    name = "GreedyBot (rule-based baseline)"
    model_name = "n/a (scripted)"

    def _call(self, fn, *args, decisions, steps, **kwargs):
        for attempt in range(3):
            try:
                result = fn(*args, **kwargs)
                decisions.append(f"tool call ok on attempt {attempt+1}: {fn.__name__}{args}")
                return result, steps + 1
            except ToolFailure as e:
                decisions.append(f"tool call failed (attempt {attempt+1}): {e} — retrying")
        raise ToolFailure("exceeded retry budget")

    def run(self, task, env) -> AgentResult:
        decisions = []
        steps = 0
        tool_calls_before = len(env.trace)
        try:
            if task.id == "t1_arithmetic":
                v, steps = self._call(env.calculator, "(17 * 23) - sqrt(49)", decisions=decisions, steps=steps)
                _, steps = self._call(env.write_file, "answer.txt", str(v), decisions=decisions, steps=steps)
            elif task.id == "t2_multi_step_calc":
                v, steps = self._call(env.calculator, "120 - 120*0.35 + 40", decisions=decisions, steps=steps)
                _, steps = self._call(env.write_file, "answer.txt", str(v), decisions=decisions, steps=steps)
            elif task.id == "t3_kb_lookup":
                v, steps = self._call(env.search_kb, "refund_policy", decisions=decisions, steps=steps)
                _, steps = self._call(env.write_file, "answer.txt", v, decisions=decisions, steps=steps)
            elif task.id == "t4_file_roundtrip":
                _, steps = self._call(env.write_file, "notes.txt", "quarterly report v1", decisions=decisions, steps=steps)
                content, steps = self._call(env.read_file, "notes.txt", decisions=decisions, steps=steps)
                wc = len(content.split())
                _, steps = self._call(env.write_file, "answer.txt", str(wc), decisions=decisions, steps=steps)
            elif task.id == "t5_chained_lookup":
                contact, steps = self._call(env.search_kb, "escalation_contact", decisions=decisions, steps=steps)
                sla, steps = self._call(env.search_kb, contact, decisions=decisions, steps=steps)
                _, steps = self._call(env.write_file, "answer.txt", sla, decisions=decisions, steps=steps)
            elif task.id == "t6_conditional_calc":
                rem, steps = self._call(env.calculator, "84 % 7", decisions=decisions, steps=steps)
                if rem == 0:
                    v, steps = self._call(env.calculator, "84/7", decisions=decisions, steps=steps)
                    _, steps = self._call(env.write_file, "answer.txt", f"even-path {v}", decisions=decisions, steps=steps)
                else:
                    _, steps = self._call(env.write_file, "answer.txt", "odd-path", decisions=decisions, steps=steps)
            elif task.id == "t7_budget_calc":
                v, steps = self._call(env.calculator, "5000 + 5000*0.18", decisions=decisions, steps=steps)
                _, steps = self._call(env.write_file, "answer.txt", str(v), decisions=decisions, steps=steps)
            elif task.id == "t8_kb_aggregate":
                a, steps = self._call(env.search_kb, "onboarding_time", decisions=decisions, steps=steps)
                b, steps = self._call(env.search_kb, "training_time", decisions=decisions, steps=steps)
                total, steps = self._call(env.calculator, f"{a}+{b}", decisions=decisions, steps=steps)
                _, steps = self._call(env.write_file, "answer.txt", str(total), decisions=decisions, steps=steps)
            else:
                raise ValueError(f"GreedyBot has no routine for {task.id}")

            tool_calls = len(env.trace) - tool_calls_before
            success = task.check(env)
            return AgentResult(success=success, steps=steps, tool_calls=tool_calls, decisions=decisions)
        except ToolFailure as e:
            tool_calls = len(env.trace) - tool_calls_before
            return AgentResult(success=False, steps=steps, tool_calls=tool_calls, error=str(e), decisions=decisions)
