from __future__ import annotations
import re
from agentbench.tools import ToolFailure
from agentbench.agents.base import Agent, AgentResult


class NaiveBot(Agent):
    """A weaker baseline, included on purpose.

    NaiveBot does the task correctly under ideal conditions, but:
      - it never retries a failed tool call (one flaky-tool failure = task failure)
      - it trusts any hint embedded in the prompt instead of re-deriving the
        answer itself (so the misleading-instructions scenario breaks it)

    If a benchmarking platform can't show daylight between GreedyBot and
    NaiveBot across scenarios, the platform isn't measuring anything — this
    agent is the sanity check for the harness itself.
    """
    name = "NaiveBot (weak baseline)"
    model_name = "n/a (scripted)"

    def run(self, task, env) -> AgentResult:
        decisions = []
        steps = 0
        tool_calls_before = len(env.trace)
        prompt = task.prompt

        # "trusts hints": if the prompt contains a suggested shortcut number, use it verbatim
        hinted = re.search(r"(?:is|just write|probably don't need to check)\D*?(-?\d+(?:\.\d+)?)", prompt)

        try:
            if hinted:
                decisions.append(f"saw a hint in the prompt, trusting it: {hinted.group(1)}")
                env.write_file("answer.txt", hinted.group(1))  # does not retry, does not verify
                steps += 1
            elif task.id == "t1_arithmetic":
                v = env.calculator("(17 * 23) - sqrt(49)"); steps += 1
                env.write_file("answer.txt", str(v)); steps += 1
            elif task.id == "t2_multi_step_calc":
                v = env.calculator("120 - 120*0.35 + 40"); steps += 1
                env.write_file("answer.txt", str(v)); steps += 1
            elif task.id == "t3_kb_lookup":
                v = env.search_kb("refund_policy"); steps += 1
                env.write_file("answer.txt", v); steps += 1
            elif task.id == "t4_file_roundtrip":
                env.write_file("notes.txt", "quarterly report v1"); steps += 1
                content = env.read_file("notes.txt"); steps += 1
                env.write_file("answer.txt", str(len(content.split()))); steps += 1
            elif task.id == "t5_chained_lookup":
                contact = env.search_kb("escalation_contact"); steps += 1
                sla = env.search_kb(contact); steps += 1
                env.write_file("answer.txt", sla); steps += 1
            elif task.id == "t6_conditional_calc":
                rem = env.calculator("84 % 7"); steps += 1
                if rem == 0:
                    v = env.calculator("84/7"); steps += 1
                    env.write_file("answer.txt", f"even-path {v}"); steps += 1
                else:
                    env.write_file("answer.txt", "odd-path"); steps += 1
            elif task.id == "t7_budget_calc":
                v = env.calculator("5000 + 5000*0.18"); steps += 1
                env.write_file("answer.txt", str(v)); steps += 1
            elif task.id == "t8_kb_aggregate":
                a = env.search_kb("onboarding_time"); steps += 1
                b = env.search_kb("training_time"); steps += 1
                total = env.calculator(f"{a}+{b}"); steps += 1
                env.write_file("answer.txt", str(total)); steps += 1
            else:
                raise ValueError(f"NaiveBot has no routine for {task.id}")

            tool_calls = len(env.trace) - tool_calls_before
            success = task.check(env)
            return AgentResult(success=success, steps=steps, tool_calls=tool_calls, decisions=decisions)
        except ToolFailure as e:
            # no retry logic: first failure ends the run
            tool_calls = len(env.trace) - tool_calls_before
            decisions.append(f"gave up immediately on tool failure: {e}")
            return AgentResult(success=False, steps=steps, tool_calls=tool_calls, error=str(e), decisions=decisions)
