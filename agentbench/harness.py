from __future__ import annotations
import time
import copy
from dataclasses import dataclass, asdict
from agentbench.tools import ToolEnvironment
from agentbench.tasks import DEFAULT_KB

SCENARIOS = ["normal", "flaky_tool", "dead_tool", "misleading_instructions", "model_unavailable"]


@dataclass
class TrialRecord:
    agent: str
    model: str
    task_id: str
    category: str
    scenario: str
    trial: int
    success: bool
    steps: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    error: str | None
    decisions: list
    tool_trace: list


def _fresh_env():
    return ToolEnvironment(files={}, knowledge_base=dict(DEFAULT_KB))


def _applies(agent, task, scenario) -> bool:
    if scenario == "misleading_instructions" and task.misleading_variant is None:
        return False
    if scenario in ("flaky_tool", "dead_tool") and not task.tools_needed:
        return False
    if scenario == "model_unavailable" and agent.model_name.startswith("n/a"):
        return False  # scripted agents have no "model" to lose
    return True


def run_benchmark(agents, tasks, trials: int = 3, force_unavailable_rate: float = 1.0):
    """Run every applicable (agent, task, scenario) combo `trials` times.

    `force_unavailable_rate` = fraction of model_unavailable trials where the
    model call is actually forced to fail (1.0 = always, for a clean signal).
    """
    records: list[TrialRecord] = []

    for agent in agents:
        for task in tasks:
            for scenario in SCENARIOS:
                if not _applies(agent, task, scenario):
                    continue
                for trial in range(trials):
                    env = _fresh_env()
                    prompt_task = task
                    target_tool = task.tools_needed[0] if task.tools_needed else None

                    if scenario == "flaky_tool" and target_tool:
                        env.flaky_tools = {target_tool: 2}
                    elif scenario == "dead_tool" and target_tool:
                        env.dead_tools = {target_tool}
                    elif scenario == "misleading_instructions":
                        prompt_task = copy.copy(task)
                        prompt_task.prompt = task.misleading_variant

                    forced_unavailable = scenario == "model_unavailable" and \
                        (trial / max(trials - 1, 1)) < force_unavailable_rate

                    t0 = time.perf_counter()
                    error = None
                    try:
                        if forced_unavailable:
                            orig_get_client = getattr(agent, "_get_client", None)
                            if orig_get_client is not None:
                                agent._get_client = lambda: (_ for _ in ()).throw(
                                    RuntimeError("simulated: model endpoint unreachable"))
                            result = agent.run(prompt_task, env)
                            if orig_get_client is not None:
                                agent._get_client = orig_get_client
                        else:
                            result = agent.run(prompt_task, env)
                    except RuntimeError as e:
                        error = str(e)
                        result = None
                    latency = time.perf_counter() - t0

                    if result is None:
                        cost = 0.0
                        records.append(TrialRecord(
                            agent=agent.name, model=agent.model_name, task_id=task.id,
                            category=task.category, scenario=scenario, trial=trial,
                            success=False, steps=0, tool_calls=0, input_tokens=0,
                            output_tokens=0, cost_usd=0.0, latency_s=latency,
                            error=error or "model_unavailable", decisions=[], tool_trace=[],
                        ))
                        continue

                    cost = (result.input_tokens / 1_000_000 * agent.cost_per_1m_input +
                            result.output_tokens / 1_000_000 * agent.cost_per_1m_output)
                    records.append(TrialRecord(
                        agent=agent.name, model=agent.model_name, task_id=task.id,
                        category=task.category, scenario=scenario, trial=trial,
                        success=result.success, steps=result.steps,
                        tool_calls=result.tool_calls, input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens, cost_usd=cost,
                        latency_s=latency, error=result.error, decisions=result.decisions,
                        tool_trace=[asdict(r) for r in env.trace],
                    ))
    return records
