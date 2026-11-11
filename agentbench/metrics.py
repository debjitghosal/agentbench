from __future__ import annotations
from collections import defaultdict
import statistics as stats


def _agg(records):
    n = len(records)
    successes = sum(1 for r in records if r.success)
    return {
        "trials": n,
        "success_rate": round(successes / n, 3) if n else 0.0,
        "avg_steps": round(stats.mean(r.steps for r in records), 2) if n else 0.0,
        "avg_tool_calls": round(stats.mean(r.tool_calls for r in records), 2) if n else 0.0,
        "avg_tokens": round(stats.mean(r.input_tokens + r.output_tokens for r in records), 1) if n else 0.0,
        "avg_cost_usd": round(stats.mean(r.cost_usd for r in records), 6) if n else 0.0,
        "avg_latency_s": round(stats.mean(r.latency_s for r in records), 4) if n else 0.0,
        "std_latency_s": round(stats.pstdev(r.latency_s for r in records), 4) if n > 1 else 0.0,
    }


def summarize(records):
    by_agent = defaultdict(list)
    by_agent_scenario = defaultdict(list)
    by_agent_task = defaultdict(list)

    for r in records:
        by_agent[r.agent].append(r)
        by_agent_scenario[(r.agent, r.scenario)].append(r)
        by_agent_task[(r.agent, r.task_id)].append(r)

    return {
        "overall": {agent: _agg(rs) for agent, rs in by_agent.items()},
        "by_scenario": {f"{a} / {s}": _agg(rs) for (a, s), rs in by_agent_scenario.items()},
        "by_task": {f"{a} / {t}": _agg(rs) for (a, t), rs in by_agent_task.items()},
    }
