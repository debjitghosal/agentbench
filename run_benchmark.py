#!/usr/bin/env python3
"""
AgentBench CLI.

Usage:
    python run_benchmark.py                 # scripted baselines only (no API key needed)
    python run_benchmark.py --with-claude    # also benchmarks a real Claude tool-use agent
                                              # (requires ANTHROPIC_API_KEY in the environment)
    python run_benchmark.py --trials 5

Writes outputs/results.json and outputs/dashboard.html.
"""
from __future__ import annotations
import argparse
import os
from agentbench.tasks import TASKS
from agentbench.agents.rule_based_agent import GreedyBot
from agentbench.agents.naive_agent import NaiveBot
from agentbench.harness import run_benchmark
from agentbench.metrics import summarize
from agentbench.report import write_results_json, write_dashboard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--with-claude", action="store_true",
                     help="also run ClaudeAgent (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    agents = [GreedyBot(), NaiveBot()]
    if args.with_claude:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("WARNING: --with-claude given but ANTHROPIC_API_KEY is not set; "
                  "ClaudeAgent trials will all record as model_unavailable.")
        from agentbench.agents.claude_agent import ClaudeAgent
        agents.append(ClaudeAgent())

    print(f"Running AgentBench: {len(agents)} agent(s) x {len(TASKS)} task(s), {args.trials} trial(s) each...")
    records = run_benchmark(agents, TASKS, trials=args.trials)
    summary = summarize(records)

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.json")
    dashboard_path = os.path.join(args.out, "dashboard.html")
    data = write_results_json(records, summary, results_path)
    write_dashboard(data, dashboard_path)

    print(f"\n{len(records)} trials recorded.")
    print(f"Results:   {results_path}")
    print(f"Dashboard: {dashboard_path}  (open in a browser)")
    print("\n=== Overall ===")
    for agent, m in summary["overall"].items():
        print(f"  {agent:35s} success={m['success_rate']*100:5.1f}%  "
              f"steps={m['avg_steps']:.2f}  tool_calls={m['avg_tool_calls']:.2f}  "
              f"cost=${m['avg_cost_usd']:.6f}  latency={m['avg_latency_s']:.4f}s")


if __name__ == "__main__":
    main()
