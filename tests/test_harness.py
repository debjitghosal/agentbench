"""Minimal pytest suite proving the measurement layer itself is trustworthy."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentbench.tools import ToolEnvironment, ToolFailure
from agentbench.tasks import TASKS, DEFAULT_KB
from agentbench.agents.rule_based_agent import GreedyBot
from agentbench.agents.naive_agent import NaiveBot
from agentbench.harness import run_benchmark
from agentbench.metrics import summarize


def fresh_env():
    return ToolEnvironment(files={}, knowledge_base=dict(DEFAULT_KB))


def test_calculator_basic():
    env = fresh_env()
    assert env.calculator("2 + 2") == 4


def test_flaky_tool_recovers_after_n_failures():
    env = fresh_env()
    env.flaky_tools = {"calculator": 2}
    for _ in range(2):
        try:
            env.calculator("1+1")
            assert False, "expected ToolFailure"
        except ToolFailure:
            pass
    assert env.calculator("1+1") == 2  # recovers on 3rd call


def test_dead_tool_always_fails():
    env = fresh_env()
    env.dead_tools = {"search_kb"}
    try:
        env.search_kb("refund_policy")
        assert False, "expected ToolFailure"
    except ToolFailure:
        pass


def test_all_tasks_are_solvable_by_greedybot_under_normal_conditions():
    bot = GreedyBot()
    for task in TASKS:
        env = fresh_env()
        result = bot.run(task, env)
        assert result.success, f"{task.id} should be solvable under normal conditions"


def test_benchmark_discriminates_agent_quality():
    records = run_benchmark([GreedyBot(), NaiveBot()], TASKS, trials=3)
    summary = summarize(records)
    greedy = summary["overall"]["GreedyBot (rule-based baseline)"]["success_rate"]
    naive = summary["overall"]["NaiveBot (weak baseline)"]["success_rate"]
    assert greedy > naive, "the benchmark should clearly separate a competent agent from a weak one"
