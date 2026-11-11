# AgentBench

**A framework-independent benchmarking platform for AI agents — not just models.**

Built by **Team debjitruns** (Debjit Ghosal) for the **HackCulture Global AI Hackathon** (Geek Room),
Theme: *AI Agents & Intelligent Systems* — Problem Statement: *Build a Benchmarking Platform for AI Agents, Not Just Models.*

---

## The problem

Every "agent leaderboard" today measures one thing: did the agent get the right answer. That tells you
almost nothing about whether the agent is deployable. In production, agents don't fail because they're
bad at arithmetic — they fail because a tool times out, a model endpoint goes down mid-task, or the
context contains a confidently-wrong instruction the agent shouldn't have trusted. None of that shows
up in a standard accuracy benchmark.

**AgentBench measures resilience, not just correctness** — and it does it for *any* agent, regardless
of what framework (LangChain, a custom loop, raw API calls, or a hand-written script) built it.

## What it actually does

1. Runs any agent that implements a single `run(task, tool_environment) -> AgentResult` method against
   an identical, auto-gradable task suite (8 tasks spanning arithmetic, multi-step reasoning, knowledge
   retrieval, chained lookups, and conditional logic).
2. Injects three real-world failure modes into the *same* tasks, without the agent's code changing at all:
   - **`flaky_tool`** — a tool fails its first N calls, then recovers (simulated timeout/rate-limit)
   - **`dead_tool`** — a tool is permanently unavailable (simulated outage)
   - **`misleading_instructions`** — the task prompt contains a confident, wrong "hint" the agent must
     see through rather than obey
   - **`model_unavailable`** — the underlying model endpoint itself is forced to fail (tests fallback behavior)
3. Measures, per trial: **task success, step count, tool-call count, input/output tokens, USD cost
   (model-specific pricing), and wall-clock latency** — across repeated runs, with mean and standard
   deviation.
4. Outputs `results.json` (the full raw data) and a self-contained `dashboard.html` — a comparison
   table across agents and scenarios, plus a **click-through execution-trace viewer** for every single
   trial, so a reviewer can see exactly which tool call failed and how the agent reacted.

## Why this is the hard track

Most hackathon teams build a single flashy agent because it demos well in 90 seconds. Building the
thing that *evaluates* agents is less flashy and much easier to get subtly wrong — the grading has to
be deterministic and auditable (no LLM-as-judge hand-waving), the failure injection has to be real
(not just a flag that's never checked), and the platform has to stay agent-agnostic or it's just a
test suite for one agent. AgentBench is deliberately designed so a judge can swap in their own agent
by implementing one method and get a fair, resilience-aware score.

## Architecture

```
agentbench/
├── tools.py                 # ToolEnvironment: calculator, read/write_file, search_kb, get_time
│                             # + failure injection (flaky/dead) + per-call trace logging
├── tasks.py                 # 8 auto-gradable tasks, each with a code-based success check
│                             # and, where relevant, a "misleading instructions" variant
├── harness.py                # Orchestrates agent x task x scenario x trial, builds a fresh
│                             # sandboxed environment per trial, forces model_unavailable
│                             # for LLM agents, records a full TrialRecord per run
├── metrics.py                # Aggregates raw trials into overall / by-scenario / by-task
│                             # summary statistics (success rate, steps, tokens, cost, latency)
├── report.py                  # Writes results.json + a self-contained dashboard.html
│                             # (inline data, no CDN — works fully offline)
└── agents/
    ├── base.py                # Agent interface every agent under test implements
    ├── rule_based_agent.py     # GreedyBot — competent baseline: retries flaky tools,
    │                          # never trusts prompt-embedded hints
    ├── naive_agent.py          # NaiveBot — weak baseline, included on purpose as a sanity
    │                          # check that the harness actually discriminates agent quality
    └── claude_agent.py         # ClaudeAgent — a real, generic (non-task-aware) tool-use
                               # agent built on the Anthropic Messages API
run_benchmark.py              # CLI entry point
tests/test_harness.py         # pytest suite covering the measurement layer itself
outputs/                      # results.json + dashboard.html land here
```

## How AI is used

`ClaudeAgent` is the agent-under-test that actually uses an LLM: it receives only the raw task prompt
and four tool schemas (no task-specific logic, unlike the two baselines), and makes real
`anthropic.messages.create()` tool-use calls to decide what to do, how to react to a tool error, and
when to stop. This is the agent AgentBench is really built to evaluate — the two scripted bots
(`GreedyBot`, `NaiveBot`) exist as fixed reference points so the platform's own measurements can be
validated (`tests/test_harness.py::test_benchmark_discriminates_agent_quality`).

Because every agent only needs to implement `run(task, env) -> AgentResult`, plugging in an agent
built on LangChain, a different provider's SDK, or a local model is a single-file addition — the
harness, grading, failure injection, and dashboard never change.

## Installation

```bash
git clone <this-repo>
cd agentbench
pip install -r requirements.txt   # only needed for --with-claude; the baselines need nothing
```

## Usage

```bash
# Run the two scripted baselines (GreedyBot, NaiveBot) — no API key required
python run_benchmark.py --trials 3

# Also benchmark a real Claude tool-use agent
export ANTHROPIC_API_KEY=sk-...
python run_benchmark.py --trials 3 --with-claude

# Open the dashboard
open outputs/dashboard.html     # or: xdg-open / just double-click it
```

Run the test suite:

```bash
pip install pytest
pytest tests/ -v
```

## Sample results (scripted baselines, 3 trials/combo, 162 trials total)

| Agent | Overall success | `normal` | `flaky_tool` | `dead_tool` | `misleading_instructions` |
|---|---|---|---|---|---|
| **GreedyBot** (competent baseline) | 70.4% | 100% | 100% | 0%¹ | 100% |
| **NaiveBot** (weak baseline) | 25.9% | 75% | 0% | 0% | 33% |

¹ *Neither baseline has a fallback tool, so a hard, permanent outage of the only tool a task needs is
unrecoverable by design — this is intentional and is itself a finding: a benchmarking platform should
surface "no graceful degradation path" as clearly as it surfaces "answer was correct."*

This is exactly the signal a pure accuracy benchmark would miss: both agents "work" under ideal
conditions, but the resilience columns show which one is actually production-ready.

## Technologies used

Python 3.12 · `anthropic` SDK (Claude tool-use) · `pytest` · vanilla HTML/CSS/JS for the dashboard
(deliberately zero frontend build step or CDN dependency, so the deliverable opens anywhere, including
air-gapped judging environments).

## What's next (beyond the hackathon window)

- Provider-agnostic `ClaudeAgent`-style wrappers for OpenAI / local (Ollama) models, so cross-provider
  comparison is a CLI flag, not a rewrite.
- A fifth scenario: concurrent rate-limiting across a batch of trials.
- Persisted historical runs so the dashboard can show drift across a model/prompt version, not just a
  single snapshot.
