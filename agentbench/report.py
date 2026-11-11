from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path


def write_results_json(records, summary, path: str):
    data = {
        "records": [asdict(r) for r in records],
        "summary": summary,
    }
    Path(path).write_text(json.dumps(data, indent=2))
    return data


DASHBOARD_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>AgentBench — Comparison Dashboard</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; padding: 2rem; background: #0b0f14; color: #e6edf3; }
  h1 { font-size: 1.4rem; margin-bottom: .2rem; }
  .sub { color: #8b949e; margin-bottom: 1.6rem; }
  h2 { font-size: 1.05rem; margin-top: 2.2rem; border-bottom: 1px solid #30363d; padding-bottom: .4rem; }
  table { border-collapse: collapse; width: 100%; margin-top: .8rem; font-size: .88rem; }
  th, td { border: 1px solid #30363d; padding: .5rem .7rem; text-align: left; }
  th { background: #161b22; position: sticky; top: 0; }
  tr:nth-child(even) { background: #11161c; }
  .ok { color: #3fb950; font-weight: 600; }
  .bad { color: #f85149; font-weight: 600; }
  .pill { display: inline-block; padding: .1rem .5rem; border-radius: 999px; font-size: .75rem; background: #21262d; }
  .row-click { cursor: pointer; }
  .row-click:hover { outline: 1px solid #58a6ff; }
  #trace { position: fixed; top: 0; right: 0; width: 420px; height: 100%; background: #0d1117;
           border-left: 1px solid #30363d; padding: 1rem; overflow-y: auto; transform: translateX(100%);
           transition: transform .15s ease; font-size: .8rem; }
  #trace.open { transform: translateX(0); }
  #trace pre { white-space: pre-wrap; word-break: break-word; background: #161b22; padding: .6rem; border-radius: 6px; }
  #closeTrace { float: right; cursor: pointer; color: #8b949e; }
</style>
</head>
<body>
<h1>AgentBench — Comparison Dashboard</h1>
<div class="sub">Built for the HackCulture Global AI Hackathon — Team debjitruns · generated locally, no external calls</div>

<h2>Overall comparison</h2>
<table id="overallTable"></table>

<h2>Resilience breakdown (by failure scenario)</h2>
<table id="scenarioTable"></table>

<h2>Per-task results</h2>
<table id="taskTable"></table>

<h2>Execution traces (click any row for the full tool-call trace)</h2>
<table id="traceTable"></table>

<div id="trace"><span id="closeTrace">&times; close</span><div id="traceContent"></div></div>

<script>
const DATA = __DATA_JSON__;

function pct(x){ return (x*100).toFixed(0) + "%"; }
function cls(ok){ return ok ? "ok" : "bad"; }

function renderOverall() {
  const rows = Object.entries(DATA.summary.overall);
  let html = "<tr><th>Agent</th><th>Success rate</th><th>Avg steps</th><th>Avg tool calls</th>" +
             "<th>Avg tokens</th><th>Avg cost (USD)</th><th>Avg latency (s)</th></tr>";
  for (const [agent, m] of rows) {
    html += `<tr><td>${agent}</td><td class="${cls(m.success_rate>0.5)}">${pct(m.success_rate)}</td>` +
            `<td>${m.avg_steps}</td><td>${m.avg_tool_calls}</td><td>${m.avg_tokens}</td>` +
            `<td>$${m.avg_cost_usd}</td><td>${m.avg_latency_s} &plusmn; ${m.std_latency_s}</td></tr>`;
  }
  document.getElementById("overallTable").innerHTML = html;
}

function renderScenario() {
  const rows = Object.entries(DATA.summary.by_scenario);
  let html = "<tr><th>Agent / Scenario</th><th>Success rate</th><th>Trials</th><th>Avg steps</th><th>Avg latency (s)</th></tr>";
  for (const [key, m] of rows) {
    html += `<tr><td><span class="pill">${key}</span></td><td class="${cls(m.success_rate>0.5)}">${pct(m.success_rate)}</td>` +
            `<td>${m.trials}</td><td>${m.avg_steps}</td><td>${m.avg_latency_s}</td></tr>`;
  }
  document.getElementById("scenarioTable").innerHTML = html;
}

function renderTask() {
  const rows = Object.entries(DATA.summary.by_task);
  let html = "<tr><th>Agent / Task</th><th>Success rate</th><th>Avg steps</th><th>Avg tool calls</th></tr>";
  for (const [key, m] of rows) {
    html += `<tr><td>${key}</td><td class="${cls(m.success_rate>0.5)}">${pct(m.success_rate)}</td>` +
            `<td>${m.avg_steps}</td><td>${m.avg_tool_calls}</td></tr>`;
  }
  document.getElementById("taskTable").innerHTML = html;
}

function renderTraces() {
  let html = "<tr><th>Agent</th><th>Task</th><th>Scenario</th><th>Trial</th><th>Result</th></tr>";
  DATA.records.forEach((r, i) => {
    html += `<tr class="row-click" data-i="${i}"><td>${r.agent}</td><td>${r.task_id}</td>` +
            `<td>${r.scenario}</td><td>${r.trial}</td><td class="${cls(r.success)}">${r.success ? "PASS" : "FAIL"}</td></tr>`;
  });
  document.getElementById("traceTable").innerHTML = html;
  document.querySelectorAll("#traceTable tr.row-click").forEach(tr => {
    tr.addEventListener("click", () => openTrace(DATA.records[+tr.dataset.i]));
  });
}

function openTrace(r) {
  const el = document.getElementById("traceContent");
  el.innerHTML = `<h3>${r.agent} — ${r.task_id} (${r.scenario}, trial ${r.trial})</h3>
    <p><b>Result:</b> <span class="${cls(r.success)}">${r.success ? "PASS" : "FAIL"}</span>
    ${r.error ? " — " + r.error : ""}</p>
    <p><b>Steps:</b> ${r.steps} &nbsp; <b>Tool calls:</b> ${r.tool_calls} &nbsp;
    <b>Tokens:</b> ${r.input_tokens + r.output_tokens} &nbsp; <b>Cost:</b> $${r.cost_usd.toFixed(6)}</p>
    <b>Agent decisions</b><pre>${(r.decisions || []).join("\\n") || "(none recorded)"}</pre>
    <b>Raw tool trace</b><pre>${JSON.stringify(r.tool_trace, null, 2)}</pre>`;
  document.getElementById("trace").classList.add("open");
}
document.getElementById("closeTrace").addEventListener("click", () =>
  document.getElementById("trace").classList.remove("open"));

renderOverall(); renderScenario(); renderTask(); renderTraces();
</script>
</body>
</html>
"""


def write_dashboard(data: dict, path: str):
    html = DASHBOARD_TEMPLATE.replace("__DATA_JSON__", json.dumps(data))
    Path(path).write_text(html)
