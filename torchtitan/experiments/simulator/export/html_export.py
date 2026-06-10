# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from typing import Any

from ..nodes import SimulationResult
from ._shared import _format_bytes, _format_time_us
from .schedule_timing import (
    _event_step,
    _inject_schedule_timing,
    _populate_des_metadata,
)

_JS_PATH = Path(__file__).parent / "trace_visualizer.js"


def _json_script_payload(result: SimulationResult) -> str:
    data = result.to_dict()
    _inject_schedule_timing(data, result)
    _populate_des_metadata(result)
    if "des_engine" in result.metadata:
        data["metadata"]["des_engine"] = result.metadata["des_engine"]
    if "des_memory" in result.metadata:
        data["metadata"]["des_memory"] = result.metadata["des_memory"]
    return escape(json.dumps(data, default=str), quote=False)


def _schedule_events_for_html(result: SimulationResult) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if result.schedule is not None:
        for ev in result.schedule.events:
            d = ev.to_dict()
            d["name"] = d["event_type"]
            events.append(d)
    for ev in result.fsdp_events:
        events.append({**ev, "name": ev.get("event_type", "fsdp")})
    for ev in result.pp_events:
        events.append({**ev, "name": ev.get("event_type", "pp")})
    for ev in result.comm_events:
        events.append(
            {
                **ev,
                "event_type": ev.get("op", "comm"),
                "name": ev.get("op", "comm"),
            }
        )
    return sorted(
        events,
        key=lambda e: (int(e.get("logical_clock", 0)), str(e.get("event_id", ""))),
    )


def _schedule_deps_for_html(result: SimulationResult) -> list[dict[str, Any]]:
    if result.schedule is None:
        return []
    return [dep.to_dict() for dep in result.schedule.deps]


def _render_swimlane_canvas(
    events: list[dict[str, Any]],
    deps: list[dict[str, Any]] | None = None,
    *,
    step: int,
) -> str:
    """Render a Chrome-trace-compatible timeline as an HTML canvas."""
    if not events:
        return '<p class="muted">No schedule events captured.</p>'
    return f"""
    <details open>
      <summary>Chrome Trace Timeline</summary>
      <p class="muted">
        Schedule events rendered in Chrome Trace format.  Each process (pid)
        groups a parallelism strategy; threads (tid) are lanes within that
        strategy.  Open <code>trace.json</code> in <code>chrome://tracing</code>
        for the full interactive viewer.
      </p>
      <div class="chart-toolbar" data-target="chrome-trace-step-{step}">
        <button type="button" data-action="zoom-in">Zoom in</button>
        <button type="button" data-action="zoom-out">Zoom out</button>
        <button type="button" data-action="reset">Reset</button>
        <span class="muted chart-note">Drag or use horizontal scrollbar to pan.</span>
      </div>
      <div class="chart-frame" id="chrome-trace-frame-step-{step}">
        <canvas id="chrome-trace-step-{step}" class="trace-chart chrome-trace-chart" data-step="{step}"></canvas>
      </div>
    </details>
    """


def _short_op_name(name: str, max_len: int = 42) -> str:
    name = name.replace("aten.", "").replace(".default", "")
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def _render_operator_dag_canvas(
    result: SimulationResult,
    phase: str,
    max_nodes: int = 220,
) -> str:
    nodes = [
        n
        for n in result.compute_graph.nodes.values()
        if (n.phase or "unknown") == phase
    ]
    if not nodes:
        return f'<p class="muted">No {escape(phase)} operators captured.</p>'
    truncated = len(nodes) > max_nodes
    note = (
        f'<p class="muted">Showing first {max_nodes} of {len(nodes)} nodes. '
        "Drag or use the horizontal scrollbar to inspect the left-to-right DAG.</p>"
        if truncated
        else '<p class="muted">Drag or use the horizontal scrollbar to inspect the left-to-right DAG.</p>'
    )
    canvas_id = "dag-" + "".join(ch if ch.isalnum() else "-" for ch in phase)
    return f"""
    {note}
    <div class="chart-toolbar" data-target="{escape(canvas_id)}">
      <button type="button" data-action="zoom-in">Zoom in</button>
      <button type="button" data-action="zoom-out">Zoom out</button>
      <button type="button" data-action="reset">Reset</button>
      <span class="muted">Canvas DAG view for {escape(phase)}</span>
    </div>
    <div class="chart-frame" id="{escape(canvas_id)}-frame">
      <canvas id="{escape(canvas_id)}" class="trace-chart dag-chart"
        data-phase="{escape(phase)}" data-max-nodes="{max_nodes}"></canvas>
    </div>
    """


def _render_memory_trace_canvas(result: SimulationResult) -> str:
    if not result.memory_events:
        return '<p class="muted">No memory events captured.</p>'
    lifetimed = sum(
        1
        for event in result.memory_events
        if event.lifetime_start is not None and event.lifetime_end is not None
    )
    return f"""
    <p class="muted">
      Memory trace uses tensor lifetime estimates from producer/consumer order.
      Events without explicit lifetimes, such as parameters, gradients, and
      optimizer state, are rendered as a steady resident baseline.
    </p>
    <div class="chart-toolbar" data-target="memory-trace">
      <button type="button" data-action="zoom-in">Zoom in</button>
      <button type="button" data-action="zoom-out">Zoom out</button>
      <button type="button" data-action="reset">Reset</button>
      <span class="muted chart-note">{lifetimed} lifetimed events, {len(result.memory_events)} total memory events.</span>
    </div>
    <div class="chart-frame" id="memory-trace-frame">
      <canvas id="memory-trace" class="trace-chart memory-chart"></canvas>
    </div>
    <div class="memory-table-wrap">
      <table class="memory-table">
        <thead>
          <tr>
            <th>Event</th>
            <th>Category</th>
            <th>Phase</th>
            <th>Bytes</th>
            <th>Lifetime</th>
            <th>Node</th>
          </tr>
        </thead>
        <tbody id="memory-events-body"></tbody>
      </table>
    </div>
    """


def export_html(
    result: SimulationResult,
    path: str | os.PathLike,
    *,
    title: str = "TorchTitan Simulation Trace",
    max_dag_nodes_per_phase: int = 220,
) -> None:
    """
    Write a self-contained HTML visualization.

    The HTML is intentionally dependency-free: it uses native ``<details>``
    sections for hierarchical drill-down and native HTML5 canvas charts for
    training schedule swimlanes and per-phase operator dependencies. Charts are
    rendered from the embedded JSON payload in the browser so large traces can
    be scrolled and zoomed without generating huge SVG documents.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    schedule_events = _schedule_events_for_html(result)
    schedule_deps = _schedule_deps_for_html(result)
    phases = sorted({n.phase or "unknown" for n in result.compute_graph.nodes.values()})
    if not phases:
        phases = ["unknown"]
    graph_summary = result.compute_graph.summary()
    memory_summary = result.metadata.get("memory", {}) or {}
    peak_memory = memory_summary.get(
        "peak_live_bytes", memory_summary.get("graph_peak_live_bytes", 0)
    )
    cost_summary = result.metadata.get("cost_model", {}) or {}
    perf_grand_total_us = cost_summary.get("e2e_step_time_us", 0)
    data_payload = _json_script_payload(result)
    steps = sorted({_event_step(ev) for ev in schedule_events}) or [0]
    has_des = any(
        n.des_start_time_us is not None for n in result.compute_graph.nodes.values()
    )
    des_cards = ""
    if has_des:
        des_util = result.metadata.get("des_engine", {})
        des_mem = result.metadata.get("des_memory", {})
        des_step = _format_time_us(des_util.get("e2e_step_time_us", 0))
        compute_pct = f"{des_util.get('compute_busy_pct', 0):.1f}%"
        comm_pct = f"{des_util.get('comm_busy_pct', 0):.1f}%"
        overlap_pct = f"{des_util.get('overlap_pct', 0):.1f}%"
        ratio = f"{des_util.get('des_vs_cp_ratio', 0):.3f}x"
        contention = str(des_util.get("contention_count", 0))
        peak_des_mem = _format_bytes(des_mem.get("peak_total_bytes", 0))
        des_cards = f"""
      <div class="card"><div class="num">{escape(des_step)}</div><div>DES step time</div></div>
      <div class="card"><div class="num">{compute_pct}</div><div>Compute utilization</div></div>
      <div class="card"><div class="num">{comm_pct}</div><div>Comm utilization</div></div>
      <div class="card"><div class="num">{overlap_pct}</div><div>Overlap</div></div>
      <div class="card"><div class="num">{ratio}</div><div>DES / Critical Path</div></div>
      <div class="card"><div class="num">{contention}</div><div>Contended ops</div></div>
      <div class="card"><div class="num">{escape(peak_des_mem)}</div><div>Peak DES memory</div></div>"""

    def _phase_sections_for_step(step: int) -> str:
        step_prefix = f"step{step}_"
        step_phases = [phase for phase in phases if phase.startswith(step_prefix)]
        if not step_phases:
            step_phases = phases
        return "\n".join(
            f"""
            <details open>
              <summary>{escape(phase)} operator dependency DAG</summary>
              {_render_operator_dag_canvas(result, phase, max_nodes=max_dag_nodes_per_phase)}
            </details>
            """
            for phase in step_phases
        )

    def _step_section(step: int) -> str:
        step_evs = [ev for ev in schedule_events if _event_step(ev) == step]
        step_ids = {ev.get("event_id") for ev in step_evs}
        step_deps = [
            dep for dep in schedule_deps
            if dep.get("from") in step_ids and dep.get("to") in step_ids
        ]
        swimlane = _render_swimlane_canvas(step_evs, step_deps, step=step)
        return f"""
        <details open>
          <summary>Train step {step}</summary>
          <details open>
            <summary>PP / FSDP2 / TP / DP / communication schedule swimlanes</summary>
            {swimlane}
          </details>
          {_phase_sections_for_step(step)}
        </details>
        """

    step_sections = "\n".join(_step_section(step) for step in steps)
    js_code = _JS_PATH.read_text(encoding="utf-8")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ --bg:#0f172a; --panel:#111827; --text:#e5e7eb; --muted:#94a3b8; --border:#334155; }}
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      background:var(--bg); color:var(--text); }}
    header {{ padding:24px 28px; background:#020617; border-bottom:1px solid var(--border); }}
    main {{ padding:20px 28px 60px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    .muted {{ color:var(--muted); font-size:13px; }}
    .cards {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:16px 0; }}
    .card {{ background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:14px; }}
    .card .num {{ font-size:24px; font-weight:700; }}
    details {{ background:var(--panel); border:1px solid var(--border); border-radius:12px; margin:14px 0; padding:12px; }}
    summary {{ cursor:pointer; font-weight:700; color:#bfdbfe; }}
    .rank-tabs {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0 8px; }}
    .rank-tabs button {{ background:#0f172a; color:var(--text); border:1px solid #475569;
      border-radius:999px; padding:6px 11px; cursor:pointer; }}
    .rank-tabs button.active {{ background:#2563eb; border-color:#60a5fa; }}
    .rank-tabs .rank-note {{ align-self:center; }}
    .chart-toolbar {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:12px 0 8px; }}
    .chart-toolbar button {{ background:#1e293b; color:var(--text); border:1px solid #475569;
      border-radius:8px; padding:6px 10px; cursor:pointer; }}
    .chart-toolbar button:hover {{ background:#334155; }}
    .chart-frame {{ overflow:auto; max-width:100%; position:relative;
      border:1px solid var(--border); border-radius:10px; background:#f8fafc; cursor:grab; }}
    .chart-frame.dragging {{ cursor:grabbing; }}
    .cursor-line {{ position:absolute; top:0; width:0; height:100%; pointer-events:none; z-index:5;
      border-left:1px dashed rgba(255,255,255,0.6); }}
    canvas.trace-chart {{ display:block; background:#f8fafc; }}
    .memory-table-wrap {{ overflow:auto; margin-top:12px; border:1px solid var(--border); border-radius:10px; }}
    .memory-table {{ width:100%; border-collapse:collapse; min-width:760px; background:#020617; }}
    .memory-table th, .memory-table td {{ padding:8px 10px; border-bottom:1px solid #1e293b; text-align:left; font-size:12px; }}
    .memory-table th {{ color:#bfdbfe; background:#0f172a; position:sticky; top:0; }}
    .memory-table td {{ color:#d1d5db; }}
    pre {{ white-space:pre-wrap; color:#d1d5db; background:#020617; padding:12px; border-radius:8px; overflow:auto; }}
  </style>
</head>
<body>
  <div id="tooltip" style="display:none;position:fixed;z-index:1000;background:#1e1e1e;
    color:#e0e0e0;padding:8px 12px;border-radius:6px;font:11px/1.4 monospace;
    max-width:360px;pointer-events:none;box-shadow:0 2px 8px rgba(0,0,0,0.3);"></div>
  <header>
    <h1>{escape(title)}</h1>
    <div class="muted">Hierarchical trace: train step &rarr; parallel schedule swimlanes
      &rarr; forward/backward operator dependency DAGs.</div>
  </header>
  <main>
    <section class="cards">
      <div class="card"><div class="num">{len(result.compute_graph.nodes)}</div><div>Operator nodes</div></div>
      <div class="card"><div class="num">{len(result.compute_graph.edges)}</div><div>Graph edges</div></div>
      <div class="card"><div class="num">{len(schedule_events)}</div><div>Schedule events</div></div>
      <div class="card"><div class="num">{len(result.comm_events)}</div><div>Communication events</div></div>
      <div class="card"><div class="num">{escape(_format_bytes(peak_memory))}</div><div>Estimated live memory peak</div></div>
      <div class="card"><div class="num">{len(result.memory_events)}</div><div>Memory events</div></div>
      <div class="card"><div class="num">{_format_time_us(perf_grand_total_us)}</div><div>Predicted step time</div></div>
      {des_cards}
    </section>
    <details open>
      <summary>Memory trace timeline and event breakdown</summary>
      {_render_memory_trace_canvas(result)}
    </details>
    {step_sections}
    <details>
      <summary>Raw graph summary</summary>
      <pre>{escape(json.dumps(graph_summary, indent=2, default=str))}</pre>
    </details>
    <details open>
      <summary>Memory estimate summary</summary>
      <pre>{escape(json.dumps(memory_summary, indent=2, default=str))}</pre>
    </details>
    <details>
      <summary>Embedded JSON trace payload</summary>
      <pre id="payload"></pre>
    </details>
  </main>
  <script type="application/json" id="trace-data">{data_payload}</script>
  <script>
{js_code}
  </script>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
