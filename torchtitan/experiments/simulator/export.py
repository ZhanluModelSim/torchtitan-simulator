# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export utilities: write :class:`SimulationResult` / :class:`ComputeGraph` to
multiple output formats.

Supported formats
-----------------
* **JSON** — full structured dump, loadable back into Python dicts.
* **DOT** — Graphviz dot format with colour-coded nodes by op type.
* **Chrome Trace** — ``chrome://tracing`` compatible JSON for timeline views.
* **HTML** — self-contained interactive visualization with expandable training
  steps, swimlane schedules, and per-phase operator DAGs.
* **Text summary** — human-readable console output with statistics.
"""

from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from typing import Any

from .nodes import ComputeGraph, OpNode, SimulationResult, TrainingSchedule

# ---------------------------------------------------------------------------
# Colour scheme for DOT export (by op_type)
# ---------------------------------------------------------------------------

_DOT_COLORS: dict[str, str] = {
    "compute": "#AED6F1",          # light blue
    "comm_collective": "#F9E79F",  # yellow
    "comm_p2p": "#FAD7A0",         # orange
    "data_move": "#A9DFBF",        # light green
    "memory": "#D7BDE2",           # light purple
    "unknown": "#D5D8DC",          # grey
}


def _node_color(op_type: str) -> str:
    return _DOT_COLORS.get(op_type, _DOT_COLORS["unknown"])


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def export_json(result: SimulationResult, path: str | os.PathLike) -> None:
    """
    Serialize a :class:`SimulationResult` to a JSON file.

    The output is pretty-printed with ``indent=2`` for readability.

    Args:
        result: The simulation result to serialize.
        path: Output file path (will be created / overwritten).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = result.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# DOT export
# ---------------------------------------------------------------------------


def _graph_to_dot(
    graph: ComputeGraph,
    title: str = "ComputeGraph",
    include_shapes: bool = True,
) -> str:
    """Render a :class:`ComputeGraph` as a Graphviz DOT string."""
    lines: list[str] = [
        f'digraph "{title}" {{',
        "  rankdir=TB;",
        '  node [shape=box fontname="Helvetica" fontsize=9];',
    ]

    for node in graph.nodes.values():
        color = _node_color(node.op_type)
        label_parts = [node.op_name]
        if include_shapes and node.outputs:
            shape_strs = [str(o.shape) for o in node.outputs[:2]]
            label_parts.append("out: " + ", ".join(shape_strs))
        if node.comm_op:
            label_parts.append(f"[{node.comm_op}]")
        label = "\\n".join(label_parts)
        node_id_safe = node.node_id.replace("-", "_")
        lines.append(
            f'  {node_id_safe} [label="{label}" fillcolor="{color}" style=filled'
            f' tooltip="{node.op_type}"];'
        )

    for edge in graph.edges:
        src = edge.src_node_id.replace("-", "_")
        dst = edge.dst_node_id.replace("-", "_")
        style = "dashed" if edge.edge_type in ("comm_dep", "sequential") else "solid"
        lines.append(f'  {src} -> {dst} [style={style}];')

    lines.append("}")
    return "\n".join(lines)


def export_dot(
    graph: ComputeGraph,
    path: str | os.PathLike,
    title: str = "ComputeGraph",
    include_shapes: bool = True,
) -> None:
    """
    Write a :class:`ComputeGraph` as a Graphviz DOT file.

    Nodes are colour-coded by op type:
    - Blue: compute
    - Yellow: collective comms
    - Orange: P2P comms
    - Green: data movement
    - Purple: memory alloc
    - Grey: unknown

    Args:
        graph: The graph to export.
        path: Output ``.dot`` file path.
        title: Graph title embedded in the DOT file.
        include_shapes: Whether to annotate nodes with output tensor shapes.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dot = _graph_to_dot(graph, title=title, include_shapes=include_shapes)
    with open(path, "w", encoding="utf-8") as f:
        f.write(dot)


# ---------------------------------------------------------------------------
# Chrome trace export
# ---------------------------------------------------------------------------


def _op_to_chrome_event(
    node: OpNode,
    pid: int = 0,
    tid: int = 0,
    ts_us: float = 0.0,
    dur_us: float = 1.0,
) -> dict[str, Any]:
    return {
        "ph": "X",
        "pid": pid,
        "tid": tid,
        "ts": ts_us,
        "dur": dur_us,
        "name": node.op_name,
        "cat": node.op_type,
        "args": {
            "node_id": node.node_id,
            "phase": node.phase,
            "pp_stage": node.pp_stage,
            "microbatch": node.microbatch_idx,
            "outputs": [str(o.shape) for o in node.outputs],
            "comm_op": node.comm_op,
        },
    }


def export_chrome_trace(
    result: SimulationResult,
    path: str | os.PathLike,
    us_per_op: float = 1.0,
) -> None:
    """
    Write a ``chrome://tracing``-compatible JSON trace file.

    Each op becomes a duration event (``"ph": "X"``).  Events are laid out
    sequentially per phase on separate *threads* (tid).  The timeline is
    purely logical (not wall-clock time): each event is assigned a
    *us_per_op* microsecond slot.

    Args:
        result: The simulation result to render.
        path: Output JSON file path.
        us_per_op: Duration in microseconds to assign each op slot.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    phase_order = ["forward", "backward", "optimizer", "unknown", "joint"]
    phase_tid: dict[str, int] = {}
    tid_counter = [0]

    def _get_tid(phase: str) -> int:
        if phase not in phase_tid:
            phase_tid[phase] = tid_counter[0]
            tid_counter[0] += 1
        return phase_tid[phase]

    phase_ts: dict[str, float] = {}

    events: list[dict[str, Any]] = []
    for node in result.compute_graph.nodes.values():
        phase = node.phase or "unknown"
        tid = _get_tid(phase)
        ts = phase_ts.get(phase, 0.0)
        events.append(_op_to_chrome_event(node, pid=0, tid=tid, ts_us=ts, dur_us=us_per_op))
        phase_ts[phase] = ts + us_per_op

    # Add FSDP events as a separate process
    for ev in result.fsdp_events:
        phase = ev.get("phase", "unknown")
        tid = _get_tid(f"fsdp_{phase}")
        ts = phase_ts.get(f"fsdp_{phase}", 0.0)
        events.append(
            {
                "ph": "X",
                "pid": 1,
                "tid": _get_tid(f"fsdp_{phase}"),
                "ts": ts,
                "dur": us_per_op,
                "name": ev.get("event_type", "fsdp_event"),
                "cat": "fsdp",
                "args": ev,
            }
        )
        phase_ts[f"fsdp_{phase}"] = ts + us_per_op

    # Metadata events
    for phase, tid in phase_tid.items():
        events.append(
            {
                "ph": "M",
                "pid": 0,
                "tid": tid,
                "name": "thread_name",
                "args": {"name": phase},
            }
        )

    trace = {"traceEvents": events, "displayTimeUnit": "us"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------


def _json_script_payload(result: SimulationResult) -> str:
    return escape(json.dumps(result.to_dict(), default=str), quote=False)


def _format_bytes(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "n/a"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _event_lane(ev: dict[str, Any]) -> str:
    event_type = str(ev.get("event_type", ""))
    metadata = ev.get("metadata", {}) or {}
    if event_type.startswith("pp_") or ev.get("pp_stage") is not None or ev.get("pp_rank") is not None:
        pp_rank = ev.get("pp_rank", ev.get("pp_stage", ev.get("rank", 0)))
        pp_stage = ev.get("pp_stage")
        if pp_stage is not None and pp_stage != pp_rank:
            return f"PP rank {pp_rank} / stage {pp_stage}"
        return f"PP rank {pp_rank}"
    strategy = metadata.get("strategy")
    if strategy:
        return f"{strategy.upper()} rank {ev.get('rank', 0)}"
    if ev.get("event_type", "").startswith("fsdp_"):
        return f"FSDP rank {ev.get('rank', 0)}"
    if event_type.startswith("tp_"):
        return f"TP rank {ev.get('rank', 0)}"
    if event_type.startswith("dp_"):
        return f"DP rank {ev.get('rank', 0)}"
    if ev.get("op"):
        return f"Comm rank {ev.get('rank', 0)}"
    return f"Rank {ev.get('rank', 0)}"


def _event_step(ev: dict[str, Any]) -> int:
    metadata = ev.get("metadata", {}) or {}
    try:
        return int(metadata.get("step", ev.get("step", 0)))
    except (TypeError, ValueError):
        return 0


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
    return sorted(events, key=lambda e: (int(e.get("logical_clock", 0)), str(e.get("event_id", ""))))


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
    if not events:
        return '<p class="muted">No schedule events captured.</p>'
    dep_count = len(deps or [])
    return f"""
    <div class="rank-tabs" data-target="schedule-step-{step}"></div>
    <div class="chart-toolbar" data-target="schedule-step-{step}">
      <button type="button" data-action="zoom-in">Zoom in</button>
      <button type="button" data-action="zoom-out">Zoom out</button>
      <button type="button" data-action="reset">Reset</button>
      <span class="muted chart-note">{len(events)} events, {dep_count} explicit deps. Drag or use the horizontal scrollbar to pan.</span>
    </div>
    <div class="chart-frame">
      <canvas id="schedule-step-{step}" class="trace-chart schedule-chart" data-step="{step}"></canvas>
    </div>
    """


def _short_op_name(name: str, max_len: int = 42) -> str:
    name = name.replace("aten.", "").replace(".default", "")
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def _render_operator_dag_canvas(
    result: SimulationResult,
    phase: str,
    max_nodes: int = 220,
) -> str:
    nodes = [n for n in result.compute_graph.nodes.values() if (n.phase or "unknown") == phase]
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
    <div class="chart-frame">
      <canvas id="{escape(canvas_id)}" class="trace-chart dag-chart" data-phase="{escape(phase)}" data-max-nodes="{max_nodes}"></canvas>
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
    <div class="chart-frame">
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
    data_payload = _json_script_payload(result)
    steps = sorted({_event_step(ev) for ev in schedule_events}) or [0]

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

    step_sections = "\n".join(
        f"""
        <details open>
          <summary>Train step {step}</summary>
          <details open>
            <summary>PP / FSDP2 / TP / DP / communication schedule swimlanes</summary>
            {_render_swimlane_canvas(
                [ev for ev in schedule_events if _event_step(ev) == step],
                [dep for dep in schedule_deps if dep.get("from") in {ev.get("event_id") for ev in schedule_events if _event_step(ev) == step} and dep.get("to") in {ev.get("event_id") for ev in schedule_events if _event_step(ev) == step}],
                step=step,
            )}
          </details>
          {_phase_sections_for_step(step)}
        </details>
        """
        for step in steps
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ --bg:#0f172a; --panel:#111827; --text:#e5e7eb; --muted:#94a3b8; --border:#334155; }}
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background:var(--bg); color:var(--text); }}
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
    .rank-tabs button {{ background:#0f172a; color:var(--text); border:1px solid #475569; border-radius:999px; padding:6px 11px; cursor:pointer; }}
    .rank-tabs button.active {{ background:#2563eb; border-color:#60a5fa; }}
    .rank-tabs .rank-note {{ align-self:center; }}
    .chart-toolbar {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:12px 0 8px; }}
    .chart-toolbar button {{ background:#1e293b; color:var(--text); border:1px solid #475569; border-radius:8px; padding:6px 10px; cursor:pointer; }}
    .chart-toolbar button:hover {{ background:#334155; }}
    .chart-frame {{ overflow:auto; max-width:100%; border:1px solid var(--border); border-radius:10px; background:#f8fafc; cursor:grab; }}
    .chart-frame.dragging {{ cursor:grabbing; }}
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
  <header>
    <h1>{escape(title)}</h1>
    <div class="muted">Hierarchical trace: train step → parallel schedule swimlanes → forward/backward operator dependency DAGs.</div>
  </header>
  <main>
    <section class="cards">
      <div class="card"><div class="num">{len(result.compute_graph.nodes)}</div><div>Operator nodes</div></div>
      <div class="card"><div class="num">{len(result.compute_graph.edges)}</div><div>Graph edges</div></div>
      <div class="card"><div class="num">{len(schedule_events)}</div><div>Schedule events</div></div>
      <div class="card"><div class="num">{len(result.comm_events)}</div><div>Communication events</div></div>
      <div class="card"><div class="num">{escape(_format_bytes(peak_memory))}</div><div>Estimated live memory peak</div></div>
      <div class="card"><div class="num">{len(result.memory_events)}</div><div>Memory events</div></div>
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
    const payload = document.getElementById('trace-data').textContent;
    const TRACE = JSON.parse(payload);
    document.getElementById('payload').textContent = JSON.stringify(TRACE, null, 2);

    const chartState = new WeakMap();
    const palette = {{
      compute: '#aed6f1',
      comm_collective: '#f9e79f',
      comm_p2p: '#fad7a0',
      data_move: '#a9dfbf',
      memory: '#d7bde2',
      unknown: '#d5d8dc',
      fwd: '#93c5fd',
      bwd: '#fca5a5',
      comm: '#fde68a',
      fsdp: '#c4b5fd',
      edge: '#94a3b8',
      explicit: '#dc2626',
    }};
    const memoryPalette = {{
      activation: '#60a5fa',
      allocation: '#c084fc',
      comm_buffer: '#f59e0b',
      comm_event_buffer: '#fbbf24',
      data_move: '#34d399',
      parameter: '#22c55e',
      gradient: '#fb7185',
      optimizer_state: '#a78bfa',
      unknown: '#94a3b8',
    }};

    function formatBytes(bytes) {{
      if (bytes === null || bytes === undefined || Number.isNaN(Number(bytes))) return 'n/a';
      let value = Number(bytes);
      const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
      for (const unit of units) {{
        if (Math.abs(value) < 1024 || unit === 'TiB') {{
          return unit === 'B' ? Math.round(value) + ' B' : value.toFixed(1) + ' ' + unit;
        }}
        value /= 1024;
      }}
      return value.toFixed(1) + ' TiB';
    }}

    function shortName(name, maxLen = 42) {{
      const cleaned = String(name || '').replace('aten.', '').replace('.default', '');
      return cleaned.length <= maxLen ? cleaned : cleaned.slice(0, maxLen - 1) + '…';
    }}

    function eventStep(ev) {{
      const metadata = ev.metadata || {{}};
      const value = metadata.step ?? ev.step ?? 0;
      const parsed = Number.parseInt(value, 10);
      return Number.isFinite(parsed) ? parsed : 0;
    }}

    function eventLane(ev) {{
      const eventType = String(ev.event_type || '');
      const metadata = ev.metadata || {{}};
      if (eventType.startsWith('pp_') || ev.pp_stage !== null && ev.pp_stage !== undefined || ev.pp_rank !== null && ev.pp_rank !== undefined) {{
        const ppRank = ev.pp_rank ?? ev.pp_stage ?? ev.rank ?? 0;
        const ppStage = ev.pp_stage;
        if (ppStage !== null && ppStage !== undefined && Number(ppStage) !== Number(ppRank)) return 'PP rank ' + ppRank + ' / stage ' + ppStage;
        return 'PP rank ' + ppRank;
      }}
      if (metadata.strategy) return String(metadata.strategy).toUpperCase() + ' rank ' + (ev.rank ?? 0);
      if (eventType.startsWith('fsdp_')) return 'FSDP rank ' + (ev.rank ?? 0);
      if (eventType.startsWith('tp_')) return 'TP rank ' + (ev.rank ?? 0);
      if (eventType.startsWith('dp_')) return 'DP rank ' + (ev.rank ?? 0);
      if (ev.op) return 'Comm rank ' + (ev.rank ?? 0);
      return 'Rank ' + (ev.rank ?? 0);
    }}

    function eventStrategy(ev) {{
      const metadata = ev.metadata || {{}};
      const eventType = String(ev.event_type || '').toLowerCase();
      if (metadata.strategy) return String(metadata.strategy).toLowerCase();
      if (eventType.startsWith('fsdp_')) return 'fsdp';
      if (eventType.startsWith('tp_')) return 'tp';
      if (eventType.startsWith('dp_')) return 'dp';
      if (eventType.startsWith('pp_') || ev.pp_stage !== null && ev.pp_stage !== undefined) return 'pp';
      return null;
    }}

    function scheduleRankViews(events) {{
      const views = [{{key: 'all', label: 'All ranks', kind: 'all'}}];
      const seen = new Set(['all']);
      function add(view) {{
        if (seen.has(view.key)) return;
        seen.add(view.key);
        views.push(view);
      }}
      const ranks = Array.from(new Set(events.map((ev) => ev.rank).filter((rank) => rank !== null && rank !== undefined))).sort((a, b) => Number(a) - Number(b));
      for (const rank of ranks) add({{key: 'global:' + rank, label: 'Global rank ' + rank, kind: 'global', rank}});
      const ppStages = Array.from(new Set(events.map((ev) => ev.pp_stage).filter((rank) => rank !== null && rank !== undefined))).sort((a, b) => Number(a) - Number(b));
      for (const stage of ppStages) add({{key: 'pp-stage:' + stage, label: 'PP stage ' + stage, kind: 'pp-stage', stage}});
      const ppRanks = Array.from(new Set(events.map((ev) => ev.pp_rank).filter((rank) => rank !== null && rank !== undefined))).sort((a, b) => Number(a) - Number(b));
      for (const rank of ppRanks) add({{key: 'pp-rank:' + rank, label: 'PP rank ' + rank, kind: 'pp-rank', ppRank: rank}});
      for (const strategy of ['tp', 'dp', 'fsdp', 'fsdp2']) {{
        const strategyRanks = Array.from(new Set(events.filter((ev) => eventStrategy(ev) === strategy).map((ev) => ev.rank).filter((rank) => rank !== null && rank !== undefined))).sort((a, b) => Number(a) - Number(b));
        const labelPrefix = strategy.toUpperCase();
        for (const rank of strategyRanks) add({{key: 'strategy:' + strategy + ':' + rank, label: labelPrefix + ' rank ' + rank, kind: 'strategy', strategy, rank}});
      }}
      return views;
    }}

    function rankViewMatches(ev, view) {{
      if (!view || view.kind === 'all') return true;
      if (view.kind === 'global') return Number(ev.rank) === Number(view.rank);
      if (view.kind === 'pp-stage') return Number(ev.pp_stage) === Number(view.stage);
      if (view.kind === 'pp-rank') return Number(ev.pp_rank ?? ev.pp_stage) === Number(view.ppRank);
      if (view.kind === 'strategy') return eventStrategy(ev) === view.strategy && Number(ev.rank) === Number(view.rank);
      return true;
    }}

    function scheduleEvents() {{
      const events = [];
      for (const ev of TRACE.schedule?.events || []) events.push({{...ev, name: ev.event_type}});
      for (const ev of TRACE.fsdp_events || []) events.push({{...ev, name: ev.event_type || 'fsdp'}});
      for (const ev of TRACE.pp_events || []) events.push({{...ev, name: ev.event_type || 'pp'}});
      for (const ev of TRACE.comm_events || []) events.push({{...ev, event_type: ev.op || 'comm', name: ev.op || 'comm'}});
      return events.sort((a, b) => (Number(a.logical_clock || 0) - Number(b.logical_clock || 0)) || String(a.event_id || '').localeCompare(String(b.event_id || '')));
    }}

    function renderRankTabs(canvas) {{
      const step = Number.parseInt(canvas.dataset.step || '0', 10);
      const events = scheduleEvents().filter((ev) => eventStep(ev) === step);
      const views = scheduleRankViews(events);
      const tabs = document.querySelector('.rank-tabs[data-target="' + canvas.id + '"]');
      if (!tabs) return;
      const state = chartState.get(canvas) || {{zoom: 1, rankView: 'all'}};
      if (!views.some((view) => view.key === state.rankView)) state.rankView = 'all';
      chartState.set(canvas, state);
      tabs.textContent = '';
      for (const view of views) {{
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = view.label;
        button.dataset.rankView = view.key;
        if (view.key === state.rankView) button.classList.add('active');
        button.addEventListener('click', () => {{
          const current = chartState.get(canvas) || {{zoom: 1}};
          current.rankView = view.key;
          chartState.set(canvas, current);
          renderRankTabs(canvas);
          redraw(canvas);
        }});
        tabs.appendChild(button);
      }}
      const globalRanks = new Set(events.map((ev) => ev.rank).filter((rank) => rank !== null && rank !== undefined));
      if (globalRanks.size <= 1) {{
        const onlyRank = Array.from(globalRanks)[0] ?? 0;
        const note = document.createElement('span');
        note.className = 'muted rank-note';
        note.textContent = 'Only local rank ' + onlyRank + ' is present in this trace; more rank tabs appear when multi-rank traces are captured or aggregated.';
        tabs.appendChild(note);
      }}
    }}

    function resizeCanvas(canvas, width, height) {{
      const dpr = window.devicePixelRatio || 1;
      canvas.style.width = width + 'px';
      canvas.style.height = height + 'px';
      canvas.width = Math.ceil(width * dpr);
      canvas.height = Math.ceil(height * dpr);
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return ctx;
    }}

    function roundedRect(ctx, x, y, w, h, r) {{
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }}

    function arrowLine(ctx, sx, sy, dx, dy, color, dashed = false, width = 1.2) {{
      ctx.save();
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = width;
      ctx.globalAlpha = 0.8;
      if (dashed) ctx.setLineDash([5, 4]);
      ctx.beginPath();
      const mid = Math.max(30, Math.abs(dx - sx) / 2);
      ctx.moveTo(sx, sy);
      ctx.bezierCurveTo(sx + mid, sy, dx - mid, dy, dx, dy);
      ctx.stroke();
      const angle = Math.atan2(dy - sy, dx - sx);
      ctx.beginPath();
      ctx.moveTo(dx, dy);
      ctx.lineTo(dx - 8 * Math.cos(angle - Math.PI / 6), dy - 8 * Math.sin(angle - Math.PI / 6));
      ctx.lineTo(dx - 8 * Math.cos(angle + Math.PI / 6), dy - 8 * Math.sin(angle + Math.PI / 6));
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }}

    function depColor(depType) {{
      if (depType === 'pp_comm') return '#ea580c';
      if (depType === 'fsdp_comm') return '#7c3aed';
      if (depType === 'tp_comm') return '#0891b2';
      if (depType === 'dp_comm') return '#16a34a';
      if (depType === 'control') return '#475569';
      return palette.explicit;
    }}

    function drawSchedule(canvas) {{
      const state = chartState.get(canvas) || {{zoom: 1, rankView: 'all'}};
      chartState.set(canvas, state);
      const step = Number.parseInt(canvas.dataset.step || '0', 10);
      const allEvents = scheduleEvents().filter((ev) => eventStep(ev) === step);
      const views = scheduleRankViews(allEvents);
      const selectedView = views.find((view) => view.key === state.rankView) || views[0];
      const events = allEvents.filter((ev) => rankViewMatches(ev, selectedView));
      const eventIds = new Set(events.map((ev) => ev.event_id));
      const deps = (TRACE.schedule?.deps || []).filter((dep) => eventIds.has(dep.from) && eventIds.has(dep.to));
      const lanes = Array.from(new Set(events.map(eventLane))).sort();
      const maxClock = Math.max(0, ...events.map((ev) => Number(ev.logical_clock || 0)));
      const scale = 58 * state.zoom;
      const width = Math.max(980, 220 + (maxClock + 3) * scale);
      const height = Math.max(160, 80 + lanes.length * 64);
      const ctx = resizeCanvas(canvas, width, height);
      ctx.clearRect(0, 0, width, height);
      ctx.font = '13px ui-sans-serif, system-ui, sans-serif';
      ctx.textBaseline = 'middle';
      const note = document.querySelector('.chart-toolbar[data-target="' + canvas.id + '"] .chart-note');
      if (note) note.textContent = events.length + ' visible events, ' + deps.length + ' visible deps. Current view: ' + selectedView.label + '. Drag or use the horizontal scrollbar to pan.';

      const laneY = new Map();
      lanes.forEach((lane, idx) => {{
        const y = 52 + idx * 64;
        laneY.set(lane, y);
        ctx.fillStyle = '#0f172a';
        ctx.font = '700 13px ui-sans-serif, system-ui, sans-serif';
        ctx.fillText(lane, 14, y);
        ctx.strokeStyle = '#cbd5e1';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(160, y);
        ctx.lineTo(width - 30, y);
        ctx.stroke();
      }});

      const positions = new Map();
      const prevByLane = new Map();
      for (const ev of events) {{
        const lane = eventLane(ev);
        const clock = Number(ev.logical_clock || 0);
        const x = 180 + clock * scale;
        const y = laneY.get(lane);
        positions.set(ev.event_id, {{x, y}});
        if (prevByLane.has(lane)) {{
          const prev = prevByLane.get(lane);
          arrowLine(ctx, prev.x + 10, prev.y, x - 12, y, '#64748b', true, 1);
        }}
        prevByLane.set(lane, {{x, y}});
      }}

      for (const dep of deps) {{
        const src = positions.get(dep.from);
        const dst = positions.get(dep.to);
        if (!src || !dst) continue;
        arrowLine(ctx, src.x + 12, src.y, dst.x - 14, dst.y, depColor(dep.type || 'data'), dep.type === 'control', 1.8);
      }}

      for (const ev of events) {{
        const pos = positions.get(ev.event_id);
        const name = String(ev.name || ev.event_type || 'event');
        const lower = name.toLowerCase();
        let fill = palette.fwd;
        if (lower.includes('fsdp')) fill = palette.fsdp;
        else if (lower.includes('send') || lower.includes('recv') || lower.includes('all_') || lower.includes('scatter')) fill = palette.comm;
        else if (lower.includes('bwd') || lower.includes('backward')) fill = palette.bwd;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 10, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.strokeStyle = '#0f172a';
        ctx.stroke();
        ctx.fillStyle = '#1f2937';
        ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
        ctx.fillText(shortName(name, 30), pos.x + 14, pos.y + 1);
      }}
    }}

    function drawDag(canvas) {{
      const state = chartState.get(canvas) || {{zoom: 1}};
      chartState.set(canvas, state);
      const phase = canvas.dataset.phase || 'unknown';
      const maxNodes = Number.parseInt(canvas.dataset.maxNodes || '220', 10);
      const allNodes = TRACE.compute_graph?.nodes || [];
      const phaseNodes = allNodes.filter((node) => (node.phase || 'unknown') === phase).slice(0, maxNodes);
      const nodeIds = new Set(phaseNodes.map((node) => node.node_id));
      const edges = (TRACE.compute_graph?.edges || []).filter((edge) => nodeIds.has(edge.src) && nodeIds.has(edge.dst));

      const preds = new Map(phaseNodes.map((node) => [node.node_id, []]));
      const succs = new Map(phaseNodes.map((node) => [node.node_id, []]));
      const indeg = new Map(phaseNodes.map((node) => [node.node_id, 0]));
      for (const edge of edges) {{
        preds.get(edge.dst)?.push(edge.src);
        succs.get(edge.src)?.push(edge.dst);
        indeg.set(edge.dst, (indeg.get(edge.dst) || 0) + 1);
      }}
      const queue = phaseNodes.filter((node) => (indeg.get(node.node_id) || 0) === 0).map((node) => node.node_id);
      const depth = new Map(phaseNodes.map((node) => [node.node_id, 0]));
      for (let i = 0; i < queue.length; i++) {{
        const id = queue[i];
        for (const dst of succs.get(id) || []) {{
          depth.set(dst, Math.max(depth.get(dst) || 0, (depth.get(id) || 0) + 1));
          indeg.set(dst, (indeg.get(dst) || 0) - 1);
          if (indeg.get(dst) === 0) queue.push(dst);
        }}
      }}
      phaseNodes.forEach((node, idx) => {{
        if (!queue.includes(node.node_id)) depth.set(node.node_id, Math.floor(idx / 8));
      }});

      const byDepth = new Map();
      for (const node of phaseNodes) {{
        const d = depth.get(node.node_id) || 0;
        if (!byDepth.has(d)) byDepth.set(d, []);
        byDepth.get(d).push(node);
      }}
      const maxDepth = Math.max(0, ...Array.from(byDepth.keys()));
      const maxLayer = Math.max(1, ...Array.from(byDepth.values()).map((items) => items.length));
      const colW = 250 * state.zoom;
      const rowH = 78;
      const width = Math.max(1100, 80 + (maxDepth + 1) * colW);
      const height = Math.max(180, 80 + maxLayer * rowH);
      const ctx = resizeCanvas(canvas, width, height);
      ctx.clearRect(0, 0, width, height);

      const positions = new Map();
      for (const [d, items] of byDepth.entries()) {{
        items.forEach((node, idx) => {{
          positions.set(node.node_id, {{x: 32 + d * colW, y: 36 + idx * rowH}});
        }});
      }}

      for (const edge of edges) {{
        const src = positions.get(edge.src);
        const dst = positions.get(edge.dst);
        if (!src || !dst) continue;
        arrowLine(ctx, src.x + 200, src.y + 26, dst.x - 6, dst.y + 26, edge.type === 'data' ? '#2563eb' : '#64748b', edge.type === 'control', 1);
      }}

      for (const node of phaseNodes) {{
        const pos = positions.get(node.node_id);
        const fill = palette[node.op_type] || palette.unknown;
        roundedRect(ctx, pos.x, pos.y, 205, 54, 8);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = '#0f172a';
        ctx.font = '700 12px ui-sans-serif, system-ui, sans-serif';
        ctx.fillText(shortName(node.op_name, 30), pos.x + 8, pos.y + 19);
        const shape = (node.outputs || []).slice(0, 1).map((t) => '[' + (t.shape || []).join(',') + ']').join(', ');
        ctx.fillStyle = '#334155';
        ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
        ctx.fillText(shortName((node.op_type || 'unknown') + ' ' + shape, 34), pos.x + 8, pos.y + 39);
      }}
    }}

    function memoryEvents() {{
      return TRACE.memory_events || [];
    }}

    function memoryCategoryColor(category) {{
      return memoryPalette[category] || memoryPalette.unknown;
    }}

    function memoryLifetimeLabel(event) {{
      const start = event.lifetime_start;
      const end = event.lifetime_end;
      if (start === null || start === undefined || end === null || end === undefined) return 'resident';
      return String(start) + ' → ' + String(end);
    }}

    function memoryCategories(events) {{
      const preferred = [
        'parameter',
        'optimizer_state',
        'gradient',
        'activation',
        'allocation',
        'data_move',
        'comm_buffer',
        'comm_event_buffer',
      ];
      const present = new Set(events.map((event) => event.category || 'unknown'));
      const ordered = preferred.filter((category) => present.has(category));
      for (const category of Array.from(present).sort()) {{
        if (!ordered.includes(category)) ordered.push(category);
      }}
      return ordered;
    }}

    function buildMemorySamples(events) {{
      const lifetimed = events.filter((event) =>
        event.lifetime_start !== null && event.lifetime_start !== undefined &&
        event.lifetime_end !== null && event.lifetime_end !== undefined
      );
      const resident = events.filter((event) =>
        event.lifetime_start === null || event.lifetime_start === undefined ||
        event.lifetime_end === null || event.lifetime_end === undefined
      );
      const categories = memoryCategories(events);
      const maxIndex = Math.max(0, ...lifetimed.map((event) => Number(event.lifetime_end || 0)));
      const residentByCategory = new Map(categories.map((category) => [category, 0]));
      for (const event of resident) {{
        const category = event.category || 'unknown';
        residentByCategory.set(category, (residentByCategory.get(category) || 0) + Number(event.bytes || 0));
      }}

      const samples = [];
      let peak = 0;
      for (let idx = 0; idx <= maxIndex; idx++) {{
        const byCategory = new Map(residentByCategory);
        for (const event of lifetimed) {{
          const start = Number(event.lifetime_start);
          const end = Number(event.lifetime_end);
          if (idx < start || idx > end) continue;
          const category = event.category || 'unknown';
          byCategory.set(category, (byCategory.get(category) || 0) + Number(event.bytes || 0));
        }}
        const total = Array.from(byCategory.values()).reduce((acc, value) => acc + value, 0);
        peak = Math.max(peak, total);
        samples.push({{idx, byCategory, total}});
      }}
      return {{samples, categories, peak, residentTotal: Array.from(residentByCategory.values()).reduce((acc, value) => acc + value, 0)}};
    }}

    function drawMemoryTrace(canvas) {{
      const state = chartState.get(canvas) || {{zoom: 1}};
      chartState.set(canvas, state);
      const events = memoryEvents();
      const {{samples, categories, peak, residentTotal}} = buildMemorySamples(events);
      const scale = 14 * state.zoom;
      const plotLeft = 90;
      const plotTop = 38;
      const plotHeight = 250;
      const legendTop = plotTop + plotHeight + 42;
      const width = Math.max(980, plotLeft + 90 + Math.max(1, samples.length) * scale);
      const height = 390;
      const ctx = resizeCanvas(canvas, width, height);
      ctx.clearRect(0, 0, width, height);
      ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
      ctx.textBaseline = 'middle';

      const note = document.querySelector('.chart-toolbar[data-target="' + canvas.id + '"] .chart-note');
      if (note) {{
        const lifetimedCount = events.filter((event) =>
          event.lifetime_start !== null && event.lifetime_start !== undefined &&
          event.lifetime_end !== null && event.lifetime_end !== undefined
        ).length;
        note.textContent = lifetimedCount + ' lifetimed events, ' + events.length +
          ' total. Estimated total peak including resident baseline: ' + formatBytes(peak) +
          ' (resident baseline ' + formatBytes(residentTotal) + ').';
      }}

      ctx.fillStyle = '#0f172a';
      ctx.font = '700 13px ui-sans-serif, system-ui, sans-serif';
      ctx.fillText('Estimated live memory by operator order', plotLeft, 18);
      ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
      ctx.fillStyle = '#475569';
      ctx.fillText('x: graph node order, y: bytes. Resident model-state estimates are drawn as a baseline.', plotLeft, 34);

      ctx.strokeStyle = '#94a3b8';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plotLeft, plotTop);
      ctx.lineTo(plotLeft, plotTop + plotHeight);
      ctx.lineTo(width - 36, plotTop + plotHeight);
      ctx.stroke();

      const safePeak = Math.max(1, peak);
      for (let tick = 0; tick <= 4; tick++) {{
        const value = safePeak * tick / 4;
        const y = plotTop + plotHeight - (value / safePeak) * plotHeight;
        ctx.strokeStyle = tick === 0 ? '#94a3b8' : '#e2e8f0';
        ctx.beginPath();
        ctx.moveTo(plotLeft, y);
        ctx.lineTo(width - 36, y);
        ctx.stroke();
        ctx.fillStyle = '#334155';
        ctx.fillText(formatBytes(value), 8, y);
      }}

      for (const sample of samples) {{
        let yTop = plotTop + plotHeight;
        const x = plotLeft + sample.idx * scale;
        const barW = Math.max(2, scale - 1);
        for (const category of categories) {{
          const bytes = sample.byCategory.get(category) || 0;
          if (bytes <= 0) continue;
          const h = bytes / safePeak * plotHeight;
          yTop -= h;
          ctx.fillStyle = memoryCategoryColor(category);
          ctx.fillRect(x, yTop, barW, h);
        }}
      }}

      ctx.fillStyle = '#334155';
      ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
      const maxIdx = Math.max(0, samples.length - 1);
      for (const idx of [0, Math.floor(maxIdx / 2), maxIdx]) {{
        const x = plotLeft + idx * scale;
        ctx.fillText(String(idx), x, plotTop + plotHeight + 18);
      }}

      let legendX = plotLeft;
      let legendY = legendTop;
      for (const category of categories) {{
        if (legendX > width - 220) {{
          legendX = plotLeft;
          legendY += 22;
        }}
        ctx.fillStyle = memoryCategoryColor(category);
        ctx.fillRect(legendX, legendY - 6, 12, 12);
        ctx.fillStyle = '#0f172a';
        ctx.fillText(category, legendX + 18, legendY);
        legendX += 150;
      }}
    }}

    function populateMemoryTable() {{
      const body = document.getElementById('memory-events-body');
      if (!body) return;
      body.textContent = '';
      const events = memoryEvents()
        .slice()
        .sort((a, b) => Number(b.bytes || 0) - Number(a.bytes || 0))
        .slice(0, 120);
      for (const event of events) {{
        const row = document.createElement('tr');
        const values = [
          event.event_id || '',
          event.category || 'unknown',
          event.phase || 'unknown',
          formatBytes(event.bytes || 0),
          memoryLifetimeLabel(event),
          event.node_id || '',
        ];
        for (const value of values) {{
          const cell = document.createElement('td');
          cell.textContent = value;
          row.appendChild(cell);
        }}
        body.appendChild(row);
      }}
    }}

    function redraw(canvas) {{
      if (canvas.classList.contains('schedule-chart')) drawSchedule(canvas);
      else if (canvas.classList.contains('memory-chart')) drawMemoryTrace(canvas);
      else drawDag(canvas);
    }}

    function installChart(canvas) {{
      chartState.set(canvas, {{zoom: 1, rankView: 'all'}});
      const frame = canvas.closest('.chart-frame');
      let dragging = false;
      let startX = 0;
      let startScroll = 0;
      frame.addEventListener('mousedown', (event) => {{
        dragging = true;
        startX = event.clientX;
        startScroll = frame.scrollLeft;
        frame.classList.add('dragging');
      }});
      window.addEventListener('mouseup', () => {{
        dragging = false;
        frame.classList.remove('dragging');
      }});
      window.addEventListener('mousemove', (event) => {{
        if (!dragging) return;
        frame.scrollLeft = startScroll - (event.clientX - startX);
      }});
      frame.addEventListener('wheel', (event) => {{
        if (!event.ctrlKey && Math.abs(event.deltaX) < Math.abs(event.deltaY)) return;
        event.preventDefault();
        frame.scrollLeft += event.deltaX || event.deltaY;
      }}, {{passive: false}});
      if (canvas.classList.contains('schedule-chart')) renderRankTabs(canvas);
      redraw(canvas);
    }}

    document.querySelectorAll('canvas.trace-chart').forEach(installChart);
    populateMemoryTable();
    document.querySelectorAll('.chart-toolbar button').forEach((button) => {{
      button.addEventListener('click', () => {{
        const toolbar = button.closest('.chart-toolbar');
        const canvas = document.getElementById(toolbar.dataset.target);
        const state = chartState.get(canvas) || {{zoom: 1}};
        if (button.dataset.action === 'zoom-in') state.zoom = Math.min(3, state.zoom * 1.25);
        if (button.dataset.action === 'zoom-out') state.zoom = Math.max(0.45, state.zoom / 1.25);
        if (button.dataset.action === 'reset') state.zoom = 1;
        chartState.set(canvas, state);
        redraw(canvas);
      }});
    }});
  </script>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Text summary export
# ---------------------------------------------------------------------------


def export_text_summary(result: SimulationResult) -> str:
    """
    Return a human-readable text summary of a :class:`SimulationResult`.

    Prints statistics about the compute graph, communication ops, FSDP
    lifecycle events, and schedule events.

    Args:
        result: The simulation result to summarise.

    Returns:
        A multi-line string.
    """
    lines: list[str] = []
    sep = "=" * 72

    def section(title: str) -> None:
        lines.append("")
        lines.append(sep)
        lines.append(f"  {title}")
        lines.append(sep)

    graph = result.compute_graph

    section("Compute Graph Summary")
    lines.append(f"  Total ops : {len(graph.nodes)}")
    lines.append(f"  Total edges: {len(graph.edges)}")

    # Count by type
    type_counts: dict[str, int] = {}
    for n in graph.nodes.values():
        type_counts[n.op_type] = type_counts.get(n.op_type, 0) + 1
    for t, c in sorted(type_counts.items()):
        lines.append(f"    {t:<22}: {c}")

    # Count by phase
    phase_counts: dict[str, int] = {}
    for n in graph.nodes.values():
        p = n.phase or "unknown"
        phase_counts[p] = phase_counts.get(p, 0) + 1
    lines.append("")
    lines.append("  By phase:")
    for p, c in sorted(phase_counts.items()):
        lines.append(f"    {p:<22}: {c}")

    section("Communication Events")
    lines.append(f"  Total comm events: {len(result.comm_events)}")
    op_counts: dict[str, int] = {}
    for ev in result.comm_events:
        op = ev.get("op", "unknown")
        op_counts[op] = op_counts.get(op, 0) + 1
    for op, c in sorted(op_counts.items()):
        lines.append(f"    {op:<22}: {c}")

    section("FSDP Events")
    lines.append(f"  Total FSDP events: {len(result.fsdp_events)}")
    ev_type_counts: dict[str, int] = {}
    for ev in result.fsdp_events:
        t = ev.get("event_type", "unknown")
        ev_type_counts[t] = ev_type_counts.get(t, 0) + 1
    for t, c in sorted(ev_type_counts.items()):
        lines.append(f"    {t:<22}: {c}")

    section("PP Events")
    lines.append(f"  Total PP events: {len(result.pp_events)}")
    pp_type_counts: dict[str, int] = {}
    for ev in result.pp_events:
        t = ev.get("event_type", ev.get("action_type", "unknown"))
        pp_type_counts[t] = pp_type_counts.get(t, 0) + 1
    for t, c in sorted(pp_type_counts.items()):
        lines.append(f"    {t:<22}: {c}")

    if result.schedule:
        section("Training Schedule")
        sched = result.schedule
        lines.append(f"  Total schedule events: {len(sched.events)}")
        lines.append(f"  Total schedule deps  : {len(sched.deps)}")
        if sched.metadata:
            for k, v in sched.metadata.items():
                lines.append(f"    {k}: {v}")

    section("Memory Estimate")
    memory = result.metadata.get("memory", {}) or {}
    lines.append(f"  Total memory events: {len(result.memory_events)}")
    if memory:
        for key in (
            "peak_live_bytes",
            "graph_peak_live_bytes",
            "parameter_bytes",
            "gradient_bytes",
            "optimizer_state_bytes",
            "model_state_total_bytes",
            "total_event_bytes",
        ):
            if key in memory:
                lines.append(f"  {key}: {_format_bytes(memory[key])}")
        for group_key in ("by_category", "by_phase", "by_device"):
            group = memory.get(group_key)
            if group:
                lines.append(f"  {group_key}:")
                for name, value in sorted(group.items()):
                    lines.append(f"    {name:<24}: {_format_bytes(value)}")

    section("Metadata")
    for k, v in result.metadata.items():
        lines.append(f"  {k}: {v}")

    return "\n".join(lines)
