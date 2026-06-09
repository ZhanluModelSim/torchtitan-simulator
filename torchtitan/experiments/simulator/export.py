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
    "compute": "#AED6F1",  # light blue
    "comm_collective": "#F9E79F",  # yellow
    "comm_p2p": "#FAD7A0",  # orange
    "data_move": "#A9DFBF",  # light green
    "memory": "#D7BDE2",  # light purple
    "unknown": "#D5D8DC",  # grey
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
        lines.append(f"  {src} -> {dst} [style={style}];")

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
        "ts": ts_us / 1000.0,
        "dur": dur_us / 1000.0,
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
    sequentially per phase on separate *threads* (tid).  When
    :attr:`OpNode.perf_result` is available the duration reflects the
    estimated compute / communication time; otherwise events fall back to
    *us_per_op* microsecond slots.

    Args:
        result: The simulation result to render.
        path: Output JSON file path.
        us_per_op: Duration in microseconds to assign each op slot when no
            :class:`PerfResult` is available.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    phase_tid: dict[str, int] = {}
    tid_counter = [0]

    def _get_tid(phase: str) -> int:
        if phase not in phase_tid:
            phase_tid[phase] = tid_counter[0]
            tid_counter[0] += 1
        return phase_tid[phase]

    phase_ts: dict[str, float] = {}

    def _node_dur_us(node: OpNode) -> float:
        """Return per-node duration from PerfResult, or fall back to us_per_op."""
        if node.perf_result is not None and node.perf_result.total_time_us > 0:
            return node.perf_result.total_time_us
        return us_per_op

    has_des = any(
        n.des_start_time_us is not None for n in result.compute_graph.nodes.values()
    )

    events: list[dict[str, Any]] = []
    if has_des:
        for node in result.compute_graph.nodes.values():
            phase = node.phase or "unknown"
            tid = _get_tid(phase)
            ts_us = (
                node.des_start_time_us if node.des_start_time_us is not None else 0.0
            )
            if (
                node.des_finish_time_us is not None
                and node.des_start_time_us is not None
            ):
                dur_us = node.des_finish_time_us - node.des_start_time_us
            else:
                dur_us = _node_dur_us(node)
            events.append(
                _op_to_chrome_event(node, pid=0, tid=tid, ts_us=ts_us, dur_us=dur_us)
            )
    else:
        for node in result.compute_graph.nodes.values():
            phase = node.phase or "unknown"
            tid = _get_tid(phase)
            ts = phase_ts.get(phase, 0.0)
            dur = _node_dur_us(node)
            events.append(
                _op_to_chrome_event(node, pid=0, tid=tid, ts_us=ts, dur_us=dur)
            )
            phase_ts[phase] = ts + dur

    # Add FSDP events as a separate process
    for ev in result.fsdp_events:
        phase = ev.get("phase", "unknown")
        ts = phase_ts.get(f"fsdp_{phase}", 0.0)
        events.append(
            {
                "ph": "X",
                "pid": 1,
                "tid": _get_tid(f"fsdp_{phase}"),
                "ts": ts / 1000.0,
                "dur": us_per_op / 1000.0,
                "name": ev.get("event_type", "fsdp_event"),
                "cat": "fsdp",
                "args": ev,
            }
        )
        phase_ts[f"fsdp_{phase}"] = ts + us_per_op

    # Add schedule events as Chrome trace duration events
    if result.schedule is not None:
        _add_schedule_trace_events(events, result, _get_tid)

    # Metadata events (thread_name for each tid)
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

    trace = {"traceEvents": events, "displayTimeUnit": "ms"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, default=str)


def _add_schedule_trace_events(
    events: list[dict[str, Any]],
    result: SimulationResult,
    _get_tid: Any,
) -> None:
    """Add schedule events + aggregated phase-level events as Chrome trace events.

    pid layout:
      pid=0:  individual OpNode events (existing)
      pid=1:  FSDP events
      pid=2:  PP schedule events
      pid=3:  FSDP schedule events
      pid=4:  TP schedule events
      pid=5:  DP schedule events
      pid=6:  Optimizer schedule events
      pid=7:  Aggregated whole-graph phase blocks (forward / backward / optimizer)
    """
    # ── pid=7: Aggregated whole-graph phase blocks ──────────────────
    _add_aggregated_phase_events(events, result)

    if result.schedule is None:
        return

    des_event_map: dict[str, dict[str, Any]] = {}
    schedule_data: dict[str, Any] = {
        "events": [ev.to_dict() for ev in result.schedule.events],
    }
    _inject_schedule_timing({"schedule": schedule_data}, result)
    for ev_dict in schedule_data["events"]:
        eid = ev_dict.get("event_id", "")
        if "perf_cumulative_start_us" in ev_dict or "perf_total_time_us" in ev_dict:
            des_event_map[eid] = ev_dict

    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for ev in result.schedule.events:
        d = ev.to_dict()
        strategy = d.get("metadata", {}).get("strategy", "")
        et = d.get("event_type", "")
        if et.startswith("pp_") or et.startswith("loss"):
            strategy = "pp"
        elif et.startswith("fsdp2_"):
            strategy = "fsdp"
        elif et.startswith("tp_"):
            strategy = "tp"
        elif et.startswith("dp_"):
            strategy = "dp"
        elif et.startswith("optimizer"):
            strategy = "optim"
        by_strategy.setdefault(strategy, []).append(d)

    pid_map = {"pp": 2, "fsdp": 3, "tp": 4, "dp": 5, "optim": 6}

    for strategy, ev_list in by_strategy.items():
        pid = pid_map.get(strategy, 7)
        # One tid per lane within strategy
        tid_map: dict[str, int] = {}
        tid_counter = [0]
        for ev in ev_list:
            lane = _schedule_event_lane_for_trace(ev, strategy)
            if lane not in tid_map:
                tid_map[lane] = tid_counter[0]
                tid_counter[0] += 1
            tid = pid * 100 + tid_map[lane]
            eid = ev.get("event_id", "")
            enriched = des_event_map.get(eid, ev)
            ts = enriched.get("perf_cumulative_start_us", ev.get("logical_clock", 0))
            dur = enriched.get("perf_total_time_us", ev.get("perf_total_time_us", 1.0))
            if dur <= 0:
                dur = 1.0
            events.append(
                {
                    "ph": "X",
                    "pid": pid,
                    "tid": tid,
                    "ts": ts / 1000.0,
                    "dur": dur / 1000.0,
                    "name": ev.get("event_type", "event"),
                    "cat": strategy,
                    "args": {
                        "pp_stage": ev.get("pp_stage"),
                        "mb": ev.get("microbatch_idx"),
                        "rank": ev.get("rank"),
                    },
                }
            )

    # Metadata events for each strategy
    strategy_names = {
        "pp": "PP Schedule",
        "fsdp": "FSDP",
        "tp": "TP",
        "dp": "DP",
        "optim": "Optimizer",
    }
    for strategy, pid in pid_map.items():
        events.append(
            {
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "name": "process_name",
                "args": {"name": strategy_names.get(strategy, strategy)},
            }
        )


def _schedule_event_lane_for_trace(ev: dict[str, Any], strategy: str) -> str:
    """One lane per physical rank (card)."""
    return f"Rank {ev.get('rank', 0)}"


def _add_aggregated_phase_events(
    events: list[dict[str, Any]],
    result: SimulationResult,
) -> None:
    """Add pid=7 aggregated whole-graph phase blocks.

    Groups all OpNodes by (phase, pp_stage, microbatch_idx) and emits one
    duration event per group so that chrome://tracing shows a coarse
    forward / backward / optimizer overview without operator-level noise.
    """
    graph = result.compute_graph

    # Group by (phase, pp_stage, microbatch)
    from collections import defaultdict

    groups: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    # Also track the earliest logical position for each group
    group_indices: dict[tuple[str, int, int], list[int]] = defaultdict(list)

    node_list = list(graph.nodes.values())
    for idx, node in enumerate(node_list):
        if node.perf_result is None:
            continue
        key = (node.phase or "unknown", node.pp_stage or 0, node.microbatch_idx or 0)
        groups[key].append(node.perf_result.total_time_us)
        group_indices[key].append(idx)

    # Phase display order and colors
    phase_order = ["forward", "backward", "optimizer"]
    phase_colors: dict[str, str] = {
        "forward": "#93c5fd",
        "backward": "#fca5a5",
        "optimizer": "#86efac",
        "unknown": "#d5d8dc",
    }

    pid = 7
    tid = 0
    cumulative_ts_us = 0.0

    for phase in phase_order:
        # Find all groups for this phase
        phase_groups = [
            (k, sum(times), min(group_indices[k]))
            for k, times in groups.items()
            if k[0] == phase
        ]
        if not phase_groups:
            continue

        for key, total_us, _ in phase_groups:
            pp_stage, mb = key[1], key[2]
            op_count_val = len(groups[key])
            name = phase
            if pp_stage or mb:
                name += f" (s{pp_stage} mb{mb})"

            events.append(
                {
                    "ph": "X",
                    "pid": pid,
                    "tid": tid,
                    "ts": cumulative_ts_us / 1000.0,
                    "dur": total_us / 1000.0,
                    "name": name,
                    "cat": "aggregated",
                    "args": {
                        "phase": phase,
                        "pp_stage": pp_stage,
                        "microbatch": mb,
                        "op_count": len(groups[key]),
                        "total_us": round(total_us, 3),
                    },
                }
            )
            cumulative_ts_us += total_us
            tid += 1

    # Metadata
    events.append(
        {
            "ph": "M",
            "pid": pid,
            "tid": 0,
            "name": "process_name",
            "args": {"name": "Aggregated Phases"},
        }
    )


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------


def _json_script_payload(result: SimulationResult) -> str:
    data = result.to_dict()
    # ── Inject schedule timing from OpNode perf_results ───────────────
    _inject_schedule_timing(data, result)
    return escape(json.dumps(data, default=str), quote=False)


def _inject_schedule_timing(data: dict[str, Any], result: SimulationResult) -> None:
    """Pre-compute per-schedule-event timing using DES results when available."""
    graph = result.compute_graph

    phase_totals: dict[str, float] = {}
    for node in graph.nodes.values():
        if node.perf_result is None:
            continue
        phase = node.phase or "unknown"
        phase_totals[phase] = (
            phase_totals.get(phase, 0.0) + node.perf_result.total_time_us
        )
    grand_total = sum(phase_totals.values())

    event_counts: dict[str, int] = {}
    schedule = data.get("schedule")
    if schedule and schedule.get("events"):
        for ev in schedule["events"]:
            ev_type = ev.get("event_type", "")
            metadata = ev.get("metadata", {}) or {}
            strategy = metadata.get("strategy", "")
            phase = _schedule_event_to_phase(ev_type, strategy)
            event_counts[phase] = event_counts.get(phase, 0) + 1

    enriched_events: list[dict[str, Any]] = []
    if schedule and schedule.get("events"):
        des_event_map: dict[str, tuple[float, float]] = {}
        if result.schedule is not None:
            for ev in result.schedule.events:
                if (
                    ev.des_start_time_us is not None
                    and ev.des_finish_time_us is not None
                ):
                    des_event_map[ev.event_id] = (
                        ev.des_start_time_us,
                        ev.des_finish_time_us,
                    )

        cumulative_per_phase: dict[str, float] = {}
        for ev in schedule["events"]:
            ev_type = ev.get("event_type", "")
            metadata = ev.get("metadata", {}) or {}
            strategy = metadata.get("strategy", "")
            phase = _schedule_event_to_phase(ev_type, strategy)
            eid = ev.get("event_id", "")

            ev_copy = dict(ev)
            if eid in des_event_map:
                start, finish = des_event_map[eid]
                ev_copy["perf_total_time_us"] = round(finish - start, 3)
                ev_copy["perf_cumulative_start_us"] = round(start, 3)
            else:
                count = event_counts.get(phase, 1)
                phase_total = phase_totals.get(phase, 0.0)
                per_event = phase_total / max(count, 1)
                ev_copy["perf_total_time_us"] = round(per_event, 3)
                ev_copy["perf_cumulative_start_us"] = round(
                    cumulative_per_phase.get(phase, 0.0), 3
                )
                cumulative_per_phase[phase] = (
                    cumulative_per_phase.get(phase, 0.0) + per_event
                )
            enriched_events.append(ev_copy)

        schedule["events"] = enriched_events
        schedule["perf_grand_total_us"] = round(grand_total, 3)

    data["perf_schedule"] = {
        "grand_total_us": round(grand_total, 3),
        "phase_totals": {p: round(t, 3) for p, t in sorted(phase_totals.items())},
    }


def _schedule_event_to_phase(event_type: str, strategy: str) -> str:
    """Map a schedule event type to an OpNode phase string."""
    et = event_type.lower()
    strategy_lower = (strategy or "").lower()
    # Direct phase matches
    if "bwd" in et or "backward" in et or "backward" in strategy_lower:
        return "backward"
    if "fwd" in et or "forward" in et:
        return "forward"
    if "optim" in et:
        return "optimizer"
    # Strategy-based mapping
    if strategy_lower in ("pp", "compute"):
        return "forward"
    # Comm/FSDP/TP events: assign to the phase of surrounding ops.
    # "reduce_scatter" and "gradient" events happen during backward.
    if "reduce" in et or "gradient" in et:
        return "backward"
    if strategy_lower in ("fsdp2", "tp", "dp"):
        return "forward"  # all-gather/reduce scatter split: default forward
    if "loss" in et:
        return "forward"
    return "forward"  # default


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
    strategy = str(metadata.get("strategy", "")).lower()

    if event_type.startswith("dp_") or strategy == "dp":
        return f"DP rank {ev.get('rank', 0)}"
    if event_type.startswith("optimizer") or "step" in event_type:
        return f"Optim rank {ev.get('rank', 0)}"
    if event_type.startswith("tp_") or strategy == "tp":
        return f"TP rank {ev.get('rank', 0)}"
    if event_type.startswith("fsdp_") or strategy in ("fsdp2", "fsdp"):
        return f"FSDP rank {ev.get('rank', 0)}"
    if event_type.startswith("pp_") or ev.get("pp_stage") is not None:
        pp_rank = ev.get("pp_rank", 0)
        pp_stage = ev.get("pp_stage")
        return f"PP stage {pp_stage or pp_rank} (rank {pp_rank})"
    if event_type.startswith("loss") or strategy == "compute":
        return f"Loss (pp rank {ev.get('pp_rank', 0)})"
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
      <div class="chart-frame">
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
    # Compute perf grand total for summary card
    cost_summary = result.metadata.get("cost_model", {}) or {}
    perf_grand_total_us = cost_summary.get("e2e_step_time_us", 0)
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
      <div class="card"><div class="num">{_format_time_us(perf_grand_total_us)}</div><div>Predicted step time</div></div>
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
      const strategy = String(metadata.strategy || '').toLowerCase();

      // DP gradient sync — separate lane per DP rank
      if (eventType.startsWith('dp_') || strategy === 'dp') {{
        return 'DP rank ' + (ev.rank ?? 0);
      }}
      // Optimizer step — separate lane per rank
      if (eventType.startsWith('optimizer') || eventType.includes('step')) {{
        return 'Optim rank ' + (ev.rank ?? 0);
      }}
      // TP all-reduce — separate lane
      if (eventType.startsWith('tp_') || strategy === 'tp') {{
        return 'TP rank ' + (ev.rank ?? 0);
      }}
      // FSDP events
      if (eventType.startsWith('fsdp_') || strategy === 'fsdp2' || strategy === 'fsdp') {{
        return 'FSDP rank ' + (ev.rank ?? 0);
      }}
      // PP events (and anything else with pp_stage / pp_rank)
      if (eventType.startsWith('pp_') || ev.pp_stage !== null && ev.pp_stage !== undefined) {{
        const ppRank = ev.pp_rank ?? 0;
        const ppStage = ev.pp_stage;
        return 'PP stage ' + (ppStage ?? ppRank) + ' (rank ' + ppRank + ')';
      }}
      // Loss compute on last stage
      if (eventType.startsWith('loss') || strategy === 'compute') {{
        return 'Loss (pp rank ' + (ev.pp_rank ?? 0) + ')';
      }}
      // Fallback
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
      if (eventType.startsWith('loss')) return 'compute';
      if (eventType.startsWith('optimizer')) return 'optim';
      return 'other';
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

    function drawChromeTrace(canvas) {{
      const state = chartState.get(canvas) || {{zoom: 1}};
      chartState.set(canvas, state);
      const step = Number.parseInt(canvas.dataset.step || '0', 10);
      const allEvents = scheduleEvents().filter((ev) => eventStep(ev) === step);
      const hasTiming = allEvents.some(ev => ev.perf_cumulative_start_us !== undefined);

      // Build Chrome-trace-style event list: group by (pid, tid) = lane
      const laneMap = new Map();
      for (const ev of allEvents) {{
        const lane = chromeTraceLane(ev);
        if (!laneMap.has(lane)) laneMap.set(lane, []);
        laneMap.get(lane).push(ev);
      }}

      // Sort lanes
      const lanes = Array.from(laneMap.keys()).sort();
      // Sort events within each lane by cumulative start time
      for (const [lane, items] of laneMap) {{
        items.sort((a, b) => {{
          const ta = a.perf_cumulative_start_us !== undefined ? a.perf_cumulative_start_us : Number(a.logical_clock || 0);
          const tb = b.perf_cumulative_start_us !== undefined ? b.perf_cumulative_start_us : Number(b.logical_clock || 0);
          return ta - tb;
        }});
      }}

      // Compute time bounds
      const maxTime = hasTiming
        ? Math.max(1, ...allEvents.map(ev => (ev.perf_cumulative_start_us || 0) + (ev.perf_total_time_us || 0)))
        : Math.max(0, ...allEvents.map(ev => Number(ev.logical_clock || 0)));
      const pixelsPerUnit = hasTiming ? Math.max(0.005, 58 * state.zoom / 50) : 58 * state.zoom;
      const laneH = 28;
      const padTop = 40;
      const labelW = 170;
      const width = Math.max(980, labelW + 80 + (maxTime + 10) * pixelsPerUnit);
      const height = Math.max(160, padTop + lanes.length * laneH + 24);
      const ctx = resizeCanvas(canvas, width, height);
      ctx.clearRect(0, 0, width, height);
      ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
      ctx.textBaseline = 'middle';

      // Note
      const note = document.querySelector('.chart-toolbar[data-target=\"' + canvas.id + '\"] .chart-note');
      const totalLabel = hasTiming ? (maxTime >= 1000 ? (maxTime / 1000).toFixed(2) + 'ms' : maxTime.toFixed(0) + 'µs') : maxTime + ' clocks';
      if (note) note.textContent = allEvents.length + ' events in ' + lanes.length + ' lanes. Total span: ' + totalLabel + '. Drag or use scrollbar to pan.';

      // Lane labels + backgrounds
      lanes.forEach((lane, idx) => {{
        const y = padTop + idx * laneH;
        if (idx % 2 === 0) {{
          ctx.fillStyle = '#f8fafc';
          ctx.fillRect(labelW, y - laneH / 2, width - labelW - 30, laneH);
        }}
        // Show rank + PP annotation
        const rankNum = lane.replace('Rank ', '');
        const sample = laneMap.get(lane)?.[0];
        let annotation = lane;
        if (sample) {{
          const ppR = sample.pp_rank;
          const ppS = sample.pp_stage;
          if (ppR !== null && ppR !== undefined) {{
            annotation = 'Rank ' + rankNum + '  [PP' + ppR;
            if (ppS !== null && ppS !== undefined && ppS !== ppR) annotation += ' v' + ppS;
            annotation += ']';
          }}
        }}
        ctx.fillStyle = '#1e293b';
        ctx.font = 'bold 10px ui-sans-serif, system-ui, sans-serif';
        ctx.fillText(annotation, 6, y);
        ctx.strokeStyle = '#e2e8f0';
        ctx.beginPath();
        ctx.moveTo(labelW, y + laneH / 2);
        ctx.lineTo(width - 30, y + laneH / 2);
        ctx.stroke();
      }});

      // Time axis
      const axisY = padTop + lanes.length * laneH + 8;
      ctx.strokeStyle = '#94a3b8';
      ctx.beginPath();
      ctx.moveTo(labelW, axisY);
      ctx.lineTo(width - 30, axisY);
      ctx.stroke();
      const numTicks = Math.min(10, Math.ceil(maxTime / (hasTiming ? 500 : 5)));
      const tickInterval = Math.max(1, maxTime / numTicks);
      for (let t = 0; t <= maxTime; t += tickInterval) {{
        const tx = labelW + t * pixelsPerUnit;
        ctx.fillStyle = '#64748b';
        ctx.font = '9px ui-sans-serif, system-ui, sans-serif';
        const tickLabel = hasTiming ? (t >= 1000 ? (t / 1000).toFixed(1) + 'ms' : Math.round(t) + 'µs') : Math.round(t);
        ctx.fillText(tickLabel, tx, axisY + 10);
        ctx.beginPath();
        ctx.moveTo(tx, axisY);
        ctx.lineTo(tx, axisY + 4);
        ctx.stroke();
      }}

      // Draw bars
      for (const [lane, items] of laneMap) {{
        const laneIdx = lanes.indexOf(lane);
        const y = padTop + laneIdx * laneH;
        for (const ev of items) {{
          const tStart = hasTiming ? (ev.perf_cumulative_start_us || 0) : Number(ev.logical_clock || 0);
          const tDur = hasTiming ? Math.max(1, ev.perf_total_time_us || 0) : 1;
          const x = labelW + tStart * pixelsPerUnit;
          const barW = Math.max(4, tDur * pixelsPerUnit);
          const barH = laneH - 6;

          const name = String(ev.name || ev.event_type || '');
          let fill = palette.fwd;
          if (name.toLowerCase().includes('bwd')) fill = palette.bwd;
          else if (name.toLowerCase().includes('fsdp')) fill = palette.fsdp;
          else if (name.toLowerCase().includes('optim')) fill = '#86efac';
          else if (name.toLowerCase().includes('gradient')) fill = '#c084fc';
          else if (name.toLowerCase().includes('all_reduce')) fill = palette.comm;
          else if (name.toLowerCase().includes('send') || name.toLowerCase().includes('recv')) fill = '#f97316';

          ctx.fillStyle = fill;
          ctx.fillRect(x, y - barH / 2, barW, barH);
          ctx.strokeStyle = '#334155';
          ctx.lineWidth = 0.5;
          ctx.strokeRect(x, y - barH / 2, barW, barH);

          // Name inside bar (if wide enough) or beside
          if (barW > 50) {{
            ctx.fillStyle = '#1e293b';
            ctx.font = '9px ui-sans-serif, system-ui, sans-serif';
            ctx.fillText(shortName(name, 18), x + 3, y);
          }}
        }}
      }}
    }}

    function chromeTraceLane(ev) {{
      // One lane per physical card (rank).  All parallelism events on the
      // same card — PP forward/backward, FSDP, TP, DP, optimizer — are
      // inherently sequential and rendered in a single horizontal lane.
      return 'Rank ' + (ev.rank ?? 0);
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
        const srcNode = phaseNodes.find(n => n.node_id === edge.src);
        const dstNode = phaseNodes.find(n => n.node_id === edge.dst);
        const srcDur = Number(((srcNode || {{}}).perf_result || {{}}).total_time_us || 0);
        const maxDur = Math.max(1, ...phaseNodes.map(n => Number((n.perf_result || {{}}).total_time_us || 0)));
        const srcW = 140 + (srcDur > 0 ? Math.max(0.15, Math.log2(1 + srcDur) / Math.log2(1 + maxDur)) * 120 : 0);
        arrowLine(ctx, src.x + srcW, src.y + 28, dst.x - 6, dst.y + 28, edge.type === 'data' ? '#2563eb' : '#64748b', edge.type === 'control', 1);
      }}

      for (const node of phaseNodes) {{
        const pos = positions.get(node.node_id);
        const fill = palette[node.op_type] || palette.unknown;
        const pr = node.perf_result || {{}};
        const durUs = Number(pr.total_time_us || 0);
        // Scale node width by relative duration (log scale clamped)
        const maxDur = Math.max(1, ...phaseNodes.map(n => Number((n.perf_result || {{}}).total_time_us || 0)));
        const logScale = durUs > 0 ? Math.max(0.15, Math.log2(1 + durUs) / Math.log2(1 + maxDur)) : 0.15;
        const nodeW = 140 + logScale * 120;
        roundedRect(ctx, pos.x, pos.y, nodeW, 56, 8);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = '#0f172a';
        ctx.font = '700 12px ui-sans-serif, system-ui, sans-serif';
        ctx.fillText(shortName(node.op_name, 28), pos.x + 8, pos.y + 17);
        const shape = (node.outputs || []).slice(0, 1).map((t) => '[' + (t.shape || []).join(',') + ']').join(', ');
        ctx.fillStyle = '#334155';
        ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
        ctx.fillText(shortName((node.op_type || 'unknown') + ' ' + shape, 32), pos.x + 8, pos.y + 32);
        // Perf timing annotation
        if (durUs > 0) {{
          const timeLabel = durUs >= 1000 ? (durUs / 1000).toFixed(2) + 'ms' : durUs.toFixed(1) + 'µs';
          ctx.fillStyle = durUs > 50 ? '#b91c1c' : '#047857';
          ctx.font = 'italic 10px ui-sans-serif, system-ui, sans-serif';
          ctx.fillText(timeLabel, pos.x + 8, pos.y + 47);
        }}
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
      if (canvas.classList.contains('chrome-trace-chart')) drawChromeTrace(canvas);
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
      if (canvas.classList.contains('chrome-trace-chart')) {{
        // No rank tabs needed for Chrome trace view
      }}
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

    # ------------------------------------------------------------------
    # Performance estimate (from CostModel / PerfResult)
    # ------------------------------------------------------------------
    cost_summary = result.metadata.get("cost_model", {}) or {}
    if cost_summary:
        section("Performance Estimate (DES Engine)")
        lines.append(
            f"  E2E step time       : {_format_time_us(cost_summary.get('e2e_step_time_us', 0))}"
        )
        lines.append(
            f"  Single-rank step    : {_format_time_us(cost_summary.get('single_rank_step_time_us', 0))}"
        )
        lines.append(
            f"  Total compute time : {_format_time_us(cost_summary.get('total_compute_time_us', 0))}"
        )
        lines.append(
            f"  Total comm time    : {_format_time_us(cost_summary.get('total_comm_time_us', 0))}"
        )
        per_phase = cost_summary.get("per_phase", {}) or {}
        if per_phase:
            lines.append("")
            lines.append("  Per-phase breakdown:")
            for phase, times in sorted(per_phase.items()):
                comp = _format_time_us(times.get("compute_time_us", 0))
                comm = _format_time_us(times.get("comm_time_us", 0))
                total = _format_time_us(times.get("total_time_us", 0))
                lines.append(
                    f"    {phase:<14}: compute={comp}  comm={comm}  total={total}"
                )

        # Annotated node count
        annotated = sum(
            1 for n in result.compute_graph.nodes.values() if n.perf_result is not None
        )
        lines.append(
            f"  Nodes with perf data: {annotated} / {len(result.compute_graph.nodes)}"
        )

    section("Metadata")
    for k, v in result.metadata.items():
        if k == "cost_model":
            continue  # already rendered above
        lines.append(f"  {k}: {v}")

    return "\n".join(lines)


def _format_time_us(us: float | int) -> str:
    """Format a time in microseconds to a human-readable string."""
    us = float(us)
    if us >= 1e6:
        return f"{us / 1e6:.3f} s"
    elif us >= 1e3:
        return f"{us / 1e3:.3f} ms"
    else:
        return f"{us:.1f} µs"
