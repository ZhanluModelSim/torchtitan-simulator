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

    Uses compact separators for large results (>10K nodes) to reduce
    file size and serialization time.  Pretty-prints small results.

    Args:
        result: The simulation result to serialize.
        path: Output file path (will be created / overwritten).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _populate_des_metadata(result)
    data = result.to_dict()
    _inject_schedule_timing(data, result)
    node_count = len(result.compute_graph.nodes)
    if node_count > 10000:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"), default=str)
    else:
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
            if node.op_type in ("comm_collective", "comm_p2p"):
                engine = "comm_engine"
            else:
                engine = "compute_engine"
            tid = _get_tid(engine)
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

        phase_starts: dict[str, float] = {}
        for node in result.compute_graph.nodes.values():
            phase = node.phase or "unknown"
            if node.des_start_time_us is not None:
                if (
                    phase not in phase_starts
                    or node.des_start_time_us < phase_starts[phase]
                ):
                    phase_starts[phase] = node.des_start_time_us
        for phase, start_us in sorted(phase_starts.items()):
            events.append(
                {
                    "ph": "i",
                    "pid": 0,
                    "tid": _get_tid("phase_markers"),
                    "ts": start_us / 1000.0,
                    "name": f"{phase} phase start",
                    "cat": "phase_boundary",
                    "s": "g",
                }
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
        name = phase
        if has_des:
            if phase == "compute_engine":
                name = "Compute Engine"
            elif phase == "comm_engine":
                name = "Comm Engine"
            elif phase == "phase_markers":
                name = "Phase Markers"
        events.append(
            {
                "ph": "M",
                "pid": 0,
                "tid": tid,
                "name": "thread_name",
                "args": {"name": name},
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


def _populate_des_metadata(result: SimulationResult) -> None:
    if "des_engine" in result.metadata:
        return
    has_des_nodes = any(
        n.des_start_time_us is not None for n in result.compute_graph.nodes.values()
    )
    has_des_events = result.schedule is not None and any(
        ev.des_start_time_us is not None for ev in result.schedule.events
    )
    if not has_des_nodes and not has_des_events:
        return
    from .des_engine import compute_des_memory_timeline, compute_des_utilization

    util = compute_des_utilization(result)
    result.metadata.setdefault("des_engine", {}).update(util)
    mem = compute_des_memory_timeline(result)
    result.metadata["des_memory"] = {
        "static_memory_bytes": mem["static_memory_bytes"],
        "peak_dynamic_bytes": mem["peak_dynamic_bytes"],
        "peak_total_bytes": mem["peak_total_bytes"],
        "timeline": mem["timeline"],
        "timeline_samples": mem.get("timeline_samples", 0),
        "phase_peak": mem["phase_peak"],
    }


def _compress_graph_by_stage_similarity(result: SimulationResult) -> dict[str, Any]:
    """Compress compute graph by detecting and merging similar PP stages.
    
    For PP models with VPP, many stages have identical operation patterns.
    This function detects repeated stage groups and represents them as a
    single representative with a multiplier.
    
    Returns a dict with:
    - nodes: compressed node list with stage ranges
    - edges: edges between compressed nodes
    - stage_groups: metadata about which stages were merged
    """
    from collections import defaultdict, Counter
    
    # Group nodes by (phase, pp_stage)
    groups = defaultdict(list)
    for node in result.compute_graph.nodes.values():
        key = (node.phase, node.pp_stage)
        groups[key].append(node)
    
    # Analyze each group's signature (operation distribution)
    group_signatures = {}
    for key, nodes in groups.items():
        phase, stage = key
        if stage is None:
            continue  # Skip phase boundary nodes
        
        # Create signature: sorted list of (op_name, count)
        op_counts = Counter(n.op_name for n in nodes)
        signature = tuple(sorted(op_counts.items()))
        group_signatures[key] = {
            "signature": signature,
            "node_count": len(nodes),
            "nodes": nodes,
        }
    
    # Detect repeated stage groups
    # Group stages by (phase, signature)
    phase_stage_groups = defaultdict(list)
    for (phase, stage), info in group_signatures.items():
        phase_stage_groups[(phase, info["signature"])].append(stage)
    
    # Build compressed representation
    compressed_nodes = []
    stage_groups = []
    kept_stages = set()
    
    for (phase, signature), stages in phase_stage_groups.items():
        stages = sorted(stages)
        
        # Find consecutive runs
        runs = []
        current_run = [stages[0]]
        for i in range(1, len(stages)):
            if stages[i] == stages[i-1] + 1:
                current_run.append(stages[i])
            else:
                runs.append(current_run)
                current_run = [stages[i]]
        runs.append(current_run)
        
        # For each run, keep only the first stage as representative
        for run in runs:
            if len(run) == 1:
                # Single stage, keep as-is
                stage = run[0]
                kept_stages.add((phase, stage))
                stage_groups.append({
                    "phase": phase,
                    "stages": [stage],
                    "count": 1,
                    "representative": stage,
                })
            else:
                # Multiple consecutive stages, keep first as representative
                representative = run[0]
                kept_stages.add((phase, representative))
                stage_groups.append({
                    "phase": phase,
                    "stages": run,
                    "count": len(run),
                    "representative": representative,
                })
    
    # Also keep phase boundary nodes (stage=None)
    for key, nodes in groups.items():
        phase, stage = key
        if stage is None:
            kept_stages.add(key)
    
    # Build compressed node list
    node_id_map = {}  # old_id -> new_id
    for (phase, stage) in kept_stages:
        nodes = groups[(phase, stage)]
        # Limit nodes per stage to keep JSON size manageable
        # Prioritize compute nodes over trivial ops
        compute_nodes = [n for n in nodes if n.op_type == 'compute']
        other_nodes = [n for n in nodes if n.op_type != 'compute']
        limited_nodes = (compute_nodes[:200] + other_nodes[:50])[:250]
        for node in limited_nodes:
            node_dict = node.to_dict()
            node_id_map[node.node_id] = node.node_id
            compressed_nodes.append(node_dict)
    
    # Build compressed edge list
    compressed_edges = []
    for edge in result.compute_graph.edges:
        if edge.src_node_id in node_id_map and edge.dst_node_id in node_id_map:
            compressed_edges.append(edge.to_dict())
    
    return {
        "nodes": compressed_nodes,
        "edges": compressed_edges,
        "stage_groups": stage_groups,
        "compressed": True,
        "total_nodes": len(result.compute_graph.nodes),
        "total_edges": len(result.compute_graph.edges),
        "kept_stages": len(kept_stages),
        "total_stages": len(group_signatures),
    }


def _json_script_payload(result: SimulationResult) -> str:
    node_count = len(result.compute_graph.nodes)
    _populate_des_metadata(result)
    if node_count > 10000:
        # For large graphs, compress by merging similar PP stages
        schedule_data = None
        if result.schedule is not None:
            schedule_data = result.schedule.to_dict()
            for ev in schedule_data.get("events", []):
                if len(ev.get("op_node_ids", [])) > 100:
                    ev["op_node_ids"] = ev["op_node_ids"][:100]
                    ev["op_node_ids_truncated"] = True
        _inject_schedule_timing(
            {"schedule": schedule_data} if schedule_data else {}, result
        )
        
        # Compress graph by stage similarity
        compressed_graph = _compress_graph_by_stage_similarity(result)
        
        compact: dict[str, Any] = {
            "metadata": result.metadata,
            "schedule": schedule_data,
            "compute_graph": compressed_graph,
            "memory_events": [e.to_dict() for e in result.memory_events[:1000]],
        }
        return escape(json.dumps(compact, default=str), quote=False)
    data = result.to_dict()
    _inject_schedule_timing(data, result)
    if "des_engine" in result.metadata:
        data["metadata"]["des_engine"] = result.metadata["des_engine"]
    if "des_memory" in result.metadata:
        data["metadata"]["des_memory"] = result.metadata["des_memory"]
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


def _short_op_name(name: str, max_len: int = 42) -> str:
    name = name.replace("aten.", "").replace(".default", "")
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def export_html(
    result: SimulationResult,
    path: str | os.PathLike,
    *,
    title: str = "TorchTitan Simulation Trace",
    max_dag_nodes_per_phase: int = 220,
) -> None:
    """
    Write a self-contained HTML visualization using ECharts and AntV G6.

    Uses ECharts for timeline/swimlane visualization and AntV G6 for DAG
    visualization. These libraries provide better performance and interactivity
    compared to custom canvas rendering.
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
    # Extract per-GPU and whole-model memory values
    per_gpu_model_state = memory_summary.get("per_gpu_model_state_bytes", 0)
    whole_model_state = memory_summary.get("model_state_total_bytes", 0)
    tp_degree = memory_summary.get("tp_degree", 1)
    fsdp_degree = memory_summary.get("fsdp_degree", 1)
    shard_factor = memory_summary.get("shard_factor", 1)
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
              <summary>{escape(phase)} operator swimlane (Cube / Vec / Communication)</summary>
              <div id="swimlane-{step}-{escape(phase)}" style="width:100%;height:700px;background:#f8fafc;border-radius:8px;border:1px solid #e5e7eb;"></div>
            </details>
            """
            for phase in step_phases
        )

    event_ids_per_step: dict[int, set[str]] = {}
    for ev in schedule_events:
        step = _event_step(ev)
        event_ids_per_step.setdefault(step, set()).add(ev.get("event_id"))

    step_sections = "\n".join(
        f"""
        <details open>
          <summary>Train step {step}</summary>
          <details open>
            <summary>PP / FSDP2 / TP / DP / communication schedule swimlanes</summary>
            <div id="timeline-{step}" style="width:100%;height:600px;background:#f8fafc;border-radius:8px;"></div>
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
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
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
      <div class="card"><div class="num">{escape(_format_bytes(per_gpu_model_state))}</div><div>Per-GPU model state</div></div>
      <div class="card"><div class="num">{escape(_format_bytes(whole_model_state))}</div><div>Whole-model state</div></div>
      <div class="card"><div class="num">{escape(_format_bytes(peak_memory))}</div><div>Activation peak</div></div>
      <div class="card"><div class="num">{len(result.memory_events)}</div><div>Memory events</div></div>
      <div class="card"><div class="num">{_format_time_us(perf_grand_total_us)}</div><div>Predicted step time</div></div>
      {des_cards}
    </section>
    <div class="muted" style="margin:-8px 0 16px 0;">
      Parallelism: TP={tp_degree}, FSDP={fsdp_degree}, shard_factor={shard_factor}
    </div>
    <details open>
      <summary>Memory trace timeline</summary>
      <div id="memory-timeline" style="width:100%;height:500px;background:#f8fafc;border-radius:8px;"></div>
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
  </main>
  <script type="application/json" id="trace-data">{data_payload}</script>
  <script>
    const TRACE = JSON.parse(document.getElementById('trace-data').textContent);

    // Helper functions
    function fmt(us) {{
      if (us === undefined || us === null) return '—';
      if (us >= 1000) return (us / 1000).toFixed(2) + ' ms';
      return us.toFixed(1) + ' µs';
    }}

    function formatBytes(bytes) {{
      if (bytes === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }}

    // Initialize ECharts for memory timeline
    const memoryChart = echarts.init(document.getElementById('memory-timeline'));
    const desMemory = TRACE.metadata?.des_memory || {{}};
    const memoryTimeline = desMemory.timeline || [];
    const memoryMeta = TRACE.metadata?.memory || {{}};
    
    // Extract parallelism info
    const tpDegree = memoryMeta.tp_degree || 1;
    const fsdpDegree = memoryMeta.fsdp_degree || 1;
    const shardFactor = memoryMeta.shard_factor || 1;
    
    // Use DES timeline data: [time_us, total_bytes]
    const memoryData = memoryTimeline.map(s => [s.time_us, s.total_bytes]);
    const staticMemory = desMemory.static_memory_bytes || 0;
    
    // Calculate whole-model static memory for reference
    const wholeModelStatic = staticMemory * shardFactor;
    
    memoryChart.setOption({{
      tooltip: {{
        trigger: 'axis',
        formatter: function(params) {{
          const time = params[0].value[0];
          const total = params[0].value[1];
          const sample = memoryTimeline[params[0].dataIndex];
          const dynamic = sample ? sample.dynamic_bytes : 0;
          const staticBytes = sample ? sample.static_bytes : 0;
          const wholeModelTotal = total * shardFactor;
          return `<b>Time:</b> ${{fmt(time)}}<br/>
                  <b>Per-GPU Memory:</b><br/>
                  &nbsp;&nbsp;Total: ${{formatBytes(total)}}<br/>
                  &nbsp;&nbsp;Static: ${{formatBytes(staticBytes)}}<br/>
                  &nbsp;&nbsp;Dynamic: ${{formatBytes(dynamic)}}<br/>
                  <b>Whole Model (×${{shardFactor}}):</b><br/>
                  &nbsp;&nbsp;Total: ${{formatBytes(wholeModelTotal)}}<br/>
                  &nbsp;&nbsp;Static: ${{formatBytes(wholeModelStatic)}}`;
        }}
      }},
      legend: {{
        data: ['Per-GPU Total', 'Per-GPU Static', 'Whole-Model Static'],
        top: 10
      }},
      xAxis: {{
        type: 'value',
        name: 'Time',
        nameLocation: 'middle',
        nameGap: 30,
        axisLabel: {{
          formatter: (val) => fmt(val)
        }},
        splitLine: {{
          lineStyle: {{ color: '#e5e7eb', type: 'dashed' }}
        }}
      }},
      yAxis: {{
        type: 'value',
        name: 'Memory',
        axisLabel: {{
          formatter: (val) => formatBytes(val)
        }},
        splitLine: {{
          lineStyle: {{ color: '#f3f4f6' }}
        }}
      }},
      series: [
        {{
          name: 'Per-GPU Total',
          type: 'line',
          data: memoryData,
          smooth: false,
          areaStyle: {{ opacity: 0.2, color: '#3b82f6' }},
          lineStyle: {{ width: 2, color: '#3b82f6' }},
          itemStyle: {{ color: '#3b82f6' }},
          showSymbol: false
        }},
        {{
          name: 'Per-GPU Static',
          type: 'line',
          data: memoryData.length > 0 ? [[memoryData[0][0], staticMemory], [memoryData[memoryData.length-1][0], staticMemory]] : [],
          lineStyle: {{ width: 2, color: '#ef4444', type: 'dashed' }},
          itemStyle: {{ color: '#ef4444' }},
          showSymbol: false,
          z: 1
        }},
        {{
          name: 'Whole-Model Static',
          type: 'line',
          data: memoryData.length > 0 ? [[memoryData[0][0], wholeModelStatic], [memoryData[memoryData.length-1][0], wholeModelStatic]] : [],
          lineStyle: {{ width: 2, color: '#f59e0b', type: 'dotted' }},
          itemStyle: {{ color: '#f59e0b' }},
          showSymbol: false,
          z: 1
        }}
      ],
      dataZoom: [
        {{ type: 'inside', xAxisIndex: 0, start: 0, end: 100 }},
        {{ type: 'slider', xAxisIndex: 0, start: 0, end: 100, height: 20, bottom: 10 }}
      ],
      grid: {{ left: '10%', right: '5%', bottom: '18%', top: '12%' }}
    }});

    // Initialize ECharts for timeline swimlanes
    {chr(10).join(f'''
    (function() {{
      const chart = echarts.init(document.getElementById('timeline-{step}'));
      const allEvents = TRACE.schedule?.events || [];
      const events = allEvents.filter(e => {{
        const evStep = e.metadata?.step ?? e.step ?? 0;
        return parseInt(evStep) === {step};
      }});
      
      if (events.length === 0) {{
        document.getElementById('timeline-{step}').innerHTML = '<div style="padding:40px;text-align:center;color:#666;">No schedule events for this step</div>';
        return;
      }}
      
      // Group events by rank
      const rankMap = {{}};
      events.forEach(ev => {{
        const rank = ev.rank || 0;
        if (!rankMap[rank]) rankMap[rank] = [];
        rankMap[rank].push(ev);
      }});
      
      const ranks = Object.keys(rankMap).sort((a, b) => parseInt(a) - parseInt(b));
      const seriesData = [];
      
      // Color mapping for event types
      const colorMap = {{
        'forward': '#3b82f6',
        'backward': '#ef4444',
        'optimizer': '#10b981',
        'comm': '#f59e0b',
        'pp_send': '#8b5cf6',
        'pp_recv': '#a855f7',
        'fsdp': '#06b6d4',
        'default': '#6b7280'
      }};
      
      function getEventColor(eventType) {{
        const type = (eventType || '').toLowerCase();
        if (type.includes('forward') || type.includes('fwd')) return colorMap.forward;
        if (type.includes('backward') || type.includes('bwd')) return colorMap.backward;
        if (type.includes('optimizer') || type.includes('optim')) return colorMap.optimizer;
        if (type.includes('pp_send') || type.includes('send')) return colorMap.pp_send;
        if (type.includes('pp_recv') || type.includes('recv')) return colorMap.pp_recv;
        if (type.includes('fsdp')) return colorMap.fsdp;
        if (type.includes('comm') || type.includes('all_') || type.includes('reduce')) return colorMap.comm;
        return colorMap.default;
      }}
      
      ranks.forEach((rank, rankIdx) => {{
        rankMap[rank].forEach(ev => {{
          const start = ev.perf_cumulative_start_us || 0;
          const duration = ev.perf_total_time_us || 0;
          const end = start + duration;
          
          seriesData.push({{
            name: ev.event_type,
            value: [rankIdx, start, end, duration],
            itemStyle: {{
              color: getEventColor(ev.event_type)
            }},
            event: ev
          }});
        }});
      }});
      
      chart.setOption({{
        tooltip: {{
          trigger: 'item',
          formatter: function(params) {{
            const ev = params.data.event;
            const duration = ev.perf_total_time_us || 0;
            const start = ev.perf_cumulative_start_us || 0;
            return `<div style="padding:8px;">
              <b style="font-size:14px;">${{ev.event_type}}</b><br/>
              <span style="color:#666;">Rank:</span> ${{ev.rank || 0}}<br/>
              <span style="color:#666;">Start:</span> ${{fmt(start)}}<br/>
              <span style="color:#666;">Duration:</span> ${{fmt(duration)}}<br/>
              <span style="color:#666;">End:</span> ${{fmt(start + duration)}}
            </div>`;
          }}
        }},
        legend: {{
          data: ['forward', 'backward', 'optimizer', 'comm', 'pp_send', 'pp_recv', 'fsdp'],
          top: 10,
          textStyle: {{ color: '#333' }}
        }},
        xAxis: {{
          type: 'value',
          name: 'Time',
          nameLocation: 'middle',
          nameGap: 30,
          axisLabel: {{ 
            formatter: (val) => fmt(val),
            color: '#666'
          }},
          splitLine: {{
            lineStyle: {{ color: '#e5e7eb' }}
          }}
        }},
        yAxis: {{
          type: 'category',
          data: ranks.map(r => 'Rank ' + r),
          inverse: true,
          axisLabel: {{ 
            color: '#333',
            fontWeight: 'bold'
          }},
          splitLine: {{
            show: true,
            lineStyle: {{ color: '#f3f4f6' }}
          }}
        }},
        series: [{{
          type: 'custom',
          renderItem: function(params, api) {{
            const rankIdx = api.value(0);
            const start = api.coord([api.value(1), rankIdx]);
            const end = api.coord([api.value(2), rankIdx]);
            const height = api.size([0, 1])[1] * 0.7;
            const width = Math.max(2, end[0] - start[0]);
            
            return {{
              type: 'rect',
              shape: {{
                x: start[0],
                y: start[1] - height / 2,
                width: width,
                height: height,
                r: 3
              }},
              style: api.style(),
              emphasis: {{
                style: {{
                  shadowBlur: 10,
                  shadowColor: 'rgba(0,0,0,0.3)'
                }}
              }}
            }};
          }},
          encode: {{ x: [1, 2], y: 0 }},
          data: seriesData
        }}],
        dataZoom: [
          {{ type: 'inside', xAxisIndex: 0, start: 0, end: 100 }},
          {{ type: 'slider', xAxisIndex: 0, start: 0, end: 100, height: 20, bottom: 10 }}
        ],
        grid: {{ left: '8%', right: '5%', bottom: '18%', top: '12%' }}
      }});
    }})();
    ''' for step in steps)}

    // Initialize ECharts for operator swimlane visualization (Cube/Vec/Communication)
    {chr(10).join(f'''
    (function() {{
      const containerId = 'swimlane-{step}-{phase}';
      const container = document.getElementById(containerId);
      if (!container) return;
      
      const phase = '{phase}';
      const nodes = (TRACE.compute_graph?.nodes || []).filter(n => n.phase === phase);
      const edges = (TRACE.compute_graph?.edges || []).filter(e => {{
        const srcNode = nodes.find(n => n.node_id === e.src);
        const dstNode = nodes.find(n => n.node_id === e.dst);
        return srcNode && dstNode;
      }});
      
      if (nodes.length === 0) {{
        container.innerHTML = '<div style="padding:40px;text-align:center;color:#666;font-size:16px;">No operators in this phase</div>';
        return;
      }}
      
      // Classify operators into Compute (Cube/Vec) or Communication
      // Cube and Vec share the same compute engine — they CANNOT run in parallel.
      // Only Communication can overlap with Compute when no data dependency exists.
      function classifyOperator(node) {{
        const opName = (node.op_name || '').toLowerCase();
        const opType = node.op_type || '';
        
        // Communication operations — runs on dedicated comm engine
        if (opType === 'comm_collective' || opType === 'comm_p2p' || 
            opName.includes('all_reduce') || opName.includes('all_gather') || 
            opName.includes('reduce_scatter') || opName.includes('broadcast') ||
            opName.includes('send') || opName.includes('recv')) {{
          return 'Communication';
        }}
        
        // Everything else runs on the shared compute engine
        // (Cube: mm/matmul/conv; Vec: add/mul/relu/norm; memory; data_move)
        return 'Compute';
      }}
      
      // Sub-classify for display color only (does NOT affect scheduling)
      function subClassify(node) {{
        const opName = (node.op_name || '').toLowerCase();
        if (opName.includes('mm') || opName.includes('matmul') || opName.includes('bmm') ||
            opName.includes('addmm') || opName.includes('linear') || opName.includes('conv') ||
            opName.includes('gemm') || opName.includes('dot')) {{
          return 'Cube';
        }}
        return 'Vec';
      }}
      
      // Build dependency graph
      const nodeMap = new Map(nodes.map(n => [n.node_id, n]));
      const inDegree = new Map(nodes.map(n => [n.node_id, 0]));
      const adjList = new Map(nodes.map(n => [n.node_id, []]));
      const reverseAdjList = new Map(nodes.map(n => [n.node_id, []]));
      
      edges.forEach(e => {{
        if (inDegree.has(e.dst)) {{
          inDegree.set(e.dst, inDegree.get(e.dst) + 1);
        }}
        if (adjList.has(e.src)) {{
          adjList.get(e.src).push(e.dst);
        }}
        if (reverseAdjList.has(e.dst)) {{
          reverseAdjList.get(e.dst).push(e.src);
        }}
      }});
      
      // Kahn's algorithm for topological sort
      const queue = [];
      inDegree.forEach((deg, nodeId) => {{
        if (deg === 0) queue.push(nodeId);
      }});
      
      const sorted = [];
      while (queue.length > 0) {{
        const nodeId = queue.shift();
        sorted.push(nodeId);
        (adjList.get(nodeId) || []).forEach(nextId => {{
          const newDeg = inDegree.get(nextId) - 1;
          inDegree.set(nextId, newDeg);
          if (newDeg === 0) queue.push(nextId);
        }});
      }}
      
      // Add any remaining nodes (cycles or disconnected)
      nodes.forEach(n => {{
        if (!sorted.includes(n.node_id)) sorted.push(n.node_id);
      }});
      
      // ── Two-resource DES scheduling ──────────────────────────────
      // Compute engine: shared by Cube + Vec (serialized, no overlap)
      // Comm engine:    dedicated to Communication (can overlap with Compute)
      //
      // For each operator in topological order:
      //   1. depEndTime = max(end time of all data-dependency predecessors)
      //   2. If Compute: start = max(depEndTime, computeLaneEndTime)
      //      If Comm:    start = max(depEndTime, commLaneEndTime)
      let computeLaneEndTime = 0;
      let commLaneEndTime = 0;
      const operatorTimes = new Map();
      
      sorted.forEach(nodeId => {{
        const node = nodeMap.get(nodeId);
        if (!node) return;
        
        const lane = classifyOperator(node);
        const duration = node.perf_result?.total_time_us || 0;
        const subType = lane === 'Compute' ? subClassify(node) : 'Communication';
        
        // Max end time of all data-dependency predecessors
        let depEndTime = 0;
        const dependencies = reverseAdjList.get(nodeId) || [];
        dependencies.forEach(depId => {{
          const depTime = operatorTimes.get(depId);
          if (depTime && depTime.end > depEndTime) {{
            depEndTime = depTime.end;
          }}
        }});
        
        let startTime;
        if (lane === 'Compute') {{
          // Compute engine is shared — must wait for both deps AND previous compute op
          startTime = Math.max(depEndTime, computeLaneEndTime);
          computeLaneEndTime = startTime + duration;
        }} else {{
          // Comm engine is independent — must wait for deps AND previous comm op
          startTime = Math.max(depEndTime, commLaneEndTime);
          commLaneEndTime = startTime + duration;
        }}
        const endTime = startTime + duration;
        
        operatorTimes.set(nodeId, {{
          nodeId: node.node_id,
          opName: (node.op_name || 'unknown').replace('aten.', '').replace('.default', ''),
          opType: node.op_type,
          subType: subType,
          start: startTime,
          end: endTime,
          duration: duration,
          lane: lane
        }});
      }});
      
      // Organize operators by display lane (Cube, Vec, Communication)
      const displayLanes = {{ 'Cube': [], 'Vec': [], 'Communication': [] }};
      operatorTimes.forEach((op) => {{
        displayLanes[op.subType].push(op);
      }});
      
      // Sort each lane by start time
      Object.keys(displayLanes).forEach(laneName => {{
        displayLanes[laneName].sort((a, b) => a.start - b.start);
      }});
      
      // Prepare ECharts data
      // Display lanes: Cube, Vec, Communication (for visual separation only)
      // Scheduling: Cube+Vec share Compute engine, Comm is independent
      const laneNames = ['Cube', 'Vec', 'Communication'];
      const laneColors = {{
        'Cube': '#3b82f6',
        'Vec': '#10b981',
        'Communication': '#f59e0b'
      }};
      
      const seriesData = [];
      let maxTime = 0;
      
      laneNames.forEach((laneName, laneIdx) => {{
        displayLanes[laneName].forEach(op => {{
          seriesData.push({{
            name: op.opName,
            value: [laneIdx, op.start, op.end, op.duration],
            itemStyle: {{ color: laneColors[laneName] }},
            op: op
          }});
          if (op.end > maxTime) maxTime = op.end;
        }});
      }});
      
      // Initialize ECharts
      const chart = echarts.init(container);
      chart.setOption({{
        tooltip: {{
          trigger: 'item',
          formatter: function(params) {{
            const op = params.data.op;
            const deps = reverseAdjList.get(op.nodeId) || [];
            const dependents = adjList.get(op.nodeId) || [];
            const engine = op.lane === 'Compute' ? 'Compute Engine' : 'Comm Engine';
            return `<div style="padding:10px;min-width:220px;">
              <div style="font-weight:bold;font-size:14px;margin-bottom:8px;color:#1f2937;">${{op.opName}}</div>
              <div style="color:#6b7280;font-size:12px;line-height:1.6;">
                <div><b>Category:</b> ${{op.subType}} (${{engine}})</div>
                <div><b>Type:</b> ${{op.opType}}</div>
                <div><b>Start:</b> ${{fmt(op.start)}}</div>
                <div><b>Duration:</b> ${{fmt(op.duration)}}</div>
                <div><b>End:</b> ${{fmt(op.end)}}</div>
                <div style="margin-top:6px;padding-top:6px;border-top:1px solid #e5e7eb;">
                  <b>Dependencies:</b> ${{deps.length}} ops<br/>
                  <b>Dependents:</b> ${{dependents.length}} ops
                </div>
                <div style="margin-top:4px;font-family:monospace;font-size:11px;color:#9ca3af;">ID: ${{op.nodeId}}</div>
              </div>
            </div>`;
          }}
        }},
        legend: {{
          data: laneNames,
          top: 10,
          textStyle: {{ color: '#333', fontSize: 13 }},
          itemWidth: 20,
          itemHeight: 14
        }},
        grid: {{
          left: '12%',
          right: '8%',
          top: '15%',
          bottom: '18%'
        }},
        xAxis: {{
          type: 'value',
          name: 'Cumulative Time',
          nameLocation: 'middle',
          nameGap: 35,
          nameTextStyle: {{ fontSize: 13, fontWeight: 'bold' }},
          axisLabel: {{
            formatter: (val) => fmt(val),
            fontSize: 11
          }},
          splitLine: {{
            lineStyle: {{ color: '#e5e7eb', type: 'dashed' }}
          }},
          max: maxTime
        }},
        yAxis: {{
          type: 'category',
          data: laneNames,
          inverse: true,
          axisLabel: {{
            fontSize: 14,
            fontWeight: 'bold',
            color: '#1f2937'
          }},
          axisTick: {{ show: false }},
          splitLine: {{
            show: true,
            lineStyle: {{ color: '#f3f4f6', width: 2 }}
          }}
        }},
        series: [{{
          type: 'custom',
          renderItem: function(params, api) {{
            const laneIdx = api.value(0);
            const start = api.coord([api.value(1), laneIdx]);
            const end = api.coord([api.value(2), laneIdx]);
            const height = api.size([0, 1])[1] * 0.75;
            const width = Math.max(2, end[0] - start[0]);
            
            return {{
              type: 'rect',
              shape: {{
                x: start[0],
                y: start[1] - height / 2,
                width: width,
                height: height,
                r: 4
              }},
              style: api.style(),
              emphasis: {{
                style: {{
                  shadowBlur: 12,
                  shadowColor: 'rgba(0,0,0,0.3)',
                  stroke: '#1f2937',
                  lineWidth: 2
                }}
              }}
            }};
          }},
          encode: {{ x: [1, 2], y: 0 }},
          data: seriesData
        }}],
        dataZoom: [
          {{ type: 'inside', xAxisIndex: 0, start: 0, end: 100 }},
          {{ type: 'slider', xAxisIndex: 0, start: 0, end: 100, height: 25, bottom: 10 }}
        ]
      }});
      
      // Add statistics panel
      const statsDiv = document.createElement('div');
      statsDiv.style.cssText = 'position:absolute;top:10px;right:10px;background:rgba(255,255,255,0.95);padding:12px 16px;border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,0.1);font-size:12px;color:#374151;min-width:220px;';
      
      const cubeOps = displayLanes['Cube'].length;
      const vecOps = displayLanes['Vec'].length;
      const commOps = displayLanes['Communication'].length;
      const totalOps = cubeOps + vecOps + commOps;
      
      // Calculate total compute time per engine
      let computeTotalTime = 0;
      let commTotalTime = 0;
      operatorTimes.forEach(op => {{
        if (op.lane === 'Compute') computeTotalTime += op.duration;
        else commTotalTime += op.duration;
      }});
      
      statsDiv.innerHTML = `
        <div style="font-weight:bold;margin-bottom:8px;font-size:13px;">Execution Statistics</div>
        <div style="line-height:1.8;">
          <div><span style="color:${{laneColors.Cube}};">●</span> Cube: <b>${{cubeOps}}</b> (${{(cubeOps/totalOps*100).toFixed(1)}}%)</div>
          <div><span style="color:${{laneColors.Vec}};">●</span> Vec: <b>${{vecOps}}</b> (${{(vecOps/totalOps*100).toFixed(1)}}%)</div>
          <div><span style="color:${{laneColors.Communication}};">●</span> Comm: <b>${{commOps}}</b> (${{(commOps/totalOps*100).toFixed(1)}}%)</div>
          <div style="margin-top:6px;padding-top:6px;border-top:1px solid #e5e7eb;">
            <b>Total:</b> ${{totalOps}} ops<br/>
            <b>Critical path:</b> ${{fmt(maxTime)}}<br/>
            <b>Compute engine:</b> ${{fmt(computeTotalTime)}}<br/>
            <b>Comm engine:</b> ${{fmt(commTotalTime)}}
          </div>
          <div style="margin-top:6px;color:#6b7280;font-size:10px;line-height:1.4;">
            Cube+Vec share the <b>Compute Engine</b> (serialized).<br/>
            Communication runs on a separate <b>Comm Engine</b>.<br/>
            Compute↔Comm overlap only when no data dependency.
          </div>
        </div>
      `;
      container.style.position = 'relative';
      container.appendChild(statsDiv);
      
      // Handle resize
      window.addEventListener('resize', () => chart.resize());
    }})();
    ''' for step in steps for phase in phases)}

    // Handle window resize
    window.addEventListener('resize', function() {{
      memoryChart.resize();
      {chr(10).join(f'echarts.getInstanceByDom(document.getElementById("timeline-{step}"))?.resize();' for step in steps)}
      {chr(10).join(f'echarts.getInstanceByDom(document.getElementById("swimlane-{step}-{phase}"))?.resize();' for step in steps for phase in phases)}
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

    des_engine = result.metadata.get("des_engine", {}) or {}
    if des_engine:
        section("DES Engine Summary")
        lines.append(
            f"  E2E step time (DES)  : {_format_time_us(des_engine.get('e2e_step_time_us', 0))}"
        )
        lines.append(
            f"  Compute busy time   : {_format_time_us(des_engine.get('compute_busy_us', 0))}"
        )
        lines.append(
            f"  Comm busy time      : {_format_time_us(des_engine.get('comm_busy_us', 0))}"
        )
        lines.append(
            f"  Overlap time        : {_format_time_us(des_engine.get('overlap_us', 0))}"
        )
        lines.append(
            f"  Compute utilization : {des_engine.get('compute_busy_pct', 0):.1f}%"
        )
        lines.append(
            f"  Comm utilization    : {des_engine.get('comm_busy_pct', 0):.1f}%"
        )
        lines.append(f"  Overlap             : {des_engine.get('overlap_pct', 0):.1f}%")
        lines.append(f"  Contended ops       : {des_engine.get('contention_count', 0)}")
        lines.append(
            f"  CP step time        : {_format_time_us(des_engine.get('cp_step_time_us', 0))}"
        )
        lines.append(
            f"  DES / CP ratio      : {des_engine.get('des_vs_cp_ratio', 0):.3f}x"
        )
        per_phase = des_engine.get("per_phase", {}) or {}
        if per_phase:
            lines.append("")
            lines.append("  Per-phase DES breakdown:")
            for phase, data in sorted(per_phase.items()):
                lines.append(f"    {phase}: {data}")

    des_memory = result.metadata.get("des_memory", {}) or {}
    if des_memory:
        section("DES Memory Estimate")
        lines.append(
            f"  Static memory       : {_format_bytes(des_memory.get('static_memory_bytes', 0))}"
        )
        lines.append(
            f"  Peak dynamic memory : {_format_bytes(des_memory.get('peak_dynamic_bytes', 0))}"
        )
        lines.append(
            f"  Peak total memory   : {_format_bytes(des_memory.get('peak_total_bytes', 0))}"
        )
        lines.append(f"  Timeline samples    : {des_memory.get('timeline_samples', 0)}")
        phase_peak = des_memory.get("phase_peak", {}) or {}
        if phase_peak:
            lines.append("")
            lines.append("  Per-phase peak memory:")
            for phase, data in sorted(phase_peak.items()):
                lines.append(
                    f"    {phase:<14}: peak={_format_bytes(data.get('peak_total_bytes', 0))}  dynamic={_format_bytes(data.get('peak_dynamic_bytes', 0))}"
                )

    section("Metadata")
    for k, v in result.metadata.items():
        if k in ("cost_model", "des_engine", "des_memory"):
            continue
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
