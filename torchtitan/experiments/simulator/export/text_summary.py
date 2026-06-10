# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from ..nodes import SimulationResult
from ._shared import _format_bytes, _format_time_us


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

    type_counts: dict[str, int] = {}
    for n in graph.nodes.values():
        type_counts[n.op_type] = type_counts.get(n.op_type, 0) + 1
    for t, c in sorted(type_counts.items()):
        lines.append(f"    {t:<22}: {c}")

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
                peak = _format_bytes(data.get("peak_total_bytes", 0))
                dyn = _format_bytes(data.get("peak_dynamic_bytes", 0))
                lines.append(f"    {phase:<14}: peak={peak}  dynamic={dyn}")

    section("Metadata")
    for k, v in result.metadata.items():
        if k in ("cost_model", "des_engine", "des_memory"):
            continue
        lines.append(f"  {k}: {v}")

    return "\n".join(lines)
