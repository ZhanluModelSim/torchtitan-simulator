# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

from .nodes import MemoryEvent, SimulationResult


def compute_des_memory_timeline(result: SimulationResult) -> dict[str, Any]:
    """Map MemoryEvent lifetimes to DES wall-clock timestamps."""
    graph = result.compute_graph
    nodes_list = list(graph.nodes.values())
    memory_events = result.memory_events

    index_to_des_time: list[float] = []
    node_id_to_des_start: dict[str, float] = {}
    node_id_to_des_finish: dict[str, float] = {}
    if result.schedule is not None:
        for ev in result.schedule.events:
            if ev.des_start_time_us is None or ev.des_finish_time_us is None:
                continue
            for nid in ev.op_node_ids:
                node_id_to_des_start[nid] = min(
                    node_id_to_des_start.get(nid, float("inf")),
                    ev.des_start_time_us,
                )
                node_id_to_des_finish[nid] = max(
                    node_id_to_des_finish.get(nid, 0.0),
                    ev.des_finish_time_us,
                )
    node_des_available = any(node.des_start_time_us is not None for node in nodes_list)
    if node_des_available:
        for node in nodes_list:
            if node.des_start_time_us is not None:
                index_to_des_time.append(node.des_start_time_us)
            else:
                if node.node_id in node_id_to_des_start:
                    index_to_des_time.append(node_id_to_des_start[node.node_id])
                else:
                    index_to_des_time.append(0.0)
    else:
        for node in nodes_list:
            if node.node_id in node_id_to_des_start:
                index_to_des_time.append(node_id_to_des_start[node.node_id])
            else:
                index_to_des_time.append(0.0)

    static_events: list[MemoryEvent] = []
    dynamic_events: list[MemoryEvent] = []
    for me in memory_events:
        if me.lifetime_start is None or me.lifetime_end is None:
            static_events.append(me)
        else:
            dynamic_events.append(me)

    static_memory_bytes = sum(me.bytes for me in static_events)

    dynamic_with_times: list[dict[str, Any]] = []
    for me in dynamic_events:
        alloc_time = index_to_des_time[me.lifetime_start]
        free_node = nodes_list[me.lifetime_end]
        if free_node.des_finish_time_us is not None:
            free_time = free_node.des_finish_time_us
        elif free_node.node_id in node_id_to_des_finish:
            free_time = node_id_to_des_finish[free_node.node_id]
        else:
            free_time = index_to_des_time[me.lifetime_end]
        dynamic_with_times.append(
            {
                "alloc_time": alloc_time,
                "free_time": free_time,
                "bytes": me.bytes,
                "category": me.category,
                "phase": me.phase,
            }
        )

    if not dynamic_with_times and static_memory_bytes == 0:
        return {
            "static_memory_bytes": 0,
            "peak_dynamic_bytes": 0,
            "peak_total_bytes": 0,
            "timeline": [],
            "timeline_samples": 0,
            "phase_peak": {},
        }

    timestamps: set[float] = set()
    for dw in dynamic_with_times:
        timestamps.add(dw["alloc_time"])
        timestamps.add(dw["free_time"])
    sorted_ts = sorted(timestamps)

    timeline: list[dict[str, Any]] = []
    peak_dynamic = 0
    peak_total = 0
    for ts in sorted_ts:
        dynamic_bytes = 0
        by_category: dict[str, int] = {}
        for dw in dynamic_with_times:
            if dw["alloc_time"] <= ts <= dw["free_time"]:
                dynamic_bytes += dw["bytes"]
                by_category[dw["category"]] = (
                    by_category.get(dw["category"], 0) + dw["bytes"]
                )
        total_bytes = static_memory_bytes + dynamic_bytes
        timeline.append(
            {
                "time_us": ts,
                "static_bytes": static_memory_bytes,
                "dynamic_bytes": dynamic_bytes,
                "total_bytes": total_bytes,
                "by_category": by_category,
            }
        )
        peak_dynamic = max(peak_dynamic, dynamic_bytes)
        peak_total = max(peak_total, total_bytes)

    if not dynamic_with_times:
        timeline.append(
            {
                "time_us": 0.0,
                "static_bytes": static_memory_bytes,
                "dynamic_bytes": 0,
                "total_bytes": static_memory_bytes,
                "by_category": {},
            }
        )
        peak_total = static_memory_bytes

    # Fill duration_us: each sample spans from its time_us to the next
    # sample's time_us (or to the step end for the last sample).  This
    # makes the timeline continuous — memory level is held constant
    # between alloc/free transitions.
    step_end = 0.0
    for node in nodes_list:
        if node.des_finish_time_us is not None:
            step_end = max(step_end, node.des_finish_time_us)
    for i, s in enumerate(timeline):
        if i < len(timeline) - 1:
            s["duration_us"] = timeline[i + 1]["time_us"] - s["time_us"]
        else:
            s["duration_us"] = max(0.0, step_end - s["time_us"])

    phase_ranges: dict[str, tuple[float, float]] = {}
    for i, node in enumerate(nodes_list):
        phase = node.phase or "unknown"
        if phase not in phase_ranges:
            if node.des_start_time_us is not None:
                phase_ranges[phase] = (node.des_start_time_us, node.des_start_time_us)
            else:
                phase_ranges[phase] = (0.0, 0.0)
        else:
            start, end = phase_ranges[phase]
            node_finish = (
                node.des_finish_time_us
                if node.des_finish_time_us is not None
                else (
                    node.des_start_time_us
                    if node.des_start_time_us is not None
                    else 0.0
                )
            )
            phase_ranges[phase] = (start, max(end, node_finish))

    phase_peak: dict[str, dict[str, Any]] = {}
    for phase, (p_start, p_end) in phase_ranges.items():
        phase_samples = [s for s in timeline if p_start <= s["time_us"] <= p_end]
        if not phase_samples:
            continue
        max_total = max(s["total_bytes"] for s in phase_samples)
        max_dynamic = max(s["dynamic_bytes"] for s in phase_samples)
        peak_cat: dict[str, int] = {}
        for s in phase_samples:
            if s["total_bytes"] == max_total:
                peak_cat = s["by_category"]
                break
        phase_peak[phase] = {
            "peak_total_bytes": max_total,
            "peak_dynamic_bytes": max_dynamic,
            "by_category": peak_cat,
        }

    return {
        "static_memory_bytes": static_memory_bytes,
        "peak_dynamic_bytes": peak_dynamic,
        "peak_total_bytes": peak_total,
        "timeline": timeline,
        "timeline_samples": len(timeline),
        "phase_peak": phase_peak,
    }
