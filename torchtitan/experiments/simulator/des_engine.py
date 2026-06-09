# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
salabim-based Discrete Event Simulation engine for step time prediction.

Models each rank as having two hardware Resource engines:
- **compute** (capacity=1): handles compute, data_move, memory ops
- **comm** (capacity=1): handles comm_collective, comm_p2p ops

Ops on separate engines can overlap (parallel execution on GPU).
Ops on the same engine are serialized (resource contention).
DAG data dependencies are modeled via salabim State signals.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import salabim as sim

sim.yieldless(False)

from .nodes import ComputeGraph, MemoryEvent, OpNode, SimulationResult

_COMM_OP_TYPES = ("comm_collective", "comm_p2p")


class _OpComponent(sim.Component):
    def setup(
        self,
        node: OpNode,
        resource: sim.Resource,
        pred_states: dict[str, sim.State],
        finish_state: sim.State,
        duration: float,
    ) -> None:
        self._node = node
        self._resource = resource
        self._pred_states = pred_states
        self._finish_state = finish_state
        self._duration = duration

    def process(self) -> Any:
        for pred_name, pred_state in self._pred_states.items():
            while not pred_state.get():
                yield self.wait(pred_state)

        yield self.request(self._resource)
        start = self.env.now()
        yield self.hold(self._duration)
        self.release(self._resource)
        finish = self.env.now()
        self._finish_state.set(True)

        if self._node is not None:
            self._node.des_start_time_us = round(start, 3)
            self._node.des_finish_time_us = round(finish, 3)


def simulate_single_rank_des(graph: ComputeGraph) -> float:
    """Run salabim DES on a single-rank compute graph.

    Returns:
        E2E step time in microseconds (max finish time across all nodes).
    """
    if not graph.nodes:
        return 0.0

    env = sim.Environment(trace=False)
    compute_resource = sim.Resource("compute", capacity=1)
    comm_resource = sim.Resource("comm", capacity=1)

    finish_states: dict[str, sim.State] = {}

    pred_map: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
    for edge in graph.edges:
        if edge.src_node_id in pred_map and edge.dst_node_id in pred_map:
            pred_map[edge.dst_node_id].append(edge.src_node_id)

    adj: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
    in_degree: dict[str, int] = {nid: 0 for nid in graph.nodes}
    for edge in graph.edges:
        if edge.src_node_id in adj and edge.dst_node_id in adj:
            adj[edge.src_node_id].append(edge.dst_node_id)
            in_degree[edge.dst_node_id] = in_degree.get(edge.dst_node_id, 0) + 1

    queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
    topo: list[str] = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in adj.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    for nid in graph.nodes:
        finish_states[nid] = sim.State(f"{nid}_finished", value=False)

    for nid in topo:
        node = graph.nodes.get(nid)
        duration = 0.0
        if node and node.perf_result:
            duration = node.perf_result.total_time_us

        resource = (
            comm_resource
            if (node and node.op_type in _COMM_OP_TYPES)
            else compute_resource
        )

        pred_state_map = {}
        for pred_id in pred_map.get(nid, []):
            pred_state_map[pred_id] = finish_states[pred_id]

        _OpComponent(
            name=nid,
            node=node,
            resource=resource,
            pred_states=pred_state_map,
            finish_state=finish_states[nid],
            duration=duration,
        )

    env.run()

    max_finish = 0.0
    for nid in graph.nodes:
        node = graph.nodes.get(nid)
        if node and node.des_finish_time_us is not None:
            max_finish = max(max_finish, node.des_finish_time_us)

    return round(max_finish, 3)


def _event_engine_type(event_type: str) -> str:
    if event_type in (
        "pp_send_activation",
        "pp_recv_activation",
        "pp_send_gradient",
        "pp_recv_gradient",
        "fsdp2_all_gather",
        "fsdp2_reduce_scatter",
        "dp_gradient_sync",
    ):
        return "comm"
    return "compute"


class _ScheduleEventComponent(sim.Component):
    def setup(
        self,
        event_id: str,
        resource: sim.Resource,
        pred_states: dict[str, sim.State],
        finish_state: sim.State,
        duration: float,
        schedule_event: Any,
    ) -> None:
        self._event_id = event_id
        self._resource = resource
        self._pred_states = pred_states
        self._finish_state = finish_state
        self._duration = duration
        self._schedule_event = schedule_event

    def process(self) -> Any:
        for pred_id, pred_state in self._pred_states.items():
            while not pred_state.get():
                yield self.wait(pred_state)

        yield self.request(self._resource)
        start = self.env.now()
        yield self.hold(self._duration)
        self.release(self._resource)
        finish = self.env.now()
        self._finish_state.set(True)

        if self._schedule_event is not None:
            self._schedule_event.des_start_time_us = round(start, 3)
            self._schedule_event.des_finish_time_us = round(finish, 3)


def simulate_multi_rank_des(result: SimulationResult) -> float:
    """Run salabim DES across all ranks using schedule dependency graph.

    Returns:
        Maximum finish time across all ranks (E2E step time in us).
    """
    from .cost_model import link_schedule_to_graph

    if result.schedule is None or len(result.schedule.events) == 0:
        return simulate_single_rank_des(result.compute_graph)

    link_schedule_to_graph(result)

    phase_duration: dict[str, float] = {}
    for node in result.compute_graph.nodes.values():
        phase = node.phase or "unknown"
        if node.perf_result:
            phase_duration.setdefault(phase, 0.0)
            phase_duration[phase] += node.perf_result.total_time_us

    event_type_counts: dict[str, int] = {}
    for event in result.schedule.events:
        event_type_counts.setdefault(event.event_type, 0)
        event_type_counts[event.event_type] += 1

    event_phase_map = {
        "pp_forward": "forward",
        "pp_backward": "backward",
        "pp_send_activation": "forward",
        "pp_recv_activation": "forward",
        "pp_send_gradient": "backward",
        "pp_recv_gradient": "backward",
        "fsdp2_all_gather": "forward",
        "fsdp2_reduce_scatter": "backward",
        "dp_gradient_sync": "backward",
        "optimizer_step": "optimizer",
    }

    event_durations: dict[str, float] = {}
    duration_key_cache: dict[tuple[str, int | None, int | None], float] = {}
    for event in result.schedule.events:
        key = (event.event_type, event.rank, event.pp_stage)
        if key in duration_key_cache:
            event_durations[event.event_id] = duration_key_cache[key]
        else:
            if event.op_node_ids:
                total = 0.0
                for nid in event.op_node_ids:
                    node = result.compute_graph.nodes.get(nid)
                    if node and node.perf_result:
                        total += node.perf_result.total_time_us
                event_durations[event.event_id] = total
                duration_key_cache[key] = total
            else:
                phase = event_phase_map.get(event.event_type, "unknown")
                phase_total = phase_duration.get(phase, 0.0)
                count = event_type_counts.get(event.event_type, 1)
                per_event = phase_total / count
                event_durations[event.event_id] = per_event
                duration_key_cache[key] = per_event

    event_pred_map: dict[str, list[str]] = {}
    for dep in result.schedule.deps:
        event_pred_map.setdefault(dep.to_event_id, []).append(dep.from_event_id)

    rank_events: dict[int, list[Any]] = {}
    for event in result.schedule.events:
        rank_events.setdefault(event.rank, []).append(event)

    env = sim.Environment(trace=False)

    finish_states: dict[str, sim.State] = {}
    for event in result.schedule.events:
        finish_states[event.event_id] = sim.State(
            f"{event.event_id}_finished", value=False
        )

    for rank in sorted(rank_events.keys()):
        compute_resource = sim.Resource(f"compute_rank{rank}", capacity=1)
        comm_resource = sim.Resource(f"comm_rank{rank}", capacity=1)

        for event in rank_events[rank]:
            duration = event_durations.get(event.event_id, 0.0)
            engine_type = _event_engine_type(event.event_type)
            resource = comm_resource if engine_type == "comm" else compute_resource

            pred_state_map = {}
            for pred_id in event_pred_map.get(event.event_id, []):
                pred_state_map[pred_id] = finish_states[pred_id]

            _ScheduleEventComponent(
                name=event.event_id,
                event_id=event.event_id,
                resource=resource,
                pred_states=pred_state_map,
                finish_state=finish_states[event.event_id],
                duration=duration,
                schedule_event=event,
            )

    env.run()

    max_finish = 0.0
    for event in result.schedule.events:
        if event.des_finish_time_us is not None:
            max_finish = max(max_finish, event.des_finish_time_us)

    return round(max_finish, 3)


class DESEngine:
    def predict_step_time_us(
        self, result: SimulationResult, cost_model: Any | None = None
    ) -> float:
        cm = cost_model
        if cm is None:
            from .cost_model import MockCostModel

            cm = MockCostModel()
        if not any(n.perf_result for n in result.compute_graph.nodes.values()):
            cm.estimate_graph(result.compute_graph)

        if result.schedule is not None and len(result.schedule.events) > 0:
            return simulate_multi_rank_des(result)
        return simulate_single_rank_des(result.compute_graph)

    def annotate(
        self, result: SimulationResult, cost_model: Any | None = None
    ) -> float:
        step_time = self.predict_step_time_us(result, cost_model)
        result.metadata.setdefault("des_engine", {})["e2e_step_time_us"] = step_time
        return step_time


def compute_des_utilization(result: SimulationResult) -> dict[str, Any]:
    """Compute DES engine utilization stats from annotated nodes/events."""
    from .cost_model import _critical_path_time_us

    graph = result.compute_graph
    nodes = list(graph.nodes.values())

    has_des = any(n.des_start_time_us is not None for n in nodes) or (
        result.schedule is not None
        and any(ev.des_start_time_us is not None for ev in result.schedule.events)
    )

    if not has_des:
        return {
            "e2e_step_time_us": 0.0,
            "single_rank_step_time_us": 0.0,
            "compute_busy_us": 0.0,
            "comm_busy_us": 0.0,
            "overlap_us": 0.0,
            "compute_busy_pct": 0.0,
            "comm_busy_pct": 0.0,
            "overlap_pct": 0.0,
            "contention_count": 0,
            "per_phase": {},
            "cp_step_time_us": 0.0,
            "des_vs_cp_ratio": 0.0,
        }

    compute_intervals: list[tuple[float, float]] = []
    comm_intervals: list[tuple[float, float]] = []
    contention_count = 0

    for node in nodes:
        if node.des_start_time_us is None or node.des_finish_time_us is None:
            continue
        start = node.des_start_time_us
        finish = node.des_finish_time_us
        dur = finish - start
        if node.op_type in ("comm_collective", "comm_p2p"):
            comm_intervals.append((start, finish))
        else:
            compute_intervals.append((start, finish))
        perf_dur = node.perf_result.total_time_us if node.perf_result else 0.0
        if dur > perf_dur + 0.1:
            contention_count += 1

    if not compute_intervals and not comm_intervals and result.schedule is not None:
        for ev in result.schedule.events:
            if ev.des_start_time_us is None or ev.des_finish_time_us is None:
                continue
            start = ev.des_start_time_us
            finish = ev.des_finish_time_us
            dur = finish - start
            engine_type = _event_engine_type(ev.event_type)
            if engine_type == "comm":
                comm_intervals.append((start, finish))
            else:
                compute_intervals.append((start, finish))
            perf_dur = (
                sum(
                    result.compute_graph.nodes.get(nid).perf_result.total_time_us
                    for nid in ev.op_node_ids
                    if nid in result.compute_graph.nodes
                    and result.compute_graph.nodes[nid].perf_result is not None
                )
                if ev.op_node_ids
                else 0.0
            )
            if dur > perf_dur + 0.1:
                contention_count += 1

    e2e_step = max(
        (n.des_finish_time_us for n in nodes if n.des_finish_time_us is not None),
        default=0.0,
    )
    if e2e_step == 0.0 and result.schedule is not None:
        e2e_step = max(
            (
                ev.des_finish_time_us
                for ev in result.schedule.events
                if ev.des_finish_time_us is not None
            ),
            default=0.0,
        )
    cp_step = _critical_path_time_us(graph)

    compute_busy = _merge_interval_total(compute_intervals)
    comm_busy = _merge_interval_total(comm_intervals)
    overlap = _compute_overlap(compute_intervals, comm_intervals)

    return {
        "e2e_step_time_us": round(e2e_step, 3),
        "single_rank_step_time_us": round(e2e_step, 3),
        "compute_busy_us": round(compute_busy, 3),
        "comm_busy_us": round(comm_busy, 3),
        "overlap_us": round(overlap, 3),
        "compute_busy_pct": round(
            compute_busy / e2e_step * 100 if e2e_step > 0 else 0.0, 2
        ),
        "comm_busy_pct": round(comm_busy / e2e_step * 100 if e2e_step > 0 else 0.0, 2),
        "overlap_pct": round(overlap / e2e_step * 100 if e2e_step > 0 else 0.0, 2),
        "contention_count": contention_count,
        "per_phase": {},
        "cp_step_time_us": round(cp_step, 3),
        "des_vs_cp_ratio": round(e2e_step / cp_step if cp_step > 0 else 0.0, 4),
    }


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    sorted_ivs = sorted(intervals)
    merged = [sorted_ivs[0]]
    for start, end in sorted_ivs[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _merge_interval_total(intervals: list[tuple[float, float]]) -> float:
    merged = _merge_intervals(intervals)
    return sum(end - start for start, end in merged)


def _compute_overlap(
    compute_intervals: list[tuple[float, float]],
    comm_intervals: list[tuple[float, float]],
) -> float:
    merged_compute = _merge_intervals(compute_intervals)
    merged_comm = _merge_intervals(comm_intervals)
    overlap = 0.0
    for cs, ce in merged_compute:
        for ms, me in merged_comm:
            os = max(cs, ms)
            oe = min(ce, me)
            if oe > os:
                overlap += oe - os
    return overlap


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
