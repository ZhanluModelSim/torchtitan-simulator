# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import deque

from .nodes import ComputeGraph, SimulationResult


def _critical_path_time_us(graph: ComputeGraph) -> float:
    """Topological longest-path algorithm on the compute graph.

    Uses ``perf_result.total_time_us`` as the node weight.  Returns the
    maximum finish time across all nodes, which is the critical-path
    duration.

    Returns:
        Longest path duration in microseconds, or 0.0 if graph is unannotated.
    """
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

    dist: dict[str, float] = {}
    max_time = 0.0
    for u in topo:
        node = graph.nodes.get(u)
        dur = 0.0
        if node and node.perf_result:
            dur = node.perf_result.total_time_us
        pred_finish = 0.0
        for edge in graph.edges:
            if edge.dst_node_id == u and edge.src_node_id in dist:
                pred_finish = max(pred_finish, dist[edge.src_node_id])
        dist[u] = pred_finish + dur
        if dist[u] > max_time:
            max_time = dist[u]

    return round(max_time, 3)


def link_schedule_to_graph(result: SimulationResult) -> None:
    """Populate ScheduleEvent.op_node_ids by matching phase (and optionally pp_stage/microbatch_idx).

    For each ScheduleEvent, find all OpNodes in result.compute_graph that
    share the same phase.  When a node has ``pp_stage=None`` or
    ``microbatch_idx=None`` (single-rank trace), it matches **any**
    schedule event with the same phase, regardless of the event's
    stage/microbatch values.  When both have concrete values, they
    must match exactly.

    For single-rank traces where nodes lack stage/mb labels, the
    matching is **proportional**: each event of the same type gets
    ``total_phase_duration / num_events_of_same_type`` as its
    duration, rather than inheriting the full set of graph nodes
    (which would duplicate time across every event).

    This bridges the coarse schedule and fine-grained compute graph,
    enabling multi-rank step time prediction.
    """
    if result.schedule is None:
        return

    graph = result.compute_graph

    node_lookup: dict[tuple[str, int | None, int | None], list[str]] = {}
    for nid, node in graph.nodes.items():
        key = (node.phase, node.pp_stage, node.microbatch_idx)
        node_lookup.setdefault(key, []).append(nid)

    phase_map = {
        "pp_forward": "forward",
        "pp_backward": "backward",
        "fsdp2_all_gather": "forward",
        "fsdp2_reduce_scatter": "backward",
        "dp_gradient_sync": "backward",
        "optimizer_step": "optimizer",
    }

    for event in result.schedule.events:
        phase = phase_map.get(event.event_type, "unknown")
        lookup_key = (phase, event.pp_stage, event.microbatch_idx)
        exact_matches = node_lookup.get(lookup_key, [])

        if exact_matches:
            event.op_node_ids = exact_matches
        else:
            event.op_node_ids = []


def predict_multi_rank_step_time_us(
    result: SimulationResult,
    cost_model: "CostModel | None" = None,
) -> float:
    """Predict step time using multi-rank salabim DES.

    Falls back to single-rank DES if no schedule events.

    Args:
        result: SimulationResult with populated schedule and compute_graph.
        cost_model: Optional CostModel for fallback duration estimation.

    Returns:
        Predicted step time in microseconds.
    """
    from .cost_model import MockCostModel
    from .des_engine import simulate_multi_rank_des

    if result.schedule is None or len(result.schedule.events) == 0:
        if cost_model is None:
            cost_model = MockCostModel()
        cost_model.estimate_graph(result.compute_graph)
        return cost_model.predict_step_time_us(result.compute_graph)

    if cost_model is None:
        cost_model = MockCostModel()
    cost_model.estimate_graph(result.compute_graph)

    return simulate_multi_rank_des(result)
