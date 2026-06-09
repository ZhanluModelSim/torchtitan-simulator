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

from .nodes import ComputeGraph, OpNode, SimulationResult

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
        self._finish_state.set(True)
        self.release(self._resource)
        finish = self.env.now()

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
        self._finish_state.set(True)
        self.release(self._resource)
        finish = self.env.now()

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
    for event in result.schedule.events:
        if event.op_node_ids:
            total = 0.0
            for nid in event.op_node_ids:
                node = result.compute_graph.nodes.get(nid)
                if node and node.perf_result:
                    total += node.perf_result.total_time_us
            event_durations[event.event_id] = total
        else:
            phase = event_phase_map.get(event.event_type, "unknown")
            phase_total = phase_duration.get(phase, 0.0)
            count = event_type_counts.get(event.event_type, 1)
            event_durations[event.event_id] = phase_total / count

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
