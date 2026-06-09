# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data classes for representing computation graphs and training schedules
captured by the TorchTitan simulator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TensorMeta:
    """Metadata about a tensor — shape, dtype, device, and optional DTensor placement info."""

    shape: tuple[int, ...]
    dtype: str
    device: str
    is_dtensor: bool = False
    # String representation of each placement (e.g. "Shard(0)", "Replicate()")
    placements: list[str] | None = None
    requires_grad: bool = False

    @classmethod
    def from_tensor(cls, t: Any) -> "TensorMeta":
        import torch
        from torch.distributed.tensor import DTensor

        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(t)}")

        is_dtensor = isinstance(t, DTensor)
        placements = None
        if is_dtensor:
            placements = [str(p) for p in t.placements]
        shape = tuple(t.shape)
        return cls(
            shape=shape,
            dtype=str(t.dtype),
            device=str(t.device),
            is_dtensor=is_dtensor,
            placements=placements,
            requires_grad=t.requires_grad,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "is_dtensor": self.is_dtensor,
            "placements": self.placements,
            "requires_grad": self.requires_grad,
        }


@dataclass
class PerfResult:
    """Performance estimation for a single :class:`OpNode`.

    Produced by a :class:`CostModel` and attached to each node.  All time
    fields are in microseconds (µs).  Fields can be left at their defaults
    when a cost model is not available.
    """

    compute_time_us: float = 0.0
    """Estimated compute time in microseconds."""

    comm_time_us: float = 0.0
    """Estimated communication time in microseconds (non-zero only for comm ops)."""

    total_time_us: float = 0.0
    """Total estimated time (compute + comm)."""

    flops: int = 0
    """Estimated floating-point operations."""

    bytes_read: int = 0
    """Estimated bytes read from memory (or received over the network)."""

    bytes_written: int = 0
    """Estimated bytes written to memory (or sent over the network)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Optional extra data from the cost model (e.g. roofline utilisation, bandwidth)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "compute_time_us": self.compute_time_us,
            "comm_time_us": self.comm_time_us,
            "total_time_us": self.total_time_us,
            "flops": self.flops,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "metadata": self.metadata,
        }


@dataclass
class OpNode:
    """
    A node in the computation graph representing a single operation.

    Categories:
      - ``compute``: arithmetic / activation / normalization ops
      - ``comm_collective``: all_reduce, all_gather, reduce_scatter, all_to_all, etc.
      - ``comm_p2p``: point-to-point send / recv (used by PP)
      - ``data_move``: cross-device copies, dtype conversions
      - ``memory``: allocation ops (empty, zeros, ones, ...)
    """

    node_id: str
    op_name: str
    op_type: str  # "compute" | "comm_collective" | "comm_p2p" | "data_move" | "memory"
    phase: str  # "forward" | "backward" | "optimizer"
    inputs: list[TensorMeta] = field(default_factory=list)
    outputs: list[TensorMeta] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    # Parallel context annotations
    pp_rank: int | None = None
    pp_stage: int | None = None
    microbatch_idx: int | None = None
    # For comm ops
    comm_op: str | None = None  # e.g. "all_reduce", "all_gather", "send", "recv"
    comm_group_size: int | None = None
    # Performance estimation (filled by CostModel)
    perf_result: PerfResult | None = None
    des_start_time_us: float | None = None
    des_finish_time_us: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "node_id": self.node_id,
            "op_name": self.op_name,
            "op_type": self.op_type,
            "phase": self.phase,
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "attrs": self.attrs,
            "pp_rank": self.pp_rank,
            "pp_stage": self.pp_stage,
            "microbatch_idx": self.microbatch_idx,
            "comm_op": self.comm_op,
            "comm_group_size": self.comm_group_size,
        }
        if self.perf_result is not None:
            d["perf_result"] = self.perf_result.to_dict()
        if self.des_start_time_us is not None:
            d["des_start_time_us"] = self.des_start_time_us
        if self.des_finish_time_us is not None:
            d["des_finish_time_us"] = self.des_finish_time_us
        return d


@dataclass
class DataEdge:
    """A directed edge encoding a data-flow dependency between two OpNodes."""

    src_node_id: str
    dst_node_id: str
    # "data" | "control" | "pp_p2p" (cross-stage pipeline send/recv)
    edge_type: str = "data"
    tensor_meta: TensorMeta | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src_node_id,
            "dst": self.dst_node_id,
            "type": self.edge_type,
            "tensor": self.tensor_meta.to_dict() if self.tensor_meta else None,
        }


@dataclass
class ComputeGraph:
    """
    Complete computation graph for one training step.

    Contains:
      - All operators (compute, comm, data-move, memory)
      - All data-flow edges between operators
      - Tensor shapes at every op boundary
      - Communication operator details (collective type, group size)

    ``nodes`` is an ordered ``dict`` keyed by ``node_id`` so that both
    index-based iteration and O(1) lookup by id are supported.
    """

    nodes: dict[str, OpNode] = field(default_factory=dict)
    edges: list[DataEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: OpNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: DataEdge) -> None:
        self.edges.append(edge)

    def fix_comm_phase_labels(self) -> None:
        """Re-label comm nodes whose data-flow predecessors are all in a
        different phase.

        In FSDP backward, ``reduce_scatter`` fires after backward
        computation finishes but may be recorded with ``phase="forward"``
        because the tracer's ``current_phase`` is set temporally.  This
        method corrects such mislabeling: if a comm node has *only*
        backward data-flow predecessors but is labeled ``"forward"``, it
        becomes ``"backward"`` (and vice versa).  Comm nodes with
        mixed-phase or no data-flow predecessors keep their original
        label.

        Must be called **before** ``add_phase_boundary_edges`` so the
        corrected labels are used for the sentinel fan-in.
        """
        COMM_TYPES = {"comm_collective", "comm_p2p"}
        data_pred_phases: dict[str, set[str]] = {}
        for edge in self.edges:
            if edge.edge_type == "data":
                src = self.nodes.get(edge.src_node_id)
                if src:
                    data_pred_phases.setdefault(edge.dst_node_id, set()).add(
                        src.phase or "unknown"
                    )

        for nid, node in self.nodes.items():
            if node.op_type not in COMM_TYPES:
                continue
            pred_phases = data_pred_phases.get(nid)
            if pred_phases is None or len(pred_phases) != 1:
                continue
            single_pred_phase = next(iter(pred_phases))
            if single_pred_phase != node.phase:
                node.phase = single_pred_phase

    def add_phase_boundary_edges(self) -> None:
        """Add control-flow edges between consecutive training phases.

        In PyTorch, ``loss.backward()`` cannot begin until the forward
        pass has completed.  The compute graph only captures data-flow
        edges, so backward nodes that have a forward data-flow
        predecessor can start as soon as *that single* predecessor
        finishes — even while other forward ops are still running.
        This method bridges the gap by inserting a synthetic
        ``phase_end`` sentinel node per phase boundary and connecting
        it to every node in the next phase, ensuring the next phase
        cannot start until the previous phase finishes.

        For each pair of consecutive phases (forward→backward,
        backward→optimizer):

        1.  Create a sentinel ``phase_end_{phase}`` node with
            ``op_type="phase_boundary"`` and zero duration.
        2.  Add a ``phase_boundary`` control edge from **every**
            node in the previous phase to the sentinel.
        3.  Add a ``phase_boundary`` control edge from the sentinel
            to **every** node in the next phase.

        The sentinel acts as a fan-in/fan-out junction: backward
        cannot start until *all* forward nodes have finished, not
        just the last one by trace order.
        """
        phase_order = ["forward", "backward", "optimizer"]
        node_ids = list(self.nodes.keys())

        cross_phase_preds: dict[str, set[str]] = {}
        for edge in self.edges:
            src_phase = self.nodes[edge.src_node_id].phase
            dst_phase = self.nodes[edge.dst_node_id].phase
            if src_phase != dst_phase and edge.edge_type != "phase_boundary":
                cross_phase_preds.setdefault(edge.dst_node_id, set()).add(
                    edge.src_node_id
                )

        for i in range(len(phase_order) - 1):
            prev_phase = phase_order[i]
            next_phase = phase_order[i + 1]

            prev_nodes = [
                nid
                for nid in node_ids
                if self.nodes[nid].phase == prev_phase and nid not in cross_phase_preds
            ]
            next_nodes = [
                nid for nid in node_ids if self.nodes[nid].phase == next_phase
            ]

            if not prev_nodes or not next_nodes:
                continue

            sentinel_id = f"phase_end_{prev_phase}"
            sentinel = OpNode(
                node_id=sentinel_id,
                op_name=f"phase_end_{prev_phase}",
                op_type="phase_boundary",
                phase=prev_phase,
                perf_result=PerfResult(
                    total_time_us=0.0,
                    metadata={"phase_boundary": True},
                ),
            )
            self.add_node(sentinel)

            for nid in prev_nodes:
                self.add_edge(
                    DataEdge(
                        src_node_id=nid,
                        dst_node_id=sentinel_id,
                        edge_type="phase_boundary",
                    )
                )

            for nid in next_nodes:
                self.add_edge(
                    DataEdge(
                        src_node_id=sentinel_id,
                        dst_node_id=nid,
                        edge_type="phase_boundary",
                    )
                )

    def summary(self) -> dict[str, int]:
        """Return op-type counts for a quick overview."""
        counts: dict[str, int] = {}
        for n in self.nodes.values():
            counts[n.op_type] = counts.get(n.op_type, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }


@dataclass
class ScheduleEvent:
    """
    A coarse-grained event in the training schedule.

    Examples: PP microbatch forward, PP microbatch backward,
    FSDP parameter allgather, FSDP gradient reduce-scatter, optimizer step.
    """

    event_id: str
    event_type: str
    rank: int
    pp_stage: int | None = None
    pp_rank: int | None = None
    microbatch_idx: int | None = None
    # Monotonically increasing logical clock; comparable within a single rank
    logical_clock: int = 0
    # Links to fine-grained OpNodes in the compute graph (populated by link_schedule_to_graph)
    op_node_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    des_start_time_us: float | None = None
    des_finish_time_us: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "rank": self.rank,
            "pp_stage": self.pp_stage,
            "pp_rank": self.pp_rank,
            "microbatch_idx": self.microbatch_idx,
            "logical_clock": self.logical_clock,
            "op_node_ids": self.op_node_ids,
            "metadata": self.metadata,
        }
        if self.des_start_time_us is not None:
            d["des_start_time_us"] = self.des_start_time_us
        if self.des_finish_time_us is not None:
            d["des_finish_time_us"] = self.des_finish_time_us
        return d


@dataclass
class ScheduleDep:
    """A dependency between two ScheduleEvents (event B cannot start before event A ends)."""

    from_event_id: str
    to_event_id: str
    # "data" | "control" | "pp_comm" | "fsdp_comm"
    dep_type: str = "data"

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_event_id,
            "to": self.to_event_id,
            "type": self.dep_type,
        }


@dataclass
class TrainingSchedule:
    """
    The training schedule: coarse-grained ordering and dependencies of events
    across a full training step (including all microbatches and all ranks).
    """

    events: list[ScheduleEvent] = field(default_factory=list)
    deps: list[ScheduleDep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: ScheduleEvent) -> None:
        self.events.append(event)

    def add_dep(self, dep: ScheduleDep) -> None:
        self.deps.append(dep)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "events": [e.to_dict() for e in self.events],
            "deps": [d.to_dict() for d in self.deps],
        }


@dataclass
class MemoryEvent:
    """
    A memory allocation or residency estimate.

    ``bytes`` is an estimate, not a device allocator measurement. Events can be
    produced from runtime tensor metadata, model parameters, communication
    buffers, or backend-specific semantic annotations.
    """

    event_id: str
    category: str
    bytes: int
    phase: str = "unknown"
    device: str = "unknown"
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    node_id: str | None = None
    lifetime_start: int | None = None
    lifetime_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "category": self.category,
            "bytes": self.bytes,
            "phase": self.phase,
            "device": self.device,
            "dtype": self.dtype,
            "shape": list(self.shape) if self.shape is not None else None,
            "node_id": self.node_id,
            "lifetime_start": self.lifetime_start,
            "lifetime_end": self.lifetime_end,
            "metadata": self.metadata,
        }


@dataclass
class SimulationResult:
    """Aggregated results from one simulation run."""

    compute_graph: ComputeGraph
    schedule: TrainingSchedule | None = None
    # Raw event lists from each interceptor (serializable dicts)
    comm_events: list[dict[str, Any]] = field(default_factory=list)
    fsdp_events: list[dict[str, Any]] = field(default_factory=list)
    pp_events: list[dict[str, Any]] = field(default_factory=list)
    memory_events: list[MemoryEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "compute_graph": self.compute_graph.to_dict(),
            "schedule": self.schedule.to_dict() if self.schedule is not None else None,
            "comm_events": self.comm_events,
            "fsdp_events": self.fsdp_events,
            "pp_events": self.pp_events,
            "memory_events": [e.to_dict() for e in self.memory_events],
        }

    def save_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
