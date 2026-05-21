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
from pathlib import Path
from dataclasses import asdict, dataclass, field
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

    def to_dict(self) -> dict[str, Any]:
        return {
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
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "rank": self.rank,
            "pp_stage": self.pp_stage,
            "pp_rank": self.pp_rank,
            "microbatch_idx": self.microbatch_idx,
            "logical_clock": self.logical_clock,
            "metadata": self.metadata,
        }


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
