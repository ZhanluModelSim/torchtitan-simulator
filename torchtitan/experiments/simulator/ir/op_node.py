# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L0 — OpNode projection.

Projects the captured :class:`~torchtitan.experiments.simulator.nodes.OpNode`
plus data-flow edges into a spec-aligned ``SpecOpNode`` whose cost fields
(``flops``/``peak_mem``/``param_mem``/``comm_bytes``) and adjacency
(``predecessors``/``successors``) are filled in.

This is a read-only projection: it never mutates the captured graph, so the
training abstraction stays derived from capture rather than re-implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..nodes import ComputeGraph, OpNode, TensorMeta

_DTYPE_BYTES: dict[str, int] = {
    "torch.float64": 8,
    "torch.float32": 4,
    "torch.float": 4,
    "torch.bfloat16": 2,
    "torch.float16": 2,
    "torch.half": 2,
    "torch.int64": 8,
    "torch.long": 8,
    "torch.int32": 4,
    "torch.int": 4,
    "torch.int16": 2,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.bool": 1,
}


def _dtype_bytes(dtype: str | None) -> int:
    if not dtype:
        return 4
    return _DTYPE_BYTES.get(dtype, 4)


def _tensor_bytes(meta: TensorMeta) -> int:
    n = 1
    for d in meta.shape:
        n *= int(d)
    return n * _dtype_bytes(meta.dtype)


@dataclass
class SpecOpNode:
    """Spec L0 node — a single operation with cost + adjacency fields."""

    op_id: str
    op_type: str
    phase: str
    flops: int = 0
    peak_mem: int = 0
    param_mem: int = 0
    comm_bytes: int = 0
    predecessors: list[str] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_id": self.op_id,
            "op_type": self.op_type,
            "phase": self.phase,
            "flops": self.flops,
            "peak_mem": self.peak_mem,
            "param_mem": self.param_mem,
            "comm_bytes": self.comm_bytes,
            "predecessors": self.predecessors,
            "successors": self.successors,
            "annotations": self.annotations,
        }


def _spec_op_type(node: OpNode) -> str:
    """Map captured op naming to the spec's canonical op name where obvious."""
    if node.comm_op:
        return node.comm_op
    return node.op_name.replace("aten.", "").replace(".default", "")


def project_op_node(
    node: OpNode,
    predecessors: list[str],
    successors: list[str],
) -> SpecOpNode:
    perf = node.perf_result
    flops = int(getattr(perf, "flops", 0) or 0) if perf is not None else 0

    peak_mem = sum(_tensor_bytes(o) for o in node.outputs)
    param_mem = sum(_tensor_bytes(i) for i in node.inputs if i.requires_grad)

    comm_bytes = 0
    if node.op_type in {"comm_collective", "comm_p2p"}:
        comm_bytes = sum(_tensor_bytes(o) for o in node.outputs) or sum(
            _tensor_bytes(i) for i in node.inputs
        )

    return SpecOpNode(
        op_id=node.node_id,
        op_type=_spec_op_type(node),
        phase=node.phase,
        flops=flops,
        peak_mem=peak_mem,
        param_mem=param_mem,
        comm_bytes=comm_bytes,
        predecessors=predecessors,
        successors=successors,
        annotations={
            "op_name": node.op_name,
            "pp_stage": node.pp_stage,
            "microbatch_idx": node.microbatch_idx,
            "comm_group_size": node.comm_group_size,
            "synthetic": bool((node.attrs or {}).get("synthetic", False)),
        },
    )


def build_adjacency(
    graph: ComputeGraph,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return (predecessors, successors) maps using only DATA edges."""
    preds: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
    succs: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
    for edge in graph.edges:
        if edge.edge_type != "data":
            continue
        if edge.src_node_id not in graph.nodes or edge.dst_node_id not in graph.nodes:
            continue
        succs[edge.src_node_id].append(edge.dst_node_id)
        preds[edge.dst_node_id].append(edge.src_node_id)
    return preds, succs


def project_op_nodes(graph: ComputeGraph) -> dict[str, SpecOpNode]:
    """Project every node in *graph* into a :class:`SpecOpNode`."""
    preds, succs = build_adjacency(graph)
    return {
        nid: project_op_node(node, preds[nid], succs[nid])
        for nid, node in graph.nodes.items()
    }
