# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L1 — StepGraph.

A bounded computation DAG for one forward / backward / optimizer step.

``StepBuilder`` partitions the *captured* :class:`ComputeGraph` by the
``phase`` label — which itself was produced by autograd / optimizer hooks
during capture — into per-step templates.  It never re-implements
torchtitan's forward/backward logic; it only projects what was captured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..nodes import ComputeGraph
from .op_node import project_op_node, SpecOpNode

_STEP_TYPES = ("forward", "backward", "optimizer")


@dataclass
class StepGraph:
    """Spec L1 — a bounded computation unit (one step's DAG template)."""

    step_id: str
    step_type: str
    nodes: dict[str, SpecOpNode] = field(default_factory=dict)
    entry_nodes: list[str] = field(default_factory=list)
    exit_nodes: list[str] = field(default_factory=list)
    tensor_lifetimes: dict[str, int] = field(default_factory=dict)
    total_flops: int = 0
    peak_active_mem: int = 0
    param_mem: int = 0
    comm_volume: int = 0
    is_acyclic: bool = True
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "entry_nodes": self.entry_nodes,
            "exit_nodes": self.exit_nodes,
            "tensor_lifetimes": self.tensor_lifetimes,
            "total_flops": self.total_flops,
            "peak_active_mem": self.peak_active_mem,
            "param_mem": self.param_mem,
            "comm_volume": self.comm_volume,
            "is_acyclic": self.is_acyclic,
            "annotations": self.annotations,
        }


def _kahn_is_acyclic(node_ids: set[str], edges: list[tuple[str, str]]) -> bool:
    indeg = {nid: 0 for nid in node_ids}
    adj: dict[str, list[str]] = {nid: [] for nid in node_ids}
    for src, dst in edges:
        if src in node_ids and dst in node_ids:
            indeg_dst = indeg.get(dst)
            if indeg_dst is None:
                continue
            adj[src].append(dst)
            indeg[dst] += 1
    queue = [nid for nid, d in indeg.items() if d == 0]
    visited = 0
    while queue:
        nid = queue.pop()
        visited += 1
        for nxt in adj[nid]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    return visited == len(node_ids)


class StepBuilder:
    """Build per-phase :class:`StepGraph` templates from a captured graph."""

    @staticmethod
    def from_compute_graph(graph: ComputeGraph) -> dict[str, StepGraph]:
        # Partition node ids by captured phase.
        by_phase: dict[str, list[str]] = {st: [] for st in _STEP_TYPES}
        for nid, node in graph.nodes.items():
            if node.phase in by_phase:
                by_phase[node.phase].append(nid)

        steps: dict[str, StepGraph] = {}
        for step_type, nids in by_phase.items():
            if not nids:
                continue
            steps[step_type] = StepBuilder._build_one(graph, step_type, nids)
        return steps

    @staticmethod
    def _build_one(graph: ComputeGraph, step_type: str, nids: list[str]) -> StepGraph:
        node_id_set = set(nids)

        # Intra-partition DATA edges only.
        intra_edges: list[tuple[str, str]] = []
        indeg: dict[str, int] = {nid: 0 for nid in nids}
        outdeg: dict[str, int] = {nid: 0 for nid in nids}
        for edge in graph.edges:
            if edge.edge_type != "data":
                continue
            s, d = edge.src_node_id, edge.dst_node_id
            if s in node_id_set and d in node_id_set:
                intra_edges.append((s, d))
                outdeg[s] += 1
                indeg[d] += 1

        preds: dict[str, list[str]] = {nid: [] for nid in nids}
        succs: dict[str, list[str]] = {nid: [] for nid in nids}
        for s, d in intra_edges:
            succs[s].append(d)
            preds[d].append(s)

        spec_nodes: dict[str, SpecOpNode] = {}
        total_flops = 0
        peak_active_mem = 0
        param_mem = 0
        comm_volume = 0
        for nid in nids:
            spec = project_op_node(graph.nodes[nid], preds[nid], succs[nid])
            spec_nodes[nid] = spec
            total_flops += spec.flops
            peak_active_mem += spec.peak_mem
            param_mem += spec.param_mem
            comm_volume += spec.comm_bytes

        entry_nodes = [nid for nid in nids if indeg[nid] == 0]
        exit_nodes = [nid for nid in nids if outdeg[nid] == 0]

        # Tensor lifetimes: producer topo-index → last-consumer topo-index span.
        topo_index = {nid: i for i, nid in enumerate(nids)}
        tensor_lifetimes: dict[str, int] = {}
        for nid in nids:
            consumers = succs[nid]
            if consumers:
                last = max(topo_index[c] for c in consumers)
                tensor_lifetimes[nid] = last - topo_index[nid]
            else:
                tensor_lifetimes[nid] = 0

        is_acyclic = _kahn_is_acyclic(node_id_set, intra_edges)

        return StepGraph(
            step_id=f"step_{step_type}",
            step_type=step_type,
            nodes=spec_nodes,
            entry_nodes=entry_nodes,
            exit_nodes=exit_nodes,
            tensor_lifetimes=tensor_lifetimes,
            total_flops=total_flops,
            peak_active_mem=peak_active_mem,
            param_mem=param_mem,
            comm_volume=comm_volume,
            is_acyclic=is_acyclic,
            annotations={"node_count": len(nids)},
        )
