# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from pathlib import Path

from ..nodes import ComputeGraph

_DOT_COLORS: dict[str, str] = {
    "compute": "#AED6F1",
    "comm_collective": "#F9E79F",
    "comm_p2p": "#FAD7A0",
    "data_move": "#A9DFBF",
    "memory": "#D7BDE2",
    "unknown": "#D5D8DC",
}


def _node_color(op_type: str) -> str:
    return _DOT_COLORS.get(op_type, _DOT_COLORS["unknown"])


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
