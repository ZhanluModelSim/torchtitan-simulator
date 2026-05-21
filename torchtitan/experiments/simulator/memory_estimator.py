# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .nodes import ComputeGraph, MemoryEvent, TensorMeta


_DTYPE_SIZES: dict[str, int] = {
    "torch.bool": 1,
    "torch.uint8": 1,
    "torch.int8": 1,
    "torch.float8_e4m3fn": 1,
    "torch.float8_e5m2": 1,
    "torch.int16": 2,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.int32": 4,
    "torch.float32": 4,
    "torch.int64": 8,
    "torch.float64": 8,
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
}


def dtype_size(dtype: str | None) -> int:
    if not dtype:
        return 0
    dtype = str(dtype)
    return _DTYPE_SIZES.get(dtype, _DTYPE_SIZES.get(dtype.replace("torch.", ""), 0))


def tensor_nbytes(meta: TensorMeta | dict[str, Any] | None) -> int:
    if meta is None:
        return 0
    if isinstance(meta, TensorMeta):
        shape = meta.shape
        dtype = meta.dtype
    else:
        shape = tuple(meta.get("shape", []) or [])
        dtype = meta.get("dtype")
    if not shape:
        return dtype_size(dtype)
    if any(dim is None or int(dim) < 0 for dim in shape):
        return 0
    return int(math.prod(int(dim) for dim in shape)) * dtype_size(dtype)


def _event_counter(prefix: str):
    count = 0

    def next_id() -> str:
        nonlocal count
        count += 1
        return f"{prefix}_{count:07d}"

    return next_id


def estimate_graph_memory(graph: ComputeGraph) -> tuple[list[MemoryEvent], dict[str, Any]]:
    """
    Estimate activation/output memory from graph node outputs and data edges.

    Lifetimes are approximated by node order: an output is live from the
    producing node until the last observed consumer edge. This is conservative
    for Python eager execution and intentionally separate from allocator-level
    peak measurements.
    """
    node_ids = list(graph.nodes.keys())
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    last_consumer: dict[str, int] = {node_id: node_index[node_id] for node_id in node_ids}
    for edge in graph.edges:
        if edge.src_node_id in node_index and edge.dst_node_id in node_index:
            last_consumer[edge.src_node_id] = max(
                last_consumer[edge.src_node_id], node_index[edge.dst_node_id]
            )

    next_id = _event_counter("mem_graph")
    events: list[MemoryEvent] = []
    for node_id, node in graph.nodes.items():
        start = node_index[node_id]
        end = last_consumer.get(node_id, start)
        category = "comm_buffer" if node.op_type.startswith("comm") else "activation"
        if node.op_type == "memory":
            category = "allocation"
        elif node.op_type == "data_move":
            category = "data_move"
        for output_idx, meta in enumerate(node.outputs):
            nbytes = tensor_nbytes(meta)
            if nbytes <= 0:
                continue
            events.append(
                MemoryEvent(
                    event_id=next_id(),
                    category=category,
                    bytes=nbytes,
                    phase=node.phase or "unknown",
                    device=meta.device,
                    dtype=meta.dtype,
                    shape=meta.shape,
                    node_id=node_id,
                    lifetime_start=start,
                    lifetime_end=end,
                    metadata={
                        "op_name": node.op_name,
                        "op_type": node.op_type,
                        "output_idx": output_idx,
                    },
                )
            )

    summary = summarize_memory_events(events)
    summary["peak_live_bytes"] = peak_live_bytes(events)
    return events, summary


def estimate_comm_memory(comm_events: list[dict[str, Any]]) -> list[MemoryEvent]:
    next_id = _event_counter("mem_comm")
    events: list[MemoryEvent] = []
    for ev in comm_events:
        total = 0
        shapes = ev.get("tensor_shapes", []) or []
        for shape_meta in shapes:
            total += tensor_nbytes(shape_meta)
        if total <= 0:
            continue
        events.append(
            MemoryEvent(
                event_id=next_id(),
                category="comm_event_buffer",
                bytes=total,
                phase=ev.get("phase", "unknown"),
                device="unknown",
                node_id=ev.get("event_id"),
                metadata={
                    "op": ev.get("op"),
                    "rank": ev.get("rank"),
                    "group_size": ev.get("group_size"),
                },
            )
        )
    return events


def estimate_model_state_memory(
    model_parts: list[Any],
    *,
    optimizer_name: str | None = None,
) -> tuple[list[MemoryEvent], dict[str, Any]]:
    import torch

    del optimizer_name
    next_id = _event_counter("mem_model")
    events: list[MemoryEvent] = []

    param_bytes = 0
    grad_bytes = 0
    optimizer_state_bytes = 0
    for part_idx, model in enumerate(model_parts):
        for name, param in model.named_parameters():
            if not isinstance(param, torch.Tensor):
                continue
            nbytes = int(param.numel() * param.element_size())
            param_bytes += nbytes
            if param.requires_grad:
                grad_bytes += nbytes
                # Adam/AdamW-style first and second moments. This is a
                # steady-state training estimate; lazy state creation may occur
                # after the first optimizer step.
                optimizer_state_bytes += nbytes * 2
            events.append(
                MemoryEvent(
                    event_id=next_id(),
                    category="parameter",
                    bytes=nbytes,
                    phase="model_state",
                    device=str(param.device),
                    dtype=str(param.dtype),
                    shape=tuple(param.shape),
                    metadata={"part_idx": part_idx, "name": name},
                )
            )

    if grad_bytes:
        events.append(
            MemoryEvent(
                event_id=next_id(),
                category="gradient",
                bytes=grad_bytes,
                phase="backward",
                metadata={"estimate": "one gradient tensor per trainable parameter"},
            )
        )
    if optimizer_state_bytes:
        events.append(
            MemoryEvent(
                event_id=next_id(),
                category="optimizer_state",
                bytes=optimizer_state_bytes,
                phase="optimizer",
                metadata={"estimate": "Adam/AdamW exp_avg + exp_avg_sq"},
            )
        )

    summary = {
        "parameter_bytes": param_bytes,
        "gradient_bytes": grad_bytes,
        "optimizer_state_bytes": optimizer_state_bytes,
        "model_state_total_bytes": param_bytes + grad_bytes + optimizer_state_bytes,
    }
    return events, summary


def summarize_memory_events(events: list[MemoryEvent]) -> dict[str, Any]:
    by_category: dict[str, int] = defaultdict(int)
    by_phase: dict[str, int] = defaultdict(int)
    by_device: dict[str, int] = defaultdict(int)
    for event in events:
        by_category[event.category] += int(event.bytes)
        by_phase[event.phase] += int(event.bytes)
        by_device[event.device] += int(event.bytes)
    return {
        "total_event_bytes": sum(int(event.bytes) for event in events),
        "by_category": dict(sorted(by_category.items())),
        "by_phase": dict(sorted(by_phase.items())),
        "by_device": dict(sorted(by_device.items())),
    }


def peak_live_bytes(events: list[MemoryEvent]) -> int:
    deltas: dict[int, int] = defaultdict(int)
    for event in events:
        if event.lifetime_start is None or event.lifetime_end is None:
            continue
        deltas[event.lifetime_start] += int(event.bytes)
        deltas[event.lifetime_end + 1] -= int(event.bytes)
    live = 0
    peak = 0
    for idx in sorted(deltas):
        live += deltas[idx]
        peak = max(peak, live)
    return peak


def merge_memory_summary(*summaries: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "total_event_bytes": 0,
        "by_category": {},
        "by_phase": {},
        "by_device": {},
    }
    for summary in summaries:
        merged["total_event_bytes"] += int(summary.get("total_event_bytes", 0))
        for key in ("by_category", "by_phase", "by_device"):
            target = merged[key]
            for name, value in (summary.get(key, {}) or {}).items():
                target[name] = target.get(name, 0) + int(value)
        for key, value in summary.items():
            if key not in merged:
                merged[key] = value
    return merged
