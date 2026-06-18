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


AGGREGATE_MEMORY_KEYS = {"total_event_bytes", "by_category", "by_phase", "by_device"}


def memory_metadata_without_aggregates(
    metadata: dict[str, Any] | None
) -> dict[str, Any]:
    if not metadata:
        return {}
    return {
        key: value
        for key, value in metadata.items()
        if key not in AGGREGATE_MEMORY_KEYS
    }


def finalize_memory_summary(
    events: list[MemoryEvent],
    *summaries: dict[str, Any],
    existing_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = merge_memory_summary(
        summarize_memory_events(events),
        memory_metadata_without_aggregates(existing_metadata),
        *summaries,
    )
    return merged


def build_runtime_memory(
    graph: ComputeGraph,
    comm_events: list[dict[str, Any]],
    *,
    existing_metadata: dict[str, Any] | None = None,
) -> tuple[list[MemoryEvent], dict[str, Any]]:
    graph_memory_events, graph_memory_summary = estimate_graph_memory(graph)
    comm_memory_events = estimate_comm_memory(comm_events)
    comm_memory_summary = merge_memory_summary(
        graph_memory_summary,
        {
            "total_event_bytes": sum(e.bytes for e in comm_memory_events),
            "by_category": {
                "comm_event_buffer": sum(e.bytes for e in comm_memory_events)
            },
        },
    )
    comm_memory_summary["graph_peak_live_bytes"] = graph_memory_summary.get(
        "peak_live_bytes", 0
    )
    memory_summary = finalize_memory_summary(
        [*graph_memory_events, *comm_memory_events],
        comm_memory_summary,
        existing_metadata=existing_metadata,
    )
    return [*graph_memory_events, *comm_memory_events], memory_summary


def attach_model_state_memory(
    result: Any,
    model_parts: list[Any],
    *,
    optimizer_name: str | None = None,
    parallelism_config: Any | None = None,
) -> dict[str, Any]:
    """Attach model state memory events to result.
    
    Args:
        result: SimulationResult to attach memory events to
        model_parts: List of model parts
        optimizer_name: Optimizer name (currently unused)
        parallelism_config: ParallelismConfig with PP/TP/FSDP/EP degrees
    """
    model_memory_events, model_memory_summary = estimate_model_state_memory(
        model_parts,
        optimizer_name=optimizer_name,
        parallelism_config=parallelism_config,
    )
    result.memory_events.extend(model_memory_events)
    result.metadata["memory"] = finalize_memory_summary(
        result.memory_events,
        model_memory_summary,
        existing_metadata=result.metadata.get("memory"),
    )
    return model_memory_summary


def estimate_graph_memory(
    graph: ComputeGraph,
) -> tuple[list[MemoryEvent], dict[str, Any]]:
    """
    Estimate activation/output memory from graph node outputs and data edges.

    Lifetimes are approximated by node order: an output is live from the
    producing node until the last observed consumer edge. This is conservative
    for Python eager execution and intentionally separate from allocator-level
    peak measurements.
    """
    node_ids = list(graph.nodes.keys())
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    last_consumer: dict[str, int] = {
        node_id: node_index[node_id] for node_id in node_ids
    }
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
    parallelism_config: Any | None = None,
) -> tuple[list[MemoryEvent], dict[str, Any]]:
    """Estimate model state memory (parameters, gradients, optimizer states).
    
    Args:
        model_parts: List of model parts (for PP, this is the local stage's layers)
        optimizer_name: Optimizer name (currently unused, assumes Adam-style)
        parallelism_config: ParallelismConfig with PP/TP/FSDP/EP degrees
    
    Returns:
        Tuple of (memory_events, summary_dict)
        
    Memory estimation accounts for parallelism sharding:
    - PP: Each GPU holds 1/PP of layers (already reflected in model_parts)
    - TP: Each GPU holds 1/TP of sharded weights
    - FSDP: Parameters sharded across dp_shard GPUs
    - EP: MoE experts distributed across EP GPUs (not modeled here)
    
    The memory events contain per-GPU bytes (after sharding), while the summary
    contains both whole-model and per-GPU values for reference.
    """
    import torch

    del optimizer_name
    next_id = _event_counter("mem_model")
    events: list[MemoryEvent] = []

    # Extract parallelism degrees
    tp_degree = 1
    fsdp_degree = 1
    if parallelism_config is not None:
        tp_degree = getattr(parallelism_config, "tensor_parallel_degree", 1) or 1
        dp_shard = getattr(parallelism_config, "data_parallel_shard_degree", 1) or 1
        fsdp_degree = max(1, dp_shard)

    # Sharding factor for parameters/gradients/optimizer states
    shard_factor = max(1, tp_degree * fsdp_degree)

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
            # Create per-GPU memory event (sharded)
            per_gpu_nbytes = nbytes // shard_factor
            events.append(
                MemoryEvent(
                    event_id=next_id(),
                    category="parameter",
                    bytes=per_gpu_nbytes,
                    phase="model_state",
                    device=str(param.device),
                    dtype=str(param.dtype),
                    shape=tuple(param.shape),
                    metadata={
                        "part_idx": part_idx,
                        "name": name,
                        "whole_model_bytes": nbytes,
                        "shard_factor": shard_factor,
                    },
                )
            )

    # Calculate per-GPU memory after parallelism sharding
    per_gpu_param_bytes = param_bytes // shard_factor
    per_gpu_grad_bytes = grad_bytes // shard_factor
    per_gpu_optimizer_bytes = optimizer_state_bytes // shard_factor

    if grad_bytes:
        events.append(
            MemoryEvent(
                event_id=next_id(),
                category="gradient",
                bytes=per_gpu_grad_bytes,
                phase="backward",
                metadata={
                    "estimate": "one gradient tensor per trainable parameter",
                    "whole_model_bytes": grad_bytes,
                    "shard_factor": shard_factor,
                },
            )
        )
    if optimizer_state_bytes:
        events.append(
            MemoryEvent(
                event_id=next_id(),
                category="optimizer_state",
                bytes=per_gpu_optimizer_bytes,
                phase="optimizer",
                metadata={
                    "estimate": "Adam/AdamW exp_avg + exp_avg_sq",
                    "whole_model_bytes": optimizer_state_bytes,
                    "shard_factor": shard_factor,
                },
            )
        )

    summary = {
        # Whole-model memory
        "parameter_bytes": param_bytes,
        "gradient_bytes": grad_bytes,
        "optimizer_state_bytes": optimizer_state_bytes,
        "model_state_total_bytes": param_bytes + grad_bytes + optimizer_state_bytes,
        # Per-GPU memory after parallelism sharding
        "per_gpu_parameter_bytes": per_gpu_param_bytes,
        "per_gpu_gradient_bytes": per_gpu_grad_bytes,
        "per_gpu_optimizer_state_bytes": per_gpu_optimizer_bytes,
        "per_gpu_model_state_bytes": per_gpu_param_bytes + per_gpu_grad_bytes + per_gpu_optimizer_bytes,
        # Parallelism info for reference
        "tp_degree": tp_degree,
        "fsdp_degree": fsdp_degree,
        "shard_factor": shard_factor,
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
