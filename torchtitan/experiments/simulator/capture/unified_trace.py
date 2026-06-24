# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified trace capture mode that combines ``FakeTensorMode`` with
``TorchDispatchMode`` to record every tensor operation **without
allocating any real memory**.

Under ``FakeTensorMode``, every dispatched op produces shape-only outputs,
so model weight and activation tensors occupy zero bytes regardless of
model size.  The ``unified_trace`` context manager also optionally
intercepts ``torch.distributed`` communication operations and attaches
FSDP lifecycle hooks — all in a single pass.

Usage::

    recorder = TraceRecorder(rank=0)
    with unified_trace(recorder, model, example_inputs):
        output = model(*example_inputs)
    result = recorder.build_result()

The recorder supports phase tracking (forward / backward / optimizer),
microbatch annotation, and PP-stage labelling — all of which propagate
into the resulting :class:`OpNode` entries.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._subclasses import FakeTensorMode
from torch.library import register_fake
from torch.utils._python_dispatch import TorchDispatchMode

from ..nodes import (
    ComputeGraph,
    DataEdge,
    OpNode,
    ScheduleDep,
    ScheduleEvent,
    SimulationResult,
    TensorMeta,
    TrainingSchedule,
)
from ..op_classification import classify_op, TRIVIAL_TARGETS


# ---------------------------------------------------------------------------
# Meta / fake-kernel registration for ops without built-in impls
# ---------------------------------------------------------------------------


@register_fake("aten::bincount")
def meta_bincount(
    self: torch.Tensor, weights: torch.Tensor | None = None, minlength: int = 0
) -> torch.Tensor:
    """Fake (shape-only) implementation of ``torch.bincount``."""
    out_len = minlength
    if out_len == 0 and self.numel() > 0:
        out_len = self.shape[0]
    return self.new_empty(out_len, dtype=torch.long)


# ---------------------------------------------------------------------------
# Tensor metadata helpers
# ---------------------------------------------------------------------------


_DTensor: type | None = None
_DTensor_RESOLVED = False


def _resolve_dtensor() -> type | None:
    global _DTensor, _DTensor_RESOLVED
    if not _DTensor_RESOLVED:
        try:
            from torch.distributed.tensor import DTensor

            _DTensor = DTensor
        except ImportError:
            _DTensor = None
        _DTensor_RESOLVED = True
    return _DTensor


def _normalize_device(device_str: str) -> str:
    if device_str == "meta":
        return "cpu"
    return device_str


def _tensor_to_meta(t: torch.Tensor) -> TensorMeta:
    dtensor_cls = _resolve_dtensor()
    is_dtensor = dtensor_cls is not None and isinstance(t, dtensor_cls)
    placements = None
    if is_dtensor:
        placements = [str(p) for p in t.placements]  # pyrefly: ignore [missing-attribute]
    device = str(t.device)
    if device == "meta":
        device = "cpu"
    return TensorMeta(
        shape=tuple(t.shape),
        dtype=str(t.dtype),
        device=device,
        is_dtensor=is_dtensor,
        placements=placements,
        requires_grad=t.requires_grad,
    )


def _collect_all(
    args: Any, kwargs: Any
) -> tuple[list[TensorMeta], list[torch.Tensor]]:
    flat, _ = pytree.tree_flatten((args, kwargs))
    metas: list[TensorMeta] = []
    tensors: list[torch.Tensor] = []
    for item in flat:
        if isinstance(item, torch.Tensor):
            tensors.append(item)
            try:
                metas.append(_tensor_to_meta(item))
            except Exception:
                pass
    return metas, tensors


def _collect_output_all(
    output: Any,
) -> tuple[list[TensorMeta], list[torch.Tensor]]:
    flat, _ = pytree.tree_flatten(output)
    metas: list[TensorMeta] = []
    tensors: list[torch.Tensor] = []
    for item in flat:
        if isinstance(item, torch.Tensor):
            tensors.append(item)
            try:
                metas.append(_tensor_to_meta(item))
            except Exception:
                pass
    return metas, tensors


# ===========================================================================
# TraceRecorder — the single recorder for all capture channels
# ===========================================================================


class TraceRecorder:
    """Thread-safe container that accumulates :class:`OpNode` entries and
    data-flow edges during a unified trace session.

    Attributes:
        rank: Process rank (for multi-rank simulation).
        nodes: Ordered list of captured :class:`OpNode` entries.
        edges: Data-flow edge triples ``(src, dst, edge_type)``.
        current_phase: Mutable phase label (``\"forward\"``, ``\"backward\"``, etc.)
        current_pp_stage: Pipeline-parallel stage index.
        current_microbatch: Microbatch index within a gradient-accumulation cycle.
    """

    def __init__(self, rank: int = 0) -> None:
        self.rank = rank
        self._counter: int = 0
        self.nodes: list[OpNode] = []
        self.edges: list[tuple[str, str, str]] = []
        self._tensor_producer: dict[int, str] = {}
        self.current_phase: str = "forward"
        self.current_pp_stage: int | None = None
        self.current_pp_rank: int | None = None
        self.current_microbatch: int | None = None
        self.comm_events: list[dict[str, Any]] = []
        self.fsdp_events: list[dict[str, Any]] = []
        self.pp_events: list[dict[str, Any]] = []
        self._pp_deps: list[dict[str, Any]] = []

    def _next_id(self) -> str:
        self._counter += 1
        return f"ut_{self._counter:07d}"

    def record(
        self,
        func_name: str,
        op_type: str,
        comm_op: str | None,
        input_metas: list[TensorMeta],
        output_metas: list[TensorMeta],
        input_tensors: list[torch.Tensor],
        output_tensors: list[torch.Tensor],
        attrs: dict[str, Any] | None = None,
    ) -> OpNode:
        node_id = self._next_id()
        node = OpNode(
            node_id=node_id,
            op_name=func_name,
            op_type=op_type,
            phase=self.current_phase,
            inputs=input_metas,
            outputs=output_metas,
            attrs=attrs or {},
            pp_stage=self.current_pp_stage,
            pp_rank=self.current_pp_rank,
            microbatch_idx=self.current_microbatch,
            comm_op=comm_op,
        )
        seen: set[str] = set()
        tp = self._tensor_producer
        edges = self.edges
        for t in input_tensors:
            producer = tp.get(id(t))
            if producer is not None and producer not in seen:
                seen.add(producer)
                edges.append((producer, node_id, "data"))
        self.nodes.append(node)
        for t in output_tensors:
            tp[id(t)] = node_id
        return node

    def get_producer(self, tensor: torch.Tensor | None) -> str | None:
        if tensor is None:
            return None
        return self._tensor_producer.get(id(tensor))

    def set_producer(self, tensor: torch.Tensor | None, node_id: str) -> None:
        if tensor is None:
            return
        self._tensor_producer[id(tensor)] = node_id

    def build_result(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> SimulationResult:
        """Assemble captured data into a :class:`SimulationResult`.

        Edges from ``self.edges`` are used directly if available;
        otherwise, sequential edges within each ``(phase, pp_stage,
        microbatch_idx)`` group are inferred.  Communication events
        from ``self.comm_events`` are merged into the compute graph
        as :class:`OpNode` entries with data-flow edges to their
        source compute nodes.
        """
        graph = ComputeGraph(metadata=metadata or {})
        for n in self.nodes:
            graph.add_node(n)

        if self.edges:
            for src, dst, edge_type in self.edges:
                graph.add_edge(
                    DataEdge(src_node_id=src, dst_node_id=dst, edge_type=edge_type)
                )
        else:
            GroupKey = tuple
            last_in_group: dict[GroupKey, str] = {}
            for n in self.nodes:
                key = GroupKey((n.phase, n.pp_stage, n.microbatch_idx))
                prev_id = last_in_group.get(key)
                if prev_id is not None:
                    graph.add_edge(
                        DataEdge(
                            src_node_id=prev_id,
                            dst_node_id=n.node_id,
                            edge_type="sequential",
                        )
                    )
                last_in_group[key] = n.node_id

        # Merge comm events as OpNode entries
        for ev in self.comm_events:
            node_id = ev.get("event_id", f"comm_{len(graph.nodes)+1:07d}")
            op_name = ev.get("op", "collective_unknown")
            phase = ev.get("phase", "unknown")
            input_metas: list[TensorMeta] = []
            output_metas: list[TensorMeta] = []
            shape_entries = ev.get("tensor_shapes") or []
            if not shape_entries:
                tm = ev.get("tensor_meta")
                if tm:
                    shape_entries = [tm]
            for entry in shape_entries:
                if entry is None:
                    continue
                meta = TensorMeta(
                    shape=tuple(entry.get("shape", [])),
                    dtype=entry.get("dtype", "unknown"),
                    device=_normalize_device(entry.get("device", "cpu")),
                    is_dtensor=entry.get("is_dtensor", False),
                    placements=entry.get("placements"),
                )
                input_metas.append(meta)
                output_metas.append(meta)
            op_type = ev.get("op_type", "comm_collective")
            comm_node = OpNode(
                node_id=node_id,
                op_name=op_name,
                op_type=op_type,
                phase=phase,
                inputs=input_metas,
                outputs=output_metas,
                comm_op=op_name,
                comm_group_size=ev.get("group_size"),
                pp_stage=ev.get("pp_stage"),
                microbatch_idx=ev.get("microbatch"),
                attrs={
                    "group": str(ev.get("group", "")),
                    "tag": str(ev.get("tag", "")),
                    "src_rank": ev.get("src_rank"),
                    "dst_rank": ev.get("dst_rank"),
                    "rank": ev.get("rank"),
                },
            )
            graph.add_node(comm_node)
            for src_id in ev.get("source_node_ids", []):
                if src_id in graph.nodes:
                    graph.add_edge(
                        DataEdge(
                            src_node_id=src_id,
                            dst_node_id=node_id,
                            edge_type="data",
                        )
                    )

        # Build schedule from FSDP + PP events
        schedule = TrainingSchedule(metadata={"rank": self.rank})
        for ev in self.fsdp_events:
            schedule.add_event(
                ScheduleEvent(
                    event_id=ev["event_id"],
                    event_type=ev["event_type"],
                    rank=self.rank,
                    logical_clock=ev["logical_clock"],
                    metadata=ev.get("metadata", {}),
                )
            )
        for ev in self.pp_events:
            schedule.add_event(
                ScheduleEvent(
                    event_id=ev["event_id"],
                    event_type=ev["event_type"],
                    rank=ev.get("rank", self.rank),
                    pp_stage=ev.get("pp_stage"),
                    microbatch_idx=ev.get("microbatch"),
                    logical_clock=ev.get("logical_clock", 0),
                )
            )
        for dep in self._pp_deps:
            schedule.add_dep(ScheduleDep(dep["from"], dep["to"], dep["type"]))

        return SimulationResult(
            compute_graph=graph,
            schedule=schedule,
            comm_events=list(self.comm_events),
            fsdp_events=list(self.fsdp_events),
            pp_events=list(self.pp_events),
            metadata=metadata or {},
        )


# ===========================================================================
# UnifiedTraceMode — TorchDispatchMode for op-level capture
# ===========================================================================


class UnifiedTraceMode(TorchDispatchMode):
    """Intercepts every tensor operation dispatched through PyTorch's
    dispatcher and records it into a :class:`TraceRecorder`.

    Works in both eager and ``FakeTensorMode`` contexts.  When used with
    a ``FakeTensorMode``, tensors are shape-only and no real memory is
    allocated.

    Device strings ``\"meta\"`` in ``TensorMeta`` are normalised to
    ``\"cpu\"`` so that downstream cost-model and export tools remain
    compatible.
    """

    def __init__(self, recorder: TraceRecorder) -> None:
        super().__init__()
        self.recorder = recorder

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        func_name = str(func)

        if func_name in TRIVIAL_TARGETS:
            return func(*args, **kwargs)

        result = func(*args, **kwargs)

        input_metas, input_tensors = _collect_all(args, kwargs)
        output_metas, output_tensors = _collect_output_all(result)

        op_type, comm_op = classify_op(func_name)

        attrs: dict[str, Any] | None = None
        for i, arg in enumerate(args):
            if isinstance(arg, (int, float, bool, str)):
                if attrs is None:
                    attrs = {}
                attrs[f"arg_{i}"] = arg

        self.recorder.record(
            func_name,
            op_type,
            comm_op,
            input_metas,
            output_metas,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs=attrs,
        )
        return result


# ===========================================================================
# Recorder stack (used by comm interceptor to find the active TraceRecorder)
# ===========================================================================


_RECORDER_STACK: list[TraceRecorder] = []


def get_current_recorder() -> TraceRecorder | None:
    """Return the innermost active :class:`TraceRecorder`, or ``None``."""
    return _RECORDER_STACK[-1] if _RECORDER_STACK else None


# ===========================================================================
# CommRecorder — distributed communication interceptor
# ===========================================================================


class CommRecorder:
    """Thread-safe recorder for distributed communication events.

    Intercepts ``torch.distributed`` collectives and P2P operations,
    recording tensor metadata, group sizes, and source node references
    from the active :class:`TraceRecorder`.
    """

    def __init__(self, rank: int = 0) -> None:
        self._lock = threading.Lock()
        self._counter: int = 0
        self.events: list[dict[str, Any]] = []
        self.rank: int = rank
        self.logical_clock: int = 0
        self.current_pp_stage: int | None = None
        self.current_microbatch: int | None = None
        self.current_phase: str = "forward"

    def _next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"comm_{self._counter:07d}"

    def _meta_or_none(self, t: Any) -> dict[str, Any] | None:
        if t is None or not isinstance(t, torch.Tensor):
            return None
        try:
            return TensorMeta.from_tensor(t).to_dict()
        except Exception:
            return None

    def _tensor_ids(self, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            return [id(value)]
        if isinstance(value, (list, tuple)):
            out: list[int] = []
            for v in value:
                if isinstance(v, torch.Tensor):
                    out.append(id(v))
            return out
        return []

    def _group_size(self, group: Any) -> int:
        try:
            if group is None:
                return dist.get_world_size()
            return dist.get_world_size(group)
        except Exception:
            return -1

    def record_collective(
        self,
        op: str,
        tensor: Any,
        group: Any,
        *,
        output_tensor: Any = None,
        **extra: Any,
    ) -> dict[str, Any]:
        input_tensor_ids = self._tensor_ids(tensor)
        output_tensor_ids = self._tensor_ids(output_tensor)
        source_node_ids: list[str] = []
        recorder = get_current_recorder()
        if recorder is not None:
            seen = set()
            for t in [tensor] if isinstance(tensor, torch.Tensor) else []:
                producer = recorder.get_producer(t)
                if producer is not None and producer not in seen:
                    source_node_ids.append(producer)
                    seen.add(producer)
            if isinstance(tensor, (list, tuple)):
                for t in tensor:
                    if isinstance(t, torch.Tensor):
                        producer = recorder.get_producer(t)
                        if producer is not None and producer not in seen:
                            source_node_ids.append(producer)
                            seen.add(producer)
        event: dict[str, Any] = {
            "event_id": self._next_id(),
            "op": op,
            "tensor_meta": self._meta_or_none(tensor),
            "group_size": self._group_size(group),
            "rank": self.rank,
            "pp_stage": self.current_pp_stage,
            "microbatch": self.current_microbatch,
            "phase": self.current_phase,
            "logical_clock": self.logical_clock,
            "input_tensor_ids": input_tensor_ids,
            "output_tensor_ids": output_tensor_ids,
            "source_node_ids": source_node_ids,
            **extra,
        }
        with self._lock:
            self.events.append(event)
            self.logical_clock += 1
        return event

    def record_p2p(
        self,
        op: str,
        tensor: Any,
        peer: int,
        group: Any,
        tag: int = 0,
    ) -> dict[str, Any]:
        input_tensor_ids = self._tensor_ids(tensor)
        source_node_ids: list[str] = []
        recorder = get_current_recorder()
        if recorder is not None and isinstance(tensor, torch.Tensor):
            producer = recorder.get_producer(tensor)
            if producer is not None:
                source_node_ids.append(producer)
        event: dict[str, Any] = {
            "event_id": self._next_id(),
            "op": op,
            "tensor_meta": self._meta_or_none(tensor),
            "peer": peer,
            "group_size": self._group_size(group),
            "rank": self.rank,
            "pp_stage": self.current_pp_stage,
            "microbatch": self.current_microbatch,
            "phase": self.current_phase,
            "tag": tag,
            "logical_clock": self.logical_clock,
            "input_tensor_ids": input_tensor_ids,
            "output_tensor_ids": [],
            "source_node_ids": source_node_ids,
        }
        with self._lock:
            self.events.append(event)
            self.logical_clock += 1
        return event


# ---------------------------------------------------------------------------
# Functional-collectives intercept
# ---------------------------------------------------------------------------


def _try_patch_functional_collectives(
    recorder: CommRecorder,
) -> list[tuple[Any, str, Any]]:
    """Patch ``torch.distributed._functional_collectives`` used by FSDP/DTensor."""
    try:
        import torch.distributed._functional_collectives as funcol
    except ImportError:
        return []

    saved: list[tuple[Any, str, Any]] = []

    def _wrap(orig_fn: Any, op_name: str, is_p2p: bool = False) -> Any:
        def wrapper(tensor: Any, *args: Any, **kwargs: Any) -> Any:
            group = kwargs.get("group") or (args[0] if args else None)
            try:
                if is_p2p:
                    peer = args[0] if args else kwargs.get("dst", kwargs.get("src", -1))
                    recorder.record_p2p(op_name, tensor, peer, group)
                else:
                    recorder.record_collective(op_name, tensor, group)
            except Exception:
                pass
            return orig_fn(tensor, *args, **kwargs)

        return wrapper

    patches = [
        ("all_reduce", "all_reduce", False),
        ("all_gather_tensor", "all_gather", False),
        ("reduce_scatter_tensor", "reduce_scatter", False),
        ("all_to_all_single", "all_to_all", False),
        ("broadcast", "broadcast", False),
        ("wait_tensor", "wait_tensor", False),
    ]

    for attr, op_name, is_p2p in patches:
        orig = getattr(funcol, attr, None)
        if orig is not None:
            saved.append((funcol, attr, orig))
            setattr(funcol, attr, _wrap(orig, op_name, is_p2p))

    return saved


# ---------------------------------------------------------------------------
# capture_comms context manager
# ---------------------------------------------------------------------------


@contextmanager
def capture_comms(
    recorder: CommRecorder,
) -> Generator[CommRecorder, None, None]:
    """Context manager that patches ``torch.distributed`` and
    ``torch.distributed._functional_collectives`` to record every
    communication operation into *recorder*.
    """
    # Save originals
    orig_all_reduce = dist.all_reduce
    orig_all_gather = dist.all_gather
    orig_all_gather_into_tensor = dist.all_gather_into_tensor
    orig_reduce_scatter = dist.reduce_scatter
    orig_reduce_scatter_tensor = dist.reduce_scatter_tensor
    orig_all_to_all = dist.all_to_all
    orig_all_to_all_single = dist.all_to_all_single
    orig_send = dist.send
    orig_recv = dist.recv
    orig_isend = dist.isend
    orig_irecv = dist.irecv
    orig_broadcast = dist.broadcast
    orig_barrier = dist.barrier

    # Patched versions
    def _all_reduce(tensor: Any, op: Any = dist.ReduceOp.SUM, group: Any = None, async_op: bool = False) -> Any:
        recorder.record_collective(
            "all_reduce", tensor, group, reduce_op=str(op), async_op=async_op
        )
        return orig_all_reduce(tensor, op=op, group=group, async_op=async_op)

    def _all_gather(tensor_list: Any, tensor: Any, group: Any = None, async_op: bool = False) -> Any:
        recorder.record_collective("all_gather", tensor, group, async_op=async_op)
        return orig_all_gather(tensor_list, tensor, group=group, async_op=async_op)

    def _all_gather_into_tensor(output_tensor: Any, input_tensor: Any, group: Any = None, async_op: bool = False) -> Any:
        ev = recorder.record_collective(
            "all_gather_into_tensor",
            input_tensor,
            group,
            output_tensor=output_tensor,
            output_shape=list(output_tensor.shape),
            async_op=async_op,
        )
        out = orig_all_gather_into_tensor(
            output_tensor, input_tensor, group=group, async_op=async_op
        )
        active = get_current_recorder()
        if active is not None:
            active.set_producer(output_tensor, ev["event_id"])
        return out

    def _reduce_scatter(output: Any, input_list: Any, op: Any = dist.ReduceOp.SUM, group: Any = None, async_op: bool = False) -> Any:
        tensor = input_list[0] if input_list else None
        ev = recorder.record_collective(
            "reduce_scatter",
            tensor,
            group,
            output_tensor=output,
            reduce_op=str(op),
            async_op=async_op,
        )
        out = orig_reduce_scatter(
            output, input_list, op=op, group=group, async_op=async_op
        )
        active = get_current_recorder()
        if active is not None:
            active.set_producer(output, ev["event_id"])
        return out

    def _reduce_scatter_tensor(output: Any, input_tensor: Any, op: Any = dist.ReduceOp.SUM, group: Any = None, async_op: bool = False) -> Any:
        ev = recorder.record_collective(
            "reduce_scatter_tensor",
            input_tensor,
            group,
            output_tensor=output,
            output_shape=list(output.shape),
            reduce_op=str(op),
            async_op=async_op,
        )
        out = orig_reduce_scatter_tensor(
            output, input_tensor, op=op, group=group, async_op=async_op
        )
        active = get_current_recorder()
        if active is not None:
            active.set_producer(output, ev["event_id"])
        return out

    def _all_to_all(output_tensor_list: Any, input_tensor_list: Any, group: Any = None, async_op: bool = False) -> Any:
        tensor = input_tensor_list[0] if input_tensor_list else None
        output_tensor = output_tensor_list[0] if output_tensor_list else None
        ev = recorder.record_collective(
            "all_to_all",
            tensor,
            group,
            output_tensor=output_tensor,
            async_op=async_op,
        )
        out = orig_all_to_all(
            output_tensor_list, input_tensor_list, group=group, async_op=async_op
        )
        active = get_current_recorder()
        if active is not None:
            for t in output_tensor_list:
                active.set_producer(t, ev["event_id"])
        return out

    def _all_to_all_single(output: Any, input: Any, *args: Any, group: Any = None, async_op: bool = False, **kwargs: Any) -> Any:
        ev = recorder.record_collective(
            "all_to_all_single",
            input,
            group,
            output_tensor=output,
            async_op=async_op,
        )
        out = orig_all_to_all_single(
            output, input, *args, group=group, async_op=async_op, **kwargs
        )
        active = get_current_recorder()
        if active is not None:
            active.set_producer(output, ev["event_id"])
        return out

    def _send(tensor: Any, dst: int, group: Any = None, tag: int = 0) -> Any:
        recorder.record_p2p("send", tensor, dst, group, tag=tag)
        return orig_send(tensor, dst, group=group, tag=tag)

    def _recv(tensor: Any, src: int | None = None, group: Any = None, tag: int = 0) -> Any:
        ev = recorder.record_p2p(
            "recv", tensor, src if src is not None else -1, group, tag=tag
        )
        out = orig_recv(tensor, src=src, group=group, tag=tag)
        active = get_current_recorder()
        if active is not None:
            active.set_producer(tensor, ev["event_id"])
        return out

    def _isend(tensor: Any, dst: int, group: Any = None, tag: int = 0) -> Any:
        recorder.record_p2p("isend", tensor, dst, group, tag=tag)
        return orig_isend(tensor, dst, group=group, tag=tag)

    def _irecv(tensor: Any, src: int | None = None, group: Any = None, tag: int = 0) -> Any:
        recorder.record_p2p(
            "irecv", tensor, src if src is not None else -1, group, tag=tag
        )
        return orig_irecv(tensor, src=src, group=group, tag=tag)

    def _broadcast(tensor: Any, src: int = 0, group: Any = None, async_op: bool = False) -> Any:
        recorder.record_collective(
            "broadcast", tensor, group, src=src, async_op=async_op
        )
        return orig_broadcast(tensor, src=src, group=group, async_op=async_op)

    def _barrier(group: Any = None, async_op: bool = False, device_ids: Any = None) -> Any:
        recorder.record_collective("barrier", None, group, async_op=async_op)
        return orig_barrier(group=group, async_op=async_op, device_ids=device_ids)

    # Apply patches
    dist.all_reduce = _all_reduce
    dist.all_gather = _all_gather
    dist.all_gather_into_tensor = _all_gather_into_tensor
    dist.reduce_scatter = _reduce_scatter
    dist.reduce_scatter_tensor = _reduce_scatter_tensor
    dist.all_to_all = _all_to_all
    dist.all_to_all_single = _all_to_all_single
    dist.send = _send
    dist.recv = _recv
    dist.isend = _isend
    dist.irecv = _irecv
    dist.broadcast = _broadcast
    dist.barrier = _barrier

    funcol_saved = _try_patch_functional_collectives(recorder)

    try:
        yield recorder
    finally:
        dist.all_reduce = orig_all_reduce
        dist.all_gather = orig_all_gather
        dist.all_gather_into_tensor = orig_all_gather_into_tensor
        dist.reduce_scatter = orig_reduce_scatter
        dist.reduce_scatter_tensor = orig_reduce_scatter_tensor
        dist.all_to_all = orig_all_to_all
        dist.all_to_all_single = orig_all_to_all_single
        dist.send = orig_send
        dist.recv = orig_recv
        dist.isend = orig_isend
        dist.irecv = orig_irecv
        dist.broadcast = orig_broadcast
        dist.barrier = orig_barrier

        for mod, attr, orig_fn in funcol_saved:
            setattr(mod, attr, orig_fn)


# ===========================================================================
# FSDPEventRecorder — FSDP lifecycle hooks
# ===========================================================================


class FSDPEventRecorder:
    """Thread-safe recorder for FSDP parameter lifecycle events."""

    def __init__(self, rank: int = 0) -> None:
        self._lock = threading.Lock()
        self._counter: int = 0
        self.events: list[dict[str, Any]] = []
        self.rank: int = rank
        self.logical_clock: int = 0
        self.current_phase: str = "forward"

    def _next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"fsdp_{self._counter:07d}"

    def record(
        self,
        event_type: str,
        module_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "event_id": self._next_id(),
            "event_type": event_type,
            "module_name": module_name,
            "rank": self.rank,
            "phase": self.current_phase,
            "logical_clock": self.logical_clock,
            "metadata": metadata or {},
        }
        with self._lock:
            self.events.append(event)
            self.logical_clock += 1
        return event


# ---------------------------------------------------------------------------
# FSDP hook factories
# ---------------------------------------------------------------------------


def _fsdp_fwd_pre_hook(recorder: FSDPEventRecorder, module_name: str) -> Any:
    def hook(module: nn.Module, input: Any) -> None:
        recorder.record(
            "fsdp_allgather_pre_fwd",
            module_name,
            {"action": "allgather_params"},
        )

    return hook


def _fsdp_fwd_post_hook(recorder: FSDPEventRecorder, module_name: str) -> Any:
    def hook(module: nn.Module, input: Any, output: Any) -> None:
        recorder.record(
            "fsdp_reshard_post_fwd",
            module_name,
            {"action": "reshard_params"},
        )

    return hook


def _fsdp_bwd_pre_hook(recorder: FSDPEventRecorder, module_name: str) -> Any:
    def hook(module: nn.Module, grad_output: Any) -> None:
        recorder.record(
            "fsdp_allgather_pre_bwd",
            module_name,
            {"action": "allgather_params_for_bwd"},
        )

    return hook


def _fsdp_bwd_post_hook(recorder: FSDPEventRecorder, module_name: str) -> Any:
    def hook(module: nn.Module, grad_input: Any, grad_output: Any) -> None:
        recorder.record(
            "fsdp_reduce_scatter_post_bwd",
            module_name,
            {"action": "reduce_scatter_grads"},
        )

    return hook


@contextmanager
def capture_fsdp_events(
    model: nn.Module,
    recorder: FSDPEventRecorder,
) -> Generator[FSDPEventRecorder, None, None]:
    """Register FSDP lifecycle hooks on every ``FSDPModule`` in *model*."""
    try:
        from torch.distributed._composable.fsdp import FSDPModule
    except ImportError:
        yield recorder
        return

    handles: list[Any] = []
    for name, module in model.named_modules():
        if isinstance(module, FSDPModule):
            handles.append(
                module.register_forward_pre_hook(_fsdp_fwd_pre_hook(recorder, name))
            )
            handles.append(
                module.register_forward_hook(_fsdp_fwd_post_hook(recorder, name))
            )
            handles.append(
                module.register_full_backward_pre_hook(_fsdp_bwd_pre_hook(recorder, name))
            )
            handles.append(
                module.register_full_backward_hook(_fsdp_bwd_post_hook(recorder, name))
            )

    try:
        yield recorder
    finally:
        for h in handles:
            h.remove()


# ===========================================================================
# unified_trace — the main context manager
# ===========================================================================


@contextmanager
def unified_trace(
    recorder: TraceRecorder,
    model: torch.nn.Module | None = None,
    example_inputs: tuple[Any, ...] | None = None,
    use_fake_mode: bool = True,
    phase: str = "forward",
    capture_comm: bool = False,
    capture_fsdp: bool = True,
    model_parts: list[torch.nn.Module] | None = None,
) -> Generator[TraceRecorder, None, None]:
    """Context manager that activates :class:`UnifiedTraceMode` and
    optionally a :class:`FakeTensorMode` for shape-only tracing.

    When ``capture_comm=True``, also activates :class:`CommRecorder` to
    intercept distributed communication operations.  When
    ``capture_fsdp=True`` and ``model_parts`` is provided, attaches
    FSDP lifecycle hooks to every :class:`FSDPModule` found.

    Args:
        recorder: Target recorder to write into.
        model: Optional model to trace (for ``use_fake_mode=True``).
        example_inputs: Optional example inputs (for ``use_fake_mode=True``).
        use_fake_mode: If ``True``, wrap in a ``FakeTensorMode`` so that
            all tensors are shape-only and no memory is allocated.
        phase: Initial phase annotation.
        capture_comm: If ``True``, activate :class:`CommRecorder` to
            intercept ``torch.distributed`` comm ops.  Required for
            gloo backend mode.
        capture_fsdp: If ``True``, attach FSDP lifecycle hooks.
        model_parts: List of model modules for FSDP hook attachment.

    Yields:
        The same ``recorder`` instance for convenience.
    """
    recorder.current_phase = phase
    _RECORDER_STACK.append(recorder)

    comm_recorder = None
    fsdp_recorder = None

    with contextlib.ExitStack() as stack:
        if use_fake_mode:
            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
            stack.enter_context(fake_mode)
        stack.enter_context(UnifiedTraceMode(recorder))

        if capture_comm:
            comm_recorder = CommRecorder(rank=recorder.rank)
            comm_recorder.current_phase = phase
            stack.enter_context(capture_comms(comm_recorder))

        if capture_fsdp and model_parts:
            fsdp_recorder = FSDPEventRecorder(rank=recorder.rank)
            fsdp_recorder.current_phase = phase
            for m in model_parts:
                stack.enter_context(capture_fsdp_events(m, fsdp_recorder))

        yield recorder

    # Transfer comm/FSDP events to the TraceRecorder after context exits
    if capture_comm and comm_recorder is not None:
        recorder.comm_events = list(comm_recorder.events)
    if capture_fsdp and fsdp_recorder is not None:
        recorder.fsdp_events = list(fsdp_recorder.events)

    _RECORDER_STACK.pop()
