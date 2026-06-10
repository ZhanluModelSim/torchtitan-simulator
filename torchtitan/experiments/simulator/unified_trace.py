# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified trace capture mode that combines ``FakeTensorMode`` with
``TorchDispatchMode`` to record every tensor operation **without
allocating any real memory**.

This replaces the previous three-level capture architecture (FX tracing,
runtime dispatch interception, and synthetic comm injection) with a
single pass that produces the same :class:`ComputeGraph` data
structure.  Under ``FakeTensorMode``, every dispatched op produces
shape-only outputs, so model weight and activation tensors occupy zero
bytes regardless of model size.

Usage::

    recorder = TraceRecorder(rank=0)
    with unified_trace(recorder, model, example_inputs):
        output = model(*example_inputs)
    result = recorder.build_result()

The recorder also supports phase tracking (forward / backward /
optimizer), microbatch annotation, and PP-stage labelling — all of
which propagate into the resulting :class:`OpNode` entries.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._subclasses import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode

from ._recorder_registry import pop_recorder, push_recorder

from .graph_assembler import comm_event_to_op_node
from .nodes import (
    ComputeGraph,
    DataEdge,
    OpNode,
    ScheduleDep,
    ScheduleEvent,
    SimulationResult,
    TensorMeta,
    TrainingSchedule,
)
from .op_classification import classify_op, TRIVIAL_TARGETS


def _normalize_device(device_str: str) -> str:
    """Map ``\"meta\"`` → ``\"cpu\"`` for output ``TensorMeta`` compatibility."""
    if device_str == "meta":
        return "cpu"
    return device_str


def _collect_tensor_metas(args: Any, kwargs: Any) -> list[TensorMeta]:
    """Extract ``TensorMeta`` from a pytree of args/kwargs, normalising device."""
    flat, _ = pytree.tree_flatten((args, kwargs))
    metas: list[TensorMeta] = []
    for item in flat:
        if isinstance(item, torch.Tensor):
            try:
                tm = TensorMeta.from_tensor(item)
                tm.device = _normalize_device(tm.device)
                metas.append(tm)
            except Exception:
                pass
    return metas


def _collect_input_tensors(args: Any, kwargs: Any) -> list[torch.Tensor]:
    flat, _ = pytree.tree_flatten((args, kwargs))
    return [item for item in flat if isinstance(item, torch.Tensor)]


def _collect_output_tensors(output: Any) -> list[torch.Tensor]:
    flat, _ = pytree.tree_flatten(output)
    return [item for item in flat if isinstance(item, torch.Tensor)]


def _collect_output_metas(output: Any) -> list[TensorMeta]:
    flat, _ = pytree.tree_flatten(output)
    metas: list[TensorMeta] = []
    for item in flat:
        if isinstance(item, torch.Tensor):
            try:
                tm = TensorMeta.from_tensor(item)
                tm.device = _normalize_device(tm.device)
                metas.append(tm)
            except Exception:
                pass
    return metas


def compute_loss(
    output: Any,
    loss_fn: Any | None = None,
    labels: Any | None = None,
) -> torch.Tensor:
    if loss_fn is not None and labels is not None:
        return loss_fn(output, labels)
    if isinstance(output, torch.Tensor):
        return output.sum()
    flat, _ = pytree.tree_flatten(output)
    return sum(t.sum() for t in flat if isinstance(t, torch.Tensor))


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
        self._lock = threading.Lock()
        self._counter: int = 0
        self.nodes: list[OpNode] = []
        self.edges: list[tuple[str, str, str]] = []
        self._tensor_producer: dict[int, str] = {}
        self.current_phase: str = "forward"
        self.current_pp_stage: int | None = None
        self.current_microbatch: int | None = None
        self.comm_events: list[dict[str, Any]] = []
        self.fsdp_events: list[dict[str, Any]] = []
        self.pp_events: list[dict[str, Any]] = []
        self._pp_deps: list[dict[str, Any]] = []

    def _next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"ut_{self._counter:07d}"

    def record(
        self,
        func: Any,
        input_metas: list[TensorMeta],
        output_metas: list[TensorMeta],
        input_tensors: list[torch.Tensor],
        output_tensors: list[torch.Tensor],
        attrs: dict[str, Any] | None = None,
    ) -> OpNode:
        """Record a dispatched op and return the corresponding :class:`OpNode`."""
        func_name = str(func)
        op_type, comm_op = classify_op(func_name)

        node = OpNode(
            node_id=self._next_id(),
            op_name=func_name,
            op_type=op_type,
            phase=self.current_phase,
            inputs=input_metas,
            outputs=output_metas,
            attrs=attrs or {},
            pp_stage=self.current_pp_stage,
            microbatch_idx=self.current_microbatch,
            comm_op=comm_op,
        )
        with self._lock:
            input_producers = set()
            for t in input_tensors:
                producer = self._tensor_producer.get(id(t))
                if producer is not None:
                    input_producers.add(producer)
            for producer in sorted(input_producers):
                self.edges.append((producer, node.node_id, "data"))

            self.nodes.append(node)
            for t in output_tensors:
                self._tensor_producer[id(t)] = node.node_id
        return node

    def get_producer(self, tensor: torch.Tensor | None) -> str | None:
        if tensor is None:
            return None
        with self._lock:
            return self._tensor_producer.get(id(tensor))

    def set_producer(self, tensor: torch.Tensor | None, node_id: str) -> None:
        if tensor is None:
            return
        with self._lock:
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

        for ev in self.comm_events:
            normalized_ev = dict(ev)
            if "tensor_shapes" in normalized_ev:
                normalized_ev["tensor_shapes"] = [
                    {**s, "device": _normalize_device(s.get("device", "cpu"))}
                    if s
                    else s
                    for s in (normalized_ev.get("tensor_shapes") or [])
                ]
            if "tensor_meta" in normalized_ev and normalized_ev["tensor_meta"]:
                tm = dict(normalized_ev["tensor_meta"])
                tm["device"] = _normalize_device(tm.get("device", "cpu"))
                normalized_ev["tensor_meta"] = tm

            node_id = normalized_ev.get("event_id", f"comm_{len(graph.nodes) + 1:07d}")
            comm_node = comm_event_to_op_node(normalized_ev, node_id=node_id)
            graph.add_node(comm_node)
            for src_id in normalized_ev.get("source_node_ids", []):
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


class UnifiedTraceMode(TorchDispatchMode):
    """Intercepts every tensor operation dispatched through PyTorch's
    dispatcher and records it into a :class:`TraceRecorder`.

    This mode works in both eager and ``FakeTensorMode`` contexts.
    When used with a ``FakeTensorMode``, tensors are shape-only and
    no real memory is allocated — making it suitable for simulating
    arbitrarily large models on a CPU-only host.

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

        result = func(*args, **kwargs)

        if func_name in TRIVIAL_TARGETS:
            return result

        input_metas = _collect_tensor_metas(args, kwargs)
        output_metas = _collect_output_metas(result)
        input_tensors = _collect_input_tensors(args, kwargs)
        output_tensors = _collect_output_tensors(result)

        attrs: dict[str, Any] = {}
        for i, arg in enumerate(args):
            if isinstance(arg, (int, float, bool, str)):
                attrs[f"arg_{i}"] = arg

        self.recorder.record(
            func,
            input_metas,
            output_metas,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs=attrs,
        )
        return result


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
            gloo backend mode; skipped for fake_backend since comm
            ops won't exist.
        capture_fsdp: If ``True``, attach FSDP lifecycle hooks.
        model_parts: List of model modules for FSDP hook attachment.

    Yields:
        The same ``recorder`` instance for convenience.
    """
    recorder.current_phase = phase
    push_recorder(recorder)

    comm_recorder = None
    fsdp_recorder = None

    with contextlib.ExitStack() as stack:
        if use_fake_mode:
            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
            stack.enter_context(fake_mode)
        stack.enter_context(UnifiedTraceMode(recorder))

        if capture_comm:
            from .comm_interceptor import capture_comms, CommRecorder

            comm_recorder = CommRecorder(rank=recorder.rank)
            comm_recorder.current_phase = phase
            stack.enter_context(capture_comms(comm_recorder))

        if capture_fsdp and model_parts:
            from .fsdp_tracer import capture_fsdp_events, FSDPEventRecorder

            fsdp_recorder = FSDPEventRecorder(rank=recorder.rank)
            fsdp_recorder.current_phase = phase
            for m in model_parts:
                stack.enter_context(capture_fsdp_events(m, fsdp_recorder))

        yield recorder

    # Transfer comm/FSDP events to the TraceRecorder after context exits
    if capture_comm and comm_recorder is not None:
        recorder.comm_events = list(comm_recorder.events)
        # Update source_node_ids to reference TraceRecorder's node IDs
        # (CommRecorder uses OpRecorder IDs via get_current_recorder,
        # which already resolves to TraceRecorder since it's on the stack)
    if capture_fsdp and fsdp_recorder is not None:
        recorder.fsdp_events = list(fsdp_recorder.events)

    pop_recorder()
