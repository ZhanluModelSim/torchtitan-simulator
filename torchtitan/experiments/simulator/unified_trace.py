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

from .nodes import (
    ComputeGraph,
    DataEdge,
    OpNode,
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
        microbatch_idx)`` group are inferred.
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

        return SimulationResult(
            compute_graph=graph,
            schedule=TrainingSchedule(metadata={"rank": self.rank}),
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


_RECORDER_STACK: list[TraceRecorder] = []


def get_current_recorder() -> TraceRecorder | None:
    """Return the innermost active :class:`TraceRecorder`, or ``None``."""
    return _RECORDER_STACK[-1] if _RECORDER_STACK else None


@contextmanager
def unified_trace(
    recorder: TraceRecorder,
    model: torch.nn.Module | None = None,
    example_inputs: tuple[Any, ...] | None = None,
    use_fake_mode: bool = True,
    phase: str = "forward",
) -> Generator[TraceRecorder, None, None]:
    """Context manager that activates :class:`UnifiedTraceMode` and
    optionally a :class:`FakeTensorMode` for shape-only tracing.

    Args:
        recorder: Target recorder to write into.
        model: Optional model to trace (for ``use_fake_mode=True``).
        example_inputs: Optional example inputs (for ``use_fake_mode=True``).
        use_fake_mode: If ``True``, wrap in a ``FakeTensorMode`` so that
            all tensors are shape-only and no memory is allocated.
        phase: Initial phase annotation.

    Yields:
        The same ``recorder`` instance for convenience.
    """
    recorder.current_phase = phase
    _RECORDER_STACK.append(recorder)

    with contextlib.ExitStack() as stack:
        if use_fake_mode:
            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
            stack.enter_context(fake_mode)
        stack.enter_context(UnifiedTraceMode(recorder))
        yield recorder

    _RECORDER_STACK.pop()
