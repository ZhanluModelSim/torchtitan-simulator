# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Static computation graph capture using ``torch.fx.make_fx`` +
``FakeTensorMode``.

``make_fx`` traces a model by recording every ATen-level operation (including
distributed collectives exposed via ``torch.ops._c10d_functional``) into an
``fx.GraphModule`` without allocating any real GPU memory.  The resulting
graph contains ``val`` metadata on every node with the FakeTensor shape and
dtype — exactly what we need to populate :class:`OpNode` and
:class:`TensorMeta`.

Supported models
----------------
- Plain ``nn.Module`` (eager)
- FSDP2 (``fully_shard``)-wrapped models
- TP-wrapped models (via DTensor)
- Models under ``activation_checkpoint`` (``torch.utils.checkpoint``)

Note: ``torch.compile``-decorated modules are handled by disabling
``error_on_nested_fx_trace`` so that make_fx inlines the compiled code.

Usage::

    forward_graph = capture_forward_fx(model, example_inputs)
    joint_graph   = capture_joint_fx(model, example_inputs, loss_fn, labels)
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from .nodes import ComputeGraph, DataEdge, OpNode, TensorMeta

# ---------------------------------------------------------------------------
# Op classification helpers (same logic as dispatch_interceptor, but operating
# on FX node targets rather than live function objects)
# ---------------------------------------------------------------------------

_COMM_MARKERS = (
    "_c10d_functional",
    "c10d_functional",
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "broadcast",
    "wait_tensor",
    "barrier",
)

_P2P_MARKERS = ("_send", "_recv")

_DATA_MOVE_MARKERS = ("_to_copy", "copy_")

_MEMORY_MARKERS = (
    "aten.empty",
    "aten.zeros",
    "aten.ones",
    "aten.full",
    "aten.arange",
)

_TRIVIAL_TARGETS = frozenset(
    [
        "aten.detach.default",
        "aten.detach_.default",
        "aten.alias.default",
        "aten.t.default",
        "aten.as_strided.default",
        "aten._unsafe_view.default",
        "aten.view.default",
        "aten.lift_fresh_copy.default",
        "aten.lift.default",
    ]
)

_COMM_OP_MAP: list[tuple[str, str]] = [
    ("reduce_scatter", "reduce_scatter"),
    ("all_gather", "all_gather"),
    ("all_reduce", "all_reduce"),
    ("all_to_all", "all_to_all"),
    ("broadcast", "broadcast"),
    ("wait_tensor", "wait"),
    ("barrier", "barrier"),
    ("_send", "send"),
    ("_recv", "recv"),
]


def _classify_fx_node(target: str) -> tuple[str, str | None]:
    """Return ``(op_type, comm_op_or_None)``."""
    if any(m in target for m in _P2P_MARKERS):
        for substr, canonical in _COMM_OP_MAP:
            if substr in target:
                return "comm_p2p", canonical
        return "comm_p2p", "p2p_unknown"

    if any(m in target for m in _COMM_MARKERS):
        for substr, canonical in _COMM_OP_MAP:
            if substr in target:
                return "comm_collective", canonical
        return "comm_collective", "collective_unknown"

    if any(m in target for m in _DATA_MOVE_MARKERS):
        return "data_move", None

    if any(target.startswith(m) or m in target for m in _MEMORY_MARKERS):
        return "memory", None

    return "compute", None


def _tensor_meta_from_val(val: Any) -> TensorMeta | None:
    """Extract TensorMeta from a FakeTensor (the ``val`` field of an FX node)."""
    if not isinstance(val, torch.Tensor):
        return None
    return TensorMeta(
        shape=tuple(val.shape),
        dtype=str(val.dtype),
        device=str(val.device),
        requires_grad=val.requires_grad,
    )


def _collect_input_metas(fx_node: fx.Node) -> list[TensorMeta]:
    metas: list[TensorMeta] = []
    for arg in fx_node.args:
        if isinstance(arg, fx.Node):
            val = arg.meta.get("val")
            m = _tensor_meta_from_val(val)
            if m is not None:
                metas.append(m)
    return metas


def _collect_output_metas(fx_node: fx.Node) -> list[TensorMeta]:
    val = fx_node.meta.get("val")
    metas: list[TensorMeta] = []
    if isinstance(val, torch.Tensor):
        m = _tensor_meta_from_val(val)
        if m is not None:
            metas.append(m)
    elif isinstance(val, (list, tuple)):
        for v in val:
            m = _tensor_meta_from_val(v)
            if m is not None:
                metas.append(m)
    return metas


# ---------------------------------------------------------------------------
# FX graph → ComputeGraph conversion
# ---------------------------------------------------------------------------


def fx_graph_to_compute_graph(
    gm: fx.GraphModule,
    phase: str = "forward",
    metadata: dict[str, Any] | None = None,
) -> ComputeGraph:
    """
    Convert an ``fx.GraphModule`` to a :class:`ComputeGraph`.

    Each ``call_function`` node becomes an :class:`OpNode`.  Data-flow edges
    are inserted for every node-argument relationship (the argument must also
    be a ``call_function`` node).

    Args:
        gm: The traced FX graph module.
        phase: Phase label (``"forward"``, ``"backward"``, ``"joint"``).
        metadata: Extra metadata dict to attach to the graph.

    Returns:
        A fully populated :class:`ComputeGraph`.
    """
    graph = ComputeGraph(metadata=metadata or {})
    counter: list[int] = [0]
    fx_name_to_node_id: dict[str, str] = {}

    for fx_node in gm.graph.nodes:
        if fx_node.op not in ("call_function", "call_method"):
            continue

        target = str(fx_node.target)
        if target in _TRIVIAL_TARGETS:
            continue

        counter[0] += 1
        node_id = f"fx_{counter[0]:07d}"
        fx_name_to_node_id[fx_node.name] = node_id

        op_type, comm_op = _classify_fx_node(target)
        input_metas = _collect_input_metas(fx_node)
        output_metas = _collect_output_metas(fx_node)

        attrs: dict[str, Any] = {}
        for i, arg in enumerate(fx_node.args):
            if isinstance(arg, (int, float, bool, str)):
                attrs[f"arg_{i}"] = arg
        for k, v in fx_node.kwargs.items():
            if isinstance(v, (int, float, bool, str)):
                attrs[k] = v

        node = OpNode(
            node_id=node_id,
            op_name=target,
            op_type=op_type,
            phase=phase,
            inputs=input_metas,
            outputs=output_metas,
            attrs=attrs,
            comm_op=comm_op,
        )
        graph.add_node(node)

        # Data-flow edges from FX argument nodes
        for arg in fx_node.args:
            if isinstance(arg, fx.Node) and arg.name in fx_name_to_node_id:
                edge = DataEdge(
                    src_node_id=fx_name_to_node_id[arg.name],
                    dst_node_id=node_id,
                    edge_type="data",
                )
                graph.add_edge(edge)

    return graph


# ---------------------------------------------------------------------------
# make_fx tracing helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _no_nested_compile():
    """Allow make_fx to trace through torch.compile'd functions."""
    prev = torch._dynamo.config.error_on_nested_fx_trace
    torch._dynamo.config.error_on_nested_fx_trace = False
    try:
        yield
    finally:
        torch._dynamo.config.error_on_nested_fx_trace = prev


def _fakeify_inputs(
    inputs: tuple[Any, ...],
    fake_mode: FakeTensorMode,
) -> tuple[Any, ...]:
    def _to_fake(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return fake_mode.from_tensor(x, static_shapes=True)
        return x

    flat, spec = pytree.tree_flatten(inputs)
    fake_flat = [_to_fake(x) for x in flat]
    return pytree.tree_unflatten(fake_flat, spec)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def capture_forward_fx(
    model: nn.Module,
    example_inputs: tuple[Any, ...],
    example_kwargs: dict[str, Any] | None = None,
    device: str = "cpu",
) -> ComputeGraph:
    """
    Trace the **forward** pass of *model* and return a :class:`ComputeGraph`.

    Uses ``make_fx`` with ``tracing_mode="fake"`` so no real computation or
    memory allocation occurs.

    Args:
        model: The model to trace (may be FSDP2- or TP-wrapped).
        example_inputs: Positional example inputs — real CPU tensors.
        example_kwargs: Optional keyword example inputs.
        device: Device hint for FakeTensor creation.

    Returns:
        :class:`ComputeGraph` with all forward ops, shapes, and comm nodes.
    """
    if example_kwargs is None:
        example_kwargs = {}

    with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
        fake_inputs = _fakeify_inputs(example_inputs, fake_mode)
        fake_kwargs = {
            k: (
                fake_mode.from_tensor(v, static_shapes=True)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in example_kwargs.items()
        }

        with _no_nested_compile():
            gm = make_fx(
                model,
                tracing_mode="fake",
                _allow_non_fake_inputs=True,
            )(*fake_inputs, **fake_kwargs)

    return fx_graph_to_compute_graph(
        gm,
        phase="forward",
        metadata={
            "trace_mode": "make_fx_fake",
            "capture": "forward",
        },
    )


def capture_joint_fx(
    model: nn.Module,
    example_inputs: tuple[Any, ...],
    loss_fn: Any | None = None,
    example_labels: torch.Tensor | None = None,
    example_kwargs: dict[str, Any] | None = None,
) -> ComputeGraph:
    """
    Trace the **joint** (forward + backward) pass using ``make_fx`` and
    ``torch.autograd.functional.vjp``.

    The backward graph is captured by wrapping ``model`` in a function that
    computes the forward pass and then calls ``loss.backward()``, then
    tracing through this combined function.

    Args:
        model: Model to trace.
        example_inputs: Positional inputs as CPU tensors.
        loss_fn: A callable ``(output, labels) -> scalar_loss``.
            If ``None``, uses ``output.sum()``.
        example_labels: Labels tensor for ``loss_fn``.
        example_kwargs: Optional keyword inputs.

    Returns:
        :class:`ComputeGraph` annotated with both ``"forward"`` and
        ``"backward"`` phases.  The phase boundaries are heuristic (nodes
        after the loss computation are labelled ``"backward"``).
    """
    if example_kwargs is None:
        example_kwargs = {}

    try:
        from torch.func import functional_call, grad
    except Exception as exc:
        raise RuntimeError(
            "joint fwd/bwd FX capture requires torch.func.functional_call and torch.func.grad"
        ) from exc

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable_names:
        raise RuntimeError(
            "joint fwd/bwd FX capture requires at least one trainable parameter"
        )

    def _loss_with_params(*param_values_and_inputs):
        n_params = len(trainable_names)
        param_values = param_values_and_inputs[:n_params]
        inputs = param_values_and_inputs[n_params:]
        param_dict = dict(zip(trainable_names, param_values))
        merged_state = {**buffers, **param_dict}
        out = functional_call(model, merged_state, inputs, kwargs=example_kwargs)
        if loss_fn is not None and example_labels is not None:
            return loss_fn(out, example_labels)
        if isinstance(out, torch.Tensor):
            return out.sum()
        flat_out, _ = pytree.tree_flatten(out)
        return sum(t.sum() for t in flat_out if isinstance(t, torch.Tensor))

    grad_fn = grad(_loss_with_params, argnums=tuple(range(len(trainable_names))))

    def _joint_fn(*all_args):
        loss = _loss_with_params(*all_args)
        grads = grad_fn(*all_args)
        if not isinstance(grads, tuple):
            grads = (grads,)
        return (loss, *grads)

    with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
        fake_trainable_params = _fakeify_inputs(
            tuple(params[name] for name in trainable_names),
            fake_mode,
        )
        fake_inputs = _fakeify_inputs(example_inputs, fake_mode)
        all_args = (*fake_trainable_params, *fake_inputs)

        with _no_nested_compile():
            gm = make_fx(
                _joint_fn,
                tracing_mode="fake",
                _allow_non_fake_inputs=True,
            )(*all_args)

    return fx_graph_to_compute_graph(
        gm,
        phase="joint",
        metadata={
            "trace_mode": "make_fx_fake",
            "capture": "joint_fwd_bwd",
        },
    )
