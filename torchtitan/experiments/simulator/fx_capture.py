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
from .op_classification import classify_op, TRIVIAL_TARGETS


def comm_event_to_op_node(
    ev: dict[str, Any],
    node_id: str | None = None,
    phase_override: str | None = None,
) -> OpNode:
    nid = node_id or ev.get("event_id", "comm_unknown")
    op_name = ev.get("op", "collective_unknown")
    phase = phase_override or ev.get("phase", "unknown")

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
            device=entry.get("device", "cpu"),
            is_dtensor=entry.get("is_dtensor", False),
            placements=entry.get("placements"),
        )
        input_metas.append(meta)
        output_metas.append(meta)

    return OpNode(
        node_id=nid,
        op_name=op_name,
        op_type=ev.get("op_type", "comm_collective"),
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


def merge_comm_events(
    graph: ComputeGraph,
    comm_events: list[dict[str, Any]],
    phase_override: str | None = None,
) -> ComputeGraph:
    for ev in comm_events:
        node_id = ev.get("event_id", f"comm_{len(graph.nodes) + 1:07d}")
        node = comm_event_to_op_node(ev, node_id=node_id, phase_override=phase_override)
        graph.add_node(node)
        for src_id in ev.get("source_node_ids", []):
            if src_id in graph.nodes:
                graph.add_edge(
                    DataEdge(
                        src_node_id=src_id,
                        dst_node_id=node.node_id,
                        edge_type="data",
                    )
                )
    return graph


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
        if target in TRIVIAL_TARGETS:
            continue

        counter[0] += 1
        node_id = f"fx_{counter[0]:07d}"
        fx_name_to_node_id[fx_node.name] = node_id

        op_type, comm_op = classify_op(target)
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
