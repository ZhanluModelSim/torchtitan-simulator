# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from torchtitan.tools.logging import logger

from .nodes import DataEdge, OpNode, TensorMeta


def inject_synthetic_comm_events(
    result: Any,
    trainer: Any,
    sim_opts: Any,
) -> None:
    if getattr(sim_opts, "comm_backend", "") == "gloo":
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_backend() == dist.Backend.GLOO:
            return

    graph = result.compute_graph
    parallelism = trainer.config.parallelism
    model_parts = trainer.model_parts

    from .schedule_inject import read_parallelism_degrees

    par = read_parallelism_degrees(trainer.config)
    tp = par.tp
    ds = par.dp_shard
    pp = par.pp

    if not (tp > 1 or ds > 1):
        return

    total_param_numel = 0
    per_module_numel: dict[str, int] = {}
    for part in model_parts:
        for name, param in part.named_parameters():
            if param.requires_grad:
                nel = param.numel()
                total_param_numel += nel
                prefix = ".".join(name.split(".")[:2])
                per_module_numel[prefix] = per_module_numel.get(prefix, 0) + nel

    from torchtitan.config import TORCH_DTYPE_MAP

    mp_param = getattr(trainer.config.training, "mixed_precision_param", "bfloat16")
    torch_dtype = TORCH_DTYPE_MAP.get(mp_param, torch.bfloat16)
    dtype_str = str(torch_dtype)
    logger.info(
        "Injecting synthetic comm events: tp=%d ds=%d dtype=%s total_param_numel=%d",
        tp,
        ds,
        dtype_str,
        total_param_numel,
    )

    shard_numel = total_param_numel // ds if ds > 1 else total_param_numel

    counter = [len(graph.nodes)]

    def _next_id() -> str:
        counter[0] += 1
        return f"comm_syn_{counter[0]:07d}"

    def _find_last_compute_node_id(phase: str) -> str | None:
        for nid in reversed(list(graph.nodes.keys())):
            n = graph.nodes[nid]
            if n.phase == phase and n.op_type == "compute":
                return nid
        return None

    if ds > 1:
        num_layers = infer_num_layers(model_parts)
        per_layer_numel = shard_numel // max(num_layers, 1)
        full_layer_numel = per_layer_numel * ds

        fwd_anchor = _find_last_compute_node_id("forward")
        bwd_anchor = _find_last_compute_node_id("backward")

        for i in range(num_layers):
            node = OpNode(
                node_id=_next_id(),
                op_name="all_gather",
                op_type="comm_collective",
                phase="forward",
                inputs=[
                    TensorMeta(shape=(per_layer_numel,), dtype=dtype_str, device="cpu")
                ],
                outputs=[
                    TensorMeta(shape=(full_layer_numel,), dtype=dtype_str, device="cpu")
                ],
                comm_op="all_gather",
                comm_group_size=ds,
                attrs={"synthetic": True},
            )
            graph.add_node(node)
            if fwd_anchor:
                graph.add_edge(DataEdge(fwd_anchor, node.node_id, "sequential"))
            result.comm_events.append(
                {
                    "event_id": node.node_id,
                    "op": "all_gather",
                    "group_size": ds,
                    "phase": "forward",
                    "tensor_meta": {
                        "shape": [per_layer_numel],
                        "dtype": dtype_str,
                        "device": "cpu",
                    },
                    "source_node_ids": [fwd_anchor] if fwd_anchor else [],
                    "synthetic": True,
                }
            )

        for i in range(num_layers):
            node = OpNode(
                node_id=_next_id(),
                op_name="reduce_scatter",
                op_type="comm_collective",
                phase="backward",
                inputs=[
                    TensorMeta(shape=(full_layer_numel,), dtype=dtype_str, device="cpu")
                ],
                outputs=[
                    TensorMeta(shape=(per_layer_numel,), dtype=dtype_str, device="cpu")
                ],
                comm_op="reduce_scatter",
                comm_group_size=ds,
                attrs={"synthetic": True},
            )
            graph.add_node(node)
            if bwd_anchor:
                graph.add_edge(DataEdge(bwd_anchor, node.node_id, "sequential"))
            result.comm_events.append(
                {
                    "event_id": node.node_id,
                    "op": "reduce_scatter",
                    "group_size": ds,
                    "phase": "backward",
                    "tensor_meta": {
                        "shape": [full_layer_numel],
                        "dtype": dtype_str,
                        "device": "cpu",
                    },
                    "source_node_ids": [bwd_anchor] if bwd_anchor else [],
                    "synthetic": True,
                }
            )

    if tp > 1:
        seq_len = trainer.config.training.seq_len
        batch_size = trainer.config.training.local_batch_size
        hidden = guess_hidden_dim(model_parts[0])
        act_numel = batch_size * seq_len * hidden
        num_layers = infer_num_layers(model_parts)
        tp_allreduce_count = num_layers * 2

        fwd_anchor = _find_last_compute_node_id("forward")
        bwd_anchor = _find_last_compute_node_id("backward")

        for _ in range(tp_allreduce_count):
            node = OpNode(
                node_id=_next_id(),
                op_name="all_reduce",
                op_type="comm_collective",
                phase="forward",
                inputs=[TensorMeta(shape=(act_numel,), dtype=dtype_str, device="cpu")],
                outputs=[TensorMeta(shape=(act_numel,), dtype=dtype_str, device="cpu")],
                comm_op="all_reduce",
                comm_group_size=tp,
                attrs={"synthetic": True},
            )
            graph.add_node(node)
            if fwd_anchor:
                graph.add_edge(DataEdge(fwd_anchor, node.node_id, "sequential"))
            result.comm_events.append(
                {
                    "event_id": node.node_id,
                    "op": "all_reduce",
                    "group_size": tp,
                    "phase": "forward",
                    "tensor_meta": {
                        "shape": [batch_size, seq_len, hidden],
                        "dtype": dtype_str,
                        "device": "cpu",
                    },
                    "source_node_ids": [fwd_anchor] if fwd_anchor else [],
                    "synthetic": True,
                }
            )

            node = OpNode(
                node_id=_next_id(),
                op_name="all_reduce",
                op_type="comm_collective",
                phase="backward",
                inputs=[TensorMeta(shape=(act_numel,), dtype=dtype_str, device="cpu")],
                outputs=[TensorMeta(shape=(act_numel,), dtype=dtype_str, device="cpu")],
                comm_op="all_reduce",
                comm_group_size=tp,
                attrs={"synthetic": True},
            )
            graph.add_node(node)
            if bwd_anchor:
                graph.add_edge(DataEdge(bwd_anchor, node.node_id, "sequential"))
            result.comm_events.append(
                {
                    "event_id": node.node_id,
                    "op": "all_reduce",
                    "group_size": tp,
                    "phase": "backward",
                    "tensor_meta": {
                        "shape": [batch_size, seq_len, hidden],
                        "dtype": dtype_str,
                        "device": "cpu",
                    },
                    "source_node_ids": [bwd_anchor] if bwd_anchor else [],
                    "synthetic": True,
                }
            )


def infer_num_layers(model_parts: list[Any]) -> int:
    for part in model_parts:
        config = getattr(part, "config", None)
        if config is not None:
            n = getattr(config, "n_layers", None)
            if n is not None:
                return n
        n = getattr(part, "n_layers", None)
        if n is not None:
            return n
        layers_attr = getattr(part, "layers", None)
        if layers_attr is not None and isinstance(
            layers_attr, (list, tuple, nn.Sequential, nn.ModuleList)
        ):
            return len(layers_attr)
    per_module_numel: dict[str, int] = {}
    for part in model_parts:
        for name, param in part.named_parameters():
            if param.requires_grad:
                prefix = ".".join(name.split(".")[:2])
                per_module_numel[prefix] = (
                    per_module_numel.get(prefix, 0) + param.numel()
                )
    return max(len(per_module_numel), 1)


def guess_hidden_dim(model: Any) -> int:
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            return mod.in_features
    return 512
