# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import time
from typing import Any

import torch
import torch.nn as nn

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.tools.logging import logger

from .cost_model import apply_cost_model, CostModel, MockCostModel
from .export import (
    export_chrome_trace,
    export_dot,
    export_html,
    export_json,
    export_text_summary,
)
from .extension_hooks import postprocess_extension_result
from .fx_capture import capture_forward_fx, capture_joint_fx
from .memory_estimator import (
    attach_model_state_memory,
    dtype_size,
    estimate_comm_memory,
    estimate_graph_memory,
    finalize_memory_summary,
    merge_memory_summary,
)
from .nodes import ComputeGraph, DataEdge, OpNode, SimulationResult, TensorMeta
from .schedule_extract import extract_schedule_from_pytorch
from .unified_trace import TraceRecorder, unified_trace


def _get_cost_model_kwargs(sim_opts: Any) -> dict[str, Any]:
    """Normalise ``cost_model_kwargs`` from config or CLI.

    Accepts both a plain Python dict (from ``config_registry``) and a JSON
    string (from ``--simulation.cost_model_kwargs '...'`` on the CLI).
    """
    raw = getattr(sim_opts, "cost_model_kwargs", {}) or {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        import json

        return json.loads(raw)
    return {}


def _import_cost_model(
    class_path: str, kwargs: dict[str, Any] | None = None
) -> CostModel:
    """Dynamically import a CostModel from a fully-qualified path.

    Supports two patterns:

    1. **Class** — ``\"my_pkg.MyCostModel\"`` → instantiated as
       ``MyCostModel(**kwargs)``.  Must be a :class:`CostModel` subclass.

    2. **Factory** — ``\"my_pkg.create_cost_model\"`` → called as
       ``create_cost_model()`` (no args).  Must return a :class:`CostModel`.

    Args:
        class_path: e.g. ``\"my_package.my_module.MyCostModel\"`` or
            ``\"my_package.my_module.create_cost_model\"``.
        kwargs: Forwarded to the constructor (class pattern only).

    Returns:
        An instance of :class:`CostModel`.
    """
    if kwargs is None:
        kwargs = {}
    module_path, _, name = class_path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"cost_model_class must be a fully-qualified path, " f'got "{class_path}"'
        )
    import importlib

    module = importlib.import_module(module_path)
    obj = getattr(module, name)

    if isinstance(obj, type) and issubclass(obj, CostModel):
        # Class pattern: instantiate with kwargs
        return obj(**kwargs)

    if callable(obj):
        # Factory pattern: call with no args
        result = obj()
        if not isinstance(result, CostModel):
            raise TypeError(
                f'Factory "{class_path}" must return a CostModel instance, '
                f"got {type(result)}"
            )
        return result

    raise TypeError(
        f'"{class_path}" must be a CostModel subclass or a callable '
        f"returning a CostModel, got {type(obj)}"
    )


def _export_result(result: Any, output_dir: str, output_formats: list[str]) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    os.makedirs(output_dir, exist_ok=True)

    if "json" in output_formats:
        export_json(result, os.path.join(output_dir, "simulation_result.json"))
    if "dot" in output_formats:
        export_dot(result.compute_graph, os.path.join(output_dir, "compute_graph.dot"))
    if "chrome_trace" in output_formats:
        export_chrome_trace(result, os.path.join(output_dir, "trace.json"))
    if "html" in output_formats:
        export_html(result, os.path.join(output_dir, "trace.html"))
    if "text" in output_formats:
        with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(export_text_summary(result))


def _inject_semantic_schedule(result: Any, config: Any) -> None:
    """Append a semantic PP / TP / DP / FSDP2 schedule to *result*.

    Reads parallelism settings from *config* and constructs a real PyTorch
    schedule object with mock stages to extract the exact action table.
    The HTML visualisation then shows the full multi-rank topology matching
    upstream PyTorch behaviour.
    """
    from .nodes import TrainingSchedule

    parallelism = getattr(config, "parallelism", None)
    if parallelism is None:
        return

    pp_degree = int(getattr(parallelism, "pipeline_parallel_degree", 1) or 1)
    tp_degree = int(getattr(parallelism, "tensor_parallel_degree", 1) or 1)
    dp_shard = int(getattr(parallelism, "data_parallel_shard_degree", 1) or 1)
    if dp_shard < 0:
        dp_shard = 1
    dp_repl = int(getattr(parallelism, "data_parallel_replicate_degree", 1) or 1)
    dp_degree = dp_shard * dp_repl

    schedule_name = str(
        getattr(parallelism, "pipeline_parallel_schedule", "1F1B") or "1F1B"
    )
    num_mb = int(getattr(parallelism, "pipeline_parallel_microbatch_size", 8) or 8)
    virtual = 2 if "Interleaved" in schedule_name else 1
    num_stages = pp_degree * virtual

    semantic = extract_schedule_from_pytorch(
        pp_degree=pp_degree,
        tp_degree=tp_degree,
        dp_degree=dp_degree,
        num_stages=num_stages,
        n_microbatches=num_mb,
        schedule_name=schedule_name,
        virtual_stages_per_rank=virtual,
    )

    existing = result.schedule
    if existing is None:
        result.schedule = semantic
    elif isinstance(existing, TrainingSchedule):
        for ev in semantic.events:
            existing.add_event(ev)
        for dep in semantic.deps:
            existing.add_dep(dep)


def _inject_synthetic_comm_events(
    result: Any,
    trainer: Any,
    sim_opts: Any,
) -> None:
    """Inject synthetic communication events for fake_backend mode.

    When running with fake_backend (no real distributed communication),
    this function creates :class:`OpNode` entries for the FSDP all-gather,
    FSDP reduce-scatter, TP all-reduce, PP send/recv, and other collectives that
    *would* be triggered by real parallelism.  Shapes and group sizes are
    derived from the model's parameter structure and the parallelism config.
    """
    graph = result.compute_graph
    parallelism = trainer.config.parallelism
    model_parts = trainer.model_parts

    # Read parallelism degrees
    tp = int(getattr(parallelism, "tensor_parallel_degree", 1) or 1)
    ds = int(getattr(parallelism, "data_parallel_shard_degree", 1) or 1)
    pp = int(getattr(parallelism, "pipeline_parallel_degree", 1) or 1)
    ep = int(getattr(parallelism, "expert_parallel_degree", 1) or 1)
    cp = int(getattr(parallelism, "context_parallel_degree", 1) or 1)
    dr = int(getattr(parallelism, "data_parallel_replicate_degree", 1) or 1)

    # Resolve ds=-1 (auto-inferred) to actual value
    if ds < 0:
        world_size = pp * tp * cp * dr
        ds = max(1, world_size // (pp * tp * cp * dr))

    if not (tp > 1 or ds > 1 or pp > 1):
        return  # No parallelism → no synthetic comm needed

    # ── Compute model parameter numel ─────────────────────────────────
    total_param_numel = 0
    per_module_numel: dict[str, int] = {}
    for part in model_parts:
        for name, param in part.named_parameters():
            if param.requires_grad:
                nel = param.numel()
                total_param_numel += nel
                prefix = ".".join(name.split(".")[:2])
                per_module_numel[prefix] = per_module_numel.get(prefix, 0) + nel

    # ── Determine dtype from config ───────────────────────────────────
    from torchtitan.config import TORCH_DTYPE_MAP

    mp_param = getattr(trainer.config.training, "mixed_precision_param", "bfloat16")
    torch_dtype = TORCH_DTYPE_MAP.get(mp_param, torch.bfloat16)
    dtype_str = str(torch_dtype)
    dtype_byte_size = (
        torch_dtype.itemsize
        if hasattr(torch_dtype, "itemsize")
        else dtype_size(dtype_str)
    )

    logger.info(
        "Injecting synthetic comm events: tp=%d ds=%d pp=%d ep=%d dtype=%s total_param_numel=%d",
        tp,
        ds,
        pp,
        ep,
        dtype_str,
        total_param_numel,
    )

    shard_numel = total_param_numel // ds if ds > 1 else total_param_numel

    # ── Helper functions ──────────────────────────────────────────────
    counter = [len(graph.nodes)]

    def _next_id() -> str:
        counter[0] += 1
        return f"comm_syn_{counter[0]:07d}"

    def _find_last_compute_node_id(phase: str, pp_stage: int | None = None) -> str | None:
        for nid in reversed(list(graph.nodes.keys())):
            n = graph.nodes[nid]
            if n.phase == phase and n.op_type == "compute":
                # If pp_stage is specified, match it or accept None
                if pp_stage is None or n.pp_stage is None or n.pp_stage == pp_stage:
                    return nid
        return None

    # ── FSDP2 all_gather + reduce_scatter events ──────────────────────
    if ds > 1:
        num_layers = _infer_num_layers(model_parts)
        per_layer_numel = shard_numel // max(num_layers, 1)
        full_layer_numel = per_layer_numel * ds

        for stage_idx in range(pp):
            pp_rank = stage_idx % pp  # Calculate pp_rank from stage_idx
            fwd_anchor = _find_last_compute_node_id("forward", stage_idx)
            bwd_anchor = _find_last_compute_node_id("backward", stage_idx)

            for i in range(num_layers):
                node = OpNode(
                    node_id=_next_id(),
                    op_name="all_gather",
                    op_type="comm_collective",
                    phase="forward",
                    pp_stage=stage_idx,
                    pp_rank=pp_rank,
                    inputs=[
                        TensorMeta(shape=(per_layer_numel,), dtype=dtype_str, device="cpu")
                    ],
                    outputs=[
                        TensorMeta(shape=(full_layer_numel,), dtype=dtype_str, device="cpu")
                    ],
                    comm_op="all_gather",
                    comm_group_size=ds,
                    attrs={"synthetic": True, "fsdp2": True},
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
                        "pp_stage": stage_idx,
                        "pp_rank": pp_rank,
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
                    pp_stage=stage_idx,
                    pp_rank=pp_rank,
                    inputs=[
                        TensorMeta(shape=(full_layer_numel,), dtype=dtype_str, device="cpu")
                    ],
                    outputs=[
                        TensorMeta(shape=(per_layer_numel,), dtype=dtype_str, device="cpu")
                    ],
                    comm_op="reduce_scatter",
                    comm_group_size=ds,
                    attrs={"synthetic": True, "fsdp2": True},
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
                        "pp_stage": stage_idx,
                        "pp_rank": pp_rank,
                        "tensor_meta": {
                            "shape": [full_layer_numel],
                            "dtype": dtype_str,
                            "device": "cpu",
                        },
                        "source_node_ids": [bwd_anchor] if bwd_anchor else [],
                        "synthetic": True,
                    }
                )

    # ── TP all_reduce events ──────────────────────────────────────────
    if tp > 1:
        seq_len = trainer.config.training.seq_len
        batch_size = trainer.config.training.local_batch_size
        hidden = _guess_hidden_dim(model_parts[0])
        act_numel = batch_size * seq_len * hidden
        num_layers = _infer_num_layers(model_parts)
        tp_allreduce_count = num_layers * 2

        for stage_idx in range(pp):
            pp_rank = stage_idx % pp  # Calculate pp_rank from stage_idx
            fwd_anchor = _find_last_compute_node_id("forward", stage_idx)
            bwd_anchor = _find_last_compute_node_id("backward", stage_idx)

            for _ in range(tp_allreduce_count):
                node = OpNode(
                    node_id=_next_id(),
                    op_name="all_reduce",
                    op_type="comm_collective",
                    phase="forward",
                    pp_stage=stage_idx,
                    pp_rank=pp_rank,
                    inputs=[TensorMeta(shape=(act_numel,), dtype=dtype_str, device="cpu")],
                    outputs=[TensorMeta(shape=(act_numel,), dtype=dtype_str, device="cpu")],
                    comm_op="all_reduce",
                    comm_group_size=tp,
                    attrs={"synthetic": True, "tp": True},
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
                        "pp_stage": stage_idx,
                        "pp_rank": pp_rank,
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
                    pp_stage=stage_idx,
                    pp_rank=pp_rank,
                    inputs=[TensorMeta(shape=(act_numel,), dtype=dtype_str, device="cpu")],
                    outputs=[TensorMeta(shape=(act_numel,), dtype=dtype_str, device="cpu")],
                    comm_op="all_reduce",
                    comm_group_size=tp,
                    attrs={"synthetic": True, "tp": True},
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
                        "pp_stage": stage_idx,
                        "pp_rank": pp_rank,
                        "tensor_meta": {
                            "shape": [batch_size, seq_len, hidden],
                            "dtype": dtype_str,
                            "device": "cpu",
                        },
                        "source_node_ids": [bwd_anchor] if bwd_anchor else [],
                        "synthetic": True,
                    }
                )

    # ── PP send/recv events ───────────────────────────────────────────
    if pp > 1:
        seq_len = trainer.config.training.seq_len
        batch_size = trainer.config.training.local_batch_size
        hidden = _guess_hidden_dim(model_parts[0])
        activation_numel = batch_size * seq_len * hidden

        # Create send/recv pairs for each microbatch
        num_microbatches = int(getattr(parallelism, "pipeline_parallel_microbatch_size", 8) or 8)

        for mb_idx in range(num_microbatches):
            for stage_idx in range(pp - 1):
                # Forward: send activation from stage_idx to stage_idx+1
                send_pp_rank = stage_idx % pp
                recv_pp_rank = (stage_idx + 1) % pp
                
                send_node = OpNode(
                    node_id=_next_id(),
                    op_name="pp_send_activation",
                    op_type="comm_p2p",
                    phase="forward",
                    pp_stage=stage_idx,
                    pp_rank=send_pp_rank,
                    microbatch_idx=mb_idx,
                    inputs=[TensorMeta(shape=(activation_numel,), dtype=dtype_str, device="cpu")],
                    outputs=[TensorMeta(shape=(activation_numel,), dtype=dtype_str, device="cpu")],
                    comm_op="send",
                    comm_group_size=2,
                    attrs={"synthetic": True, "pp": True, "dst_stage": stage_idx + 1},
                )
                graph.add_node(send_node)

                recv_node = OpNode(
                    node_id=_next_id(),
                    op_name="pp_recv_activation",
                    op_type="comm_p2p",
                    phase="forward",
                    pp_stage=stage_idx + 1,
                    pp_rank=recv_pp_rank,
                    microbatch_idx=mb_idx,
                    inputs=[TensorMeta(shape=(activation_numel,), dtype=dtype_str, device="cpu")],
                    outputs=[TensorMeta(shape=(activation_numel,), dtype=dtype_str, device="cpu")],
                    comm_op="recv",
                    comm_group_size=2,
                    attrs={"synthetic": True, "pp": True, "src_stage": stage_idx},
                )
                graph.add_node(recv_node)

                # Add edge: send -> recv
                graph.add_edge(DataEdge(send_node.node_id, recv_node.node_id, "pp_p2p"))

                result.comm_events.append(
                    {
                        "event_id": send_node.node_id,
                        "op": "send",
                        "group_size": 2,
                        "phase": "forward",
                        "pp_stage": stage_idx,
                        "pp_rank": send_pp_rank,
                        "microbatch": mb_idx,
                        "tensor_meta": {
                            "shape": [batch_size, seq_len, hidden],
                            "dtype": dtype_str,
                            "device": "cpu",
                        },
                        "source_node_ids": [],
                        "synthetic": True,
                    }
                )
                result.comm_events.append(
                    {
                        "event_id": recv_node.node_id,
                        "op": "recv",
                        "group_size": 2,
                        "phase": "forward",
                        "pp_stage": stage_idx + 1,
                        "pp_rank": recv_pp_rank,
                        "microbatch": mb_idx,
                        "tensor_meta": {
                            "shape": [batch_size, seq_len, hidden],
                            "dtype": dtype_str,
                            "device": "cpu",
                        },
                        "source_node_ids": [send_node.node_id],
                        "synthetic": True,
                    }
                )

            # Backward: send gradient from stage_idx+1 to stage_idx
            for stage_idx in range(pp - 1, 0, -1):
                send_pp_rank = stage_idx % pp
                recv_pp_rank = (stage_idx - 1) % pp
                
                send_node = OpNode(
                    node_id=_next_id(),
                    op_name="pp_send_gradient",
                    op_type="comm_p2p",
                    phase="backward",
                    pp_stage=stage_idx,
                    pp_rank=send_pp_rank,
                    microbatch_idx=mb_idx,
                    inputs=[TensorMeta(shape=(activation_numel,), dtype=dtype_str, device="cpu")],
                    outputs=[TensorMeta(shape=(activation_numel,), dtype=dtype_str, device="cpu")],
                    comm_op="send",
                    comm_group_size=2,
                    attrs={"synthetic": True, "pp": True, "dst_stage": stage_idx - 1},
                )
                graph.add_node(send_node)

                recv_node = OpNode(
                    node_id=_next_id(),
                    op_name="pp_recv_gradient",
                    op_type="comm_p2p",
                    phase="backward",
                    pp_stage=stage_idx - 1,
                    pp_rank=recv_pp_rank,
                    microbatch_idx=mb_idx,
                    inputs=[TensorMeta(shape=(activation_numel,), dtype=dtype_str, device="cpu")],
                    outputs=[TensorMeta(shape=(activation_numel,), dtype=dtype_str, device="cpu")],
                    comm_op="recv",
                    comm_group_size=2,
                    attrs={"synthetic": True, "pp": True, "src_stage": stage_idx},
                )
                graph.add_node(recv_node)

                # Add edge: send -> recv
                graph.add_edge(DataEdge(send_node.node_id, recv_node.node_id, "pp_p2p"))

                result.comm_events.append(
                    {
                        "event_id": send_node.node_id,
                        "op": "send",
                        "group_size": 2,
                        "phase": "backward",
                        "pp_stage": stage_idx,
                        "pp_rank": send_pp_rank,
                        "microbatch": mb_idx,
                        "tensor_meta": {
                            "shape": [batch_size, seq_len, hidden],
                            "dtype": dtype_str,
                            "device": "cpu",
                        },
                        "source_node_ids": [],
                        "synthetic": True,
                    }
                )
                result.comm_events.append(
                    {
                        "event_id": recv_node.node_id,
                        "op": "recv",
                        "group_size": 2,
                        "phase": "backward",
                        "pp_stage": stage_idx - 1,
                        "pp_rank": recv_pp_rank,
                        "microbatch": mb_idx,
                        "tensor_meta": {
                            "shape": [batch_size, seq_len, hidden],
                            "dtype": dtype_str,
                            "device": "cpu",
                        },
                        "source_node_ids": [send_node.node_id],
                        "synthetic": True,
                    }
                )


def _infer_num_layers(model_parts: list[Any]) -> int:
    """Derive the number of transformer layers from model config or structure.

    Tries in order:
    1. model.config.n_layers (all TorchTitan models define this)
    2. len(model.layers) attribute
    3. Fallback: unique parameter-prefix count (approximate)
    """
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
    # Fallback: count unique 2-prefix groups from parameter names
    per_module_numel: dict[str, int] = {}
    for part in model_parts:
        for name, param in part.named_parameters():
            if param.requires_grad:
                prefix = ".".join(name.split(".")[:2])
                per_module_numel[prefix] = (
                    per_module_numel.get(prefix, 0) + param.numel()
                )
    return max(len(per_module_numel), 1)


def _guess_hidden_dim(model: Any) -> int:
    """Guess the hidden dimension from a model's first Linear layer."""
    import torch.nn as nn

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            return mod.in_features
    return 512  # fallback


def run_trainer_simulation(trainer: Any, sim_opts: Any) -> None:
    """Run one simulated training step using an already-built Trainer instance."""
    rank = int(os.environ.get("RANK", "0"))
    comm_backend = getattr(sim_opts, "comm_backend", "") or ""
    actual_comm_mode = getattr(trainer.config.comm, "mode", "") or ""
    if actual_comm_mode == "fake_backend":
        comm_backend = ""

    data_iterator = trainer.batch_generator(trainer.dataloader)
    trainer.optimizers.zero_grad()

    microbatches: list[tuple[dict[str, torch.Tensor], torch.Tensor]] = []
    local_valid_tokens = torch.tensor(0, dtype=torch.int64)
    for _ in range(trainer.gradient_accumulation_steps):
        input_dict, labels = next(data_iterator)
        local_valid_tokens += (labels != IGNORE_INDEX).sum()
        microbatches.append((input_dict, labels))

    use_fake = comm_backend != "gloo"
    capture_comm = comm_backend == "gloo"

    first_input_dict, first_labels = microbatches[0]

    sim_t0 = time.monotonic()

    # -- Decide if we should multi-stage trace ----------
    model_parts = getattr(trainer, "_pp_model_parts", None) or trainer.model_parts
    num_virtual_stages = len(model_parts)
    pp_config_degree = int(getattr(trainer.config.parallelism, "pipeline_parallel_degree", 1) or 1)
    
    # Calculate VPP (virtual pipeline parallelism) factor
    vpp = num_virtual_stages // pp_config_degree if pp_config_degree > 1 else 1

    if num_virtual_stages > 1:
        # === Multi-stage per-PP trace ===
        # Move model parts back to meta device to avoid OOM with
        # large models (1T+ parameters materialized on CPU)
        for m in model_parts:
            m.to_empty(device="meta")

        from collections import Counter

        all_nodes: list[OpNode] = []
        all_edges: list[tuple[str, str, str]] = []

        # Stage 0 uses real token ids; later stages use previous stage's output
        prev_stage_output = None

        logger.info(
            "Multi-stage tracing: %d virtual stages, PP=%d, VPP=%d",
            num_virtual_stages,
            pp_config_degree,
            vpp,
        )

        for stage_idx, model_part in enumerate(model_parts):
            # Calculate which physical PP rank this virtual stage belongs to
            # For Interleaved1F1B with PP=8, VPP=2:
            #   stages 0-7 -> ranks 0-7 (first virtual stage)
            #   stages 8-15 -> ranks 0-7 (second virtual stage)
            pp_rank = stage_idx % pp_config_degree
            
            logger.info(
                "Tracing virtual stage %d/%d (PP rank %d, VPP chunk %d/%d, %d params)",
                stage_idx + 1,
                num_virtual_stages,
                pp_rank,
                (stage_idx // pp_config_degree) + 1,
                vpp,
                sum(p.numel() for p in model_part.parameters()),
            )

            # All model parts are on meta device (we moved them above to
            # avoid OOM). FakeTensorMode works with meta parameters on all
            # stages so use_fake is always True for fake_backend mode.
            use_fake_for_stage = use_fake

            if stage_idx == 0:
                if use_fake:
                    stage_input = (first_input_dict["input"].to("meta"),)
                else:
                    stage_input = (first_input_dict["input"],)
            else:
                # For later stages: use prev stage's output directly.
                stage_input = (prev_stage_output,)

            recorder = TraceRecorder(rank=rank)
            recorder._counter = (
                stage_idx + 1
            ) * 100000  # avoid node_id collision (0 is default initial)
            recorder.current_pp_stage = stage_idx
            recorder.current_pp_rank = pp_rank

            with unified_trace(
                recorder,
                model_part,
                stage_input,
                use_fake_mode=use_fake_for_stage,
                phase="forward",
                capture_comm=capture_comm,
                capture_fsdp=True,
                model_parts=model_parts,
            ):
                output = model_part(*stage_input)

                if isinstance(output, torch.Tensor):
                    loss = output.sum()
                else:
                    import torch.utils._pytree as pytree

                    flat, _ = pytree.tree_flatten(output)
                    loss = sum(t.sum() for t in flat if isinstance(t, torch.Tensor))

                recorder.current_phase = "backward"
                loss.backward()

            # Save output for next stage input
            if isinstance(output, torch.Tensor):
                prev_stage_output = output.detach()
            elif isinstance(output, (list, tuple)):
                tensors = [t for t in output if isinstance(t, torch.Tensor)]
                prev_stage_output = tensors[0].detach() if tensors else None

            logger.info(
                "  Virtual stage %d (PP rank %d): %d nodes (%d fwd + %d bwd), %d edges",
                stage_idx,
                pp_rank,
                len(recorder.nodes),
                sum(1 for n in recorder.nodes if n.phase == "forward"),
                sum(1 for n in recorder.nodes if n.phase == "backward"),
                len(recorder.edges),
            )

            all_nodes.extend(recorder.nodes)
            all_edges.extend(recorder.edges)

        # -- Merge per-stage ComputeGraphs -------------------
        merged_graph = ComputeGraph()
        for n in all_nodes:
            merged_graph.add_node(n)
        for src, dst, etype in all_edges:
            merged_graph.add_edge(
                DataEdge(src_node_id=src, dst_node_id=dst, edge_type=etype)
            )

        merged_graph.fix_comm_phase_labels()
        merged_graph.add_phase_boundary_edges()

        result = SimulationResult(
            compute_graph=merged_graph,
            comm_events=[],  # multi-stage doesn't capture comm events yet
            metadata={
                "mode": "unified_trace",
                "device_mode": "meta",
                "rank": rank,
                "num_virtual_stages": num_virtual_stages,
                "pp_degree": pp_config_degree,
                "vpp": vpp,
            },
        )

        stage_counts = Counter(n.pp_stage for n in merged_graph.nodes.values())
        logger.info(
            "Merged graph: %d nodes across %d stages: %s",
            len(merged_graph.nodes),
            len(stage_counts),
            dict(stage_counts),
        )
        logger.info(
            "Multi-stage tracing completed in %.2fs (%d nodes, %d edges)",
            time.monotonic() - sim_t0,
            len(merged_graph.nodes),
            len(merged_graph.edges),
        )
    else:
        # === Single-stage trace (original logic) ===
        recorder = TraceRecorder(rank=rank)
        model_part = model_parts[0]

        if use_fake:
            example_inputs = (first_input_dict["input"].to("meta"),)
        else:
            example_inputs = (first_input_dict["input"],)

        with unified_trace(
            recorder,
            model_part,
            example_inputs,
            use_fake_mode=use_fake,
            phase="forward",
            capture_comm=capture_comm,
            capture_fsdp=True,
            model_parts=trainer.model_parts,
        ):
            output = model_part(*example_inputs)

            if isinstance(output, torch.Tensor):
                loss = output.sum()
            else:
                import torch.utils._pytree as pytree

                flat, _ = pytree.tree_flatten(output)
                loss = sum(t.sum() for t in flat if isinstance(t, torch.Tensor))

            recorder.current_phase = "backward"
            loss.backward()

        result = recorder.build_result(
            metadata={
                "mode": "unified_trace",
                "device_mode": "meta" if use_fake else "cpu",
                "rank": rank,
            }
        )

        result.compute_graph.fix_comm_phase_labels()
        result.compute_graph.add_phase_boundary_edges()
        logger.info(
            "Single-stage tracing completed in %.2fs (%d nodes, %d edges)",
            time.monotonic() - sim_t0,
            len(result.compute_graph.nodes),
            len(result.compute_graph.edges),
        )

    attach_model_state_memory(
        result,
        trainer.model_parts,
        optimizer_name=getattr(trainer.config.optimizer, "name", None),
    )

    graph_mem_events, graph_mem_summary = estimate_graph_memory(result.compute_graph)
    comm_mem_events = estimate_comm_memory(result.comm_events)
    result.memory_events.extend(graph_mem_events)
    result.memory_events.extend(comm_mem_events)
    merged_summary = merge_memory_summary(
        graph_mem_summary,
        {
            "total_event_bytes": sum(e.bytes for e in comm_mem_events),
            "by_category": {"comm_event_buffer": sum(e.bytes for e in comm_mem_events)},
        },
    )
    merged_summary["graph_peak_live_bytes"] = graph_mem_summary.get(
        "peak_live_bytes", 0
    )
    result.metadata["memory"] = finalize_memory_summary(
        result.memory_events,
        merged_summary,
        existing_metadata=result.metadata.get("memory"),
    )

    if comm_backend != "gloo":
        try:
            _inject_synthetic_comm_events(result, trainer, sim_opts)
            synth_comm = [n for n in result.compute_graph.nodes.values() if n.attrs.get("synthetic")]
            logger.info("Synthetic comm events injected: %d", len(synth_comm))
        except Exception as exc:
            import traceback
            logger.warning("Failed to inject synthetic comm events: %s", exc)
            traceback.print_exc()

    # ── Semantic schedule (must precede CostModel) ────────────────────
    if sim_opts.semantic_schedule:
        _inject_semantic_schedule(result, trainer.config)

    # ── CostModel ──────────────────────────────────────────────────────
    cost_model_enabled = getattr(sim_opts, "cost_model", False)
    if cost_model_enabled:
        cost_model_cls = getattr(sim_opts, "cost_model_class", "") or ""
        cost_model_kwargs = _get_cost_model_kwargs(sim_opts)
        if cost_model_cls:
            cost_model = _import_cost_model(cost_model_cls, cost_model_kwargs)
        else:
            cost_model = MockCostModel()
        cost_summary = apply_cost_model(result, cost_model)
        result.metadata["cost_model"] = cost_summary
        logger.info(
            "CostModel: e2e_step=%.1f us, single_rank_step=%.1f us, "
            "compute=%.1f us, comm=%.1f us",
            cost_summary["e2e_step_time_us"],
            cost_summary["single_rank_step_time_us"],
            cost_summary["total_compute_time_us"],
            cost_summary["total_comm_time_us"],
        )

    if not microbatches:
        raise RuntimeError("simulation requires at least one microbatch")
    first_input_dict, first_labels = microbatches[0]
    example_inputs = (first_input_dict["input"],)
    # Skip fx_forward_graph capture for multi-stage traces - the per-stage
    # model parts are on meta device and make_fx on 1T-param models is
    # extremely slow (~minutes).  The 99998-node merged graph already
    # contains all ops.
    if num_virtual_stages <= 1:
        try:
            result.metadata["fx_forward_graph"] = capture_forward_fx(
                trainer.model_parts[0],
                example_inputs,
            ).to_dict()
        except Exception as exc:
            result.metadata["fx_forward_graph_error"] = str(exc)
    if sim_opts.capture_joint_fx:

        def _trainer_loss_adapter(pred: Any, labels: torch.Tensor) -> torch.Tensor:
            try:
                valid_tokens = (labels != IGNORE_INDEX).sum().to(dtype=torch.float32)
                return trainer.loss_fn(pred, labels, valid_tokens)
            except TypeError:
                return trainer.loss_fn(pred, labels)

        try:
            result.metadata["fx_joint_graph"] = capture_joint_fx(
                trainer.model_parts[0],
                example_inputs,
                loss_fn=_trainer_loss_adapter,
                example_labels=first_labels.to(trainer.device),
            ).to_dict()
        except Exception as exc:
            result.metadata["fx_joint_graph_error"] = str(exc)

    result = postprocess_extension_result(result, trainer, sim_opts)

    output_formats = sim_opts.output_formats or [
        "json",
        "dot",
        "chrome_trace",
        "html",
        "text",
    ]
    t_export = time.monotonic()
    _export_result(result, sim_opts.output_dir, output_formats)
    logger.info(
        "Simulation outputs written to %s (export %.2fs, total %.2fs)",
        sim_opts.output_dir,
        time.monotonic() - t_export,
        time.monotonic() - sim_t0,
    )
    result.metadata["timing"] = {
        "tracing_s": round(t_export - sim_t0, 2),
        "export_s": round(time.monotonic() - t_export, 2),
        "total_s": round(time.monotonic() - sim_t0, 2),
        "node_count": len(result.compute_graph.nodes),
        "edge_count": len(result.compute_graph.edges),
    }
