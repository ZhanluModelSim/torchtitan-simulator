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

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.tools.logging import logger

from .cost_model import apply_cost_model, CostModel, MockCostModel
from .export import export_result
from .extension_hooks import postprocess_extension_result
from .fx_capture import capture_forward_fx, capture_joint_fx
from .memory_estimator import (
    attach_model_state_memory,
    estimate_comm_memory,
    estimate_graph_memory,
    finalize_memory_summary,
    merge_memory_summary,
)
from .nodes import ComputeGraph, DataEdge, OpNode, SimulationResult
from .schedule_inject import inject_semantic_schedule
from .synthetic_comm import inject_synthetic_comm_events
from .unified_trace import compute_loss, TraceRecorder, unified_trace


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


def run_trainer_simulation(trainer: Any, sim_opts: Any) -> None:
    """Run one simulated training step using an already-built Trainer instance."""
    rank = int(os.environ.get("RANK", "0"))
    comm_backend = getattr(sim_opts, "comm_backend", "") or ""

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
    pp_degree = len(model_parts)

    if pp_degree > 1:
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

        for stage_idx, model_part in enumerate(model_parts):
            logger.info(
                "Tracing PP stage %d/%d (%d params)",
                stage_idx + 1,
                pp_degree,
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
                loss = compute_loss(output)

                recorder.current_phase = "backward"
                loss.backward()

            # Save output for next stage input
            if isinstance(output, torch.Tensor):
                prev_stage_output = output.detach()
            elif isinstance(output, (list, tuple)):
                tensors = [t for t in output if isinstance(t, torch.Tensor)]
                prev_stage_output = tensors[0].detach() if tensors else None

            logger.info(
                "  Stage %d: %d nodes (%d fwd + %d bwd), %d edges",
                stage_idx,
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
                "pp_degree": pp_degree,
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
            loss = compute_loss(output)

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
            inject_synthetic_comm_events(result, trainer, sim_opts)
        except Exception as exc:
            logger.warning("Failed to inject synthetic comm events: %s", exc)

    # ── Semantic schedule (must precede CostModel) ────────────────────
    if sim_opts.semantic_schedule:
        inject_semantic_schedule(result, trainer.config)

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
    if pp_degree <= 1:
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
    export_result(result, sim_opts.output_dir, output_formats, log_fn=logger.info)
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
