# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from typing import Any

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.tools.logging import logger

from .cost_model import apply_cost_model, CostModel, MockCostModel
from .export import export_result as _export_result
from .extension_hooks import postprocess_extension_result
from .fx_capture import capture_forward_fx, capture_joint_fx
from .memory_estimator import (
    attach_model_state_memory,
    estimate_comm_memory,
    estimate_graph_memory,
    finalize_memory_summary,
    merge_memory_summary,
)
from .schedule_inject import inject_semantic_schedule
from .synthetic_comm import inject_synthetic_comm_events
from .unified_trace import compute_loss, TraceRecorder, unified_trace


def _get_cost_model_kwargs(sim_opts: Any) -> dict[str, Any]:
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
        return obj(**kwargs)

    if callable(obj):
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

    recorder = TraceRecorder(rank=rank)
    model_part = trainer.model_parts[0]
    use_fake = comm_backend != "gloo"
    capture_comm = comm_backend == "gloo"

    if use_fake:
        first_input_dict, first_labels = microbatches[0]
        example_inputs = (first_input_dict["input"].to("meta"),)
    else:
        first_input_dict, first_labels = microbatches[0]
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

    if sim_opts.semantic_schedule:
        inject_semantic_schedule(result, trainer.config)

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
    _export_result(result, sim_opts.output_dir, output_formats, log_fn=logger.info)
    logger.info("Simulation outputs written to %s", sim_opts.output_dir)
