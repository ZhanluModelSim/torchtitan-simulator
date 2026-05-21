# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools.logging import logger

from .export import (
    export_chrome_trace,
    export_dot,
    export_html,
    export_json,
    export_text_summary,
)
from .extension_hooks import collect_extension_metadata, postprocess_extension_result
from .fx_capture import capture_forward_fx, capture_joint_fx
from .memory_estimator import (
    estimate_model_state_memory,
    merge_memory_summary,
    summarize_memory_events,
)
from .pp_schedule_extractor import PPScheduleExtractor
from .runtime_capture import RuntimeCapture


def _export_result(result: Any, output_dir: str, output_formats: list[str]) -> None:
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


def run_trainer_simulation(trainer: Any, sim_opts: Any) -> None:
    """
    Run one simulated training step using an already-built Trainer instance.

    This keeps full compatibility with the native TorchTitan entry path:
    model build, distributed init, dataloader, and parallelism setup are
    unchanged; only execution is switched from `trainer.train()` to one-step
    instrumented capture.
    """
    rank = int(os.environ.get("RANK", "0"))
    capture = RuntimeCapture(rank=rank)
    data_iterator = trainer.batch_generator(trainer.dataloader)
    trainer.optimizers.zero_grad()

    microbatches: list[tuple[dict[str, torch.Tensor], torch.Tensor]] = []
    local_valid_tokens = torch.tensor(0, dtype=torch.int64)
    for _ in range(trainer.gradient_accumulation_steps):
        input_dict, labels = next(data_iterator)
        local_valid_tokens += (labels != IGNORE_INDEX).sum()
        microbatches.append((input_dict, labels))

    local_valid_tokens = local_valid_tokens.to(trainer.device)
    if trainer.parallel_dims.dp_enabled:
        batch_mesh = trainer.parallel_dims.get_mesh("batch")
        global_valid_tokens = dist_utils.dist_sum(local_valid_tokens, batch_mesh)
    else:
        global_valid_tokens = local_valid_tokens.float()

    pp_schedule = getattr(trainer, "pp_schedule", None)
    with capture.activate(
        trainer.model_parts,
        phase="forward",
        pp_schedule=pp_schedule,
        pp_stages=trainer.model_parts,
    ):
        for mb_idx, (input_dict, labels) in enumerate(microbatches):
            capture.set_microbatch(mb_idx)
            capture.set_phase("forward")
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(trainer.device)
            labels = labels.to(trainer.device)
            trainer.forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                global_valid_tokens=global_valid_tokens,
            )

        capture.set_phase("optimizer")
        dist_utils.clip_grad_norm_(
            [p for m in trainer.model_parts for p in m.parameters()],
            trainer.config.training.max_norm,
            foreach=True,
            pp_mesh=trainer.parallel_dims.get_optional_mesh("pp"),
            ep_enabled=trainer.parallel_dims.ep_enabled,
        )
        trainer.optimizers.step()
        trainer.lr_schedulers.step()

    result = capture.build_result(
        metadata={
            "mode": "simulation",
            "rank": rank,
            **collect_extension_metadata(trainer, capture),
        }
    )

    model_memory_events, model_memory_summary = estimate_model_state_memory(
        trainer.model_parts,
        optimizer_name=getattr(trainer.config.optimizer, "name", None),
    )
    result.memory_events.extend(model_memory_events)
    memory_metadata = {
        key: value
        for key, value in (result.metadata.get("memory", {}) or {}).items()
        if key not in {"total_event_bytes", "by_category", "by_phase", "by_device"}
    }
    result.metadata["memory"] = merge_memory_summary(
        summarize_memory_events(result.memory_events),
        memory_metadata,
        model_memory_summary,
    )

    pp_schedule = getattr(trainer, "pp_schedule", None)
    if pp_schedule is not None:
        extractor = PPScheduleExtractor(
            schedule=pp_schedule,
            pp_rank=rank,
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
        )
        result.schedule = extractor.extract()

    if not microbatches:
        raise RuntimeError("simulation requires at least one microbatch")
    first_input_dict, first_labels = microbatches[0]
    example_inputs = (first_input_dict["input"],)
    result.metadata["fx_forward_graph"] = capture_forward_fx(
        trainer.model_parts[0],
        example_inputs,
    ).to_dict()
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
    _export_result(result, sim_opts.output_dir, output_formats)
    logger.info("Simulation outputs written to %s", sim_opts.output_dir)
