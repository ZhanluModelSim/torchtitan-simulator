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
from .memory_estimator import attach_model_state_memory
from .pp_schedule_extractor import PPScheduleExtractor
from .runtime_capture import RuntimeCapture
from .schedule_generator import generate_interleaved_1f1b_schedule


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


def _inject_semantic_schedule(result: Any, config: Any) -> None:
    """Append a semantic PP / TP / DP / FSDP2 schedule to *result*.

    Reads parallelism settings from *config* so the HTML visualisation
    shows the full multi-rank topology even when the simulator runs on a
    single CPU process.
    """
    from .nodes import TrainingSchedule

    parallelism = getattr(config, "parallelism", None)
    if parallelism is None:
        return

    pp_degree = int(getattr(parallelism, "pipeline_parallel_degree", 1) or 1)
    tp_degree = int(getattr(parallelism, "tensor_parallel_degree", 1) or 1)
    dp_shard = int(getattr(parallelism, "data_parallel_shard_degree", 1) or 1)
    # dp_shard == -1 means "use remaining ranks"
    if dp_shard < 0:
        dp_shard = 1
    dp_repl = int(getattr(parallelism, "data_parallel_replicate_degree", 1) or 1)
    num_mb = int(getattr(parallelism, "pipeline_parallel_microbatch_size", 8) or 8)

    schedule = getattr(parallelism, "pipeline_parallel_schedule", "1F1B") or "1F1B"
    virtual = 2 if "Interleaved" in str(schedule) else 1

    training = getattr(config, "training", None)
    num_steps = int(getattr(training, "steps", 1) or 1) if training else 1

    semantic = generate_interleaved_1f1b_schedule(
        pp_degree=pp_degree,
        tp_degree=tp_degree,
        dp_shard_degree=dp_shard,
        dp_replicate_degree=dp_repl,
        num_microbatches=num_mb,
        num_steps=num_steps,
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

    pp_stages = getattr(trainer, "_sim_stages", None) or trainer.model_parts
    with capture.activate(
        trainer.model_parts,
        phase="forward",
        pp_schedule=None,
        pp_stages=pp_stages,
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

    attach_model_state_memory(
        result,
        trainer.model_parts,
        optimizer_name=getattr(trainer.config.optimizer, "name", None),
    )

    if sim_opts.semantic_schedule:
        _inject_semantic_schedule(result, trainer.config)
        _inject_semantic_schedule(result, trainer.config)

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
    _export_result(result, sim_opts.output_dir, output_formats)
    logger.info("Simulation outputs written to %s", sim_opts.output_dir)
