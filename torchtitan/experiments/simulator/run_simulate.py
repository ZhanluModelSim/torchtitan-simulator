#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CLI entry point for the TorchTitan CPU simulator.

Mirrors the structure of ``torchtitan/train.py`` but instead of training it:

1. Sets up a CPU-only environment (no GPUs needed).
2. Builds the model specified in the config (Llama3 / Qwen3 / etc.).
3. Runs simulation in one of three modes:
   - ``fx``       — static graph trace via ``make_fx`` + FakeTensorMode
   - ``runtime``  — one real CPU training step with full op / comm capture
   - ``schedule`` — PP schedule extraction only (no model forward pass)
   - ``all``      — fx + runtime + schedule

Usage::

    # Single-process simulation (no PP)
    python -m torchtitan.experiments.simulator.run_simulate \\
        --job.config_file ./train_configs/llama3_8b.toml \\
        --simulate.mode all \\
        --simulate.output_dir ./sim_out \\
        --simulate.output_format json,dot,chrome_trace,text

    # Multi-process PP simulation (torchrun)
    torchrun --nproc_per_node 4 \\
        -m torchtitan.experiments.simulator.run_simulate \\
        --job.config_file ./train_configs/llama3_8b.toml \\
        --training.data_parallel_replicate_degree 1 \\
        --training.pipeline_parallel_degree 4 \\
        --simulate.mode all

The simulator uses the standard TorchTitan config system (TOML + overrides).
All parallelism is set up on CPU using the gloo backend.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch

# Force CPU-only before any torchtitan import so device_type defaults to "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _parse_simulate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TorchTitan CPU simulator",
        add_help=False,  # torchtitan's config parser also uses argparse; we add ours first
    )
    parser.add_argument(
        "--simulate.mode",
        dest="simulate_mode",
        default="all",
        choices=["fx", "runtime", "schedule", "all"],
        help="Simulation mode (default: all)",
    )
    parser.add_argument(
        "--simulate.output_dir",
        dest="simulate_output_dir",
        default="./simulator_output",
        help="Directory to write output files (default: ./simulator_output)",
    )
    parser.add_argument(
        "--simulate.output_format",
        dest="simulate_output_format",
        default="json,dot,chrome_trace,html,text",
        help="Comma-separated list of output formats (default: json,dot,chrome_trace,html,text)",
    )
    parser.add_argument(
        "--simulate.max_seq_len",
        dest="simulate_max_seq_len",
        type=int,
        default=128,
        help="Sequence length for example inputs (default: 128)",
    )
    parser.add_argument(
        "--simulate.batch_size",
        dest="simulate_batch_size",
        type=int,
        default=2,
        help="Batch size for example inputs (default: 2)",
    )
    args, remaining = parser.parse_known_args()
    # Put remaining args back so torchtitan's config system can parse them
    sys.argv = [sys.argv[0]] + remaining
    return args


def _build_example_inputs(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> tuple[torch.Tensor, ...]:
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    return (tokens,)


def main() -> None:  # noqa: C901 (complex but straightforward)
    sim_args = _parse_simulate_args()
    output_formats = [f.strip() for f in sim_args.simulate_output_format.split(",")]

    # ------------------------------------------------------------------
    # 1.  TorchTitan config + distributed init
    # ------------------------------------------------------------------
    from torchtitan.config_manager import JobConfig

    config = JobConfig()
    config.parse_args()

    from torchtitan.experiments.simulator.cpu_env import (
        cpu_distributed_context,
        patch_device_type_to_cpu,
    )

    patch_device_type_to_cpu()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    print(f"[run_simulate] rank={rank}/{world_size}, mode={sim_args.simulate_mode}")

    with cpu_distributed_context(rank=rank, world_size=world_size):
        _run_simulation(config, sim_args, rank, world_size, output_formats)


def _run_simulation(
    config: Any,
    sim_args: argparse.Namespace,
    rank: int,
    world_size: int,
    output_formats: list[str],
) -> None:
    from torchtitan.experiments.simulator.simulator import Simulator

    sim = Simulator(rank=rank, world_size=world_size, verbose=(rank == 0))

    # ------------------------------------------------------------------
    # 2.  Build model according to config
    # ------------------------------------------------------------------
    model_parts, pp_schedule, pp_stages, vocab_size = _build_model_cpu(config, rank)

    # ------------------------------------------------------------------
    # 3.  Example inputs
    # ------------------------------------------------------------------
    example_inputs = _build_example_inputs(
        batch_size=sim_args.simulate_batch_size,
        seq_len=sim_args.simulate_max_seq_len,
        vocab_size=vocab_size,
    )

    # ------------------------------------------------------------------
    # 4.  Run simulation
    # ------------------------------------------------------------------
    mode = sim_args.simulate_mode
    output_dir = sim_args.simulate_output_dir

    if mode == "fx":
        result = sim.simulate_fx(model_parts[0], example_inputs)
        _export_result(result, output_dir, output_formats, sim)

    elif mode == "runtime":
        result = sim.simulate_runtime(
            model_parts,
            example_inputs,
            pp_schedule=pp_schedule,
            pp_stages=pp_stages,
        )
        _export_result(result, output_dir, output_formats, sim)

    elif mode == "schedule":
        if pp_schedule is None:
            print(
                "[run_simulate] No PP schedule found; nothing to export for 'schedule' mode."
            )
            return
        result = sim.simulate_pp_schedule(pp_schedule)
        _export_result(result, output_dir, output_formats, sim)

    else:  # "all"
        result = sim.simulate_all(
            model_parts,
            example_inputs,
            pp_schedule=pp_schedule,
            pp_stages=pp_stages,
            output_dir=output_dir,
            output_formats=output_formats,
        )


def _export_result(
    result: Any, output_dir: str, output_formats: list[str], sim: Any
) -> None:
    from torchtitan.experiments.simulator.export import (
        export_chrome_trace,
        export_dot,
        export_html,
        export_json,
        export_text_summary,
    )

    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    if "json" in output_formats:
        p = os.path.join(output_dir, "simulation_result.json")
        export_json(result, p)
        sim._log(f"JSON → {p}")
    if "dot" in output_formats:
        p = os.path.join(output_dir, "compute_graph.dot")
        export_dot(result.compute_graph, p)
        sim._log(f"DOT  → {p}")
    if "chrome_trace" in output_formats:
        p = os.path.join(output_dir, "trace.json")
        export_chrome_trace(result, p)
        sim._log(f"Chrome trace → {p}")
    if "html" in output_formats:
        p = os.path.join(output_dir, "trace.html")
        export_html(result, p)
        sim._log(f"HTML trace → {p}")
    if "text" in output_formats:
        summary = export_text_summary(result)
        p = os.path.join(output_dir, "summary.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(summary)
        sim._log(f"Text summary → {p}")
        if sim.verbose:
            print(summary)


def _build_model_cpu(
    config: Any,
    rank: int,
) -> tuple[list[Any], Any | None, list[Any] | None, int]:
    """
    Build TorchTitan model on CPU.

    Returns:
        (model_parts, pp_schedule_or_None, pp_stages_or_None, vocab_size)
    """
    import torch.distributed as dist

    from torchtitan.distributed import ParallelDims

    # Resolve model name
    model_name = getattr(getattr(config, "model", None), "name", "llama3")

    # Build parallel dims
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    parallel_dims = ParallelDims(
        dp_shard=getattr(config.training, "data_parallel_shard_degree", 1),
        dp_replicate=getattr(config.training, "data_parallel_replicate_degree", 1),
        cp=getattr(config.training, "context_parallel_degree", 1),
        tp=getattr(config.training, "tensor_parallel_degree", 1),
        pp=getattr(config.training, "pipeline_parallel_degree", 1),
        world_size=world_size,
        enable_loss_parallel=getattr(config.training, "enable_loss_parallel", False),
    )

    device = torch.device("cpu")

    # Build DeviceMesh on CPU
    from torchtitan.distributed import utils as dist_utils

    world_mesh = parallel_dims.build_mesh(device_type="cpu")

    # Build model
    from torchtitan.models import (
        model_name_to_cls,
        model_name_to_tokenizer,
        models_config,
    )

    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][config.model.flavor]
    model_config.update_from_config(config, parallel_dims)

    tokenizer_type = model_name_to_tokenizer[model_name]
    vocab_size = getattr(model_config, "vocab_size", 32000)

    with torch.device("cpu"):
        model = model_cls.from_model_args(model_config)

    pp_degree = parallel_dims.pp
    if pp_degree > 1:
        from torchtitan.distributed.pipeline_parallel import pipeline_llm

        pp_mesh = world_mesh["pp"] if "pp" in world_mesh.mesh_dim_names else None
        model_parts, pp_schedule, pp_stages = pipeline_llm(
            model,
            pp_mesh,
            parallel_dims,
            config,
            device,
            model_config,
        )
    else:
        # Apply FSDP / TP if configured
        from torchtitan.distributed.fsdp import parallelize_llm

        parallelize_llm(model, world_mesh, parallel_dims, config)
        model_parts = [model]
        pp_schedule = None
        pp_stages = None

    # Move to CPU explicitly (should already be there)
    for m in model_parts:
        m.to(device)

    return model_parts, pp_schedule, pp_stages, vocab_size


if __name__ == "__main__":
    # Support both ``python run_simulate.py`` and ``python -m ...``
    main()
