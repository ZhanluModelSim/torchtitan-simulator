# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan.trainer import Trainer

from .cpu_env import patch_device_type_to_cpu
from .trainer_runner import run_trainer_simulation


@dataclass(kw_only=True, slots=True)
class SimulationConfig:
    output_dir: str = "./simulator_output"
    output_formats: list[str] = field(
        default_factory=lambda: ["json", "dot", "chrome_trace", "html", "text"]
    )
    capture_joint_fx: bool = False
    semantic_schedule: bool = False


def _cpu_noop_parallelize(model, **__):
    """CPU-only parallelize stub: return model unchanged.

    The real ``parallelize_llama`` calls ``apply_fsdp`` / ``fully_shard``
    which allocate CUDA tensors that cannot be materialised on CPU-only
    builds.  Skipping FSDP/TP is safe because the interception-based
    runtime capture records the actual ops that execute.
    """
    return model


def _cpu_noop_pipeline(model, **__):
    """CPU-only pipelining stub: return single-part list.

    The real ``pipeline_llm`` shards the model across pipeline stages,
    which triggers the same meta-tensor problem as ``parallelize_llama``.
    For simulation we treat the whole model as a single stage.
    """
    return None, [model], True, True


def _set_fake_world_size(config: Any) -> None:
    """Set ``NGPU``/``WORLD_SIZE`` from parallelism config for semantic schedule mode.

    The simulator runs on a single CPU process, but the semantic schedule
    needs ``ParallelDims`` to validate against the full topology size.
    """
    import os

    p = config.parallelism
    world = 1
    world *= int(getattr(p, "pipeline_parallel_degree", 1) or 1)
    world *= int(getattr(p, "tensor_parallel_degree", 1) or 1)
    dp_shard = int(getattr(p, "data_parallel_shard_degree", -1) or -1)
    dp_repl = int(getattr(p, "data_parallel_replicate_degree", 1) or 1)
    if dp_shard < 0:
        dp_shard = 1
    world *= dp_shard * dp_repl
    os.environ["NGPU"] = str(world)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"


class SimulationTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def __init__(self, config: Config):
        patch_device_type_to_cpu()

        pp = int(getattr(config.parallelism, "pipeline_parallel_degree", 1) or 1)
        tp = int(getattr(config.parallelism, "tensor_parallel_degree", 1) or 1)
        ds = int(getattr(config.parallelism, "data_parallel_shard_degree", -1) or -1)
        dr = int(getattr(config.parallelism, "data_parallel_replicate_degree", 1) or 1)
        if ds < 0:
            ds = 1
        if pp * tp * ds * dr > 1:
            _set_fake_world_size(config)

        config.model_spec.parallelize_fn = _cpu_noop_parallelize
        config.model_spec.pipelining_fn = _cpu_noop_pipeline

        super().__init__(config)

        self.parallel_dims.pp = 0
        self.parallel_dims.tp = 1
        self.parallel_dims.dp_shard = 1
        self.parallel_dims.dp_replicate = 1

    def train(self):
        patch_device_type_to_cpu()
        run_trainer_simulation(self, self.config.simulation)

    def train(self):
        patch_device_type_to_cpu()
        run_trainer_simulation(self, self.config.simulation)
