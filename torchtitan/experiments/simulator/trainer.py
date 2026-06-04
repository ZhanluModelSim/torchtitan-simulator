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
    which allocate meta tensors that cannot be materialised on CPU-only
    builds.  Simulation captures parallelisation semantics through the
    ``TrainingSchedule`` instead, so skipping FSDP/TP here is safe.
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

        # When semantic_schedule is requested, set WORLD_SIZE to the
        # product of parallelism degrees so that ParallelDims validates
        # correctly even though we run on a single CPU process.
        if config.simulation.semantic_schedule:
            _set_fake_world_size(config)

        # Override the model's parallelize / pipelining callables so that
        # Trainer.__init__ does not invoke FSDP/TP/PP wrappers which rely
        # on CUDA device handles deep inside PyTorch.
        config.model_spec.parallelize_fn = _cpu_noop_parallelize
        config.model_spec.pipelining_fn = _cpu_noop_pipeline

        super().__init__(config)

        # After Trainer.__init__, disable PP/tp flags so that
        # forward_backward_step uses the non-PP code path (single model,
        # no schedule.step call).  The semantic schedule injected later
        # carries the full topology; the runtime only needs one CPU pass.
        self.parallel_dims.pp = 0
        self.parallel_dims.tp = 1
        self.parallel_dims.dp_shard = 1
        self.parallel_dims.dp_replicate = 1

    def train(self):
        patch_device_type_to_cpu()
        run_trainer_simulation(self, self.config.simulation)
