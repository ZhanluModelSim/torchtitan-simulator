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
    return [model], None, False, False


class SimulationTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def __init__(self, config: Config):
        patch_device_type_to_cpu()

        # Override the model's parallelize / pipelining callables so that
        # Trainer.__init__ does not invoke FSDP/TP/PP wrappers which rely
        # on CUDA device handles deep inside PyTorch.
        config.model_spec.parallelize_fn = _cpu_noop_parallelize
        config.model_spec.pipelining_fn = _cpu_noop_pipeline

        super().__init__(config)

    def train(self):
        patch_device_type_to_cpu()
        run_trainer_simulation(self, self.config.simulation)
