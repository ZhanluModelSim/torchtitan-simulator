# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from torchtitan.trainer import Trainer

from .cpu_env import patch_device_type_to_cpu
from .meta_env import patch_device_type_to_meta
from .trainer_runner import run_trainer_simulation


@dataclass(kw_only=True, slots=True)
class SimulationConfig:
    output_dir: str = "./simulator_output"
    output_formats: list[str] = field(
        default_factory=lambda: ["json", "dot", "chrome_trace", "html", "text"]
    )
    capture_joint_fx: bool = False
    semantic_schedule: bool = False
    cost_model: bool = False
    """When ``True``, run a :class:`CostModel` over the compute graph.

    The model class is determined by ``cost_model_class`` (defaults to
    :class:`MockCostModel`)."""
    cost_model_class: str = ""
    """Fully-qualified Python path for a custom :class:`CostModel` class or
    factory function.  The path must resolve to either:

    * a :class:`CostModel` subclass (instantiated with
      ``cost_model_kwargs``), or
    * a **factory function** that takes no arguments and returns a
      :class:`CostModel` instance (useful for complex setup).

    Example class path: ``\"my_package.MyCostModel\"``
    Example factory path: ``\"my_package.create_cost_model\"``

    If empty and ``cost_model=True``, :class:`MockCostModel` is used.
    """
    cost_model_kwargs: str = ""
    """Keyword arguments forwarded to the ``cost_model_class`` constructor,
    as a JSON string.  When set via ``config_registry``, use a Python dict
    assigned directly; :meth:`~_get_cost_model_kwargs` normalises both forms.

    CLI example::
      --simulation.cost_model_kwargs '{"compute_tflops":312.0,"nvlink_gb_per_s":600.0}'

    config_registry example::
      cost_model_kwargs={"compute_tflops": 312.0, "nvlink_gb_per_s": 600.0}
    """
    comm_backend: str = ""
    """Distributed backend for communication capture.

    ``""`` (empty, default) uses fake_backend (shape-only, no real
    comm).  ``"gloo"`` applies FSDP1 wrapping on CPU tensors and
    captures all-gather / reduce-scatter / all-reduce events via
    ``CommRecorder`` interception.  Uses ``FakeProcessGroup`` for
    ``init_distributed`` so single-process execution suffices — no
    ``torchrun`` required.
    """
    device_mode: str = ""
    """Device mode for model construction and trace capture.
    ``\"\"`` (empty) auto-selects: ``\"meta\"`` for fake_backend, ``\"cpu\"``
    for gloo.  ``\"meta\"`` creates shape-only parameters (0 bytes memory),
    suitable for simulating arbitrarily large models.  ``\"cpu\"`` creates
    real CPU tensors (required for gloo comm capture)."""


def _cpu_noop_parallelize(model, **__):
    """CPU-only parallelize stub: return model unchanged.

    The real ``parallelize_llama`` calls ``apply_fsdp`` / ``fully_shard``
    which allocate CUDA tensors that cannot be materialised on CPU-only
    builds.  Skipping FSDP/TP is safe because the interception-based
    runtime capture records the actual ops that execute.
    """
    return model


def _cpu_gloo_parallelize_llama(model: Any, **__: Any) -> Any:
    """CPU+gloo parallelize stub: return model unchanged.

    FSDP1 wrapping is applied **after** ``Trainer.__init__`` completes
    (when parameters are fully materialised on CPU), not here during
    model construction where parameters are still on ``meta`` device.
    """
    return model


def _cpu_gloo_parallelize_dsv4(model: Any, **__: Any) -> Any:
    """CPU+gloo DeepSeek V4 parallelize stub (see ``_cpu_gloo_parallelize_llama``)."""
    return _cpu_gloo_parallelize_llama(model, **__)


def _apply_fsdp1_on_cpu(model: Any) -> Any:
    """Wrap a fully-materialised CPU model with FSDP1 for comm capture.

    FSDP1 ``SHARD_GRAD_OP`` sharding on CPU creates real all-gather /
    reduce-scatter calls whose shapes the ``CommRecorder`` intercepts.
    Requires ``dist.is_initialized()`` with ``world_size > 1``.
    """
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return model
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
        )

        return FSDP(
            model,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            device_id=torch.device("cpu"),
        )
    except Exception:
        return model


def _cpu_noop_pipeline(model, parallelize_fn=None, **__):
    """CPU-only pipelining stub: apply parallelize_fn then return single-part list.

    The real ``pipeline_llm`` shards the model across pipeline stages,
    which triggers the same meta-tensor problem as ``parallelize_llama``.
    For simulation we treat the whole model as a single stage, but
    still apply the ``parallelize_fn`` (e.g. FSDP1 wrapping for gloo
    mode) so that communication ops are present in the forward pass.
    """
    if parallelize_fn is not None:
        model = parallelize_fn(model, **__)
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
        sim_opts = config.simulation
        comm_backend = getattr(sim_opts, "comm_backend", "") or ""
        device_mode = getattr(sim_opts, "device_mode", "") or ""
        if not device_mode:
            device_mode = "meta" if comm_backend != "gloo" else "cpu"
        sim_opts.device_mode = device_mode

        if device_mode == "meta":
            patch_device_type_to_meta()
        else:
            patch_device_type_to_cpu()

        pp = int(getattr(config.parallelism, "pipeline_parallel_degree", 1) or 1)
        tp = int(getattr(config.parallelism, "tensor_parallel_degree", 1) or 1)
        ds = int(getattr(config.parallelism, "data_parallel_shard_degree", -1) or -1)
        dr = int(getattr(config.parallelism, "data_parallel_replicate_degree", 1) or 1)
        if ds < 0:
            ds = 1
        if pp * tp * ds * dr > 1:
            _set_fake_world_size(config)

        # Force comm.mode to fake_backend so init_distributed uses the
        # fake process group (no NCCL/gloo rendezvous, no multi-process
        # requirement).  The simulator captures communication separately
        # via CommRecorder/FSDP hooks, not through init_distributed.
        config.comm.mode = "fake_backend"

        if comm_backend == "gloo":
            model_name = getattr(config.model_spec, "name", "")
            if "deepseek" in model_name.lower():
                config.model_spec.parallelize_fn = _cpu_gloo_parallelize_dsv4
            else:
                config.model_spec.parallelize_fn = _cpu_gloo_parallelize_llama
        else:
            config.model_spec.parallelize_fn = _cpu_noop_parallelize
        config.model_spec.pipelining_fn = _cpu_noop_pipeline

        super().__init__(config)

        self.parallel_dims.pp = 0
        self.parallel_dims.tp = 1
        self.parallel_dims.dp_shard = 1
        self.parallel_dims.dp_replicate = 1

        # Apply FSDP1 wrapping after model is fully initialised on CPU.
        # Must happen after super().__init__() because the Trainer builds
        # the model on meta, then calls to_empty + init_weights to
        # materialise CPU tensors.  FSDP1 on meta/empty tensors crashes.
        if comm_backend == "gloo":
            self.model_parts = [_apply_fsdp1_on_cpu(m) for m in self.model_parts]

    def train(self):
        comm_backend = getattr(self.config.simulation, "comm_backend", "") or ""
        if comm_backend == "gloo":
            patch_device_type_to_cpu()
        else:
            patch_device_type_to_meta()
        run_trainer_simulation(self, self.config.simulation)
