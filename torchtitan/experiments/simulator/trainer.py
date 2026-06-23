# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
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
    mode: str = "all"
    max_seq_len: int = 128
    batch_size: int = 2
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
    operator_swimlane_comm_scope: str = "model_only"
    """Communication scope for forward/backward operator swimlanes in HTML.

    ``"model_only"`` (default) hides synthetic scheduling comm nodes
    (FSDP/PP/DP) from operator swimlanes while keeping them available in
    schedule swimlanes and raw JSON. TP/CP/EP comm remains visible.
    ``"all"`` shows every comm node.
    """


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


def _cpu_pp_module_split(model: Any, config: Any, model_config: Any) -> list[Any]:
    """PP-split model on CPU without PipelineStage/DeviceMesh.

    Reuses upstream ``_generate_llm_fqn_per_model_part`` and
    ``_split_module`` but skips ``_get_pipeline_metadata`` (which
    depends on ``model_config.layers``).
    """
    from torchtitan.distributed.pipeline_parallel import (
        _generate_llm_fqn_per_model_part,
        _split_module,
    )

    pp_degree = int(getattr(config.parallelism, "pipeline_parallel_degree", 1) or 1)
    schedule_name = str(
        getattr(config.parallelism, "pipeline_parallel_schedule", "1F1B") or "1F1B"
    )

    from torch.distributed.pipelining.schedules import (
        get_schedule_class,
        PipelineScheduleSingle,
    )

    schedule_class = get_schedule_class(schedule_name)
    is_single = issubclass(schedule_class, PipelineScheduleSingle)
    vpp = 1 if is_single else 2
    num_virtual_stages = pp_degree * vpp

    num_layers = getattr(model_config, "num_layers", 0)
    if not num_layers:
        # Fallback: count from model.layers (ModuleDict or ModuleList)
        layers_attr = getattr(model, "layers", None)
        if layers_attr is not None:
            num_layers = len(layers_attr)
        else:
            num_layers = 0

    module_names = _generate_llm_fqn_per_model_part(
        num_virtual_stages, num_layers, input_weight=1, output_weight=1
    )

    model_parts = []
    for stage_module_names in module_names:
        part = _split_module(model, stage_module_names)
        model_parts.append(part)
    return model_parts


def _cpu_semantic_pipeline(
    model: Any,
    parallelize_fn: Any = None,
    **kwargs: Any,
) -> tuple[Any, list[Any], bool, bool]:
    """PP-split model on CPU for multi-stage tracing

    Accepts ``config`` and ``model_parts_holder`` via ``functools.partial``.
    The upstream ``trainer.py`` passes ``model_config`` through kwargs
    (the ``model_spec.model`` dataclass config with training overrides applied).
    """
    config = kwargs.get("config")
    model_parts_holder = kwargs.get("model_parts_holder")
    model_config = kwargs.get("model_config")

    pp_degree = (
        int(getattr(config.parallelism, "pipeline_parallel_degree", 1) if config else 1)
        or 1
    )

    if parallelize_fn is not None:
        model = parallelize_fn(model, **kwargs)

    if pp_degree <= 1 or model_parts_holder is None:
        return None, [model], True, True

    model_parts = _cpu_pp_module_split(model, config, model_config)
    # Move parts back to meta device immediately to avoid OOM for large
    # models (1T+ parameters).  The Trainer loop that follows will call
    # to_empty + init_weights on model_parts, which would materialize
    # 1T+ floats on CPU.  We skip that path by setting parallel_dims.pp=0
    # after super().__init__() and replacing model_parts with these meta
    # copies.
    for part in model_parts:
        part.to_empty(device="meta")
    model_parts_holder.clear()
    model_parts_holder.extend(model_parts)

    class MockSchedule:
        def step(self, *args, **kwargs):
            if "losses" in kwargs and isinstance(kwargs["losses"], list):
                kwargs["losses"].append(torch.tensor(0.0, device="meta"))
            return torch.tensor(0.0, device="meta")

    return MockSchedule(), model_parts, True, True



def _set_fake_world_size(config: Any) -> None:
    """Set ``NGPU``/``WORLD_SIZE`` from parallelism config for semantic schedule mode.

    The simulator runs on a single CPU process, but the semantic schedule
    needs ``ParallelDims`` to validate against the full topology size.
    """
    import os

    p = config.parallelism
    pp = int(getattr(p, "pipeline_parallel_degree", 1) or 1)
    tp = int(getattr(p, "tensor_parallel_degree", 1) or 1)
    cp = int(getattr(p, "context_parallel_degree", 1) or 1)
    ep = int(getattr(p, "expert_parallel_degree", 1) or 1)
    dp_shard = int(getattr(p, "data_parallel_shard_degree", -1) or -1)
    dp_repl = int(getattr(p, "data_parallel_replicate_degree", 1) or 1)
    if dp_shard < 0:
        min_dp_shard = max(1, -(-ep // (cp * tp))) if ep > cp * tp else 1
        dp_shard = min_dp_shard
    world = pp * tp * cp * dp_shard * dp_repl
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

        # Override comm_backend: sim_opts.comm_backend defaults to "gloo" in
        # config_registry, but the actual --comm.mode CLI override tells us
        # the real backend.  Treat fake_backend as "no real comm".
        actual_comm_mode = getattr(config.comm, "mode", "") or ""
        if actual_comm_mode == "fake_backend":
            comm_backend = ""
        sim_opts.comm_backend = comm_backend

        # Determine device_mode AFTER comm_backend finalization so that
        # sim_opts.comm_backend="gloo" with --comm.mode=fake_backend correctly
        # selects meta mode instead of being derailed by the default.
        device_mode = getattr(sim_opts, "device_mode", "") or ""
        if not device_mode:
            device_mode = "meta" if comm_backend != "gloo" else "cpu"
        sim_opts.device_mode = device_mode

        if device_mode == "meta":
            patch_device_type_to_meta()
        else:
            patch_device_type_to_cpu()

        # When running in meta device mode, set deterministic seed explicitly
        # to avoid set_determinism trying to broadcast a meta seed tensor.
        if device_mode == "meta":
            if config.debug.seed is None:
                config.debug.seed = 42

        if comm_backend == "gloo":
            model_name = getattr(config.model_spec, "name", "")
            if "deepseek" in model_name.lower():
                config.model_spec.parallelize_fn = _cpu_gloo_parallelize_dsv4
            else:
                config.model_spec.parallelize_fn = _cpu_gloo_parallelize_llama
        else:
            config.model_spec.parallelize_fn = _cpu_noop_parallelize

        # Use PP-semantic pipeline when PP > 1 and not gloo mode
        self._pp_model_parts: list[Any] = []
        if pp > 1 and comm_backend != "gloo":
            config.model_spec.pipelining_fn = partial(
                _cpu_semantic_pipeline,
                config=config,
                model_parts_holder=self._pp_model_parts,
            )
        else:
            config.model_spec.pipelining_fn = _cpu_noop_pipeline

        super().__init__(config)
        
        if self._pp_model_parts:
            self.model_parts = self._pp_model_parts

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
