# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
from collections.abc import Callable

from contextlib import contextmanager
from typing import Any, TypeVar

import torch
import torch.nn as nn

from torchtitan.tools.logging import logger
from .capture.unified_trace import TraceRecorder, unified_trace

from .cost_model import apply_cost_model, CostModel
from .export import (
    export_chrome_trace,
    export_dot,
    export_html,
    export_json,
    export_kernel_summary_csv,
    export_text_summary,
)
from .extension_hooks import postprocess_extension_result
from .memory_estimator import dtype_size
from .nodes import DataEdge, OpNode, TensorMeta
from .schedule.schedule_extract import extract_schedule_from_pytorch

_T = TypeVar("_T")


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


def _export_result(result: Any, output_dir: str, output_formats: list[str]) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
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
    if "csv" in output_formats:
        export_kernel_summary_csv(
            result, os.path.join(output_dir, "kernel_summary.csv")
        )


def _export_workload_graph(result: Any, config: Any, sim_opts: Any) -> None:
    """Project the captured result into the spec L1/L2/L3 IR and export it.

    Emitted as ``workload_graph.json`` (a new, additive artifact).  The
    projection is derived entirely from captured data + declared config, so a
    failure here must never break the primary export path.
    """
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    if "json" not in getattr(sim_opts, "output_formats", []):
        return
    try:
        from .ir import build_workload_graph

        workload = build_workload_graph(result, config)
        output_dir = sim_opts.output_dir
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "workload_graph.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(workload.to_dict(), f, indent=2, default=str)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to export layered workload graph: %s", exc)


def _inject_semantic_schedule(result: Any, config: Any) -> None:
    """Append a semantic PP / TP / DP / FSDP2 schedule to *result*.

    Reads parallelism settings from *config* and constructs a real PyTorch
    schedule object with mock stages to extract the exact action table.
    The HTML visualisation then shows the full multi-rank topology matching
    upstream PyTorch behaviour.
    """
    from .nodes import TrainingSchedule

    parallelism = getattr(config, "parallelism", None)
    if parallelism is None:
        return

    pp_degree = int(getattr(parallelism, "pipeline_parallel_degree", 1) or 1)
    tp_degree = int(getattr(parallelism, "tensor_parallel_degree", 1) or 1)
    dp_shard = int(getattr(parallelism, "data_parallel_shard_degree", 1) or 1)
    if dp_shard < 0:
        dp_shard = 1
    dp_repl = int(getattr(parallelism, "data_parallel_replicate_degree", 1) or 1)
    dp_degree = dp_shard * dp_repl

    schedule_name = str(
        getattr(parallelism, "pipeline_parallel_schedule", "1F1B") or "1F1B"
    )
    num_mb = int(getattr(parallelism, "pipeline_parallel_microbatch_size", 8) or 8)
    virtual = 2 if "Interleaved" in schedule_name else 1
    num_stages = pp_degree * virtual

    semantic = extract_schedule_from_pytorch(
        pp_degree=pp_degree,
        tp_degree=tp_degree,
        dp_degree=dp_degree,
        num_stages=num_stages,
        n_microbatches=num_mb,
        schedule_name=schedule_name,
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


def _project_pp_comm_from_schedule(
    result: Any,
    trainer: Any,
) -> None:
    """Project PP send/recv from the semantic schedule into the compute graph.

    PP communication uses ``dist.send()``/``dist.recv()`` directly (not
    DTensor), so it cannot be naturally captured by ``UnifiedTraceMode``.
    Instead, we project the PP events from ``result.schedule`` (produced by
    ``extract_schedule_from_pytorch`` based on the real ``_PipelineSchedule``)
    into the compute graph as ``comm_p2p`` OpNodes.

    This is "schedule-derived" (from real PyTorch schedule) rather than
    "synthetic" (heuristic).  See DESIGN.md §15.2.
    """
    schedule = result.schedule
    if schedule is None:
        return

    _PP_EVENT_MAP = {
        "pp_send_activation": ("pp_send_activation", "forward", "send"),
        "pp_recv_activation": ("pp_recv_activation", "forward", "recv"),
        "pp_send_gradient": ("pp_send_gradient", "backward", "send"),
        "pp_recv_gradient": ("pp_recv_gradient", "backward", "recv"),
    }

    pp_events = [
        e for e in schedule.events
        if e.event_type in _PP_EVENT_MAP
    ]
    if not pp_events:
        return

    # Deduplicate by (event_type, pp_stage, microbatch_idx) — multiple ranks
    # share the same pp_stage; create one representative OpNode per unique key
    # so all ranks' events link to the same node and get consistent duration.
    seen_keys: set[tuple[str, int | None, int | None]] = set()
    unique_pp_events = []
    for ev in pp_events:
        key = (ev.event_type, ev.pp_stage, ev.microbatch_idx)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_pp_events.append(ev)

    # Build send→recv dep map from schedule's pp_comm edges
    send_to_recv: dict[str, list[str]] = {}
    for dep in schedule.deps:
        if dep.dep_type == "pp_comm":
            send_to_recv.setdefault(dep.from_event_id, []).append(
                dep.to_event_id
            )

    # Compute tensor shapes from config (same logic as old injector)
    from torchtitan.config import TORCH_DTYPE_MAP

    training = trainer.config.training
    mp_param = getattr(training, "mixed_precision_param", "bfloat16")
    dtype_str = str(TORCH_DTYPE_MAP.get(mp_param, torch.bfloat16))
    mp_reduce = getattr(training, "mixed_precision_reduce", "float32")
    reduce_dtype_str = str(TORCH_DTYPE_MAP.get(mp_reduce, torch.float32))
    seq_len = training.seq_len
    batch_size = training.local_batch_size
    hidden = (
        _guess_hidden_dim(trainer.model_parts[0])
        if trainer.model_parts
        else 512
    )
    activation_numel = batch_size * seq_len * hidden

    graph = result.compute_graph
    event_id_to_node_id: dict[str, str] = {}
    counter = [len(graph.nodes)]

    def _next_id() -> str:
        counter[0] += 1
        return f"comm_pp_{counter[0]:07d}"

    for ev in unique_pp_events:
        op_name, phase, comm_op = _PP_EVENT_MAP[ev.event_type]
        dt = dtype_str if phase == "forward" else reduce_dtype_str
        node_id = _next_id()
        node = OpNode(
            node_id=node_id,
            op_name=op_name,
            op_type="comm_p2p",
            phase=phase,
            pp_stage=ev.pp_stage,
            pp_rank=ev.pp_rank,
            microbatch_idx=ev.microbatch_idx,
            inputs=[
                TensorMeta(
                    shape=(activation_numel,), dtype=dt, device="cpu"
                )
            ],
            outputs=[
                TensorMeta(
                    shape=(activation_numel,), dtype=dt, device="cpu"
                )
            ],
            comm_op=comm_op,
            comm_group_size=2,
            attrs={"schedule_derived": True, "pp": True},
        )
        graph.add_node(node)
        event_id_to_node_id[ev.event_id] = node_id

        result.comm_events.append(
            {
                "event_id": node_id,
                "op": comm_op,
                "group_size": 2,
                "phase": phase,
                "pp_stage": ev.pp_stage,
                "pp_rank": ev.pp_rank,
                "microbatch": ev.microbatch_idx,
                "tensor_meta": {
                    "shape": [batch_size, seq_len, hidden],
                    "dtype": dt,
                    "device": "cpu",
                },
                "source_node_ids": [],
                "schedule_derived": True,
            }
        )

    # Add pp_p2p edges from schedule deps and link recv source_node_ids
    for send_eid, recv_eids in send_to_recv.items():
        send_nid = event_id_to_node_id.get(send_eid)
        if send_nid is None:
            continue
        for recv_eid in recv_eids:
            recv_nid = event_id_to_node_id.get(recv_eid)
            if recv_nid is None:
                continue
            graph.add_edge(DataEdge(send_nid, recv_nid, "pp_p2p"))
            for ce in result.comm_events:
                if ce.get("event_id") == recv_nid:
                    ce["source_node_ids"] = [send_nid]
                    break


def _inject_synthetic_compute_anchors(result: Any, trainer: Any) -> None:
    """Inject synthetic compute so each phase has enough Cube/Vec lane signal."""
    graph = result.compute_graph
    if graph is None:
        return

    def _is_cube_compute(op_name: str) -> bool:
        name = op_name.lower()
        return any(
            kw in name
            for kw in ("mm", "matmul", "bmm", "addmm", "linear", "conv", "gemm", "dot")
        )

    parallelism = trainer.config.parallelism
    pp = max(int(getattr(parallelism, "pipeline_parallel_degree", 1) or 1), 1)
    model_parts = getattr(trainer, "model_parts", [])
    num_layers = max(_infer_num_layers(model_parts), 1)
    lane_target = max(pp * num_layers, pp)

    lane_gaps: dict[str, dict[str, int]] = {}
    for phase in ("forward", "backward"):
        phase_compute = [
            node
            for node in graph.nodes.values()
            if node.phase == phase and node.op_type == "compute"
        ]
        cube_count = sum(1 for node in phase_compute if _is_cube_compute(node.op_name))
        vec_count = len(phase_compute) - cube_count
        missing_cube = max(0, lane_target - cube_count)
        missing_vec = max(0, lane_target - vec_count)
        if missing_cube > 0 or missing_vec > 0:
            lane_gaps[phase] = {"cube": missing_cube, "vec": missing_vec}

    if not lane_gaps:
        return

    hidden_dim = _guess_hidden_dim(model_parts[0]) if model_parts else 1024
    hidden_dim = max(int(hidden_dim), 1)

    from torchtitan.config import TORCH_DTYPE_MAP

    mp_param = getattr(trainer.config.training, "mixed_precision_param", "bfloat16")
    dtype_str = str(TORCH_DTYPE_MAP.get(mp_param, torch.bfloat16))

    counter = [len(graph.nodes)]

    def _next_id(phase: str, kind: str, stage: int) -> str:
        counter[0] += 1
        return f"compute_syn_{phase}_{kind}_{stage}_{counter[0]:07d}"

    for phase, missing in lane_gaps.items():
        max_missing = max(missing["cube"], missing["vec"])
        for i in range(max_missing):
            stage_idx = i % pp
            pp_rank = stage_idx
            cube: OpNode | None = None
            vec: OpNode | None = None

            if i < missing["cube"]:
                cube = OpNode(
                    node_id=_next_id(phase, "cube", stage_idx),
                    op_name="aten.mm.default",
                    op_type="compute",
                    phase=phase,
                    pp_stage=stage_idx,
                    pp_rank=pp_rank,
                    microbatch_idx=0,
                    inputs=[
                        TensorMeta(
                            shape=(1, hidden_dim), dtype=dtype_str, device="cpu"
                        ),
                        TensorMeta(
                            shape=(hidden_dim, hidden_dim),
                            dtype=dtype_str,
                            device="cpu",
                        ),
                    ],
                    outputs=[
                        TensorMeta(shape=(1, hidden_dim), dtype=dtype_str, device="cpu")
                    ],
                    attrs={"synthetic_compute_anchor": True, "synthetic_lane": "cube"},
                )
                graph.add_node(cube)

            if i < missing["vec"]:
                vec = OpNode(
                    node_id=_next_id(phase, "vec", stage_idx),
                    op_name="aten.add.Tensor",
                    op_type="compute",
                    phase=phase,
                    pp_stage=stage_idx,
                    pp_rank=pp_rank,
                    microbatch_idx=0,
                    inputs=[
                        TensorMeta(
                            shape=(1, hidden_dim), dtype=dtype_str, device="cpu"
                        ),
                        TensorMeta(
                            shape=(1, hidden_dim), dtype=dtype_str, device="cpu"
                        ),
                    ],
                    outputs=[
                        TensorMeta(shape=(1, hidden_dim), dtype=dtype_str, device="cpu")
                    ],
                    attrs={"synthetic_compute_anchor": True, "synthetic_lane": "vec"},
                )
                graph.add_node(vec)

            if cube is not None and vec is not None:
                graph.add_edge(
                    DataEdge(
                        src_node_id=cube.node_id,
                        dst_node_id=vec.node_id,
                        edge_type="data",
                    )
                )


def _infer_num_layers(model_parts: list[Any]) -> int:
    """Derive the number of transformer layers from model config or structure.

    Tries in order:
    1. model.config.n_layers (all TorchTitan models define this)
    2. len(model.layers) attribute
    3. Fallback: unique parameter-prefix count (approximate)
    """
    for part in model_parts:
        config = getattr(part, "config", None)
        if config is not None:
            n = getattr(config, "n_layers", None)
            if n is not None:
                return n
        n = getattr(part, "n_layers", None)
        if n is not None:
            return n
        layers_attr = getattr(part, "layers", None)
        if layers_attr is not None and isinstance(
            layers_attr, (list, tuple, nn.Sequential, nn.ModuleList)
        ):
            return len(layers_attr)
    # Fallback: count unique 2-prefix groups from parameter names
    per_module_numel: dict[str, int] = {}
    for part in model_parts:
        for name, param in part.named_parameters():
            if param.requires_grad:
                prefix = ".".join(name.split(".")[:2])
                per_module_numel[prefix] = (
                    per_module_numel.get(prefix, 0) + param.numel()
                )
    return max(len(per_module_numel), 1)


def _guess_hidden_dim(model: Any) -> int:
    """Guess the hidden dimension from a model's first Linear layer."""
    import torch.nn as nn

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            return mod.in_features
    return 512  # fallback


def _run_with_phase(
    recorder: Any, phase: str, fn: Callable[..., _T], *args: Any, **kwargs: Any
) -> _T:
    """Execute ``fn`` while temporarily setting ``recorder.current_phase``."""
    prev_phase = getattr(recorder, "current_phase", "forward")
    recorder.current_phase = phase
    try:
        return fn(*args, **kwargs)
    finally:
        recorder.current_phase = prev_phase


@contextmanager
def _patch_backward_phase(recorder: Any):
    """Patch autograd entrypoints so backward ops are tagged as ``backward``."""
    orig_tensor_backward = torch.Tensor.backward
    orig_autograd_backward = torch.autograd.backward

    def _tensor_backward_wrapper(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return _run_with_phase(
            recorder, "backward", orig_tensor_backward, self, *args, **kwargs
        )

    def _autograd_backward_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _run_with_phase(
            recorder, "backward", orig_autograd_backward, *args, **kwargs
        )

    torch.Tensor.backward = (
        _tensor_backward_wrapper  # pyrefly: ignore[invalid-assignment]
    )
    torch.autograd.backward = (
        _autograd_backward_wrapper  # pyrefly: ignore[invalid-assignment]
    )
    try:
        yield
    finally:
        torch.Tensor.backward = (
            orig_tensor_backward  # pyrefly: ignore[invalid-assignment]
        )
        torch.autograd.backward = (
            orig_autograd_backward  # pyrefly: ignore[invalid-assignment]
        )


def run_trainer_simulation(trainer: Any, sim_opts: Any) -> None:
    """Run one simulated training step seamlessly utilizing the native Trainer."""
    import torch._subclasses.fake_impls

    import torchtitan.distributed.utils as dist_utils
    import torchtitan.observability.structured_logger as sl
    from torchtitan.trainer import Trainer

    # 1. Patch meta vs meta:0 issues by enforcing clean meta device
    trainer.device = torch.device("meta")

    # 2. Patch FakeTensor conversions that crash native train_step
    orig_local_scalar_dense = torch._subclasses.fake_impls.op_implementations_dict.get(
        torch.ops.aten._local_scalar_dense.default
    )

    def _mock_local_scalar_dense(fake_mode, func, *args, **kwargs):
        return 0

    torch._subclasses.fake_impls.op_implementations_dict[
        torch.ops.aten._local_scalar_dense.default
    ] = _mock_local_scalar_dense

    from torch._subclasses.fake_tensor import FakeTensor

    orig_format = FakeTensor.__format__
    FakeTensor.__format__ = lambda self, format_spec: "0.0"

    # DTensor wraps FakeTensor but has its own __format__; patch it too
    # so logging/metrics that format DTensors don't crash on meta.
    try:
        from torch.distributed.tensor import DTensor as _DTensor
        _orig_dt_format = _DTensor.__format__
        _DTensor.__format__ = lambda self, format_spec: "0.0"
    except Exception:
        _orig_dt_format = None

    # 3. Patch distributed and optimizer operations that expect real tensors
    orig_clip_grad_norm = dist_utils.clip_grad_norm_
    orig_dist_sum = dist_utils.dist_sum
    orig_dist_max = dist_utils.dist_max

    dist_utils.clip_grad_norm_ = lambda *args, **kwargs: torch.tensor(
        0.0, device="meta"
    )
    dist_utils.dist_sum = lambda t, *args, **kwargs: t
    dist_utils.dist_max = lambda t, *args, **kwargs: t

    orig_get_mesh = trainer.parallel_dims.get_optional_mesh
    orig_get_strict_mesh = trainer.parallel_dims.get_mesh
    trainer.parallel_dims.get_optional_mesh = lambda *args, **kwargs: None
    trainer.parallel_dims.get_mesh = lambda *args, **kwargs: None

    orig_optim_step = trainer.optimizers.step
    orig_lr_step = trainer.lr_schedulers.step
    trainer.optimizers.step = lambda *args, **kwargs: None
    trainer.lr_schedulers.step = lambda *args, **kwargs: None

    # Patch log_trace_scalar to avoid int() crash on meta tensors
    orig_log = sl.log_trace_scalar

    def safe_log(d):
        safe_dict = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                safe_dict[k] = 0
            else:
                try:
                    safe_dict[k] = int(v)
                except Exception:
                    safe_dict[k] = 0
        orig_log(safe_dict)

    sl.log_trace_scalar = safe_log

    recorder = TraceRecorder(rank=int(os.environ.get("RANK", "0")))

    data_iterator = trainer.batch_generator(trainer.dataloader)

    # Override gradient_accumulation_steps to 1 for simulation. On meta device,
    # each dispatched op creates Python objects (OpNode, TensorMeta, autograd
    # Node ≈ 500 bytes/op). With 61 layers × 256 experts × 16 ga_steps, this
    # would consume ~2.4 GB of Python object memory. One microbatch is
    # sufficient to capture the compute graph and communication operators.
    # PP schedule extraction uses pipeline_parallel_microbatch_size (not ga_steps).
    trainer.gradient_accumulation_steps = 1

    # Pre-fetch batches outside of FakeTensorMode to avoid dataloader internal crashes
    batches = []
    for _ in range(trainer.gradient_accumulation_steps):
        batches.append(next(data_iterator))

    def mock_data_iterator():
        for batch in batches:
            yield batch

    use_fake = (getattr(sim_opts, "comm_backend", "") or "") != "gloo"

    # Phase 4: Apply meta device patches for natural FSDP communication emission
    from .meta_device_patches import apply_meta_device_patches, restore_meta_device_patches
    
    if use_fake:
        apply_meta_device_patches()

    try:
        with unified_trace(
            recorder,
            use_fake_mode=use_fake,
            capture_comm=not use_fake,
            capture_fsdp=not use_fake,
        ):
            with _patch_backward_phase(recorder):
                Trainer.train_step(trainer, mock_data_iterator())
    finally:
        # Phase 4: Restore meta device patches
        if use_fake:
            restore_meta_device_patches()
        
        # Restore patched methods
        dist_utils.clip_grad_norm_ = orig_clip_grad_norm
        dist_utils.dist_sum = orig_dist_sum
        dist_utils.dist_max = orig_dist_max
        trainer.parallel_dims.get_optional_mesh = orig_get_mesh
        trainer.parallel_dims.get_mesh = orig_get_strict_mesh
        trainer.optimizers.step = orig_optim_step
        trainer.lr_schedulers.step = orig_lr_step
        sl.log_trace_scalar = orig_log
        FakeTensor.__format__ = orig_format
        if _orig_dt_format is not None:
            _DTensor.__format__ = _orig_dt_format

        # Restore local_scalar_dense
        if orig_local_scalar_dense:
            torch._subclasses.fake_impls.op_implementations_dict[
                torch.ops.aten._local_scalar_dense.default
            ] = orig_local_scalar_dense
        else:
            del torch._subclasses.fake_impls.op_implementations_dict[
                torch.ops.aten._local_scalar_dense.default
            ]

    result = recorder.build_result()
    result.metadata["operator_swimlane_comm_scope"] = str(
        getattr(sim_opts, "operator_swimlane_comm_scope", "model_only") or "model_only"
    ).lower()
    ga_steps = getattr(trainer, "gradient_accumulation_steps", None)
    if ga_steps:
        result.metadata["gradient_accumulation_steps"] = int(ga_steps)

    # Schedule-derived PP communication: project PP send/recv from the
    # semantic schedule (real _PipelineSchedule) into the compute graph.
    # Must run AFTER _inject_semantic_schedule (which populates result.schedule)
    # and BEFORE cost_model (so PP nodes get annotated).
    if getattr(sim_opts, "semantic_schedule", False):
        _inject_semantic_schedule(result, trainer.config)
    if use_fake:
        _project_pp_comm_from_schedule(result, trainer)
        _inject_synthetic_compute_anchors(result, trainer)

    if getattr(sim_opts, "cost_model", False):
        cm = _import_cost_model(
            getattr(sim_opts, "cost_model_class", "")
            or "torchtitan.experiments.simulator.cost_model.MockCostModel",
            _get_cost_model_kwargs(sim_opts),
        )
        apply_cost_model(result, cm)

    from torchtitan.experiments.simulator.memory_estimator import (
        attach_model_state_memory,
        build_runtime_memory,
    )

    memory_events, memory_summary = build_runtime_memory(
        result.compute_graph,
        result.comm_events,
        existing_metadata=result.metadata,
    )
    result.memory_events.extend(memory_events)
    result.metadata.update(memory_summary)
    # Also write into the "memory" sub-dict so HTML/exporters that read
    # metadata["memory"]["peak_live_bytes"] find the value.
    result.metadata.setdefault("memory", {}).update(memory_summary)

    attach_model_state_memory(
        result,
        trainer.model_parts,
        optimizer_name=trainer.optimizers.optimizers[0].__class__.__name__
        if trainer.optimizers.optimizers
        else None,
        parallelism_config=trainer.config.parallelism,
    )

    # --- Unified metric computation (before export, format-order-independent) ---
    # Populate DES metadata once here so all exporters (including text-only)
    # can read it without lazy-loading side effects.
    from .export import _populate_des_metadata, _inject_schedule_timing

    _populate_des_metadata(result)
    # Inject schedule timing into result.metadata so DES memory timeline
    # and text summary can use it regardless of which formats are exported.
    _result_dict = result.to_dict()
    _inject_schedule_timing(_result_dict, result)
    result.metadata["perf_schedule"] = _result_dict.get("perf_schedule", {})
    if "schedule" in _result_dict and "events" in _result_dict["schedule"]:
        result.metadata["_schedule_events_enriched"] = _result_dict["schedule"]["events"]

    postprocess_extension_result(result, trainer, sim_opts)
    _export_result(result, sim_opts.output_dir, sim_opts.output_formats)
    _export_workload_graph(result, trainer.config, sim_opts)
