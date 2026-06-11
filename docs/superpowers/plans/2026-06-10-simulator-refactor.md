# Simulator Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `torchtitan/experiments/simulator/` for readability, maintainability, and extensibility while preserving all functionality and output.

**Architecture:** Layered incremental refactoring in 4 phases: Foundation (eliminate duplication) → Capture (unify architecture) → Analysis (split large files, decouple) → Output (split export.py, extract JS).

**Tech Stack:** Python 3.10+, PyTorch, salabim, pre-commit (ufmt, flake8, pyrefly, codespell)

**Spec:** `docs/superpowers/specs/2026-06-10-simulator-refactor-design.md`

---

## File Structure (Target State)

```
torchtitan/experiments/simulator/
  __init__.py                    (public API, unchanged exports)
  _recorder_registry.py          (NEW: recorder stack, breaks circular dep)
  simulator.py                   (Simulator class, uses unified_trace internally)
  trainer.py                     (SimulationTrainer, streamlined)
  trainer_runner.py              (run_trainer_simulation orchestrator, ~200 lines)
  run_simulate.py                (CLI entry, uses export.export_result)
  cpu_env.py                     (CPU env + shared device patching factory)
  meta_env.py                    (thin wrapper over cpu_env helpers)
  nodes.py                       (data model, unchanged)
  op_classification.py           (unchanged)
  unified_trace.py               (unified trace, canonical tensor helpers)
  comm_interceptor.py            (imports from _recorder_registry)
  fsdp_tracer.py                 (unchanged)
  fx_capture.py                  (FX capture + from_fx + merge_comm_events)
  cost_model.py                  (CostModel ABC + MockCostModel + apply, ~150 lines)
  cost_estimators.py             (NEW: FLOPs/bytes estimation + overlap strategies)
  schedule_analysis.py           (NEW: schedule-graph linking + critical path)
  des_engine.py                  (DES core + utilization, ~450 lines)
  des_memory.py                  (NEW: compute_des_memory_timeline)
  memory_estimator.py            (canonical dtype sizes, unchanged logic)
  schedule_extract.py            (merged: schedule_extract + pp_schedule_extractor)
  schedule_generator.py          (semantic schedule, uses shared replication)
  synthetic_comm.py              (NEW: synthetic comm injection + model helpers)
  schedule_inject.py             (NEW: semantic schedule injection + parallelism utils)
  extension_hooks.py             (unchanged)
  synthetic_dataloader.py        (unchanged)
  export/                        (NEW: sub-package)
    __init__.py                  (re-exports all public symbols)
    json_export.py
    dot_export.py
    chrome_trace.py
    html_export.py
    text_summary.py
    schedule_timing.py
    export_utils.py
    trace_visualizer.js
  llama3/                        (unchanged)
  deepseek_v4/                   (unchanged)
  tests/test_simulator.py        (updated imports for renamed classes)
```

**Deleted files:** `dispatch_interceptor.py`, `runtime_capture.py`, `graph_assembler.py`, `pp_schedule_extractor.py`

---

## Phase 1: Foundation Layer

### Task 1: Consolidate Device Environment Patching

**Files:**
- Modify: `torchtitan/experiments/simulator/cpu_env.py`
- Modify: `torchtitan/experiments/simulator/meta_env.py`
- Test: `torchtitan/experiments/simulator/tests/test_simulator.py` (TestMetaDevicePatch unchanged)

- [ ] **Step 1: Add shared factory and helpers to cpu_env.py**

Replace `_make_cpu_device_module()` (lines 110-132), `_PATCHED_MODULES` (lines 158-162), and `_patch_torch_cuda_for_cpu()` (lines 182-212) with shared parameterized versions. Add these new functions after the existing `cpu_distributed_context()`:

```python
_PATCHED_MODULES: dict[str, tuple[str, ...]] = {
    "torchtitan.components.metrics": ("device_module",),
    "torchtitan.distributed.parallel_dims": ("device_type",),
    "torchtitan.distributed.utils": ("device_module", "device_type"),
}


def make_device_module(
    device_count: int = 1,
    device_name: str = "CPU_Simulator",
    total_memory: int = 1,
) -> "types.SimpleNamespace":
    """Build a namespace that quacks like torch.cuda on a simulated device."""
    import types

    return types.SimpleNamespace(
        set_device=lambda device: None,
        current_device=lambda: 0,
        device_count=lambda: device_count,
        device_capability=lambda device=None: (0, 0),
        get_device_name=lambda device=None: device_name,
        get_device_properties=lambda device=None: types.SimpleNamespace(
            name=device_name, total_memory=total_memory
        ),
        get_arch_list=lambda: [],
        synchronize=lambda: None,
        memory_allocated=lambda device=None: 0,
        max_memory_allocated=lambda device=None: 0,
        memory_reserved=lambda device=None: 0,
        max_memory_reserved=lambda device=None: 0,
        reset_peak_memory_stats=lambda device=None: None,
        memory_stats=lambda device=None: {},
        empty_cache=lambda: None,
    )


def _patch_downstream_modules(device_type: str, device_module) -> None:
    """Rebind device_module/device_type in downstream torchtitan modules."""
    for mod_name, attrs in _PATCHED_MODULES.items():
        try:
            mod = __import__(mod_name, fromlist=list(attrs))
        except ImportError:
            continue
        for attr in attrs:
            if hasattr(mod, attr):
                if attr == "device_module":
                    setattr(mod, attr, device_module)
                else:
                    setattr(mod, attr, device_type)


def _patch_torch_cuda(device_module) -> None:
    """Replace key torch.cuda entrypoints with stubs from device_module."""
    import torch
    import torch.cuda

    torch.cuda.is_available = lambda: False
    torch.cuda._lazy_init = lambda: None
    torch.cuda.current_device = device_module.current_device
    torch.cuda.device_count = device_module.device_count
    torch.cuda.get_device_name = device_module.get_device_name
    torch.cuda.get_device_properties = device_module.get_device_properties
    torch.cuda.synchronize = device_module.synchronize
    torch.cuda.memory_allocated = device_module.memory_allocated
    torch.cuda.max_memory_allocated = device_module.max_memory_allocated
    torch.cuda.memory_reserved = device_module.memory_reserved
    torch.cuda.max_memory_reserved = device_module.max_memory_reserved
    torch.cuda.reset_peak_memory_stats = device_module.reset_peak_memory_stats
    torch.cuda.memory_stats = device_module.memory_stats
    torch.cuda.empty_cache = device_module.empty_cache
    if not hasattr(torch.cuda, "set_device"):
        torch.cuda.set_device = device_module.set_device
    if not hasattr(torch.cuda, "get_arch_list"):
        torch.cuda.get_arch_list = device_module.get_arch_list
    if not hasattr(torch.cuda, "device_capability"):
        torch.cuda.device_capability = device_module.device_capability
```

Update `_make_cpu_device_module()` to call the factory:

```python
def _make_cpu_device_module():
    """Lazily build a namespace that quacks like torch.cuda / torch.npu on CPU."""
    return make_device_module(device_count=1, device_name="CPU_Simulator", total_memory=1)
```

Update `patch_device_type_to_cpu()` to use the shared helpers:

```python
def patch_device_type_to_cpu() -> None:
    try:
        import torchtitan.tools.utils as tt_utils
        tt_utils.device_type = "cpu"
        tt_utils.device_module = _make_cpu_device_module()
    except ImportError:
        pass
    _patch_downstream_modules("cpu", _make_cpu_device_module())
    _patch_torch_cuda(_make_cpu_device_module())
```

- [ ] **Step 2: Rewrite meta_env.py to use shared helpers**

Replace the entire content of `meta_env.py` (keeping license header and module docstring):

```python
from __future__ import annotations

from .cpu_env import make_device_module, _patch_downstream_modules, _patch_torch_cuda


def _make_meta_device_module():
    """Build a namespace that quacks like torch.cuda but reports zero devices."""
    return make_device_module(
        device_count=0, device_name="Meta_Simulator", total_memory=0
    )


def patch_device_type_to_meta() -> None:
    meta_mod = _make_meta_device_module()
    try:
        import torchtitan.tools.utils as tt_utils
        tt_utils.device_type = "meta"
        tt_utils.device_module = meta_mod
    except ImportError:
        pass
    _patch_downstream_modules("meta", meta_mod)
    _patch_torch_cuda(meta_mod)
```

- [ ] **Step 3: Run tests**

```bash
pytest torchtitan/experiments/simulator/tests/test_simulator.py -v -k "MetaDevice"
```

- [ ] **Step 4: Run full test suite and pre-commit**

```bash
pytest torchtitan/experiments/simulator/tests/test_simulator.py -v
pre-commit run --all-files
```

- [ ] **Step 5: Commit**

```bash
git add torchtitan/experiments/simulator/cpu_env.py torchtitan/experiments/simulator/meta_env.py
git commit -m "refactor(simulator): consolidate device environment patching"
```

---

### Task 2: Extract Shared Utilities (loss, export, dtype, parallelism)

**Files:**
- Create: `torchtitan/experiments/simulator/schedule_inject.py`
- Create: `torchtitan/experiments/simulator/synthetic_comm.py`
- Modify: `torchtitan/experiments/simulator/trainer_runner.py`
- Modify: `torchtitan/experiments/simulator/simulator.py`
- Modify: `torchtitan/experiments/simulator/run_simulate.py`
- Modify: `torchtitan/experiments/simulator/cost_model.py`
- Test: `torchtitan/experiments/simulator/tests/test_simulator.py`

- [ ] **Step 1: Create schedule_inject.py with parallelism reader and semantic schedule injection**

Create `torchtitan/experiments/simulator/schedule_inject.py`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torchtitan.tools.logging import logger


@dataclass
class ParallelismDegrees:
    pp: int
    tp: int
    dp_shard: int
    dp_replicate: int
    dp: int


def read_parallelism_degrees(config: Any) -> ParallelismDegrees:
    parallelism = getattr(config, "parallelism", None)
    if parallelism is None:
        return ParallelismDegrees(pp=1, tp=1, dp_shard=1, dp_replicate=1, dp=1)
    pp = int(getattr(parallelism, "pipeline_parallel_degree", 1) or 1)
    tp = int(getattr(parallelism, "tensor_parallel_degree", 1) or 1)
    ds = int(getattr(parallelism, "data_parallel_shard_degree", 1) or 1)
    if ds < 0:
        ds = 1
    dr = int(getattr(parallelism, "data_parallel_replicate_degree", 1) or 1)
    return ParallelismDegrees(pp=pp, tp=tp, dp_shard=ds, dp_replicate=dr, dp=ds * dr)


def inject_semantic_schedule(result: Any, config: Any) -> None:
    from .nodes import TrainingSchedule
    from .schedule_extract import extract_schedule_from_pytorch

    parallelism = getattr(config, "parallelism", None)
    if parallelism is None:
        return

    par = read_parallelism_degrees(config)
    schedule_name = str(
        getattr(parallelism, "pipeline_parallel_schedule", "1F1B") or "1F1B"
    )
    num_mb = int(getattr(parallelism, "pipeline_parallel_microbatch_size", 8) or 8)
    virtual = 2 if "Interleaved" in schedule_name else 1
    num_stages = par.pp * virtual

    semantic = extract_schedule_from_pytorch(
        pp_degree=par.pp,
        tp_degree=par.tp,
        dp_degree=par.dp,
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
```

- [ ] **Step 2: Create synthetic_comm.py with comm injection and model helpers**

Move `_inject_synthetic_comm_events()`, `_infer_num_layers()`, and `_guess_hidden_dim()` from `trainer_runner.py` (lines 178-449) to `torchtitan/experiments/simulator/synthetic_comm.py`. Use `read_parallelism_degrees()` from `schedule_inject`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from torchtitan.tools.logging import logger

from .memory_estimator import dtype_size
from .nodes import DataEdge, OpNode, TensorMeta
from .schedule_inject import read_parallelism_degrees


Copy `_infer_num_layers` (trainer_runner.py:408-439) as `infer_num_layers`, `_guess_hidden_dim` (trainer_runner.py:442-449) as `guess_hidden_dim`, and `_inject_synthetic_comm_events` (trainer_runner.py:178-406) as `inject_synthetic_comm_events`.

Changes from the original:
1. Replace the parallelism degree reading block (lines 202-204) with:

```python
    par = read_parallelism_degrees(trainer.config)
    tp = par.tp
    ds = par.dp_shard
    pp = par.pp
```

2. Replace `_infer_num_layers(model_parts)` calls with `infer_num_layers(model_parts)`
3. Replace `_guess_hidden_dim(model_parts[0])` calls with `guess_hidden_dim(model_parts[0])`
4. Remove the `if getattr(sim_opts, "comm_backend", "") == "gloo":` guard block (lines 191-195) — it stays in the caller (`trainer_runner.run_trainer_simulation`) instead.
```

- [ ] **Step 3: Add compute_loss helper to unified_trace.py**

Add to `torchtitan/experiments/simulator/unified_trace.py` after the existing helper functions (after line 100):

```python
def compute_loss(
    output: Any,
    loss_fn: Any | None = None,
    labels: Any | None = None,
) -> "torch.Tensor":
    if loss_fn is not None and labels is not None:
        return loss_fn(output, labels)
    if isinstance(output, torch.Tensor):
        return output.sum()
    flat, _ = pytree.tree_flatten(output)
    return sum(t.sum() for t in flat if isinstance(t, torch.Tensor))
```

- [ ] **Step 4: Add export_result to export.py**

Add to `torchtitan/experiments/simulator/export.py` after the existing export functions:

```python
def export_result(
    result: Any,
    output_dir: str,
    output_formats: list[str],
    log_fn: Any | None = None,
    print_summary: bool = False,
) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    os.makedirs(output_dir, exist_ok=True)

    if "json" in output_formats:
        p = os.path.join(output_dir, "simulation_result.json")
        export_json(result, p)
        if log_fn:
            log_fn(f"JSON → {p}")
    if "dot" in output_formats:
        p = os.path.join(output_dir, "compute_graph.dot")
        export_dot(result.compute_graph, p)
        if log_fn:
            log_fn(f"DOT  → {p}")
    if "chrome_trace" in output_formats:
        p = os.path.join(output_dir, "trace.json")
        export_chrome_trace(result, p)
        if log_fn:
            log_fn(f"Chrome trace → {p}")
    if "html" in output_formats:
        p = os.path.join(output_dir, "trace.html")
        export_html(result, p)
        if log_fn:
            log_fn(f"HTML trace → {p}")
    if "text" in output_formats:
        summary = export_text_summary(result)
        p = os.path.join(output_dir, "summary.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(summary)
        if log_fn:
            log_fn(f"Text summary → {p}")
        if print_summary:
            print(summary)
```

- [ ] **Step 5: Unify dtype mapping in cost_model.py**

In `cost_model.py`, replace the `_tensor_bytes` function's inline `dtype_bytes` dict (lines 215-233) with an import from `memory_estimator`:

```python
from .memory_estimator import dtype_size as _dtype_size


def _tensor_bytes(
    shape: tuple[int, ...], dtype: str, default_seq_len: int = 4096
) -> int:
    return _numel(shape, default_seq_len) * _dtype_size(dtype) or _numel(shape, default_seq_len) * 2
```

- [ ] **Step 6: Update trainer_runner.py to use new modules**

Replace the `_inject_semantic_schedule` function (lines 129-175) with an import:

```python
from .schedule_inject import inject_semantic_schedule
```

Replace the `_inject_synthetic_comm_events`, `_infer_num_layers`, `_guess_hidden_dim` functions (lines 178-449) with imports:

```python
from .synthetic_comm import inject_synthetic_comm_events, infer_num_layers, guess_hidden_dim
```

Replace `_export_result` (lines 110-127) with an import:

```python
from .export import export_result as _export_result
```

Update `run_trainer_simulation()` to use the new names:
- Line 491-497: Replace loss computation with `from .unified_trace import compute_loss; ... loss = compute_loss(output)`
- Line 541: `inject_synthetic_comm_events(result, trainer, sim_opts)`
- Line 547: `inject_semantic_schedule(result, trainer.config)`
- Line 608: `_export_result(result, sim_opts.output_dir, output_formats)`

- [ ] **Step 7: Update simulator.py to use shared utilities**

Replace the loss computation blocks in `simulate_runtime()` (lines 207-215) and `simulate_unified()` (lines 347-355):

```python
from .unified_trace import compute_loss
# ...
loss = compute_loss(output, loss_fn=loss_fn, labels=example_labels)
```

Replace the export block in `simulate_all()` (lines 448-478):

```python
from .export import export_result
# ...
export_result(
    rt_result, output_dir, output_formats,
    log_fn=self._log if self.verbose else None,
    print_summary=self.verbose,
)
```

- [ ] **Step 8: Update run_simulate.py to use shared export**

Replace `_export_result()` (lines 200-238) and its calls with:

```python
from torchtitan.experiments.simulator.export import export_result
# In _run_simulation:
export_result(result, output_dir, output_formats, log_fn=sim._log if sim.verbose else None, print_summary=sim.verbose)
```

- [ ] **Step 9: Run tests and pre-commit**

```bash
pytest torchtitan/experiments/simulator/tests/test_simulator.py -v
pre-commit run --all-files
```

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "refactor(simulator): extract shared utilities for loss, export, dtype, parallelism"
```

---

### Task 3: Extract Comm-Event-to-OpNode and Rank Replication Helpers

**Files:**
- Modify: `torchtitan/experiments/simulator/graph_assembler.py`
- Modify: `torchtitan/experiments/simulator/unified_trace.py`
- Modify: `torchtitan/experiments/simulator/schedule_extract.py`
- Modify: `torchtitan/experiments/simulator/schedule_generator.py`
- Test: `torchtitan/experiments/simulator/tests/test_simulator.py`

- [ ] **Step 1: Extract comm_event_to_op_node() in graph_assembler.py**

Add a module-level function before the `GraphAssembler` class:

```python
def comm_event_to_op_node(
    ev: dict[str, Any],
    node_id: str | None = None,
    phase_override: str | None = None,
) -> OpNode:
    """Convert a comm event dict to an OpNode."""
    nid = node_id or ev.get("event_id", "comm_unknown")
    op_name = ev.get("op", "collective_unknown")
    phase = phase_override or ev.get("phase", "unknown")

    input_metas: list[TensorMeta] = []
    output_metas: list[TensorMeta] = []
    shape_entries = ev.get("tensor_shapes") or []
    if not shape_entries:
        tm = ev.get("tensor_meta")
        if tm:
            shape_entries = [tm]
    for entry in shape_entries:
        if entry is None:
            continue
        meta = TensorMeta(
            shape=tuple(entry.get("shape", [])),
            dtype=entry.get("dtype", "unknown"),
            device=entry.get("device", "cpu"),
            is_dtensor=entry.get("is_dtensor", False),
            placements=entry.get("placements"),
        )
        input_metas.append(meta)
        output_metas.append(meta)

    return OpNode(
        node_id=nid,
        op_name=op_name,
        op_type=ev.get("op_type", "comm_collective"),
        phase=phase,
        inputs=input_metas,
        outputs=output_metas,
        comm_op=op_name,
        comm_group_size=ev.get("group_size"),
        pp_stage=ev.get("pp_stage"),
        microbatch_idx=ev.get("microbatch"),
        attrs={
            "group": str(ev.get("group", "")),
            "tag": str(ev.get("tag", "")),
            "src_rank": ev.get("src_rank"),
            "dst_rank": ev.get("dst_rank"),
            "rank": ev.get("rank"),
        },
    )
```

Update `GraphAssembler.merge_comm_events()` to use it:

```python
    @staticmethod
    def merge_comm_events(graph, comm_events, phase_override=None):
        for ev in comm_events:
            node_id = ev.get("event_id", f"comm_{len(graph.nodes)+1:07d}")
            node = comm_event_to_op_node(ev, node_id=node_id, phase_override=phase_override)
            graph.add_node(node)
            for src_id in ev.get("source_node_ids", []):
                if src_id in graph.nodes:
                    graph.add_edge(DataEdge(src_node_id=src_id, dst_node_id=node.node_id, edge_type="data"))
        return graph
```

- [ ] **Step 2: Update unified_trace.py build_result() to use comm_event_to_op_node**

In `TraceRecorder.build_result()` (lines 226-278), replace the inline comm-event-to-OpNode conversion:

```python
        from .graph_assembler import comm_event_to_op_node

        for ev in self.comm_events:
            node_id = ev.get("event_id", f"comm_{len(graph.nodes)+1:07d}")
            comm_node = comm_event_to_op_node(ev, node_id=node_id)
            graph.add_node(comm_node)
            for src_id in ev.get("source_node_ids", []):
                if src_id in graph.nodes:
                    graph.add_edge(DataEdge(src_node_id=src_id, dst_node_id=node_id, edge_type="data"))
```

- [ ] **Step 3: Extract replicate_events_to_ranks() in schedule_extract.py**

Add a module-level function before the existing replication code (before line 400):

```python
def replicate_events_to_ranks(
    schedule: TrainingSchedule,
    group_size: int,
    strategies: set[str],
    per_rank_prev: dict[Any, str],
    next_id_fn: Any,
) -> None:
    """Copy events from base ranks to sibling TP/DP ranks."""
    if group_size <= 1:
        return
    original_events = [e for e in schedule.events if e.metadata.get("strategy") in strategies]
    original_deps = list(schedule.deps)
    eid_remap: dict[str, dict[int, str]] = {}

    for ev in original_events:
        base_rank = ev.rank
        group_base = (base_rank // group_size) * group_size
        for r_offset in range(1, group_size):
            r = group_base + r_offset
            new_eid = next_id_fn(ev.event_type)
            new_ev = ScheduleEvent(
                event_id=new_eid,
                event_type=ev.event_type,
                rank=r,
                pp_rank=ev.pp_rank,
                pp_stage=ev.pp_stage,
                microbatch_idx=ev.microbatch_idx,
                logical_clock=ev.logical_clock,
                metadata=dict(ev.metadata),
            )
            schedule.add_event(new_ev)
            if ev.event_id not in eid_remap:
                eid_remap[ev.event_id] = {}
            eid_remap[ev.event_id][r_offset] = new_eid
            if r in per_rank_prev:
                schedule.add_dep(ScheduleDep(per_rank_prev[r], new_eid, "control"))
            per_rank_prev[r] = new_eid

    for dep in original_deps:
        remap_from = eid_remap.get(dep.from_event_id, {})
        remap_to = eid_remap.get(dep.to_event_id, {})
        for r_offset, to_copy in remap_to.items():
            from_copy = remap_from.get(r_offset)
            if from_copy:
                schedule.add_dep(ScheduleDep(from_copy, to_copy, dep.dep_type))
```

Update the inline replication code (lines 400-446) to call this function.

- [ ] **Step 4: Update schedule_generator.py to use shared replication**

Replace the post-hoc replication block (lines 370-420) with a call to `replicate_events_to_ranks()`.

- [ ] **Step 5: Run tests and pre-commit**

```bash
pytest torchtitan/experiments/simulator/tests/test_simulator.py -v
pre-commit run --all-files
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(simulator): extract comm_event_to_op_node and replicate_events_to_ranks"
```

---

## Phase 2: Capture Layer

### Task 4: Create _recorder_registry.py to Break Circular Dependency

**Files:**
- Create: `torchtitan/experiments/simulator/_recorder_registry.py`
- Modify: `torchtitan/experiments/simulator/unified_trace.py`
- Modify: `torchtitan/experiments/simulator/comm_interceptor.py`

- [ ] **Step 1: Create _recorder_registry.py**

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Lightweight recorder stack registry.

Shared by ``unified_trace`` and ``comm_interceptor`` to avoid circular imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .unified_trace import TraceRecorder

_RECORDER_STACK: list[Any] = []


def get_current_recorder() -> Any | None:
    return _RECORDER_STACK[-1] if _RECORDER_STACK else None


def push_recorder(recorder: Any) -> None:
    _RECORDER_STACK.append(recorder)


def pop_recorder() -> Any | None:
    return _RECORDER_STACK.pop() if _RECORDER_STACK else None
```

- [ ] **Step 2: Update unified_trace.py to use _recorder_registry**

Remove `_RECORDER_STACK` and `get_current_recorder()` from `unified_trace.py`. Replace with:

```python
from ._recorder_registry import get_current_recorder, push_recorder, pop_recorder
```

In `unified_trace()` context manager, replace `_RECORDER_STACK.append(recorder)` with `push_recorder(recorder)` and `_RECORDER_STACK.pop()` with `pop_recorder()`.

Keep the `get_current_recorder` re-export for backward compatibility:

```python
from ._recorder_registry import get_current_recorder  # noqa: F401
```

- [ ] **Step 3: Update comm_interceptor.py to import from _recorder_registry**

Change line 32 from:

```python
from .unified_trace import get_current_recorder
```

to:

```python
from ._recorder_registry import get_current_recorder
```

- [ ] **Step 4: Run tests**

```bash
pytest torchtitan/experiments/simulator/tests/test_simulator.py -v
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(simulator): extract _recorder_registry to break circular dependency"
```

---

### Task 5: Migrate simulate_runtime to unified_trace and Delete Old Path

**Files:**
- Modify: `torchtitan/experiments/simulator/simulator.py`
- Delete: `torchtitan/experiments/simulator/dispatch_interceptor.py`
- Delete: `torchtitan/experiments/simulator/runtime_capture.py`
- Delete: `torchtitan/experiments/simulator/graph_assembler.py` (merge into fx_capture.py)
- Modify: `torchtitan/experiments/simulator/fx_capture.py`
- Modify: `torchtitan/experiments/simulator/tests/test_simulator.py`

- [ ] **Step 1: Move GraphAssembler.from_fx and merge_comm_events into fx_capture.py**

Add to `fx_capture.py`:

```python
from .graph_assembler import comm_event_to_op_node


def merge_comm_events(
    graph: "ComputeGraph",
    comm_events: list[dict[str, Any]],
    phase_override: str | None = None,
) -> "ComputeGraph":
    for ev in comm_events:
        node_id = ev.get("event_id", f"comm_{len(graph.nodes)+1:07d}")
        node = comm_event_to_op_node(ev, node_id=node_id, phase_override=phase_override)
        graph.add_node(node)
        for src_id in ev.get("source_node_ids", []):
            if src_id in graph.nodes:
                graph.add_edge(DataEdge(src_node_id=src_id, dst_node_id=node.node_id, edge_type="data"))
    return graph
```

- [ ] **Step 2: Rewrite simulate_runtime() in simulator.py**

Replace `simulate_runtime()` (lines 149-234) to use `TraceRecorder` + `unified_trace()`:

```python
    def simulate_runtime(
        self,
        model_parts: list[nn.Module],
        example_inputs: tuple[Any, ...],
        loss_fn: Any | None = None,
        example_labels: torch.Tensor | None = None,
        pp_schedule: Any | None = None,
        pp_stages: list[Any] | None = None,
        optimizer: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SimulationResult:
        self._log("Starting runtime capture …")
        patch_device_type_to_cpu()

        recorder = TraceRecorder(rank=self.rank)

        with unified_trace(
            recorder,
            model_parts[0],
            example_inputs,
            use_fake_mode=False,
            phase="forward",
            capture_comm=True,
            capture_fsdp=True,
            model_parts=model_parts,
        ):
            if pp_schedule is not None:
                self._log("  running PP schedule step …")
                pp_schedule.step(*example_inputs)
            else:
                model = model_parts[0]
                self._log("  running forward pass …")
                output = model(*example_inputs)

                loss = compute_loss(output, loss_fn=loss_fn, labels=example_labels)

                self._log("  running backward pass …")
                recorder.current_phase = "backward"
                loss.backward()

                if optimizer is not None:
                    self._log("  running optimizer step …")
                    recorder.current_phase = "optimizer"
                    optimizer.step()
                    optimizer.zero_grad()

        result = recorder.build_result(metadata={"mode": "runtime", **(metadata or {})})

        self._log(
            f"  captured {len(result.compute_graph.nodes)} ops, "
            f"{len(result.comm_events)} comm events, "
            f"{len(result.fsdp_events)} FSDP events"
        )
        return result
```

Remove imports of `RuntimeCapture`, `capture_comms`, `CommRecorder` from `simulator.py`.

- [ ] **Step 3: Delete old capture path files**

```bash
rm torchtitan/experiments/simulator/dispatch_interceptor.py
rm torchtitan/experiments/simulator/runtime_capture.py
rm torchtitan/experiments/simulator/graph_assembler.py
```

- [ ] **Step 4: Update test imports**

In `tests/test_simulator.py`:
- `TestOpCaptureMode`: Replace `from .dispatch_interceptor import OpRecorder, OpCaptureMode, capture_ops` with `from .unified_trace import TraceRecorder, UnifiedTraceMode, unified_trace`. Update test methods to use `TraceRecorder` and `UnifiedTraceMode`.
- `TestGraphAssembler`: Remove `from_runtime` test, update `merge_comm` test to use `fx_capture.merge_comm_events`.
- Remove any imports of `dispatch_interceptor`, `runtime_capture`, `graph_assembler`.

- [ ] **Step 5: Update __init__.py if needed**

No changes needed — `__init__.py` doesn't export any of the deleted modules.

- [ ] **Step 6: Run tests and pre-commit**

```bash
pytest torchtitan/experiments/simulator/tests/test_simulator.py -v
pre-commit run --all-files
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(simulator): unify capture architecture, delete old dispatch/runtime path"
```

---

## Phase 3: Analysis Layer

### Task 6: Split cost_model.py into Three Files

**Files:**
- Create: `torchtitan/experiments/simulator/cost_estimators.py`
- Create: `torchtitan/experiments/simulator/schedule_analysis.py`
- Modify: `torchtitan/experiments/simulator/cost_model.py`
- Modify: `torchtitan/experiments/simulator/des_engine.py`
- Test: `torchtitan/experiments/simulator/tests/test_simulator.py`

- [ ] **Step 1: Create cost_estimators.py**

Move from `cost_model.py`:
- `OverlapStrategy`, `NoOverlap`, `FixedOverlap` classes (lines 31-53)
- Mock hardware parameter constants (lines 62-65)
- `_estimate_flops()` (lines 68-157)
- `_estimate_bytes()` (lines 160-172)
- `_estimate_comm_bytes()` (lines 175-193)
- `_numel()` (lines 196-208)
- `_tensor_bytes()` (lines 211-233)

Copy these from `cost_model.py` verbatim:
- `OverlapStrategy`, `NoOverlap`, `FixedOverlap` (lines 31-53)
- Constants `_DEFAULT_MOCK_TFLOPS`, `_DEFAULT_MOCK_GB_PER_S`, `_DEFAULT_MOCK_COMM_GB_PER_S`, `_DEFAULT_MOCK_COMM_LATENCY_US` (lines 62-65)
- `_estimate_flops()` (lines 68-157)
- `_estimate_bytes()` (lines 160-172)
- `_estimate_comm_bytes()` (lines 175-193)
- `_numel()` (lines 196-208)
- `_tensor_bytes()` (lines 211-233) — replace the inline `dtype_bytes` dict with:

```python
from .memory_estimator import dtype_size as _dtype_size

def _tensor_bytes(shape, dtype, default_seq_len=4096):
    size = _dtype_size(dtype)
    return _numel(shape, default_seq_len) * (size if size > 0 else 2)
```

- [ ] **Step 2: Create schedule_analysis.py**

Move from `cost_model.py`:
- `_critical_path_time_us()` (lines 438-484)
- `link_schedule_to_graph()` (lines 492-570)
- `predict_multi_rank_step_time_us()` (lines 577-604)

Copy these three functions verbatim from `cost_model.py`:
- `_critical_path_time_us()` (lines 438-484)
- `link_schedule_to_graph()` (lines 492-570)
- `predict_multi_rank_step_time_us()` (lines 577-604)

In `predict_multi_rank_step_time_us`, update the lazy imports:

```python
    from .cost_model import MockCostModel  # one-directional, no cycle
    from .des_engine import simulate_multi_rank_des
```

- [ ] **Step 3: Slim down cost_model.py**

Keep only:
- `CostModel` class (lines 241-300) — update `predict_step_time_us` to import from `des_engine`
- `MockCostModel` class (lines 307-431) — import estimation functions from `cost_estimators`
- `apply_cost_model()` (lines 612-671) — import `predict_multi_rank_step_time_us` from `schedule_analysis`

Add re-exports for backward compatibility:

```python
from .cost_estimators import (
    OverlapStrategy, NoOverlap, FixedOverlap,
    _estimate_flops, _estimate_bytes, _estimate_comm_bytes,
    _numel, _tensor_bytes,
)
from .schedule_analysis import (
    link_schedule_to_graph, predict_multi_rank_step_time_us, _critical_path_time_us,
)
```

- [ ] **Step 4: Update des_engine.py imports**

Change `from .cost_model import link_schedule_to_graph` (line 195) to:

```python
from .schedule_analysis import link_schedule_to_graph
```

Change `from .cost_model import _critical_path_time_us` (line 325) to:

```python
from .schedule_analysis import _critical_path_time_us
```

Change `from .cost_model import MockCostModel` (line 305) to remain as-is (one-directional import, no cycle).

- [ ] **Step 5: Update test imports**

In `tests/test_simulator.py`, update any imports that reference moved symbols. The re-exports in `cost_model.py` should keep most tests working, but verify:

```bash
pytest torchtitan/experiments/simulator/tests/test_simulator.py -v -k "CostModel or ScheduleGraph or MultiRank or Overlap or CriticalPath"
```

- [ ] **Step 6: Run full tests and pre-commit**

```bash
pytest torchtitan/experiments/simulator/tests/test_simulator.py -v
pre-commit run --all-files
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(simulator): split cost_model into cost_estimators and schedule_analysis"
```

---

### Task 7: Split des_engine.py — Extract Memory Timeline

**Files:**
- Create: `torchtitan/experiments/simulator/des_memory.py`
- Modify: `torchtitan/experiments/simulator/des_engine.py`
- Test: `torchtitan/experiments/simulator/tests/test_simulator.py`

- [ ] **Step 1: Create des_memory.py**

Move `compute_des_memory_timeline()` (lines 465-647 of `des_engine.py`) to `des_memory.py`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DES memory timeline computation."""

from __future__ import annotations

from typing import Any

from .nodes import MemoryEvent, SimulationResult


Copy `compute_des_memory_timeline` verbatim from `des_engine.py` lines 465-647. No changes to the function body — only the module location changes.
```

- [ ] **Step 2: Update des_engine.py**

Remove `compute_des_memory_timeline()` and add re-export:

```python
from .des_memory import compute_des_memory_timeline  # noqa: F401
```

- [ ] **Step 3: Run tests and pre-commit**

```bash
pytest torchtitan/experiments/simulator/tests/test_simulator.py -v -k "DESMemory"
pytest torchtitan/experiments/simulator/tests/test_simulator.py -v
pre-commit run --all-files
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(simulator): extract compute_des_memory_timeline to des_memory.py"
```

---

### Task 8: Split trainer_runner.py

**Files:**
- Modify: `torchtitan/experiments/simulator/trainer_runner.py`
- Modify: `torchtitan/experiments/simulator/synthetic_comm.py` (already created in Task 2)
- Modify: `torchtitan/experiments/simulator/schedule_inject.py` (already created in Task 2)

- [ ] **Step 1: Verify synthetic_comm.py and schedule_inject.py contain all moved code**

These files were created in Task 2. Verify they contain:
- `synthetic_comm.py`: `inject_synthetic_comm_events`, `infer_num_layers`, `guess_hidden_dim`
- `schedule_inject.py`: `inject_semantic_schedule`, `read_parallelism_degrees`, `ParallelismDegrees`

- [ ] **Step 2: Slim down trainer_runner.py**

After Task 2, `trainer_runner.py` should only contain:
- `_get_cost_model_kwargs()` (lines 41-54)
- `_import_cost_model()` (lines 57-107)
- `run_trainer_simulation()` (lines 452-609, streamlined)

Remove the now-imported functions. Update `run_trainer_simulation()` to use the imported names.

- [ ] **Step 3: Run tests and pre-commit**

```bash
pytest torchtitan/experiments/simulator/tests/test_simulator.py -v
pre-commit run --all-files
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(simulator): split trainer_runner into focused modules"
```

---

### Task 9: Merge Schedule Extraction Files

**Files:**
- Modify: `torchtitan/experiments/simulator/schedule_extract.py`
- Delete: `torchtitan/experiments/simulator/pp_schedule_extractor.py`
- Modify: `torchtitan/experiments/simulator/simulator.py`
- Test: `torchtitan/experiments/simulator/tests/test_simulator.py`

- [ ] **Step 1: Merge PPScheduleExtractor into schedule_extract.py**

Add the `PPScheduleExtractor` class from `pp_schedule_extractor.py` to the end of `schedule_extract.py`. Update its internal import:

```python
class PPScheduleExtractor:
    def extract(self) -> TrainingSchedule:
        # Primary path uses _convert_pipeline_order_to_training_schedule (same file now)
        ...
```

- [ ] **Step 2: Delete pp_schedule_extractor.py**

```bash
rm torchtitan/experiments/simulator/pp_schedule_extractor.py
```

- [ ] **Step 3: Update simulator.py import**

Change line 53:

```python
from .schedule_extract import PPScheduleExtractor
```

- [ ] **Step 4: Update test imports**

In `tests/test_simulator.py`, change:

```python
from torchtitan.experiments.simulator.pp_schedule_extractor import PPScheduleExtractor
```

to:

```python
from torchtitan.experiments.simulator.schedule_extract import PPScheduleExtractor
```

- [ ] **Step 5: Run tests and pre-commit**

```bash
pytest torchtitan/experiments/simulator/tests/test_simulator.py -v
pre-commit run --all-files
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(simulator): merge pp_schedule_extractor into schedule_extract"
```

---

## Phase 4: Output Layer

### Task 10: Create export/ Sub-Package and Extract JS

**Files:**
- Create: `torchtitan/experiments/simulator/export/__init__.py`
- Create: `torchtitan/experiments/simulator/export/json_export.py`
- Create: `torchtitan/experiments/simulator/export/dot_export.py`
- Create: `torchtitan/experiments/simulator/export/chrome_trace.py`
- Create: `torchtitan/experiments/simulator/export/html_export.py`
- Create: `torchtitan/experiments/simulator/export/text_summary.py`
- Create: `torchtitan/experiments/simulator/export/schedule_timing.py`
- Create: `torchtitan/experiments/simulator/export/export_utils.py`
- Create: `torchtitan/experiments/simulator/export/trace_visualizer.js`
- Delete: `torchtitan/experiments/simulator/export.py`

- [ ] **Step 1: Create export/ directory and __init__.py**

```bash
mkdir -p torchtitan/experiments/simulator/export
```

`export/__init__.py`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .json_export import export_json
from .dot_export import export_dot
from .chrome_trace import export_chrome_trace
from .html_export import export_html
from .text_summary import export_text_summary
from .export_utils import export_result

__all__ = [
    "export_json",
    "export_dot",
    "export_chrome_trace",
    "export_html",
    "export_text_summary",
    "export_result",
]
```

- [ ] **Step 2: Split export.py into sub-modules**

For each sub-module, copy the relevant functions from the original `export.py`:

- `json_export.py`: `export_json()` + `_populate_des_metadata()` (import from schedule_timing)
- `dot_export.py`: `export_dot()`, `_graph_to_dot()`, `_DOT_COLORS`, `_node_color()`
- `chrome_trace.py`: `export_chrome_trace()` + Chrome trace helpers
- `html_export.py`: `export_html()` + HTML template (reads JS from file)
- `text_summary.py`: `export_text_summary()` + text formatting helpers
- `schedule_timing.py`: `_inject_schedule_timing()`, `_populate_des_metadata()`, `_schedule_event_to_phase()`, `_event_lane()`, `_inject_schedule_timing_into_dict()`
- `export_utils.py`: `export_result()` (already defined in Task 2)

Each file starts with the BSD license header and imports from `..nodes`.

- [ ] **Step 3: Extract JavaScript to trace_visualizer.js**

Extract the ~1100 lines of JavaScript from the `export_html()` function's f-string into `export/trace_visualizer.js`. Replace Python f-string `{{` / `}}` with standard JS `{` / `}`. Use `__SIMULATOR_DATA__` as the data injection placeholder.

In `html_export.py`:

```python
from pathlib import Path

_JS_PATH = Path(__file__).parent / "trace_visualizer.js"


def export_html(result, path):
    js_code = _JS_PATH.read_text(encoding="utf-8")
    data_json = json.dumps(result.to_dict(), default=str)
    js_code = js_code.replace("__SIMULATOR_DATA__", data_json)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Simulator Trace</title></head>
<body><script>{js_code}</script></body></html>"""
    Path(path).write_text(html, encoding="utf-8")
```

- [ ] **Step 4: Delete old export.py**

```bash
rm torchtitan/experiments/simulator/export.py
```

- [ ] **Step 5: Update __init__.py imports**

The top-level `simulator/__init__.py` already imports from `.export`. Since `export` is now a sub-package with `__init__.py` re-exporting everything, no changes needed.

Verify:

```python
from .export import (
    export_chrome_trace,
    export_dot,
    export_html,
    export_json,
    export_text_summary,
)
```

This still resolves correctly.

- [ ] **Step 6: Update all internal imports**

Files that import from `export`:
- `simulator.py`: `from .export import ...` (unchanged)
- `trainer_runner.py`: `from .export import ...` (unchanged)
- `run_simulate.py`: `from torchtitan.experiments.simulator.export import ...` (unchanged)

All should work without changes since the sub-package re-exports.

- [ ] **Step 7: Run tests and pre-commit**

```bash
pytest torchtitan/experiments/simulator/tests/test_simulator.py -v
pre-commit run --all-files
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor(simulator): split export.py into sub-package, extract JS to file"
```

---

## Final Verification

### Task 11: End-to-End Verification

- [ ] **Step 1: Run full unit test suite**

```bash
pytest torchtitan/experiments/simulator/tests/test_simulator.py -v
```

Expected: All ~70 tests pass.

- [ ] **Step 2: Run pre-commit on all files**

```bash
pre-commit run --all-files
```

Expected: All checks pass (ufmt, flake8, pyrefly, codespell, license headers).

- [ ] **Step 3: Run E2E smoketest (if GPU/multi-process available)**

```bash
MODULE=simulator.deepseek_v4 CONFIG=deepseek_v4_sim_smoketest ./run_train.sh
```

Expected: Output files generated and content matches baseline.

- [ ] **Step 4: Verify public API unchanged**

```python
from torchtitan.experiments.simulator import (
    Simulator, SimulationResult, ComputeGraph, TrainingSchedule,
    OpNode, DataEdge, TensorMeta, ScheduleEvent, ScheduleDep,
    DESEngine, simulate_single_rank_des, simulate_multi_rank_des,
    CostModel, MockCostModel, PerfResult, apply_cost_model,
    export_json, export_dot, export_chrome_trace, export_html, export_text_summary,
)
```

- [ ] **Step 5: Final commit (if any fixups needed)**

```bash
git add -A
git commit -m "refactor(simulator): final verification fixups"
```
