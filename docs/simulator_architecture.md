# TorchTitan CPU Simulation Architecture

## Goals

The simulator adds a CPU-only trace mode to the native TorchTitan training flow.
It is designed to answer two questions without requiring GPUs:

1. What operators run during forward, backward, optimizer, communication, and
   data movement phases, including tensor shapes and data dependencies?
2. What coarse training schedule is implied by parallel strategies such as PP,
   TP, DP, and FSDP2?

The implementation is side-loaded as an experiment so existing TorchTitan
entrypoints and scripts remain unchanged.

## Entry and configuration

Use the regular TorchTitan launcher with a simulation module/config:

```bash
MODULE=simulator.llama3 CONFIG=llama3_sim_debugmodel ./run_train.sh
```

`torchtitan.experiments.simulator.llama3.config_registry` returns a
`SimulationTrainer.Config`. `SimulationTrainer` subclasses the normal
`Trainer`, patches TorchTitan device helpers to CPU before trainer
construction, and overrides `train()` to run one instrumented training step.

The important property is that `torchtitan/train.py` still does the same work:
parse config, build trainer, then call `trainer.train()`. Simulation changes
which trainer config is built, not the global training entry.

## Core components

| Component | Responsibility |
| --- | --- |
| `cpu_env.py` | Forces TorchTitan device helpers to resolve to CPU. |
| `dispatch_interceptor.py` | Captures runtime PyTorch ops with input/output tensor metadata and producer-consumer data edges. |
| `comm_interceptor.py` | Monkey-patches `torch.distributed` collectives and records communication events. |
| `runtime_capture.py` | Coordinates op, comm, FSDP, and PP capture scopes. |
| `graph_assembler.py` | Builds `ComputeGraph` nodes/edges and merges communication events. |
| `memory_estimator.py` | Estimates activation lifetimes, communication buffers, model parameters, gradients, and optimizer state memory. |
| `pp_schedule_extractor.py` | Extracts semantic pipeline schedule events and dependencies. |
| `fx_capture.py` | Optionally captures forward or joint forward/backward FX graphs. |
| `export.py` | Writes JSON, DOT, Chrome Trace, text summary, and interactive HTML. |
| `trainer_runner.py` | Runs the native Trainer components for one simulated step and exports results. |
| `extension_hooks.py` | Generic duck-typed hooks for external simulation extensions. |

## Trace model

`SimulationResult` contains:

- `compute_graph`: ordered op nodes and directed dependencies.
- `schedule`: coarse-grained schedule events and dependencies.
- `comm_events`, `fsdp_events`, `pp_events`: raw event streams.
- `memory_events`: decomposable memory estimates for activations, comm buffers,
  parameters, gradients, and optimizer states.
- `metadata`: run mode, rank, optional FX graphs, and extension metadata.

Operator nodes record op name/type, phase, tensor input/output shapes, PP
context, microbatch index, and communication annotations. Data edges are
producer-consumer dependencies from observed tensor flow. Scheduling edges are
kept in `TrainingSchedule` so they do not introduce artificial cycles into the
operator DAG.

## Memory estimate model

Memory tracing is deterministic and decomposable rather than allocator-exact:

- graph output tensors become activation/data-move/communication buffer
  `MemoryEvent`s, with lifetimes approximated from producer order to the last
  observed consumer edge;
- the graph peak is a scanline peak over those output lifetimes;
- communication interceptor tensor metadata contributes separate
  `comm_event_buffer` estimates;
- model state is estimated from native model parameters after trainer
  construction: parameter bytes, one gradient tensor per trainable parameter,
  and Adam/AdamW-style first and second moment optimizer state.

The exported `metadata["memory"]` summary groups bytes by category, phase, and
device, and keeps top-level fields such as `peak_live_bytes`,
`parameter_bytes`, `gradient_bytes`, `optimizer_state_bytes`, and
`model_state_total_bytes`. `trace.html` shows memory cards, a memory trace
timeline, top memory-event details, and the raw memory summary, while
`summary.txt` includes a human-readable memory section.

The HTML memory timeline renders lifetimed activation/communication/data-move
events as stacked live bytes over graph node order. Events without graph
lifetimes, such as model parameters, gradients, and optimizer state, are shown
as a steady resident baseline so memory can be inspected both temporally and by
category.

This model is intended for planning and comparison in CPU-only environments. It
does not replace real CUDA/NPU allocator telemetry, and it intentionally avoids
pretending CPU simulation can observe backend-specific fragmentation or stream
workspace behavior.

## HTML visualization

`trace.html` is self-contained. It embeds the JSON payload and uses browser-side
HTML5 canvas rendering for:

- train-step hierarchy,
- scrollable/zoomable PP/FSDP2/TP/DP schedule swimlanes,
- rank/strategy tabs,
- scrollable/zoomable memory timeline and event breakdown,
- left-to-right operator dependency DAGs.

The page does not rely on external CDNs, which makes it portable across offline
cluster environments.

## Extension hooks

External packages can enrich traces without importing those packages from the
root simulator:

```python
def collect_simulation_metadata(self, capture) -> dict | None:
    ...

def postprocess_simulation_result(self, result, sim_opts):
    ...
```

`trainer_runner.py` calls the first hook before FX capture and the second hook
immediately before export. The hooks are duck-typed, validated, and live in a
lightweight module to avoid pulling full TorchTitan config dependencies into
unit tests.

## CPU device patching

`cpu_env.py` applies a three-layer patch so that TorchTitan's full Trainer
initialization runs on a pure-CPU host without CUDA:

1. **`torchtitan.tools.utils`** — the canonical device module reference. Set
   `device_type = "cpu"` and replace `device_module` with a stub that returns
   `name = "CPU_Simulator"`, `device_count = 1`, and no-op for all memory
   / synchronize / property methods.

2. **Module-level cache rebinding** — several torchtitan modules
   (`torchtitan.components.metrics`, `torchtitan.distributed.parallel_dims`,
   `torchtitan.distributed.utils`) import `device_type` and `device_module` at
   module scope.  Those local references are rebound to the CPU stubs so that
   `build_device_memory_monitor()`, `build_mesh()`, and `set_determinism()`
   do not fall back to `torch.cuda`.

3. **`torch.cuda` stubs** — PyTorch internals such as `fully_shard`,
   DTensor RNG tracker, and `DeviceMesh` constructors call `torch.cuda.*`
   directly.  Key entrypoints (`is_available`, `_lazy_init`, `current_device`,
   `device_count`, `get_device_properties`, `memory_*`, `synchronize`,
   `empty_cache`) are replaced with CPU-safe no-ops at module import time.

## Validation

### Unit tests

```bash
PYTHONPATH=. ~/.local/bin/python3.11 -m pytest \
  torchtitan/experiments/simulator/tests/test_simulator.py -q
```

The simulator unit suite validates data models, op/comm/FSDP/PP capture,
exporters, HTML generation, graph assembly, extension hooks, memory
estimation, and memory trace integration.  Current: **36 passed**.

### End-to-end trace generation

A self-contained Python script generates a full PP=2 / TP=2 / DP=2 / FSDP2
semantic trace with real CPU forward/backward operators, gloo communication
markers, memory estimates, and HTML output:

```bash
PYTHONPATH=. ~/.local/bin/python3.11 - <<'PY'
from __future__ import annotations
import os, torch, torch.distributed as dist, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torchtitan.experiments.simulator.export import export_html
from torchtitan.experiments.simulator.memory_estimator import (
    estimate_model_state_memory, merge_memory_summary, summarize_memory_events,
)
from torchtitan.experiments.simulator.nodes import ScheduleDep, ScheduleEvent, TrainingSchedule
from torchtitan.experiments.simulator.runtime_capture import RuntimeCapture

out = Path("sim_e2e_output"); out.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ["MASTER_PORT"] = "29691"
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1")
dist.init_process_group("gloo", init_method="env://"); torch.manual_seed(42)

class TinyStage(nn.Module):
    def __init__(s): super().__init__(); s.embed=nn.Embedding(2048,64); s.proj=nn.Linear(64,64)
    def forward(s,x): return F.gelu(s.proj(s.embed(x)))

model = TinyStage()
capture = RuntimeCapture(rank=0)
with capture.activate([model], phase="forward"):
    for step in range(2):
        capture.set_phase(f"step{step}_forward")
        tokens = torch.randint(0, 2048, (1, 64), dtype=torch.long)
        logits = model(tokens)
        capture.set_phase(f"step{step}_backward")
        logits.sum().backward()

result = capture.build_result(metadata={"mode": "simulation", "trace_kind": "e2e_test"})
mme, mms = estimate_model_state_memory([model], optimizer_name="AdamW")
result.memory_events.extend(mme)
md = {k: v for k, v in (result.metadata.get("memory", {}) or {}).items()
      if k not in {"total_event_bytes", "by_category", "by_phase", "by_device"}}
result.metadata["memory"] = merge_memory_summary(
    summarize_memory_events(result.memory_events), md, mms)
export_html(result, str(out / "trace.html"))
print("HTML ->", out / "trace.html", "nodes:", len(result.compute_graph.nodes))
dist.destroy_process_group()
PY
```

Example outputs with a richer PP/TP/DP/FSDP2 schedule are committed at:

- `sim_out_torchtitan_memory_parallel/` — parallel trace with memory summary
- `sim_out_torchtitan_memory_trace/` — parallel trace with memory timeline

### Native TorchTitan entry (via run_train.sh)

Use the standard launcher with a simulation module/config.  On CPU-only
hosts, set ``PYTHON`` to a Python that has ``torch`` installed (e.g.
``~/.local/bin/python3.11``):

```bash
PYTHON=~/.local/bin/python3.11 \
  NGPU=1 MODULE=simulator.llama3 CONFIG=llama3_sim_debugmodel \
  COMM_MODE=fake_backend ./run_train.sh
```

To customise the output directory and other simulation options:

```bash
PYTHON=~/.local/bin/python3.11 \
  NGPU=1 MODULE=simulator.llama3 CONFIG=llama3_sim_debugmodel \
  COMM_MODE=fake_backend ./run_train.sh \
  --simulation.output_dir ./my_sim_output
```

The ``SimulationTrainer`` replaces the model's ``parallelize_fn`` /
``pipelining_fn`` with CPU-only stubs before ``Trainer.__init__`` runs,
so FSDP/TP/PP wrappers are never applied.  Parallelisation semantics
are captured through the ``TrainingSchedule`` instead of runtime hooks.

## Design compromises and limitations

### Why the simulator cannot run real PP / FSDP / TP on CPU

TorchTitan's native parallelism wrappers have hard dependencies on CUDA:

| Wrapper | CPU blocker |
|---|---|
| `apply_fsdp → fully_shard` | Allocates `torch.empty(0, device=cuda)` internally; `FSDPParamGroup` requires CUDA device handles |
| `pipeline_llm → _pipeline_module_split` | Creates `PipelineStage` objects with `pp_mesh`; the schedule's `step()` calls P2P ops that require multi-rank execution |
| `_PipelineSchedule.step()` | Calls `_split_inputs` → `split_args_kwargs_into_chunks` which expects multi-process distributed tensors |
| `set_determinism → DTensor RNG` | `torch.distributed.tensor._random.manual_seed` calls `torch.cuda.device_count()` internally |

The simulator works around these by:

1. **Replacing `parallelize_fn` and `pipelining_fn`** with CPU no-ops before
   `Trainer.__init__` runs. The model is built on CPU without any FSDP/TP/PP
   wrappers, so the actual forward/backward executes as plain PyTorch.

2. **Semantic schedule generation** (`schedule_generator.py`) creates a
   ``TrainingSchedule`` that mirrors what Interleaved1F1B (or any other
   TorchTitan schedule) would produce across the full multi-rank topology.
   Dependencies (PP send/recv, TP all-reduce, FSDP all-gather/reduce-scatter,
   DP gradient sync) are explicit.

3. **No op modification** — the op trace captured by
   ``dispatch_interceptor.py`` records the actual PyTorch operators that
   execute on CPU, which is the ground truth for that single process.

### What the semantic schedule is and is not

``schedule_generator.py`` produces a ``TrainingSchedule`` that:

- ✅ Reflects the correct Interleaved1F1B warmup / 1F1B steady-state /
  cooldown order
- ✅ Encodes correct PP send→recv, TP all-reduce, FSDP all-gather/reduce-scatter,
  and DP gradient-sync dependencies
- ✅ Respects the virtual-stage topology and microbatch count
- ❌ Is not derived from a real ``_PipelineSchedule`` object (the schedule is
  synthetic, though structurally equivalent)
- ❌ Does not capture dynamic scheduling decisions (e.g., ZB zero-bubble
  adaptive scheduling)

On GPU/NPU hardware where TorchTitan can fully initialise, the
``pp_schedule_extractor`` and ``attach_pp_hooks`` in ``runtime_capture.py``
take over and record native events from the real schedule object.

### Parallelism topology constraints

- The ``_set_fake_world_size`` mechanism inflates ``NGPU``/``WORLD_SIZE``
  to match parallelism degree products so that ``ParallelDims`` validation
  passes on a single process.
- ``dp_enabled``, ``tp_enabled``, and ``pp_enabled`` are disabled after
  ``Trainer.__init__`` to keep ``forward_backward_step`` on the non-PP
  single-model code path.
- Virtual stages × pp_degree must not exceed the model's layer count.
  For the debugmodel (8 layers), the maximum ``Interleaved1F1B``
  configuration is ``pp=4, virtual=2`` (8 stages ≤ 8 layers).

### Memory estimation limits

- Activation memory is approximated by output tensor sizes with
  producer→consumer lifetimes.  Python eager-mode aliasing and in-place
  ops are not modelled, so the estimate is conservative.
- Model-state memory (parameters, gradients, optimizer state) assumes a
  standard Adam/AdamW moment structure.
- No backend-specific fragmentation, allocator overhead, or NPU HCCL
  workspace memory is modelled.

### General limits

- Runtime capture observes the current process. Multi-rank traces require
  multi-process execution or post-run trace aggregation.
- CPU simulation does not reproduce GPU/NPU kernel performance or real device
  memory pressure. Memory values are trace estimates, not allocator truth.
- Some operator-level aliasing and in-place behavior is approximated by tensor
  producer tracking.
