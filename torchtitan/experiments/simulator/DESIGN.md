# TorchTitan Simulator -- Architecture Design Document

> **Status note:** This document reflects the **as-built** implementation as of the
> latest commit.

## 1. Overview

TorchTitan Simulator is a CPU-only training trace/simulation system built as a
side-loaded experiment on top of the upstream torchtitan LLM training platform. It
captures forward/backward computation graphs, communication patterns, and training
schedules (PP, FSDP, TP, DP) **without any GPU hardware**, enabling:

- **Training step profiling** at arbitrary scale (e.g. 1024-GPU topology) on a single
  CPU machine
- **Performance prediction** via cost models and discrete-event simulation (DES)
- **Parallelism strategy exploration** (PP/TP/DP/FSDP degree combinations) without real
  hardware
- **Workload graph export** for downstream hardware simulators (ZhanluModelSim etc.)

### Design Principles

1. **Side-loaded experiment** -- `train.py` remains unchanged; all simulator code lives
   under `torchtitan/experiments/simulator/`
2. **Capture-faithful** -- schedules and compute graphs are derived from *captured* data
   (or from real PyTorch schedule objects), not re-implemented training logic
3. **PyTorch-native** -- reuses upstream PyTorch schedule objects (`PipelineSchedule`),
   FX tracing, and `TorchDispatchMode`
4. **Model-agnostic core** -- model-specific code is isolated in per-model
   subdirectories (`llama3/`, `deepseek_v4/`)

---

## 2. Directory Structure

```
simulator/
  __init__.py              # Public API, module aliases for backward compat
  trainer.py               # SimulationTrainer (subclasses Trainer), SimulationConfig
  trainer_runner.py        # run_trainer_simulation() -- main execution orchestrator
  nodes.py                 # Core data model (OpNode, ComputeGraph, TrainingSchedule, ...)
  export.py                # Multi-format export (JSON, DOT, Chrome Trace, HTML, CSV, Text)
  cost_model.py            # CostModel ABC + MockCostModel (roofline-style estimation)
  des_engine.py            # salabim-based Discrete Event Simulation engine
  memory_estimator.py      # Activation / model state / comm buffer memory estimation
  op_classification.py     # Unified op classification (compute/comm/data_move/memory)
  cpu_env.py               # CPU device patching (monkey-patches torch.cuda -> CPU stubs)
  meta_env.py              # Meta device patching (0-byte tensors for large model sim)
  meta_device_patches.py   # Phase 4: 7 meta device patches enabling natural comm on meta
  synthetic_dataloader.py  # SyntheticTokenDataLoader (random token generation)
  extension_hooks.py       # Extension points for NPU/other side-loads

  capture/                 # Trace capture (single unified module)
    __init__.py
    unified_trace.py       # FakeTensorMode + TorchDispatchMode + CommRecorder + FSDP hooks

  schedule/                # Training schedule extraction & generation
    __init__.py            # Exports PPScheduleExtractor, extract_schedule_from_pytorch
    schedule_extract.py    # Extract schedule from real PyTorch PipelineSchedule objects
    pp_schedule_extractor.py # PPScheduleExtractor class (reads pipeline_order tables)

  ir/                      # Layered IR (L0-L3) for workload graph export
    __init__.py
    op_node.py             # L0: SpecOpNode projection
    step_graph.py          # L1: StepGraph (per-phase DAG template)
    schedule_graph.py      # L2: ScheduleGraph (orchestration: instances, data passes)
    workload_graph.py      # L3: WorkloadGraph (iteration semantics + data flow)
    builder.py             # Top-level orchestrator for IR projection

  llama3/                  # Llama3-specific simulation configs
    __init__.py
    config_registry.py     # llama3_sim_debugmodel, llama3_sim_1024gpu

  deepseek_v4/             # DeepSeek V4-specific simulation configs
    __init__.py
    config_registry.py     # deepseek_v4_sim_smoketest, deepseek_v4_pro_sim_smoketest

  tests/                   # Unit tests
    __init__.py
    test_simulator.py      # Core simulator + capture + DES + export tests
    test_ir.py             # Layered IR (L0-L3) projection tests
```

---

## 3. Entry Points

### 3.1 SimulationTrainer (via `run_train.sh`)

The primary entry point for end-to-end simulation. `SimulationTrainer` subclasses the
upstream `Trainer`:

```
run_train.sh  -->  torchrun  -->  torchtitan.train_main()
                                    |
                                    v
                            config.build()  -->  SimulationTrainer.__init__()
                                    |
                                    v
                            SimulationTrainer.train()
                                    |
                                    v
                            run_trainer_simulation()
```

**Key behaviors in `__init__` (`trainer.py`):**

1. Read `sim_opts = config.simulation` and resolve `comm_backend`.
2. Read parallelism degrees (`pp`, `tp`, `dp_shard`, `dp_replicate`); auto-resolve
   `dp_shard=-1` -> 1.
3. If topology `pp*tp*dp_shard*dp_replicate > 1` -> call `_set_fake_world_size(config)`.
4. **Force** `config.comm.mode = "fake_backend"` so `init_distributed` uses the fake
   process group (no NCCL/gloo rendezvous, no torchrun required).
5. If the actual CLI `comm.mode == "fake_backend"`, override `comm_backend = ""`.
6. Resolve `device_mode`: empty auto-selects `"meta"` unless `comm_backend == "gloo"`.
7. Call `patch_device_type_to_meta()` or `patch_device_type_to_cpu()` accordingly.
8. For meta mode, default `config.debug.seed = 42` if unset.
9. **Parallelize-fn selection** (see §15, §16):
   - `comm_backend == "gloo"` -> `_cpu_gloo_parallelize_llama` /
     `_cpu_gloo_parallelize_dsv4` (model-name based), then FSDP1 wrapping after init.
   - Otherwise (default) -> `_meta_parallelize_with_skip_fsdp(real_parallelize_fn)` --
     the Phase 4 wrapper that applies the *real* parallelization (TP/EP/CP/FSDP2) on
     meta device (enabled by `apply_meta_device_patches()`).
10. **Pipelining-fn selection**:
    - `pp > 1` and not gloo -> `partial(_cpu_semantic_pipeline, ...)` (meta PP split).
    - Else -> `_cpu_noop_pipeline` (single-stage).
11. Call `super().__init__(config)` (the real `Trainer`).
12. If `_cpu_semantic_pipeline` filled `self._pp_model_parts`, overwrite
    `self.model_parts`.
13. If `comm_backend == "gloo"`: apply `_apply_fsdp1_on_cpu` to each model part.

`train()` re-patches the device and calls `run_trainer_simulation(self, sim_opts)`.

**`SimulationConfig`** (`@dataclass(kw_only=True, slots=True)` in `trainer.py`):

| Field | Default | Description |
|-------|---------|-------------|
| `output_dir` | `"./simulator_output"` | Export directory |
| `output_formats` | `["json","dot","chrome_trace","html","text","csv"]` | Export formats |
| `mode` | `"all"` | `"all"`, `"runtime"`, or `"schedule"` |
| `max_seq_len` | `128` | Sequence length used for dynamic-dim resolution |
| `batch_size` | `2` | Batch size used for dynamic-dim resolution |
| `capture_joint_fx` | `False` | Joint fwd+bwd FX capture |
| `semantic_schedule` | `False` | Generate full PP/TP/DP schedule from config |
| `cost_model` | `False` | Run cost model over compute graph |
| `cost_model_class` | `""` | Custom CostModel class/factory path (empty = MockCostModel) |
| `cost_model_kwargs` | `""` | JSON string (CLI) or dict (registry) of CostModel kwargs |
| `comm_backend` | `""` | `""` (fake) or `"gloo"` (real CPU comm) |
| `device_mode` | `""` | `""` (auto), `"meta"`, or `"cpu"` |
| `operator_swimlane_comm_scope` | `"model_only"` | `"model_only"` hides synthetic PP/DP/FSDP comm from operator swimlanes; `"all"` shows everything |

### 3.2 Programmatic Capture API (as-built)

The capture layer can be used directly for programmatic tracing. This is the only
programmatic API currently implemented:

```python
from torchtitan.experiments.simulator import TraceRecorder, unified_trace

recorder = TraceRecorder(rank=0)
with unified_trace(recorder, use_fake_mode=True, phase="forward"):
    output = model(*inputs)
    recorder.current_phase = "backward"
    output.sum().backward()
result = recorder.build_result()
```

---

## 4. Capture Mode

The simulator uses a single unified capture mode via `unified_trace()`. This context
manager combines `FakeTensorMode` with `TorchDispatchMode` in a single pass, and
optionally activates communication interception and FSDP lifecycle hooks:

```
unified_trace() context manager
  |
  +-- FakeTensorMode (shape-only, 0 bytes) -- when use_fake_mode=True
  |     |
  |     +-- Every op produces FakeTensor outputs (shape/dtype metadata only)
  |
  +-- UnifiedTraceMode (TorchDispatchMode)
  |     |
  |     +-- Every dispatched op is recorded as OpNode in TraceRecorder
  |     +-- Tensor producer-consumer tracking via id(tensor) -> node_id map
  |     +-- Data-flow edges built automatically
  |     +-- Phase tracking (forward/backward/optimizer)
  |
  +-- [optional] CommRecorder -- when capture_comm=True (gloo mode)
  |     |
  |     +-- Monkey-patches torch.distributed collectives and _functional_collectives
  |
  +-- [optional] FSDPEventRecorder -- when capture_fsdp=True and model_parts given
        |
        +-- Attaches module hooks for FSDP allgather/reshard/reduce-scatter lifecycle
```

**Signature:**
```python
def unified_trace(
    recorder: TraceRecorder,
    model: torch.nn.Module | None = None,
    example_inputs: tuple | None = None,
    use_fake_mode: bool = True,
    phase: str = "forward",
    capture_comm: bool = False,
    capture_fsdp: bool = True,
    model_parts: list[torch.nn.Module] | None = None,
) -> Generator[TraceRecorder, None, None]
```
(`model` / `example_inputs` are accepted for API symmetry with FX tracing but are not
used inside the context manager; the model is invoked in the caller's `with` body.)

**TraceRecorder** is the single recorder that accumulates:
- `nodes: list[OpNode]` -- one per dispatched op
- `edges: list[(src, dst, type)]` -- data-flow dependencies (deduped per call)
- `_tensor_producer: dict[int, str]` -- `id(tensor) -> node_id`, the producer-tracking core
- `comm_events`, `fsdp_events`, `pp_events` -- specialized event lists
- Phase/PP-stage/microbatch context via mutable fields

**`build_result()`** assembles everything into a `SimulationResult` with a populated
`ComputeGraph`. If no explicit edges were recorded, it infers sequential edges within each
`(phase, pp_stage, microbatch_idx)` group. It merges `comm_events` as `OpNode` entries
and builds a `TrainingSchedule` from `fsdp_events`/`pp_events`.

A module-global recorder stack (`_RECORDER_STACK` / `get_current_recorder()`) decouples
`CommRecorder`/`FSDPEventRecorder` from the active `TraceRecorder`, letting comm events
form data-flow edges back to compute nodes via `get_producer`/`set_producer`.

---

## 5. Core Data Model (`nodes.py`)

```
SimulationResult
  |
  +-- compute_graph: ComputeGraph
  |     +-- nodes: dict[str, OpNode]     # ordered by insertion
  |     +-- edges: list[DataEdge]         # data / control / phase_boundary
  |     +-- metadata: dict
  |
  +-- schedule: TrainingSchedule | None
  |     +-- events: list[ScheduleEvent]
  |     +-- deps: list[ScheduleDep]
  |
  +-- comm_events: list[dict]            # raw communication events
  +-- fsdp_events: list[dict]            # raw FSDP lifecycle events
  +-- pp_events: list[dict]              # raw PP events
  +-- memory_events: list[MemoryEvent]   # memory allocation/residency estimates
  +-- metadata: dict
```

**OpNode** -- a single operation in the compute graph:
- `op_type`: `"compute"` | `"comm_collective"` | `"comm_p2p"` | `"data_move"` | `"memory"`
- `phase`: `"forward"` | `"backward"` | `"optimizer"`
- `pp_stage`, `pp_rank`, `microbatch_idx`: parallel context
- `comm_op`, `comm_group_size`: for communication ops
- `perf_result: PerfResult | None`: cost model output (compute_time, comm_time, FLOPs, bytes)
- `des_start_time_us`, `des_finish_time_us`: DES engine timestamps

**PerfResult** -- cost model output (all times in µs):
- `compute_time_us`, `comm_time_us`, `total_time_us`, `flops`, `bytes_read`,
  `bytes_written`, `metadata`

**DataEdge** -- `src_node_id`, `dst_node_id`, `edge_type` (`"data"` / `"control"` /
`"pp_p2p"`), `tensor_meta`.

**ScheduleEvent** -- coarse-grained schedule event: `event_id`, `event_type`, `rank`,
`pp_stage`, `pp_rank`, `microbatch_idx`, `logical_clock`, `op_node_ids` (populated by
`link_schedule_to_graph`), `des_start_time_us`, `des_finish_time_us`.

**ScheduleDep** -- `from_event_id`, `to_event_id`, `dep_type` (`"data"` / `"control"` /
`"pp_comm"` / `"fsdp_comm"`).

**MemoryEvent** -- `event_id`, `category`, `bytes`, `phase`, `device`, `dtype`,
`shape`, `node_id`, `lifetime_start`, `lifetime_end`, `metadata`.

**ComputeGraph** provides:
- `fix_comm_phase_labels()`: corrects mislabeled comm node phases (e.g. an FSDP
  `reduce_scatter` that fires after backward but is temporally recorded as forward).
  Re-labels a comm node only if it has exactly one distinct predecessor phase that
  differs from its own. **Must run before `add_phase_boundary_edges`.**
- `add_phase_boundary_edges()`: inserts `phase_end_{phase}` sentinel nodes as
  fan-in/fan-out junctions to enforce forward->backward->optimizer ordering (the
  captured graph only has data-flow edges, so a backward node could otherwise start as
  soon as a single forward predecessor finishes).
- `summary()`: op-type counts

---

## 6. Execution Flow (`trainer_runner.py`)

`run_trainer_simulation()` is the main orchestrator:

```
1. Setup patches (before capture)
   - trainer.device = torch.device("meta")
   - Mock _local_scalar_dense -> 0, FakeTensor.__format__ -> "0.0"
   - No-op clip_grad_norm_ -> meta tensor(0.0), dist_sum/dist_max -> identity
   - No-op parallel_dims get_mesh/get_optional_mesh -> None
   - No-op optimizer.step(), lr_schedulers.step()
   - Wrap sl.log_trace_scalar to coerce tensor values to 0

2. Capture
   - Pre-fetch gradient_accumulation_steps batches (outside FakeTensorMode)
   - Build mock_data_iterator() generator
   - use_fake = comm_backend != "gloo"
   - [use_fake] apply_meta_device_patches()   # Phase 4
   - unified_trace(recorder, use_fake_mode=use_fake,
                   capture_comm=not use_fake, capture_fsdp=not use_fake)
     + _patch_backward_phase(recorder)   # tags backward ops
     + Trainer.train_step(mock_data_iterator())  # one full step
   - [finally] restore_meta_device_patches(); restore all patches

3. Post-processing
   - result = recorder.build_result()
   - Set metadata (operator_swimlane_comm_scope, gradient_accumulation_steps)
   - [semantic_schedule] _inject_semantic_schedule(result, config) # real PyTorch schedule
   - [use_fake] _project_pp_comm_from_schedule(result, trainer)   # PP schedule projection
   - [use_fake] _inject_synthetic_compute_anchors(result, trainer) # conditional (usually skipped)
   - [cost_model] apply_cost_model(result, cm)
   - Memory estimation (build_runtime_memory + attach_model_state_memory)
   - postprocess_extension_result(result, trainer, sim_opts)       # duck-typed hook

4. Export
   - _export_result() -> JSON, DOT, Chrome Trace, HTML, Text, CSV (rank 0 only)
   - _export_workload_graph() -> workload_graph.json (L0-L3 IR)
```

## 7. Device Environment Layer

Two patching modes enable GPU-free execution:

### 7.1 Meta Mode (`meta_env.py`)

- Used for `fake_backend` (the default). Global, largely irreversible monkey-patch.
- `patch_device_type_to_meta()` -- redirects `device_type` to `"meta"`, creates 0-byte
  tensors.
- Enables simulating arbitrarily large models (e.g. 1T+ parameters) with minimal RAM.
- Patches `torchtitan.tools.utils.device_type`, `device_module`, and downstream modules
  (`metrics`, `parallel_dims`, `distributed.utils`).
- Patches FSDP2 internals for meta device: `_get_device_from_mesh` -> `torch.device("meta")`
  when `mesh.device_type == "meta"`; re-exports into `_fsdp_init`, `_fully_shard`,
  `_fsdp_param_group`, `_fsdp_state`, `_fsdp_collectives`; patches
  `_get_device_handle`; patches `FSDPParamGroup._validate_no_meta_params` -> no-op.
- Patches `Decoder._init_self_buffers` (buffer_device=None on meta).
- Patches `torch.cuda.*` entrypoints with meta stubs (0 devices, 80GB reported).

### 7.2 CPU Mode (`cpu_env.py`)

- Used for `gloo` backend (real CPU communication capture).
- `patch_device_type_to_cpu()` -- redirects `device_type` to `"cpu"`, creates real CPU
  tensors (required for gloo tensor exchange).
- `init_cpu_distributed()` -- sets up gloo process group for single-process sim
  (`master_addr`/`master_port` via env, `dist.init_process_group(backend="gloo")`).
- Patches `torch.cuda.*` entrypoints with CPU stubs (1 device, `CPU_Simulator`).

### Device Mode Selection Logic

```
comm_backend == "gloo"  -->  device_mode = "cpu"     # needs real tensors
comm_backend == ""      -->  device_mode = "meta"    # 0 bytes, huge models
(overridable via SimulationConfig.device_mode)
```

---

## 8. Capture Layer (`capture/unified_trace.py`)

A single module consolidates all trace capture. Components:

### 8.1 TraceRecorder

The central recorder (see §4). `record()` builds an `OpNode`, constructs "data" edges
from input tensors' producers, and updates `_tensor_producer` for each output tensor.

### 8.2 UnifiedTraceMode

`TorchDispatchMode` subclass that intercepts every dispatched op:
- **Short-circuits** ops in `TRIVIAL_TARGETS` (detach, alias, view, as_strided,
  unsafe_view, lift, lift_fresh_copy, t) -- returns unrecorded.
- Runs the op, then records it.
- Captures scalar Python args as `arg_{i}` attrs.
- Classifies ops via `op_classification.classify_op()`.

### 8.3 CommRecorder

Thread-safe (`threading.Lock`). Intercepts `torch.distributed` functions:
- `all_reduce`, `all_gather`, `all_gather_into_tensor`, `reduce_scatter`,
  `reduce_scatter_tensor`, `all_to_all`, `all_to_all_single`, `send`, `recv`, `isend`,
  `irecv`, `broadcast`, `barrier` (13 functions via the `capture_comms` context).
- Also patches `torch.distributed._functional_collectives` (used by FSDP2/DTensor):
  `all_reduce`, `all_gather_tensor`, `reduce_scatter_tensor`, `all_to_all_single`,
  `broadcast`, `wait_tensor`.
- Records tensor metadata, group size, PP stage, microbatch.
- Resolves source node references from the active `TraceRecorder` via
  `get_current_recorder()`.
- `all_gather_into_tensor`, `reduce_scatter`, etc. call `set_producer(output, event_id)`
  to link comm events into the data-flow graph.

### 8.4 FSDPEventRecorder

Thread-safe. Attaches PyTorch module hooks to FSDP-wrapped modules:
- `forward_pre_hook` -> `fsdp_allgather_pre_fwd` (allgather_params)
- `forward_hook` -> `fsdp_reshard_post_fwd` (reshard_params)
- `backward_pre_hook` -> `fsdp_allgather_pre_bwd` (allgather_params_for_bwd)
- `backward_hook` -> `fsdp_reduce_scatter_post_bwd` (reduce_scatter_grads)

### 8.5 unified_trace() Context Manager

Orchestrates all components (see §4 signature). Conditionally enters `FakeTensorMode`
+ `UnifiedTraceMode`, then optionally `CommRecorder` (gloo) and, per model part,
`FSDPEventRecorder`. After exit, transfers `comm_events`/`fsdp_events` onto the recorder.

---

## 9. Schedule Layer (`schedule/`)

### 9.1 Schedule Extraction (`schedule_extract.py`) -- the runtime path

**Core strategy**: Construct a real PyTorch `_PipelineSchedule` with `MockPipelineStage`
instances and read its `pipeline_order_with_comms` action table.

This guarantees the extracted schedule matches upstream PyTorch behavior exactly, and
automatically picks up schedule algorithm changes without simulator code
modifications. This is the path actually used by `_inject_semantic_schedule()` when
`semantic_schedule=True`.

`MockPipelineStage` -- duck-typed mock that satisfies `_PipelineSchedule.__init__`
attribute reads without calling `dist.get_rank()` / `dist.get_world_size()`.

Supported schedules: `Schedule1F1B`, `ScheduleGPipe`, `ScheduleInterleaved1F1B`,
`ScheduleLoopedBFS`, `ScheduleInterleavedZeroBubble`, `ScheduleZBVZeroBubble`,
`ScheduleDualPayV`, and CSV-driven `_PipelineScheduleRuntime` -- anything registered in
`torch.distributed.pipelining.schedules`.

Action type mapping (`_ComputationType` -> `event_type`):

| Action | Event Type |
|--------|-----------|
| `F` | `pp_forward` |
| `B` | `pp_backward` |
| `I` | `pp_backward_input` |
| `W` | `pp_backward_weight` |
| `UNSHARD` | `fsdp2_all_gather` |
| `RESHARD` | `fsdp2_reduce_scatter` |
| `SEND_F` | `pp_send_activation` |
| `RECV_F` | `pp_recv_activation` |
| `SEND_B` | `pp_send_gradient` |
| `RECV_B` | `pp_recv_gradient` |
| `REDUCE_GRAD` | `dp_gradient_sync` |
| `OVERLAP_F_B` | `overlap_forward_backward` |

`_convert_pipeline_order_to_training_schedule()` is the shared conversion core
(imported by both extraction entry points). It: converts actions to `ScheduleEvent`s;
adds sequential `control` deps within each PP rank; adds cross-rank PP comm deps
(`SEND_F(stage=s)->RECV_F(stage=s+1)`, `SEND_B(stage=s)->RECV_B(stage=s-1)`); replicates
PP-group events across TP/DP sibling ranks; appends an `optimizer_step` event per rank.

### 9.2 PP Schedule Extractor (`pp_schedule_extractor.py`)

`PPScheduleExtractor` class for extracting from a pre-existing `_PipelineSchedule`
instance. Primary path delegates to `schedule_extract._convert_pipeline_order_to_training_schedule()`
(with `tp_degree=1, dp_degree=1`). Fallback: heuristic 1F1B reconstruction.

---

## 10. Cost Model (`cost_model.py`)

### Architecture

```
CostModel (ABC)
  +-- estimate_node(OpNode) -> PerfResult            # abstract
  +-- estimate_graph(ComputeGraph)                    # annotates every node in-place
  +-- predict_step_time_us(ComputeGraph)              # critical-path analysis (O(V+E))
  +-- estimate_result(SimulationResult)               # convenience over compute_graph
  |
  +-- MockCostModel (concrete)
        - compute_time = flops / (tflops * 1e6)
        - comm_time = latency + bytes / (comm_gb_per_s * 1e3)
        - memory-bound ceiling when arithmetic intensity < threshold
        - Gaussian noise for realism
        - optional OverlapStrategy
```

### `MockCostModel` constructor args

`tflops=10.0` (FP16/BF16), `gb_per_s=100.0` (HBM), `comm_gb_per_s=50.0`,
`comm_latency_us=5.0`, `arithmetic_intensity_threshold=10.0`, `noise_std=0.05`,
`seed=42`, `default_seq_len=4096`, `overlap_strategy: OverlapStrategy | None = None`.

### OverlapStrategy

- `OverlapStrategy` (ABC): `overlap_factor(compute_us, comm_us) -> float`.
- `NoOverlap`: sums (`compute + comm`).
- `FixedOverlap(factor=0.5)`: `compute + max(0, comm - compute*factor)`.

### FLOPs Estimation (`_estimate_flops`)

Heuristic rules per op category:
- **Matmul-like** (`mm`, `matmul`, `bmm`, `addmm`, `linear`): `2 * batch * M * K * N`
- **Attention** (`scaled_dot_product_attention`): `2 * numel(Q) * K_seq + 2 * numel(Q) * V_dim`
- **DeepEP/MoE** (`deepep.dispatch`, `deepep.combine`): rough approximation
- **Activations** (gelu, silu, sigmoid, ...): 1-5 FLOPs per output element
- **Norms** (rms_norm, layer_norm): 5 FLOPs per element

### Communication Bytes Estimation (`_estimate_comm_bytes`)

Ring algorithm scaling:
- `all_gather`: `input_bytes * (P-1)/P`
- `reduce_scatter`: `input_bytes * (P-1)/P`
- `all_reduce`: `output_bytes * 2*(P-1)/P`
- `send`/`recv`: raw tensor bytes

### DimResolver

Resolves symbolic shape dims to concrete values via a map (`hidden_dim`, `seq_len`,
`num_heads`, `num_experts`, `vocab_size`) extracted from model config. Used so dynamic
shapes (e.g. `-1` batch/seq dims) still produce concrete FLOPs/bytes.

### Schedule linking

- **`link_schedule_to_graph(result)`**: populates `ScheduleEvent.op_node_ids` by matching
  `(phase, pp_stage, microbatch_idx)`. For single-rank traces (nodes lack stage/mb
  labels), assigns empty `op_node_ids` to avoid duplicating time across events.
- **`predict_multi_rank_step_time_us(result, cost_model)`**: lazily imports
  `simulate_multi_rank_des`; falls back to single-rank critical path if no schedule events.

### `apply_cost_model(result, cost_model=None)`

The function called from `trainer_runner.py`:
1. If `cost_model is None` -> `MockCostModel()`.
2. `cost_model.estimate_graph(result.compute_graph)`.
3. `single_rank_step` = critical-path time.
4. `e2e_step` = `predict_multi_rank_step_time_us` (multi-rank DES if schedule present).
5. Builds per-phase breakdown (`compute_time_us`, `comm_time_us`, `total_time_us`).
6. Returns dict with `e2e_step_time_us`, `single_rank_step_time_us`,
   `total_compute_time_us`, `total_comm_time_us`, `per_phase`.

---

## 11. DES Engine (`des_engine.py`)

Uses **salabim** discrete-event simulation library to model resource contention:

```
Per-rank resources (capacity=1 each):
  - compute: handles compute, data_move, memory ops
  - comm:    handles comm_collective, comm_p2p ops
```

Ops on separate engines can overlap (modeling GPU compute/comm parallelism). Ops on the
same engine are serialized (resource contention). DAG dependencies are modeled via
salabim `State` signals.

### Two Simulation Levels

1. **Single-rank DES** (`simulate_single_rank_des`):
   - Topological sort of compute graph; each node becomes an `_OpComponent` with a
     resource request. Returns max finish time across all nodes.

2. **Multi-rank DES** (`simulate_multi_rank_des`):
   - Uses `TrainingSchedule` events and dependencies. Each event becomes a
     `_ScheduleEventComponent`.
   - Duration derived from linked `OpNode.perf_result` via `link_schedule_to_graph()`:
     summed across matched nodes then divided by `events_per_key` (proportional split
     across microbatches).
   - Separate compute/comm resources per rank.
   - Cross-rank dependencies (PP send/recv) create inter-rank synchronization.

`_event_engine_type(event_type)` maps PP/FSDP2/DP event types to `"comm"`
(`pp_send_activation`, `pp_recv_activation`, `pp_send_gradient`, `pp_recv_gradient`,
`fsdp2_all_gather`, `fsdp2_reduce_scatter`, `dp_gradient_sync`); everything else ->
`"compute"`.

### `DESEngine` Class

```python
engine = DESEngine()
step_time = engine.predict_step_time_us(result, cost_model)  # multi-rank if schedule present
engine.annotate(result)  # writes result.metadata["des_engine"]["e2e_step_time_us"]
```

### Utilization Analysis (`compute_des_utilization`)

Computes from annotated nodes/events (lazily cached into `result.metadata["des_engine"]`
by `_populate_des_metadata` in `export.py`):
- `e2e_step_time_us`, `single_rank_step_time_us`
- `compute_busy_pct` / `comm_busy_pct`: engine utilization
- `overlap_pct`: compute/comm overlap (via interval merging)
- `contention_count`: ops whose actual duration exceeds perf duration by > 0.1µs
- `des_vs_cp_ratio`: DES time vs critical-path time (quantifies contention impact)

### DES Memory Timeline (`compute_des_memory_timeline`)

Maps `MemoryEvent` lifetimes to DES wall-clock timestamps (cached into
`result.metadata["des_memory"]`). Builds a sweep-line timeline of alloc/free events;
computes `peak_dynamic_bytes`, `peak_total_bytes`, and `phase_peak` (peak per phase with
category breakdown). Used by the HTML memory-trace chart.

---

## 12. Memory Estimation (`memory_estimator.py`)

Three estimation sources:

1. **Graph memory** (`estimate_graph_memory`):
   - Activation/output memory from node outputs. Lifetime approximated by node order
     (producer index to last-consumer index).
   - Categories: `activation`, `comm_buffer`, `allocation`, `data_move`.

2. **Comm memory** (`estimate_comm_memory`):
   - Communication buffer memory from comm events.
   - Category: `comm_event_buffer`.

3. **Model state memory** (`estimate_model_state_memory`):
   - Parameter memory from `model.named_parameters()`.
   - Optimizer state memory (Adam: 2x param size for momentum + variance).
   - Accounts for sharding: `shard_factor = max(1, tp_degree * fsdp_degree)`;
     per-GPU parameter event = `nbytes // shard_factor`.
   - **EP/MoE expert distribution is explicitly NOT modeled.**

`build_runtime_memory()` combines graph + comm memory; `attach_model_state_memory()`
extends `result.memory_events` and finalizes `result.metadata["memory"]`.

---

## 13. Layered IR (`ir/`)

A four-level intermediate representation that projects captured data into a spec-aligned
format (mirroring `ZhanluModelSim/workload-model-platform`) for downstream hardware
simulators. **All layers are read-only projections** -- they never mutate the captured
graph and never re-implement torchtitan logic.

```
L0: SpecOpNode       -- single op with cost + adjacency
      |
L1: StepGraph        -- per-phase DAG template (forward/backward/optimizer)
      |
L2: ScheduleGraph    -- StepInstance orchestration, DataPass, parallelism degrees
      |
L3: WorkloadGraph    -- iteration semantics, data flow, cross-iteration passes
```

### Projection Pipeline

```
SimulationResult
  +-- build_step_graphs(result)     --> dict[str, StepGraph]    (L1)
  +-- build_schedule_graph(result)  --> ScheduleGraph            (L2)
  +-- build_workload_graph(result) --> WorkloadGraph             (L3, the full pipeline)
```

`_gradient_accumulation(result, config)` is capture-faithful: the value captured in
`result.metadata["gradient_accumulation_steps"]` takes precedence over the declared
config value.

### L0: SpecOpNode (`ir/op_node.py`)

- `flops`, `peak_mem` (output tensor volumes), `param_mem` (grad-requiring inputs),
  `comm_bytes` (output or input volumes for comm ops).
- `predecessors`, `successors` -- adjacency built from DATA edges only.
- `project_op_nodes(graph)` builds the full L0 projection.

### L1: StepGraph (`ir/step_graph.py`)

- `StepBuilder.from_compute_graph` partitions the captured graph by phase.
- Computes `entry_nodes` (indeg=0), `exit_nodes` (outdeg=0), `tensor_lifetimes`
  (span metric using capture order as a topological proxy), aggregates (`total_flops`,
  `peak_active_mem`, `param_mem`, `comm_volume`).
- Validates acyclicity via Kahn's algorithm (`_kahn_is_acyclic`).

### L2: ScheduleGraph (`ir/schedule_graph.py`)

- `ScheduleBuilder.from_capture(step_templates, schedule, parallelism, ...)`.
- `StepInstance` -- (step_ref, micro_batch_idx, pipeline_stage, device_ids, dp_group).
- `DataPass` -- tensor flow between instances (`TensorSlot`, `comm_primitive`).
- If a captured PP schedule is present: builds instances from fwd/bwd events and wires
  `p2p_send_recv` data passes between adjacent stages. Fallback: a single
  fwd->bwd->opt chain per gradient-accumulation micro-batch.
- Parallelism degrees: pp, tp, dp, ep, cp.

### L3: WorkloadGraph (`ir/workload_graph.py`)

- `WorkloadBuilder.from_capture(schedule_graph, step_templates, training)`.
- `IterationSpec` wraps ScheduleGraph with microbatch count.
- `DataFlow` for the dataloader (volume from `batch * seq_len * dtype_bytes * ga`).
- `cross_iter_passes` -- e.g. optimizer -> first forward of the next iteration.

---

## 14. Export System (`export.py`)

| Format | File | Description |
|--------|------|-------------|
| JSON | `simulation_result.json` | Full structured dump; pretty for <=10K nodes, compact otherwise |
| DOT | `compute_graph.dot` | Graphviz with color-coded nodes by op type |
| Chrome Trace | `trace.json` | `chrome://tracing` compatible timeline (DES dual-track when present) |
| HTML | `trace.html` | Interactive visualization (data embedded inline; **ECharts loaded from CDN**) |
| Text | `summary.txt` | Human-readable statistics |
| CSV | `kernel_summary.csv` | Per-operator kernel trace (nsys/msprof compatible) |
| Workload Graph | `workload_graph.json` | L0-L3 IR projection |

> **CDN caveat:** The HTML trace embeds all result data inline, but renders charts with
> **ECharts 5.4.3 loaded from `cdn.jsdelivr.net`**. It is therefore not fully offline /
> self-contained. The `export_html` docstring's mention of "AntV G6" is stale -- G6 is
> not used.

### Chrome Trace pid layout

0=OpNodes, 1=FSDP, 2=PP, 3=FSDP sched, 4=TP sched, 5=DP sched, 6=Optimizer, 7=aggregated
phase blocks. When DES timing is present, ops split into `compute_engine` /
`comm_engine` dual tracks.

### HTML Visualization

The generated HTML contains:
- Summary metric cards (node/edge/schedule counts, memory, step time; extra DES cards
  when timing present).
- Parallelism line (TP, FSDP, shard_factor).
- Memory-trace timeline chart (per-GPU total/static, whole-model static).
- Per train-step sections with:
  - **PP/FSDP2/TP/DP/comm schedule swimlanes** (ECharts Gantt, per rank).
  - **Per-phase operator swimlanes (Cube / Vec / Communication)** implementing an
    **in-browser two-engine DES scheduler** (Cube+Vec share a serialized Compute engine;
    Communication on a separate Comm engine; Kahn's topological sort). Filters synthetic
    cluster-parallel comm nodes unless `operator_swimlane_comm_scope == "all"`.

`export_text_summary` returns a string (the runner writes it to `summary.txt`).

---

## 15. Communication Capture Strategies

There are **three** layers of communication in a simulation result, in increasing order
of "naturalness":

### 15.1 Natural capture (default, Phase 4) -- TP / EP / FSDP2

The default `fake_backend` / meta path runs the **real** parallelization
(`_meta_parallelize_with_skip_fsdp`) on meta device. With `apply_meta_device_patches()`
active (7 patches -- see §20), DTensor TP and FSDP2 **naturally emit** their
communication operators through the dispatcher, which `unified_trace` captures exactly:

- **DTensor TP** on meta + `FakeProcessGroup(world_size>1)`: emits `all_reduce` +
  `wait_tensor` per TP reduction.
- **FSDP2** on meta: emits `all_gather` (forward unshard) and `reduce_scatter`
  (backward gradient reduction), with shapes derived from actual parameter dimensions.
- **EP** all-to-all: emitted naturally by `AllToAllTokenDispatcher` (with forced
  load-balance mock on meta -- see §20.6).

The model is actually **run** during the trace via `SequentialPipelineSchedule`
(§16), which executes all model parts sequentially and calls `loss.backward()`.
This triggers FSDP2 lifecycle hooks (unshard/reshard/reduce-scatter) and DTensor
TP reductions, producing real comm ops in the captured graph.

Verified on DeepSeek V4 smoketest (PP=2, TP=2, DP=2): **223 natural comm ops**
(all_gather×85, reduce_scatter×14, all_reduce×19, wait×105), bwd/fwd ratio = 2.15.

### 15.2 Schedule-derived PP communication (replaces synthetic injection)

Pipeline parallelism uses `dist.send()`/`dist.recv()` directly (not DTensor dispatch),
so PP send/recv cannot be naturally captured by `UnifiedTraceMode`.  Instead of
heuristic injection (`_inject_pp_send_recv`, now removed), PP communication is
**projected from the semantic schedule** via `_project_pp_comm_from_schedule()`:

1. `_inject_semantic_schedule()` runs first, populating `result.schedule` with PP
   events (`pp_send_activation`, `pp_recv_activation`, `pp_send_gradient`,
   `pp_recv_gradient`) derived from the real `_PipelineSchedule.pipeline_order_with_comms`
   action table.
2. `_project_pp_comm_from_schedule()` reads rank-0 PP events from `result.schedule`
   and projects them into the compute graph as `comm_p2p` `OpNode`s with
   `attrs={"schedule_derived": True, "pp": True}`.
3. PP send→recv data edges are created from the schedule's `pp_comm` dependencies.

This is "schedule-derived" (from real PyTorch schedule) — not "captured" (from
execution) but not "synthetic" (heuristic) either.  The design principle says
schedules should come from "captured data **or real PyTorch schedule objects**" —
this fits.

### 15.3 Compute anchors (conditional, usually skipped)

`_inject_synthetic_compute_anchors(result, trainer)` checks if each phase has enough
"Cube" (matmul-like) and "Vec" (`aten.add.Tensor`) lane signal for the HTML operator
swimlanes. If the natural capture already exceeds the threshold (`pp * num_layers`),
it returns immediately without injecting anything.

With Phase 4, the model is actually run (via `SequentialPipelineSchedule`), producing
thousands of real compute nodes. The anchors are **effectively always skipped** —
verified on smoketest: 5935 natural compute nodes vs threshold of 4.

### 15.4 gloo Mode

Real CPU communication capture:
- FSDP1 wrapping applied post-init via `_apply_fsdp1_on_cpu()`.
- `CommRecorder` intercepts real all-gather/reduce-scatter calls.
- Requires `init_cpu_distributed()` for process group setup.
- Single-process sufficient (uses `FakeProcessGroup` for `init_distributed`).

> **Note:** gloo mode is effectively unreachable from `run_train.sh` today because
> `SimulationTrainer.__init__` forces `config.comm.mode = "fake_backend"`, which then
> overrides `comm_backend = ""`.

---

## 16. Pipeline Parallelism on CPU

### 16.1 Semantic Pipeline (`_cpu_semantic_pipeline`)

For PP > 1 in fake_backend mode:
1. Applies real parallelization (TP/EP/FSDP2) to the **full** model via
   `_meta_parallelize_with_skip_fsdp`. Does **not** PP-split the model — the split
   uses `copy.deepcopy` which strips the FSDP2 wrapper class and breaks DTensor
   mesh alignment.
2. Returns a `SequentialPipelineSchedule` that runs all model parts sequentially
   (flattened PP: `part_0(input) → part_1(out) → … → loss.backward()`). This
   triggers FSDP2 lifecycle hooks on each part, emitting `all_gather` /
   `reduce_scatter` naturally during the trace.
3. The real PP schedule is extracted separately via `_inject_semantic_schedule()`
   (uses `extract_schedule_from_pytorch` with mock pipeline stages — no model
   execution needed).

**Key design choice:** PP split is skipped to preserve FSDP2's wrapper class and
DTensor mesh alignment. The PP schedule for visualization/DES comes from
`extract_schedule_from_pytorch`, not from actual PP execution.

### Gloo Pipeline

For PP > 1 in gloo mode: `_cpu_pp_module_split` splits the model into stages on CPU;
upstream schedule objects with real gloo communication are used.

---

## 17. Model Configuration Registries

Each model has its own `config_registry.py` returning `SimulationTrainer.Config`. Both
reuses upstream model code (`torchtitan.models.*`); there are no `parallelize.py` /
architecture files in the simulator model dirs.

### Llama3

| Config | Topology | Notes |
|--------|----------|-------|
| `llama3_sim_debugmodel` | 1 GPU (default parallelism) | Small model, gloo comm, cost model, seq=64, bs=1 |
| `llama3_sim_1024gpu` | PP=4, TP=8, dp_shard=4, dp_replicate=8 (1024 GPUs), Interleaved1F1B, vpp=2 | semantic_schedule=True, seq=64, bs=8, 8 microbatches |

### DeepSeek V4

| Config | Topology | Notes |
|--------|----------|-------|
| `deepseek_v4_sim_smoketest` | PP=2, TP=2, dp_shard=2, dp_replicate=1 (8 ranks), Interleaved1F1B | 2-layer smoketest, vocab=129280, seq=128, bs=4 |
| `deepseek_v4_pro_sim_smoketest` | PP=8, TP=8, dp_shard=-1(auto), EP=192, Interleaved1F1B | DeepSeek V4 Pro 61-layer, seq=4096, bs=1 |

All configs register all six output formats and enable `cost_model=True`.

---

## 18. Extension System (`extension_hooks.py`)

Two duck-typed hooks for external side-loads (e.g. torchtitan-npu):

1. `collect_extension_metadata(trainer, capture)` -- calls
   `trainer.collect_simulation_metadata(capture)` if implemented. Returns `{}` if
   missing/None/non-dict. **Defined but not called in `trainer_runner.py`** -- utility
   for extension packages.
2. `postprocess_extension_result(result, trainer, sim_opts)` -- calls
   `trainer.postprocess_simulation_result(result, sim_opts)` if implemented. **Called**
   in the post-processing step (before export).

---

## 19. Key Design Decisions

### 19.1 Why a single entry point (`SimulationTrainer`)?

- **Consistency**: All simulation runs go through the same `Trainer` initialization path,
  ensuring the model is constructed, parallelized, and configured identically to real
  training.
- **Config-driven**: `SimulationConfig` integrates with torchtitan's config system.
- **No API drift**: A single entry point prevents divergence.

### 19.2 Why a single capture module (`unified_trace.py`)?

- **No circular dependencies**: `CommRecorder` and `FSDPEventRecorder` are inlined with
  `TraceRecorder`.
- **Single source of truth** for all capture logic.

### 19.3 Why `FakeTensorMode` + `TorchDispatchMode`?

- **Zero memory**: Shape-only tensors enable 1T+ parameter models.
- **Dispatcher-level**: Captures all ATen ops including those hidden by autograd/FSDP.

### 19.4 Why Mock PyTorch Schedule Objects?

- Reuses upstream schedule algorithms exactly; auto-adapts to new schedule types
  (ZeroBubble, DualPipeV, ...).

### 19.5 Why natural capture over synthetic injection?

- TP/FSDP communication shapes are exact (from DTensor/FSDP2 dispatch), not heuristic.
- Automatic support for EP/CP without simulator code changes.
- ~50 lines of meta patches replace ~400 lines of injection logic.

### 19.6 Why salabim DES?

- Models compute/comm resource contention.
- Handles cross-rank synchronization (PP send/recv dependencies).

### 19.7 Why Layered IR (L0-L3)?

- Separation of concerns: op -> step -> schedule -> workload.
- Spec-aligned for downstream hardware simulators.

---

## 20. Phase 4 -- Natural Communication Capture (COMPLETED & VERIFIED)

> This section documents the as-built Phase 4 work. The code is integrated, active in
> the default path, and **verified via E2E smoketest** (223 natural comm ops captured).

### 20.1 Problem it solved

The earlier heuristic communication-injection path used assumptions such as uniform layer
distribution, manual tensor-size calculation, and hardcoded patterns (e.g. 2 TP all_reduce
per layer). This was not capture-faithful.

### 20.2 Meta device patches (`meta_device_patches.py`) — 7 patches

| # | Patch | Effect |
|---|-------|--------|
| 1 | `FSDPParamGroup._validate_no_meta_params = lambda self: None` | Skip validation that rejects meta parameters |
| 2 | `FakeTensor._find_common_device = _patched_find_common_device` | Prefer meta device for FSDP mixed meta/cpu ops |
| 3 | `FakeTensorMode.wrap_meta_outputs_with_default_device_logic = _patched_wrap` | Convert CPU tensors to meta for FSDP internal buffers |
| 4 | `foreach_reduce` dtype coercer | Coerce mixed-dtype gradients to uniform dtype (avoids FSDP2 reduce-scatter dtype assertion on meta) |
| 5 | `_unimplemented_deepcopy` → `_fsdp_meta_deepcopy` | Allow FSDP module deepcopy for PP module splitting (bypass `__deepcopy__` block + `disable_fsdp_module_new_init` + identity-copy `ProcessGroup`/`DeviceMesh`) |
| 6 | `nn.Module.to_empty` no-op on meta | Preserve FSDP2 sharding state (to_empty creates fresh tensors, discarding FSDP2's DTensor placements) |
| 7 | `repeat_interleave` fake impl override | Return placeholder shape instead of raising `DynamicOutputShapeException` (MoE dispatch uses dynamic-shape `repeat_interleave`) |

### 20.3 Additional fixes for E2E natural capture

Beyond the 7 meta patches, the following fixes enable the model to **actually run**
during the trace (triggering FSDP2/TP hooks):

| Fix | File | Effect |
|-----|------|--------|
| **`apply_fsdp` from llama4** | `models/deepseek_v4/parallelize.py` | Import EP-aware `apply_fsdp` from `llama4` (supports `edp_mesh`/`ep_degree`) instead of llama3 (which doesn't). Filters `gradient_divide_factor`. |
| **`SequentialPipelineSchedule`** | `trainer.py` | Replaces `MockSchedule` (no-op). Runs all model parts sequentially + `loss.backward()`, triggering FSDP2 hooks. |
| **Skip PP split** | `trainer.py` | Don't `deepcopy`/split the model — preserves FSDP2 wrapper class and DTensor mesh alignment. PP schedule comes from `extract_schedule_from_pytorch` separately. |
| **`mixed_precision_param=fp32`** | `trainer.py` | Force fp32 on meta (FSDP2's bfloat16 casting causes DTensor dtype mismatch on meta). |
| **Disable activation checkpointing** | `trainer.py` | AC's mutation check raises on FakeTensors; AC has no memory benefit on meta (0-byte tensors). |
| **`DTensor.__format__` patch** | `trainer_runner.py` | Prevents format crash when logging/metrics format DTensors. |
| **EP expert padding** | `models/common/token_dispatcher.py` | Pad `num_tokens_per_expert` to be divisible by `ep_size` when `num_experts` is not (e.g. 256 experts, EP=192). |
| **Forced load-balance MoE dispatch** | `models/common/token_dispatcher.py` | On meta (FakeTensor detected), bypass dynamic `bincount`/`all_to_all`/`.tolist()` (all return zeros) and use uniform token distribution. Gives concrete non-zero split lists so all downstream shapes are static. Identity permutation (uniform = no reordering needed). |

### 20.4 Integration points

- **`trainer.py`** — `_meta_parallelize_with_skip_fsdp.wrapper`: applies meta patches
  before calling the real parallelize fn, forces fp32 + disables AC, restores in `finally`.
- **`trainer_runner.py`** — `run_trainer_simulation`: applies meta patches around the
  `unified_trace(...)` + `Trainer.train_step(...)` block, patches `DTensor.__format__`,
  restores in `finally`.
- **`models/deepseek_v4/parallelize.py`** — imports `apply_fsdp` from `llama4` (not
  `llama3`), enabling EP-aware FSDP2 with per-param mesh routing.
- **`models/common/token_dispatcher.py`** — `AllToAllTokenDispatcher.dispatch/combine`
  have a `_is_fake` branch that uses forced load-balance on meta.

### 20.5 As-built communication taxonomy

After Phase 4, a simulation result's communication comes from:
- **Natural** (TP/FSDP2/EP): exact shapes from DTensor/FSDP2 dispatch. Verified: 223
  ops on smoketest (all_gather, reduce_scatter, all_reduce, wait_tensor).
- **Schedule-derived PP** (`_project_pp_comm_from_schedule`): projected from real
  `_PipelineSchedule`, shapes estimated from config.
- **Synthetic compute anchors** (`_inject_synthetic_compute_anchors`): Cube/Vec lane
  padding for visualization.

### 20.6 Forced load-balance MoE dispatch (meta-only)

On meta device, FakeTensors carry no real values. The MoE EP dispatch chain
(`bincount` → `all_to_all` → `.tolist()` → dynamic splits) produces all zeros,
leading to zero-sized tensors and shape crashes. The forced load-balance mock
bypasses this by computing **uniform** token counts from known static quantities:

```
total_routed = num_tokens * top_k           (static)
tokens_per_rank = total_routed // ep_size   (uniform)
tokens_per_expert = tokens_per_rank // num_local_experts
```

Split lists are constructed as Python ints (not `.tolist()` on FakeTensors).
`_permute` is replaced with identity permutation (uniform distribution = no
reordering). `_unpermute` in `combine` is skipped (identity).

This is a **simulation approximation** — real training has non-uniform token
distribution. But for comm op capture (all_to_all, FSDP2 all_gather/reduce_scatter),
the exact token counts don't affect the communication shapes (which come from
parameter dimensions, not token counts).

### 20.7 Remaining limitations

- **Meta device patches are fragile** — depend on PyTorch internal implementation.
- **Forced load-balance is an approximation** — real MoE routing is non-uniform.
- **Pro 61-layer E2E is slow** (~8 min/step) — use 4-layer config for faster testing.
- **`repeat_interleave` fake impl returns placeholder shape** — may not match
  downstream expectations in all cases.

---

## 21. Tech Debt & Known Discrepancies

1. **Event-type taxonomy inconsistent** across schedule files. Unify on `pp_*`/`fsdp2_*`.
2. **`Simulator` programmatic API does not exist.** See §22.
3. **HTML trace not self-contained.** ECharts from CDN.
4. **gloo mode unreachable from `run_train.sh`.** Trainer forces fake_backend.
5. **Forced load-balance is a simulation approximation.** MoE EP dispatch uses
   uniform token distribution on meta (§20.6).
6. **`mixed_precision_param` forced to fp32 on meta.** bfloat16 causes DTensor
   dtype mismatch.
7. **Activation checkpointing disabled on meta.** AC mutation check raises on
   FakeTensors.
8. **`apply_fsdp` imported from llama4.** Cross-model dependency.
9. **DeepSeek-specific logic in `trainer.py`.** `expert_parallel_comm_backend`,
   `fsdp_gradient_divide_factor`, `get_optional_mesh` patches, and model-name
   string checks are DeepSeek-specific but live in the generic simulator wrapper.
   Should be moved to a model adapter layer.
10. **Simulator logic in core model files.** `token_dispatcher.py` has a
    `_is_fake` branch for forced load-balance. Annotated as simulator-only but
    not isolated to `experiments/simulator/`.
11. **`collect_extension_metadata` defined but not called.** Extension packages
    cannot collect metadata during capture.
12. **Config fields reserved but unimplemented.** `mode`, `capture_joint_fx`,
    `max_seq_len`, `batch_size` are defined but not consumed. Annotated as
    "Reserved for future use."

---

## 22. Roadmap: Programmatic `Simulator` API

A higher-level programmatic API is planned but **not yet implemented**. The intended
design (referenced in `AGENTS.md`) is a `Simulator` class with three modes:

| Mode | Purpose | Basis |
|------|---------|-------|
| `simulate_fx` | Static FX trace capture | `torch.fx` symbolic trace of the model |
| `simulate_runtime` | Dynamic 1-step capture | The current `unified_trace` + train-step path |
| `simulate_pp_schedule` | Schedule extraction only | `extract_schedule_from_pytorch` |

```python
from torchtitan.experiments.simulator import Simulator

result = Simulator(model, config).simulate_runtime()   # intended
```

Today, the equivalent is the `TraceRecorder` + `unified_trace` direct usage (§3.2) for
capture, or `SimulationTrainer` (§3.1) for end-to-end runs.
