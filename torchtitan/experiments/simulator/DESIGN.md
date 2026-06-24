# TorchTitan Simulator -- Architecture Design Document

## 1. Overview

TorchTitan Simulator is a CPU-only training trace/simulation system built as a side-loaded experiment on top of the upstream torchtitan LLM training platform. It captures forward/backward computation graphs, communication patterns, and training schedules (PP, FSDP, TP, DP) **without any GPU hardware**, enabling:

- **Training step profiling** at arbitrary scale (e.g., 1024-GPU topology) on a single CPU machine
- **Performance prediction** via cost models and discrete-event simulation (DES)
- **Parallelism strategy exploration** (PP/TP/DP/FSDP degree combinations) without real hardware
- **Workload graph export** for downstream hardware simulators (ZhanluModelSim etc.)

### Design Principles

1. **Side-loaded experiment** -- `train.py` remains unchanged; all simulator code lives under `torchtitan/experiments/simulator/`
2. **Capture-faithful** -- schedules and compute graphs are derived from *captured* data, not re-implemented training logic
3. **PyTorch-native** -- reuses upstream PyTorch schedule objects (`PipelineSchedule`), FX tracing, and `TorchDispatchMode`
4. **Model-agnostic core** -- model-specific code is isolated in per-model subdirectories (`llama3/`, `deepseek_v4/`)

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
  synthetic_dataloader.py  # SyntheticTokenDataLoader (random token generation)
  extension_hooks.py       # Extension points for NPU/other side-loads

  capture/                 # Trace capture (single unified module)
    unified_trace.py       # FakeTensorMode + TorchDispatchMode + CommRecorder + FSDP hooks

  schedule/                # Training schedule extraction & generation
    schedule_extract.py    # Extract schedule from real PyTorch PipelineSchedule objects
    schedule_generator.py  # Generate semantic Interleaved1F1B schedules from config
    pp_schedule_extractor.py # PPScheduleExtractor class (reads pipeline_order tables)

  ir/                      # Layered IR (L0-L3) for workload graph export
    op_node.py             # L0: SpecOpNode projection
    step_graph.py          # L1: StepGraph (per-phase DAG template)
    schedule_graph.py      # L2: ScheduleGraph (orchestration: instances, data passes)
    workload_graph.py      # L3: WorkloadGraph (iteration semantics + data flow)
    builder.py             # Top-level orchestrator for IR projection

  llama3/                  # Llama3-specific simulation configs
    config_registry.py     # llama3_sim_debugmodel, llama3_sim_1024gpu, ...

  deepseek_v4/             # DeepSeek V4-specific simulation configs
    config_registry.py     # deepseek_v4_sim_smoketest, deepseek_v4_pro_sim_smoketest

  tests/                   # Unit tests
    test_simulator.py
    test_ir.py
```

---

## 3. Entry Points

### 3.1 SimulationTrainer (via `run_train.sh`)

The primary entry point for end-to-end simulation. `SimulationTrainer` subclasses the upstream `Trainer`:

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

**Key behaviors in `__init__`:**
- Forces `config.comm.mode = "fake_backend"` (no real NCCL/gloo rendezvous needed for single-process)
- Replaces `parallelize_fn` with CPU stubs (`_cpu_noop_parallelize` or `_cpu_gloo_parallelize_*`)
- Replaces `pipelining_fn` with CPU stubs (`_cpu_noop_pipeline` or `_cpu_semantic_pipeline`)
- Sets up fake world size from parallelism config via `_set_fake_world_size()`
- Applies device patching (meta or CPU) based on `comm_backend`
- After `super().__init__()`, optionally wraps model with FSDP1 for gloo comm capture

**`SimulationConfig`** (dataclass in `trainer.py`):

| Field | Default | Description |
|-------|---------|-------------|
| `output_dir` | `"./simulator_output"` | Export directory |
| `output_formats` | `["json","dot","chrome_trace","html","text","csv"]` | Export formats |
| `mode` | `"all"` | `"all"`, `"runtime"`, or `"schedule"` |
| `capture_joint_fx` | `False` | Joint fwd+bwd FX capture |
| `semantic_schedule` | `False` | Generate full PP/TP/DP schedule from config |
| `cost_model` | `False` | Run cost model over compute graph |
| `cost_model_class` | `""` | Custom CostModel class path (empty = MockCostModel) |
| `comm_backend` | `""` | `""` (fake) or `"gloo"` (real CPU comm) |
| `device_mode` | `""` | `""` (auto), `"meta"`, or `"cpu"` |

### 3.2 Programmatic API

The capture layer can be used directly for programmatic tracing:

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

The simulator uses a single unified capture mode via `unified_trace()`. This context manager combines `FakeTensorMode` with `TorchDispatchMode` in a single pass, and optionally activates communication interception and FSDP lifecycle hooks:

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
  +-- [optional] FSDPEventRecorder -- when capture_fsdp=True (gloo mode)
        |
        +-- Attaches module hooks for FSDP allgather/reshard/reduce-scatter lifecycle
```

**TraceRecorder** is the single recorder that accumulates:
- `nodes: list[OpNode]` -- one per dispatched op
- `edges: list[(src, dst, type)]` -- data-flow dependencies
- `comm_events`, `fsdp_events`, `pp_events` -- specialized event lists
- Phase/PP-stage/microbatch context via mutable fields

**`build_result()`** assembles everything into a `SimulationResult` with a populated `ComputeGraph`.

**CommRecorder** (inlined into `unified_trace.py`) intercepts `torch.distributed` collectives and P2P operations, recording tensor metadata, group sizes, and source node references from the active `TraceRecorder`.

**FSDPEventRecorder** (inlined into `unified_trace.py`) attaches PyTorch module hooks to FSDP-wrapped modules to capture the parameter lifecycle (allgather/reshard/reduce-scatter).

---

## 5. Core Data Model (`nodes.py`)

```
SimulationResult
  |
  +-- compute_graph: ComputeGraph
  |     +-- nodes: dict[str, OpNode]     # ordered by insertion
  |     +-- edges: list[DataEdge]         # data-flow / control / phase_boundary
  |     +-- metadata: dict
  |
  +-- schedule: TrainingSchedule | None
  |     +-- events: list[ScheduleEvent]   # coarse-grained schedule events
  |     +-- deps: list[ScheduleDep]       # dependencies between events
  |
  +-- comm_events: list[dict]             # raw communication events
  +-- fsdp_events: list[dict]             # raw FSDP lifecycle events
  +-- pp_events: list[dict]               # raw PP events
  +-- memory_events: list[MemoryEvent]    # memory allocation/residency estimates
  +-- metadata: dict
```

**OpNode** -- a single operation in the compute graph:
- `op_type`: `"compute"` | `"comm_collective"` | `"comm_p2p"` | `"data_move"` | `"memory"`
- `phase`: `"forward"` | `"backward"` | `"optimizer"`
- `pp_stage`, `pp_rank`, `microbatch_idx`: parallel context
- `comm_op`, `comm_group_size`: for communication ops
- `perf_result: PerfResult | None`: cost model output (compute_time, comm_time, FLOPs, bytes)
- `des_start_time_us`, `des_finish_time_us`: DES engine timestamps

**ComputeGraph** provides:
- `fix_comm_phase_labels()`: corrects mislabeled comm node phases
- `add_phase_boundary_edges()`: inserts sentinel nodes to enforce forward->backward->optimizer ordering
- `summary()`: op-type counts

---

## 6. Execution Flow (`trainer_runner.py`)

`run_trainer_simulation()` is the main orchestrator:

```
1. Setup patches
   - Meta device for trainer.device
   - Mock _local_scalar_dense, FakeTensor.__format__
   - No-op clip_grad_norm_, dist_sum, dist_max
   - No-op optimizer.step(), lr_schedulers.step()
   - No-op parallel_dims mesh access

2. Capture
   - Pre-fetch batches from dataloader (outside FakeTensorMode)
   - unified_trace() context with TraceRecorder
   - _patch_backward_phase() to set recorder phase to "backward"
   - Trainer.train_step() executes one full step

3. Post-processing
   - result = recorder.build_result()
   - [fake_backend only] Inject synthetic comm events:
     - FSDP2 all_gather / reduce_scatter per layer
     - TP all_reduce per layer
     - PP send/recv per microbatch
   - [semantic_schedule] Inject full PP/TP/DP schedule from config
   - [cost_model] Apply CostModel (MockCostModel or custom)
   - Memory estimation (graph + comm + model state)
   - Extension hooks (postprocess_extension_result)

4. Export
   - _export_result() -> JSON, DOT, Chrome Trace, HTML, Text, CSV
   - _export_workload_graph() -> workload_graph.json (L0-L3 IR)
```

---

## 7. Device Environment Layer

Two patching modes enable GPU-free execution:

### 7.1 Meta Mode (`meta_env.py`)

- Used for `fake_backend` (no real communication)
- `patch_device_type_to_meta()` -- redirects `device_type` to `"meta"`, creates 0-byte tensors
- Enables simulating arbitrarily large models (e.g., 1T+ parameters) with minimal RAM
- Patches `torchtitan.tools.utils.device_type`, `device_module`, and downstream modules
- Patches `torch.cuda.*` entrypoints with meta stubs

### 7.2 CPU Mode (`cpu_env.py`)

- Used for `gloo` backend (real CPU communication capture)
- `patch_device_type_to_cpu()` -- redirects `device_type` to `"cpu"`
- Creates real CPU tensors (required for gloo tensor exchange)
- `init_cpu_distributed()` -- sets up gloo process group for single-process sim
- Patches `torch.cuda.*` entrypoints with CPU stubs

### Device Mode Selection Logic

```
comm_backend == "gloo"  -->  device_mode = "cpu"
comm_backend == ""      -->  device_mode = "meta"
(overridable via SimulationConfig.device_mode)
```

---

## 8. Capture Layer (`capture/`)

The capture layer is a single module (`unified_trace.py`) that consolidates all trace capture functionality:

### 8.1 TraceRecorder

The central recorder that accumulates:
- `nodes: list[OpNode]` -- dispatched ops with metadata
- `edges: list[(src, dst, type)]` -- data-flow dependencies via tensor producer-consumer tracking
- `comm_events`, `fsdp_events`, `pp_events` -- specialized event lists
- `build_result()` -- assembles everything into a `SimulationResult`

### 8.2 UnifiedTraceMode

`TorchDispatchMode` subclass that intercepts every dispatched op:
- Records input/output tensor metadata
- Tracks tensor producer-consumer via `id(tensor) -> node_id`
- Classifies ops via `op_classification.classify_op()`
- Filters trivial ops (detach, alias, view, etc.)

### 8.3 CommRecorder

Intercepts `torch.distributed` functions:
- `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `send`, `recv`, `broadcast`, `barrier`
- Also patches `torch.distributed._functional_collectives` (used by FSDP2 and DTensor)
- Records tensor metadata, group size, PP stage, microbatch
- Resolves source node references from the active `TraceRecorder` via `get_current_recorder()`

### 8.4 FSDPEventRecorder

Attaches PyTorch module hooks to FSDP-wrapped modules:
- `forward_pre_hook` -> allgather event
- `forward_hook` -> reshard event
- `backward_pre_hook` -> allgather event
- `backward_hook` -> reduce-scatter event

### 8.5 unified_trace() Context Manager

Orchestrates all capture components:

```python
with unified_trace(recorder, use_fake_mode=True, capture_comm=False):
    output = model(*inputs)
    recorder.current_phase = "backward"
    output.sum().backward()
```

- Activates `FakeTensorMode` (if `use_fake_mode=True`) for shape-only computation
- Activates `UnifiedTraceMode` for op-level capture
- Optionally activates `CommRecorder` (for gloo backend mode)
- Optionally activates `FSDPEventRecorder` (for gloo backend mode)
- Phase tracking via `recorder.current_phase` (mutable, set by `_patch_backward_phase`)

---

## 9. Schedule Layer (`schedule/`)

### 9.1 Schedule Extraction (`schedule_extract.py`)

**Core strategy**: Construct a real PyTorch `_PipelineSchedule` with `MockPipelineStage` instances and read its `pipeline_order_with_comms` action table.

This guarantees the extracted schedule matches upstream PyTorch behavior exactly, and automatically picks up schedule algorithm changes without simulator code modifications.

`MockPipelineStage` -- duck-typed mock that satisfies `_PipelineSchedule.__init__` attribute reads without calling `dist.get_rank()` / `dist.get_world_size()`.

Supported schedules: all registered in `torch.distributed.pipelining.schedules` -- 1F1B, GPipe, Interleaved1F1B, LoopedBFS, InterleavedZeroBubble, ZBVZeroBubble, DualPipeV, and CSV-driven runtime schedules.

Action type mapping (`_ComputationType` -> `event_type`):

| Action | Event Type |
|--------|-----------|
| `F` | `pp_forward` |
| `B` | `pp_backward` |
| `UNSHARD` | `fsdp2_all_gather` |
| `RESHARD` | `fsdp2_reduce_scatter` |
| `SEND_F` | `pp_send_activation` |
| `RECV_F` | `pp_recv_activation` |
| `SEND_B` | `pp_send_gradient` |
| `RECV_B` | `pp_recv_gradient` |
| `REDUCE_GRAD` | `dp_gradient_sync` |

### 9.2 Schedule Generator (`schedule_generator.py`)

Generates semantic Interleaved1F1B schedules from parallelism config without any real schedule object. Used when `semantic_schedule=True`.

Produces the full multi-rank topology:
- PP send/recv pairs across ranks
- FSDP2 all-gather/reduce-scatter per DP group
- TP all-reduce per TP group
- DP gradient sync

### 9.3 PP Schedule Extractor (`pp_schedule_extractor.py`)

`PPScheduleExtractor` class for extracting from an existing `_PipelineSchedule` instance. Primary path delegates to `schedule_extract._convert_pipeline_order_to_training_schedule()`. Fallback: heuristic 1F1B reconstruction.

---

## 10. Cost Model (`cost_model.py`)

### Architecture

```
CostModel (ABC)
  |
  +-- estimate_node(OpNode) -> PerfResult
  +-- estimate_graph(ComputeGraph) -- annotates every node in-place
  +-- predict_step_time_us(ComputeGraph) -- critical-path analysis
  |
  +-- MockCostModel (concrete)
        - compute_time = flops / (tflops * 1e6)
        - comm_time = latency + bytes / (comm_gb_per_s * 1e3)
        - memory-bound ceiling when arithmetic intensity < threshold
        - Gaussian noise for realism
```

### FLOPs Estimation

Heuristic rules per op category:
- **Matmul-like** (`mm`, `matmul`, `bmm`, `addmm`): `2 * batch * M * K * N`
- **Attention** (`scaled_dot_product`, `flash_attention`): `2 * numel(Q) * K_seq + 2 * numel(Q) * V_dim`
- **Activations** (gelu, silu, sigmoid, ...): 1-5 FLOPs per output element
- **Norms** (rms_norm, layer_norm): 5 FLOPs per element
- **DeepEP/MoE** (`deepep.dispatch`, `deepep.combine`): rough approximation

### Communication Bytes Estimation

Ring algorithm scaling:
- `all_gather`: `input_bytes * (P-1)/P`
- `reduce_scatter`: `input_bytes * (P-1)/P`
- `all_reduce`: `output_bytes * 2*(P-1)/P`
- `send`/`recv`: raw tensor bytes

### `apply_cost_model()`

Convenience function that:
1. Runs `cost_model.estimate_result(result)` to annotate all nodes
2. Computes per-phase timing aggregates
3. Runs `_critical_path_time_us()` for step time prediction
4. Stores results in `result.metadata["cost_model"]`

---

## 11. DES Engine (`des_engine.py`)

### Architecture

Uses **salabim** discrete-event simulation library to model resource contention:

```
Per-rank resources:
  - compute (capacity=1): handles compute, data_move, memory ops
  - comm    (capacity=1): handles comm_collective, comm_p2p ops
```

Ops on separate engines can overlap (modeling GPU compute/comm parallelism).
Ops on the same engine are serialized (resource contention).
DAG dependencies are modeled via salabim `State` signals.

### Two Simulation Levels

1. **Single-rank DES** (`simulate_single_rank_des`):
   - Topological sort of compute graph
   - Each node becomes a `_OpComponent` with resource request
   - Returns max finish time across all nodes

2. **Multi-rank DES** (`simulate_multi_rank_des`):
   - Uses `TrainingSchedule` events and dependencies
   - Each event becomes a `_ScheduleEventComponent`
   - Duration derived from linked `OpNode.perf_result` via `link_schedule_to_graph()`
   - Separate compute/comm resources per rank
   - Cross-rank dependencies (PP send/recv) create inter-rank synchronization

### `DESEngine` Class

```python
engine = DESEngine()
step_time = engine.predict_step_time_us(result, cost_model)
engine.annotate(result)  # writes to result.metadata["des_engine"]
```

### Utilization Analysis

`compute_des_utilization()` computes:
- `e2e_step_time_us`: total simulated step time
- `compute_busy_pct` / `comm_busy_pct`: engine utilization
- `overlap_pct`: compute/comm overlap
- `contention_count`: number of ops delayed by resource contention
- `des_vs_cp_ratio`: DES time vs critical-path time (quantifies contention impact)

---

## 12. Memory Estimation (`memory_estimator.py`)

Three estimation sources:

1. **Graph memory** (`estimate_graph_memory`):
   - Activation/output memory from node outputs
   - Lifetime = producer node index to last-consumer node index
   - Categories: `activation`, `comm_buffer`, `allocation`, `data_move`

2. **Comm memory** (`estimate_comm_memory`):
   - Communication buffer memory from comm events
   - Category: `comm_event_buffer`

3. **Model state memory** (`estimate_model_state_memory`):
   - Parameter memory from `model.named_parameters()`
   - Optimizer state memory (Adam: 2x param size for momentum + variance)
   - Accounts for FSDP sharding (divides by `dp_shard` degree)

---

## 13. Layered IR (`ir/`)

A four-level intermediate representation that projects captured data into a spec-aligned format for downstream hardware simulators:

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
  |
  +-- build_step_graphs()     --> dict[str, StepGraph]       (L1)
  +-- build_schedule_graph()  --> ScheduleGraph              (L2)
  +-- build_workload_graph()  --> WorkloadGraph              (L3)
```

All projections are **read-only** -- they never mutate the captured graph.

### L0: SpecOpNode

- `flops`, `peak_mem`, `param_mem`, `comm_bytes` -- cost fields
- `predecessors`, `successors` -- adjacency (data edges only)

### L1: StepGraph

- Partitions captured graph by phase
- Computes `entry_nodes`, `exit_nodes`, `tensor_lifetimes`
- Validates acyclicity via Kahn's algorithm

### L2: ScheduleGraph

- `StepInstance` -- (step_ref, micro_batch_idx, pipeline_stage, device_ids, dp_group)
- `DataPass` -- tensor flow between step instances (with `comm_primitive`)
- Parallelism degrees: pp, tp, dp, ep, cp

### L3: WorkloadGraph

- `IterationSpec` -- wraps ScheduleGraph with microbatch count
- `DataFlow` -- dataloader input/output descriptions
- `cross_iter_passes` -- tensor passes spanning iterations (e.g., optimizer -> next forward)

---

## 14. Export System (`export.py`)

| Format | File | Description |
|--------|------|-------------|
| JSON | `simulation_result.json` | Full structured dump, compact for >10K nodes |
| DOT | `compute_graph.dot` | Graphviz with color-coded nodes by op type |
| Chrome Trace | `trace.json` | `chrome://tracing` compatible timeline |
| HTML | `trace.html` | Self-contained interactive visualization |
| Text | `summary.txt` | Human-readable statistics |
| CSV | `kernel_summary.csv` | Per-operator kernel trace (nsys/msprof compatible) |
| Workload Graph | `workload_graph.json` | L0-L3 IR projection |

### HTML Visualization

The HTML trace is self-contained (no CDN dependency) and includes:
- Swimlane schedules (per-phase, per-strategy: PP/FSDP/TP/DP/Optimizer)
- Operator DAGs with expandable details
- Phase boundary markers
- DES timing overlays when cost model is active

---

## 15. Communication Capture Strategies

### 15.1 fake_backend Mode (default)

No real distributed communication. Synthetic comm events are **injected** post-capture:

```
_inject_synthetic_comm_events()
  |
  +-- FSDP2 all_gather / reduce_scatter (per layer, per PP stage)
  |     shape: shard_numel / num_layers per shard
  |     group_size: dp_shard degree
  |
  +-- TP all_reduce (2x per layer, forward + backward)
  |     shape: batch * seq_len * hidden
  |     group_size: tp degree
  |
  +-- PP send/recv (per microbatch, per stage boundary)
        shape: batch * seq_len * hidden
        group_size: 2 (adjacent stages)
```

### 15.2 gloo Mode

Real CPU communication capture:
- FSDP1 wrapping applied post-init via `_apply_fsdp1_on_cpu()`
- `CommRecorder` intercepts real all-gather/reduce-scatter calls
- Requires `init_cpu_distributed()` for process group setup
- Single-process sufficient (uses `FakeProcessGroup` for `init_distributed`)

---

## 16. Pipeline Parallelism on CPU

### Semantic Pipeline (`_cpu_semantic_pipeline`)

For PP > 1 in fake_backend mode:
1. Splits model into PP stages using upstream `_generate_llm_fqn_per_model_part` + `_split_module`
2. Moves parts to meta device immediately (avoids OOM for 1T+ models)
3. Returns `MockSchedule` (no-op step) and model parts list
4. The real PP schedule is extracted later via `_inject_semantic_schedule()`

### Gloo Pipeline

For PP > 1 in gloo mode:
- `_cpu_pp_module_split` splits the model into stages on CPU
- Uses upstream schedule objects with real gloo communication

---

## 17. Model Configuration Registries

Each model has its own `config_registry.py` that returns `SimulationTrainer.Config`:

### Llama3

| Config | Topology | Description |
|--------|----------|-------------|
| `llama3_sim_debugmodel` | 1 GPU | Small model, gloo comm, cost model |
| `llama3_sim_1024gpu` | PP=4, TP=8, DP_shard=4, DP_repl=8 | 1024-GPU semantic simulation |

### DeepSeek V4

| Config | Topology | Description |
|--------|----------|-------------|
| `deepseek_v4_sim_smoketest` | PP=2, TP=2, DP=2 | 2-layer smoketest (8 ranks) |
| `deepseek_v4_pro_sim_smoketest` | PP=8, TP=8, EP=192 | 61-layer with full parallelism |

---

## 18. Extension System (`extension_hooks.py`)

Two hooks for external side-loads (e.g., torchtitan-npu):

1. `collect_extension_metadata(trainer, capture)` -- calls `trainer.collect_simulation_metadata(capture)` if implemented
2. `postprocess_extension_result(result, trainer, sim_opts)` -- calls `trainer.postprocess_simulation_result(result, sim_opts)` if implemented

These use duck typing to avoid import dependencies from core simulator code.

---

## 19. Key Design Decisions

### 19.1 Why a single entry point (`SimulationTrainer`) instead of a `Simulator` class?

- **Consistency**: All simulation runs go through the same `Trainer` initialization path, ensuring the model is constructed, parallelized, and configured identically to real training
- **Config-driven**: `SimulationConfig` is a dataclass integrated with torchtitan's config system, supporting CLI overrides and config registries
- **No API drift**: A single entry point prevents the two APIs from diverging over time

### 19.2 Why a single capture module (`unified_trace.py`)?

- **No circular dependencies**: `CommRecorder` and `FSDPEventRecorder` are inlined into the same file as `TraceRecorder`, eliminating cross-module import cycles
- **Single source of truth**: All capture logic (dispatch interception, communication recording, FSDP hooks) lives in one place
- **Simplified maintenance**: No need to synchronize interfaces across multiple capture modules

### 19.3 Why `FakeTensorMode` + `TorchDispatchMode`?

- **Zero memory**: Shape-only tensors enable simulating 1T+ parameter models
- **Single pass**: All capture components composed in one context manager
- **Dispatcher-level**: Captures all ATen ops including those hidden by autograd/FSDP

### 19.4 Why Mock PyTorch Schedule Objects?

- Reuses upstream schedule algorithm implementations exactly
- Automatically picks up new schedule types (ZeroBubble, DualPipeV, etc.)
- Avoids fragile re-implementation of complex scheduling logic

### 19.5 Why Synthetic Comm Injection?

- `fake_backend` mode avoids multi-process complexity
- Communication shapes/sizes derived from actual model parameters
- Sufficient for performance estimation without real data exchange

### 19.6 Why salabim DES?

- Models compute/comm resource contention (GPU has separate engines)
- Handles cross-rank synchronization (PP send/recv dependencies)
- Produces realistic step time predictions accounting for overlap and contention

### 19.7 Why Layered IR (L0-L3)?

- Separation of concerns: op-level -> step-level -> schedule-level -> workload-level
- Spec-aligned for downstream hardware simulators
- All projections derived from captured data, not re-implemented logic
