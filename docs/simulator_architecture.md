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

## Validation

Run:

```bash
PYTHONPATH=. ~/.local/bin/python3.11 -m pytest \
  torchtitan/experiments/simulator/tests/test_simulator.py -q
```

The current simulator unit suite validates data models, op/comm/FSDP/PP capture,
exporters, HTML generation, graph assembly, and extension hooks.

## Limitations

- Runtime capture observes the current process. Multi-rank traces require
  multi-process execution or post-run trace aggregation.
- CPU simulation does not reproduce GPU/NPU kernel performance or real device
  memory pressure. Memory values are trace estimates, not allocator truth.
- Some operator-level aliasing and in-place behavior is approximated by tensor
  producer tracking.
- Parallel schedules can be semantic when the actual backend cannot run in the
  current environment.
