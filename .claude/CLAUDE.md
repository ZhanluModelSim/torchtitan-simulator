# TorchTitan Simulator Development Guide

This is the **simulator fork** of torchtitan — it adds a CPU-only training trace/simulation experiment (`torchtitan/experiments/simulator/`) to the upstream torchtitan LLM training platform. The NPU simulator fork lives in a separate sub-repo at `torchtitan-npu-simulator/` (its own git root).

## Build & Test

```bash
pip install -r requirements.txt -r requirements-dev.txt
pre-commit run --all-files
pytest tests/unit_tests -x
```

Pre-commit enforces: ufmt (black 22.12 + usort 1.0), flake8 (with torchfix, bugbear, pep585, new-union-types), pydoclint, codespell, license headers, and **pyrefly** type checking (not mypy/pyright). CI also runs lychee link checking.

Run a single test file:
```bash
pytest tests/unit_tests/test_config_manager.py -s
```

Run simulator unit tests:
```bash
pytest torchtitan/experiments/simulator/tests/test_simulator.py -v
```

Integration tests require GPUs and use a custom runner:
```bash
python -m tests.integration_tests.run_tests <output_dir> [--module MODULE] [--config CONFIG] [--test_suite features]
```

## Validating Numerics

Non-computation changes must produce **identical loss** before vs. after with `--debug.seed=42` and `--debug.deterministic`. Use `scripts/loss_compare.py` and TensorBoard for full-precision comparison — stdout only shows 5 significant digits.

**NEVER** use `--debug.deterministic_warn_only`.

## Running Training & Simulation

Normal training (8 GPU, Llama 3 debug):
```bash
MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh
```

CPU simulation (side-loaded experiment):
```bash
MODULE=simulator.llama3 CONFIG=llama3_sim_debugmodel ./run_train.sh
```

NPU simulation (separate sub-repo):
```bash
cd torchtitan-npu-simulator
TRAIN_FILE=torchtitan_npu.entry MODULE=torchtitan_npu.simulator.llama3 CONFIG=llama3_npu_sim_debugmodel ./scripts/run_train.sh --training.steps=1
```

Debug modes (no GPU needed):
```bash
NGPU=32 COMM_MODE="fake_backend" ./run_train.sh   # config validation, no execution
NGPU=32 COMM_MODE="local_tensor" ./run_train.sh    # single-GPU debug with simulated multi-GPU
```

## Simulator Architecture

The simulator is side-loaded as an experiment — `train.py` remains unchanged. `SimulationTrainer` subclasses `Trainer`, patches device helpers to CPU, and overrides `train()` for one instrumented step. Entry: config_registry returns `SimulationTrainer.Config` which the normal config system builds.

Key components in `torchtitan/experiments/simulator/`:

| File | Role |
| --- | --- |
| `cpu_env.py` | Forces device helpers to CPU |
| `dispatch_interceptor.py` | Captures runtime ops with tensor metadata |
| `comm_interceptor.py` | Monkey-patches `torch.distributed` collectives |
| `runtime_capture.py` | Coordinates op/comm/FSDP/PP capture |
| `graph_assembler.py` | Builds ComputeGraph nodes/edges |
| `memory_estimator.py` | Estimates activation, comm buffer, model state memory |
| `pp_schedule_extractor.py` | Extracts pipeline schedule events |
| `fx_capture.py` | Optional FX graph capture |
| `export.py` | Writes JSON, DOT, Chrome Trace, HTML, text |
| `trainer_runner.py` | Runs one simulated step and exports |
| `extension_hooks.py` | Duck-typed hooks for external extensions |

Outputs: `simulation_result.json`, `compute_graph.dot`, `trace.json`, `trace.html`, `summary.txt`. The HTML trace is self-contained (no CDN dependency).

## Core Principles

1. **PyTorch-native.** Core training/parallelism code must not depend on non-PyTorch libraries. Complex techniques belong upstream.
2. **Root cause first.** Don't land band-aid fixes. Understand *why* before proposing.
3. **Reuse over duplication.** Unify across models. Use upstream (torchao, PyTorch) when available.
4. **Don't leak experiments into core.** No `if experiment_x:` in core files. Experiment code stays in `torchtitan/experiments/`.
5. **Protect converged paths.** Flag silent breakage of checkpoints or user code.
6. **Audit all callsites.** Shared code changes must update every model variant: llama3, llama4, qwen3, deepseek_v3, gpt_oss, flux.

## Code Style

### `axis` vs `dim` (critical convention)
- **`axis`/`axes`** = specific `DeviceMesh` axis (TP axis, `dp_shard` axis, axes a spec references)
- **`dim`/`dimensional`** = mesh *shape* ("1D mesh", "multi-dimensional SPMD mesh") and tensor dimensions
- Exception: match upstream API spelling at callsite (`DeviceMesh.mesh_dim_names`, `DataParallelMeshDims`), then assign into locally named `mesh_axis_names`

### Other naming
- `num_` prefix for counts (`num_expert_groups` not `n_expert_groups`) unless matching upstream API
- Match torchao/PyTorch naming (e.g. `Float8Linear` not `Float8Config`)
- No "toy/test/temp" in production names — context goes in docstrings

### Code placement
- Model-agnostic parallelism helpers → `torchtitan/distributed/`
- Shared model components (attention, MoE, embeddings) → `torchtitan/models/common/`
- Model-specific code → that model's folder
- Experiment code → `torchtitan/experiments/<name>/`

### Error handling
- `ValueError` for user-facing errors; `assert` only for internal invariants
- Validate mesh axes, placements, config values explicitly — never assume 1D mesh
- Silently skipped config → emit a warning

### Comments
- Only for non-obvious things: dimension semantics, gradient placements, workaround reasons
- Descriptions in docstrings, not names

### Config system
- `Configurable` base class with `Config` dataclass pattern
- No `None` defaults for required fields
- `dataclasses.replace()` is shallow — be explicit about deep copies
- Experiments must use torchtitan's existing config system, not custom arg parsing

### Model folder pattern
Each model: `config_registry.py` → `parallelize.py` → architecture files

## Pre-commit / CI Gotchas

- CI installs PyTorch nightly + torchao nightly + torchdata nightly (CPU wheels)
- CI unit tests path is `tests/unit_tests` (not `tests/`)
- CI lint runs on changed files only; `pre-commit run --all-files` is the local equivalent
- Lychee link checker is optional locally (gracefully skipped if not installed)
- Pre-commit blocks commits to `main` branch (`no-commit-to-branch` hook)
- Max file size: 500KB (`check-added-large-files`)

## Experiments Rules

- Must still pass `pre-commit run --all-files`
- Must not modify core torchtitan code
- Must use torchtitan's config system
- Keep distinct features in separate folders