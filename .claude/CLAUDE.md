# TorchTitan Simulator Development Guide

This is the **simulator fork** of torchtitan — it adds a CPU-only training trace/simulation experiment (`torchtitan/experiments/simulator/`) to the upstream torchtitan LLM training platform. The NPU simulator fork lives in a separate sub-repo at `torchtitan-npu-simulator/` (its own git root — don't modify it when working on this repo).

## Build & Test

```bash
pip install -r requirements.txt -r requirements-dev.txt
pre-commit run --all-files
pytest tests/unit_tests -x
```

Pre-commit enforces: ufmt (black 22.12 + usort 1.0), flake8 (with torchfix, bugbear, pep585, new-union-types), pydoclint, codespell, license headers (BSD-style from `assets/license_header.txt`), and **pyrefly** type checking (not mypy/pyright). CI also runs lychee link checking.

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

Debug modes (no GPU needed):
```bash
NGPU=32 COMM_MODE="fake_backend" ./run_train.sh   # config validation, no execution
NGPU=32 COMM_MODE="local_tensor" ./run_train.sh    # single-GPU debug with simulated multi-GPU
```

Gloo-based comm capture (real CPU collectives with FSDP all-gather/reduce-scatter):
```bash
--simulation.comm_backend=gloo  # requires torchrun or mp.spawn for multi-process
```

## Simulator Architecture

The simulator is side-loaded as an experiment — `train.py` remains unchanged. Two entry paths:

1. **SimulationTrainer** (via `run_train.sh`): subclasses `Trainer`, patches device helpers to CPU, overrides `train()` for one instrumented step. Config registry returns `SimulationTrainer.Config` which the normal config system builds.
2. **Simulator class** (programmatic API): `from torchtitan.experiments.simulator import Simulator` — three modes: `simulate_fx` (static FX trace), `simulate_runtime` (dynamic 1-step capture), `simulate_pp_schedule` (schedule extraction only).

Simulation modes controlled by `SimulationConfig`:
- `comm_backend=""` (default): fake_backend, no real communication
- `comm_backend="gloo"`: real CPU communication capture via gloo backend
- `capture_joint_fx`: joint fwd+bwd FX capture
- `semantic_schedule`: generate full PP/TP/DP schedule from parallelism config without multi-rank execution
- `cost_model`: annotate ops with performance estimates (default: `MockCostModel`)

Per-model config registries live in subdirectories: `simulator/llama3/config_registry.py`, `simulator/deepseek_v4/config_registry.py`.

Outputs: `simulation_result.json`, `compute_graph.dot`, `trace.json`, `trace.html`, `summary.txt`. The HTML trace is self-contained (no CDN dependency).

## Core Principles

1. **PyTorch-native.** Core training/parallelism code must not depend on non-PyTorch libraries. Complex techniques belong upstream.
2. **Root cause first.** Don't land band-aid fixes. Understand *why* before proposing.
3. **Reuse over duplication.** Unify across models. Use upstream (torchao, PyTorch) when available.
4. **Don't leak experiments into core.** No `if experiment_x:` in core files. Experiment code stays in `torchtitan/experiments/`.
5. **Protect converged paths.** Flag silent breakage of checkpoints or user code.
6. **Audit all callsites.** Shared code changes must update every model variant: llama3, llama4, qwen3, qwen3_vl, deepseek_v3, deepseek_v4, gpt_oss, flux.

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
- `Configurable` base class with `@dataclass(kw_only=True, slots=True)` `Config` — enforced by `__init_subclass__`, not optional
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
- License header is enforced by pre-commit (`insert-license` hook using `assets/license_header.txt`)

## Experiments Rules

- Must still pass `pre-commit run --all-files`
- Must not modify core torchtitan code
- Must use torchtitan's config system
- Keep distinct features in separate folders

## Reference

- Domain-specific rules: `.claude/rules/` (config, distributed, experiments, models)
- NPU sub-repo: `torchtitan-npu-simulator/` — own git root, own `AGENTS.md`, don't cross-modify
