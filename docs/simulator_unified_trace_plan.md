# Simulator: Unified Dispatch Trace + Meta Device Patching

## Problem Statement

Two systemic issues in the simulator architecture:

### Issue 1: Three-Level Compute Graph Tracing Redundancy

Currently, `compute_graph` is populated through **three separate mechanisms** that overlap significantly:

| Mechanism | Entry | What it captures | OpNode ID prefix | Phase granularity |
|-----------|-------|-------------------|-------------------|-------------------|
| FX tracing | `capture_forward_fx` / `capture_joint_fx` | ATen ops via `make_fx` + `FakeTensorMode` | `fx_` | `"forward"` or `"joint"` (no backward separation) |
| Runtime capture (dispatch) | `OpCaptureMode` + `CommRecorder` + `FSDPEventRecorder` | Dispatched ops + monkey-patched comm + FSDP hooks | `op_` / `comm_` | `"forward"` / `"backward"` / `"optimizer"` |
| Synthetic injection | `_inject_synthetic_comm_events` | Heuristic FSDP/TP comm nodes based on model param numel | `comm_syn_` | `"forward"` / `"backward"` |

**Redundancies:**
1. Op classification logic duplicated: `_classify_fx_node()` (fx_capture.py) vs `_categorize_op()` (dispatch_interceptor.py) — nearly identical marker lists with minor differences (`"broadcast_"` vs `"broadcast"`, `"aten.rand"` missing from FX).
2. Comm ops captured in 3 overlapping ways: (a) as dispatched c10d_functional ATen ops via `OpCaptureMode`, (b) as monkey-patched `dist.*` calls via `CommRecorder`, (c) as synthetic heuristic nodes.
3. In `trainer_runner.py`, FX forward/joint graphs are captured **after** runtime capture but stored in `result.metadata["fx_forward_graph"]` — a second complete graph that's never merged with the primary compute_graph.
4. The FX path produces no backward-only ops; runtime path does. The joint FX path labels everything as `"joint"`.
5. In fake_backend mode, `OpCaptureMode` captures compute ops on CPU tensors (with real memory allocation), while `_inject_synthetic_comm_events` creates comm nodes based on heuristic model structure analysis. These two are conceptually one step that should be done together.

### Issue 2: CPU Device Patching Memory Pressure

The simulator patches devices from GPU → **CPU** to avoid GPU dependency. This works for small models but:
- Large models (Llama 3 70B: ~70B params × 2 bytes = ~140GB) exceed CPU RAM
- Even small debug models allocate real tensors, wasting memory and slowing capture
- The FX path already uses `FakeTensorMode` (shape-only, no allocation), but the runtime path allocates real CPU tensors

**Opportunity:** Patch devices to **meta** instead of CPU. Meta tensors have:
- `.shape`, `.dtype`, `.device` — all the metadata we need for `TensorMeta`
- No data allocation (0 bytes memory)
- PyTorch's `torch.device("meta")` context manager for model construction
- TorchTitan core already uses `with torch.device("meta"):` for model init

**Constraint:** `compute_graph` and communication operator information must not change significantly. The `OpNode` fields (op_name, op_type, phase, inputs, outputs, comm_op, etc.) must remain identical.

---

## Solution Design

### Part A: Unified Dispatch-Based Trace Model

**Core idea:** Replace the three-level mechanism with a **single dispatch-based trace** that captures everything needed in one pass, using `TorchDispatchMode` + `FakeTensorMode` together.

#### A1: Unify Op Classification

Create a single `op_classification.py` module with shared marker lists, `_COMM_MARKERS`, `_P2P_MARKERS`, `_DATA_MOVE_MARKERS`, `_MEMORY_MARKERS`, `_TRIVIAL_TARGETS`, and `_COMM_OP_MAP`. Both FX and dispatch paths call the same `_classify_op(target: str) -> (op_type, comm_op)` function.

```python
# torchtitan/experiments/simulator/op_classification.py
_COMM_MARKERS = ("_c10d_functional", "c10d_functional", "all_reduce", "all_gather", "reduce_scatter", "all_to_all", "broadcast", "wait_tensor", "barrier")
_P2P_MARKERS = ("_send", "_recv", ".send", ".recv")
_DATA_MOVE_MARKERS = ("_to_copy", "copy_", ".to.")
_MEMORY_MARKERS = ("aten.empty", "aten.zeros", "aten.ones", "aten.full", "aten.arange", "aten.rand")
_TRIVIAL_TARGETS = frozenset(["aten.detach.default", "aten.detach_.default", "aten.alias.default", "aten.t.default", "aten.as_strided.default", "aten._unsafe_view.default", "aten.view.default", "aten.lift_fresh_copy.default", "aten.lift.default"])
_COMM_OP_MAP = [("reduce_scatter", "reduce_scatter"), ("all_gather", "all_gather"), ("all_reduce", "all_reduce"), ("all_to_all", "all_to_all"), ("broadcast", "broadcast"), ("wait_tensor", "wait"), ("barrier", "barrier"), ("_send", "send"), ("_recv", "recv")]

def classify_op(target: str) -> tuple[str, str | None]:
    """Return (op_type, comm_op_or_None) for any op target string."""
    ...
```

Both `fx_capture.py` and `dispatch_interceptor.py` import from this module.

#### A2: Unified TraceCaptureMode

Create a new `UnifiedTraceMode` that combines `FakeTensorMode` + `TorchDispatchMode` into one context manager. This mode:
- Uses `FakeTensorMode` internally so all tensors are shape-only (no memory allocation)
- Intercepts every dispatched op via `TorchDispatchMode.__torch_dispatch__`
- Records each op as an `OpNode` with phase, pp_stage, microbatch context
- Tracks data-flow edges via tensor identity (same mechanism as current `OpRecorder._tensor_producer`)
- Detects backward phase via autograd hooks (same mechanism as current `torch.Tensor.backward` monkey-patch)

```python
# torchtitan/experiments/simulator/unified_trace.py
class UnifiedTraceMode(TorchDispatchMode):
    def __init__(self, recorder: OpRecorder, fake_mode: FakeTensorMode | None = None):
        self.recorder = recorder
        self.fake_mode = fake_mode or FakeTensorMode(allow_non_fake_inputs=True)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 1. Execute under fake_mode (shape-only, no memory)
        # 2. Classify op via unified classify_op()
        # 3. Record as OpNode with TensorMeta from FakeTensor metadata
        # 4. Track data-flow edges
        ...
```

**Key benefit:** This single mode produces the **same** compute_graph as the current runtime path, but without allocating real tensors. It replaces both `OpCaptureMode` + `FakeTensorMode` as separate contexts.

#### A3: Merge Synthetic Comm Injection into Dispatch Capture

In fake_backend mode, instead of post-hoc `_inject_synthetic_comm_events()` that creates heuristic comm nodes, the `UnifiedTraceMode` captures FSDP/TP/DP comm ops as they **would** be dispatched by a properly-parallelized model. Two strategies:

**Strategy 1 (preferred):** Run the model under `FakeTensorMode` with the **real parallelize function** (FSDP2, TP). Since FSDP2/TP work correctly on meta/FakeTensors (TorchTitan core does this), the `all_gather`, `reduce_scatter`, `all_reduce` ops appear naturally in the dispatch trace. No synthetic injection needed.

**Strategy 2 (fallback):** If Strategy 1 fails for some model configurations, keep `_inject_synthetic_comm_events()` as a fallback, but invoke it **inside** the capture context rather than as a post-processing step. This ensures the synthetic nodes get proper edge connections via the existing `_tensor_producer` tracking.

#### A4: Remove FX Capture as Separate Path

The FX path (`capture_forward_fx`, `capture_joint_fx`) becomes a **legacy/optional** mode. The unified dispatch path produces equivalent (or better) data because:
- It captures backward ops with correct phase labels (FX joint path labels everything as `"joint"`)
- It captures FSDP lifecycle events (FX path doesn't)
- It captures PP stage/microbatch context (FX path doesn't)
- It captures DTensor placement info (FX path doesn't)

The FX path is kept only for:
1. Static graph export (producing an `fx.GraphModule` for downstream compilation tools)
2. Verifying that the dispatch trace matches the static FX graph (cross-validation)

These use cases store the FX graph in `result.metadata`, not in the primary `compute_graph`.

#### A5: File Changes Summary

| File | Change |
|------|--------|
| `op_classification.py` | **New**: shared `_COMM_MARKERS`, `classify_op()`, `_TRIVIAL_TARGETS`, `_COMM_OP_MAP` |
| `unified_trace.py` | **New**: `UnifiedTraceMode`, `TraceRecorder` (combines OpRecorder + phase tracking) |
| `dispatch_interceptor.py` | **Modified**: import `classify_op` from `op_classification`, remove local `_categorize_op` and marker lists |
| `fx_capture.py` | **Modified**: import `classify_op` from `op_classification`, remove local `_classify_fx_node` and marker lists |
| `runtime_capture.py` | **Modified**: use `UnifiedTraceMode` instead of separate `OpCaptureMode` + `CommRecorder`; for fake_backend, use Strategy 1 (parallelize under FakeTensor) or Strategy 2 (synthetic injection within capture context) |
| `trainer_runner.py` | **Modified**: remove `_inject_synthetic_comm_events()` call (Strategy 1 handles it); keep FX capture in metadata only; remove FX→primary-compute_graph merge logic |
| `graph_assembler.py` | **Modified**: `merge_comm_events` becomes optional (only needed for gloo path); unify `from_fx` and `from_runtime` to both use the same `classify_op` |
| `nodes.py` | **No change**: OpNode, TensorMeta, ComputeGraph stay identical |

---

### Part B: Meta Device Patching

**Core idea:** Replace CPU device patching with meta device patching so the simulator allocates **zero bytes** for model parameters and activations.

#### B1: The Challenge with Meta Tensors

Meta tensors (`torch.device("meta")`) have shape/dtype/device metadata but **no data storage**. This means:
- `tensor.shape` works → `TensorMeta.shape` works
- `tensor.dtype` works → `TensorMeta.dtype` works
- `tensor.device` works → `TensorMeta.device` = `"meta"` (needs mapping to `"cpu"` in output)
- `tensor.numel()` works → cost_model FLOPs computation works
- `tensor.requires_grad` works → phase tracking works

**But:**
- **Operations on meta tensors crash**: `aten.mm.default(meta, meta)` raises RuntimeError because there's no data to compute
- **FakeTensorMode solves this**: FakeTensors carry shape/dtype metadata and **simulate** operation outputs without computing. This is exactly what the FX path already does.
- **The `UnifiedTraceMode` from Part A uses FakeTensorMode internally**: So Part B is naturally solved by Part A's unified dispatch mode.

#### B2: Patching Strategy

Replace `patch_device_type_to_cpu()` with `patch_device_type_to_meta()`:

```python
# torchtitan/experiments/simulator/cpu_env.py (or new meta_env.py)

def patch_device_type_to_meta() -> None:
    """Monkey-patch torchtitan device helpers to 'meta'."""
    # Same pattern as patch_device_type_to_cpu(), but:
    # - tt_utils.device_type = "meta"
    # - tt_utils.device_module = _make_meta_device_module()
    # - _PATCHED_MODULES rebind to "meta"
    # - torch.cuda patches remain (meta models don't need CUDA)

def _make_meta_device_module():
    """Namespace that quacks like torch.cuda but reports meta device."""
    return types.SimpleNamespace(
        set_device=lambda device: None,
        current_device=lambda: 0,
        device_count=lambda: 0,  # No real devices
        ...
    )
```

#### B3: Model Construction on Meta

Replace `with torch.device("cpu"):` with `with torch.device("meta"):`:

```python
# In run_simulate.py / trainer.py
with torch.device("meta"):
    model = model_cls.from_model_args(model_config)
```

This creates all parameters as meta tensors (shape-only, 0 bytes). Then:
1. **Parallelize on meta**: TorchTitan's parallelize functions (FSDP2, TP) are designed to work on meta tensors. This means `fully_shard` and TP wrapping produce correct sharding metadata without allocating any data.
2. **No `to_empty()` or `init_weights()`**: We never materialize the model. We only need shape/dtype/device metadata for the compute_graph.
3. **Run under `UnifiedTraceMode`**: The `FakeTensorMode` inside `UnifiedTraceMode` converts meta tensors to FakeTensors at the first op dispatch. All subsequent ops are traced symbolically.

#### B4: Communication in Meta/FakeTensor Mode

- **Fake_backend**: FSDP2's `all_gather` / `reduce_scatter` on FakeTensors produce FakeTensor outputs with correct shapes. These ops appear naturally in the dispatch trace. The `UnifiedTraceMode` records them as `OpNode(op_type="comm_collective", comm_op="all_gather")`. No synthetic injection needed.
- **Gloo backend**: Real `torch.distributed` comm on FakeTensors is NOT possible (FakeTensors have no data to send). For gloo mode, we must **materialize** tensors to CPU for the comm ops only. Two approaches:
  1. **Selective materialization**: Detect comm ops in `UnifiedTraceMode.__torch_dispatch__`, materialize just the comm input tensors to CPU for the actual gloo operation, then convert outputs back to FakeTensor metadata. This preserves the zero-alloc approach for 99% of ops while allowing real comm capture.
  2. **Hybrid mode**: For gloo, fall back to the current CPU path (real CPU tensors for everything). Meta patching is only for fake_backend mode.

**Decision:** Use approach 2 (hybrid) for simplicity. Meta patching applies only when `comm_backend=""` (fake_backend). When `comm_backend="gloo"`, keep CPU patching as-is.

#### B5: TensorMeta.device Normalization

Meta tensors have `device="meta"`. In the output `TensorMeta`, we need `device="cpu"` for backward compatibility with cost_model and export formats. Add a normalization step in `UnifiedTraceMode`:

```python
def _normalize_device(device_str: str) -> str:
    """Map 'meta' → 'cpu' for output TensorMeta compatibility."""
    if device_str == "meta":
        return "cpu"
    return device_str
```

This is applied in `_collect_tensor_metas()` when recording OpNode inputs/outputs. The compute_graph and all downstream tools see `"cpu"` devices, preserving backward compatibility.

#### B6: Phase Transition Detection on Meta/FakeTensors

The current backward-phase detection uses `torch.Tensor.backward` monkey-patch. On FakeTensors:
- `backward()` still works (it dispatches through `__torch_dispatch__`)
- The `UnifiedTraceMode` can intercept the backward call and set phase

Alternative: Use `torch.autograd.graph.Node` hooks instead of monkey-patching `torch.Tensor.backward`. This is cleaner and works on both real and FakeTensors.

#### B7: Memory Estimation on Meta

Meta tensors have `tensor.numel()` and `tensor.dtype` but `tensor.element_size()` may not work. The memory estimator (`memory_estimator.py`) needs updating:
- Use `dtype_size(dtype_str)` (already exists) instead of `tensor.element_size()`
- Calculate memory as `numel * dtype_size` (shape-only calculation, no tensor data needed)
- This is actually **more correct** than the current approach which relies on real tensor allocation

#### B8: File Changes Summary

| File | Change |
|------|--------|
| `meta_env.py` | **New**: `patch_device_type_to_meta()`, `_make_meta_device_module()`, `_normalize_device()` |
| `cpu_env.py` | **No change**: kept for gloo backend fallback |
| `unified_trace.py` | **New**: `UnifiedTraceMode` with FakeTensorMode integration, device normalization |
| `trainer.py` | **Modified**: use `patch_device_type_to_meta()` for fake_backend, `patch_device_type_to_cpu()` for gloo |
| `run_simulate.py` | **Modified**: `with torch.device("meta"):` for fake_backend model construction |
| `memory_estimator.py` | **Modified**: use `dtype_size()` + `numel` instead of `tensor.element_size()` |
| `nodes.py` | **No change**: TensorMeta stays the same (device="cpu" in output) |
| `cost_model.py` | **No change**: `_numel()` already works on tuple shapes; `_estimate_comm_bytes()` already uses `numel * dtype_size` |

---

## Implementation Phases

### Phase 1: Unify Op Classification (Low risk, immediate value)

1. Create `op_classification.py` with shared markers and `classify_op()`
2. Update `fx_capture.py` to import from `op_classification`
3. Update `dispatch_interceptor.py` to import from `op_classification`
4. Test: existing 69 tests unchanged, new `TestOpClassification` tests

### Phase 2: Create UnifiedTraceMode (Core change)

1. Create `unified_trace.py` with `UnifiedTraceMode(TorchDispatchMode)` + `TraceRecorder`
2. `UnifiedTraceMode.__torch_dispatch__`:
   - Enter `FakeTensorMode` if not already active
   - Execute op (shape-only output)
   - Record OpNode via unified `classify_op()`
   - Track data-flow edges
   - Normalize `device="meta"` → `"cpu"` in TensorMeta
3. Phase detection: `torch.Tensor.backward` monkey-patch (same as current, works on FakeTensors)
4. Test: `TestUnifiedTrace` — trace a small model, verify compute_graph matches current runtime path output

### Phase 3: Meta Device Patching (Part B)

1. Create `meta_env.py` with `patch_device_type_to_meta()`
2. Add `device_mode` to `SimulationConfig` (values: `"cpu"`, `"meta"`)
3. Update `trainer.py`: fake_backend → `patch_device_type_to_meta()` + `torch.device("meta")`; gloo → keep CPU
4. Update `run_simulate.py`: same logic
5. Update `memory_estimator.py`: use shape-only calculation
6. Test: `TestMetaDevicePatch` — verify model on meta has 0-byte params, TensorMeta has `"cpu"` device, compute_graph matches

### Phase 4: Remove Synthetic Comm Injection (Part A3, Strategy 1)

1. In `trainer_runner.py`, for fake_backend mode:
   - Parallelize model under `FakeTensorMode` (FSDP2 on meta/FakeTensors)
   - Run `UnifiedTraceMode` on the parallelized model
   - Comm ops appear naturally in the trace
   - Remove `_inject_synthetic_comm_events()` call for fake_backend
2. Keep `_inject_synthetic_comm_events()` as fallback for configurations where Strategy 1 fails
3. Test: `TestFakeBackendCommCapture` — verify FSDP all_gather/reduce_scatter appear in dispatch trace without synthetic injection

### Phase 5: Integrate and Clean Up

1. Replace `RuntimeCapture.activate()` to use `UnifiedTraceMode` instead of separate `OpCaptureMode` + `CommRecorder`
2. Remove FX capture from primary compute_graph (keep in metadata only)
3. Update all config_registry files for `device_mode` option
4. Run full 69-test suite + new tests, pre-commit
5. DeepSeek V4 smoketest validation

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| FSDP2 on meta/FakeTensor may not dispatch all_gather/reduce_scatter as visible ATen ops | Keep `_inject_synthetic_comm_events()` as fallback; test with llama3_debugmodel first |
| `UnifiedTraceMode` (FakeTensorMode + TorchDispatchMode) may conflict with nested mode handling | PyTorch supports nested dispatch modes; test with existing models |
| Meta tensor backward may not work for all op types | FakeTensorMode handles this; if specific ops fail, add to `_TRIVIAL_TARGETS` |
| `param.numel()` on meta tensors returns correct value but `param.data_ptr()` is invalid | Never call `data_ptr()` in simulator code; use `numel()` only |
| Existing gloo mode must continue working | Keep CPU path for gloo; meta only for fake_backend |

---

## Expected Outcomes

| Metric | Current | After |
|--------|---------|-------|
| Memory for Llama 3 8B debug model | ~16GB CPU RAM (real params + activations) | ~0 bytes (meta params, FakeTensor activations) |
| Memory for Llama 3 70B | ~140GB CPU RAM (impossible on most machines) | ~0 bytes (feasible on any machine) |
| Capture speed (fake_backend) | Seconds (real CPU computation) | Milliseconds (shape-only dispatch) |
| Op classification code | 2 copies (fx_capture + dispatch_interceptor) | 1 copy (op_classification) |
| Synthetic comm injection | Heuristic post-processing | Natural dispatch capture (or fallback) |
| FX graph duplication | Stored in metadata AND as separate capture | Only in metadata (optional) |
| backward phase labels | Correct in runtime, `"joint"` in FX | Correct everywhere (unified dispatch) |
| DTensor placement info | Missing in FX path | Captured via `TensorMeta.from_tensor()` on FakeTensors |

---

## Compatibility Guarantees

1. **OpNode fields unchanged**: node_id, op_name, op_type, phase, inputs, outputs, attrs, comm_op, comm_group_size, pp_stage, microbatch_idx, perf_result — all identical
2. **TensorMeta fields unchanged**: shape, dtype, device (always `"cpu"` in output), is_dtensor, placements, requires_grad — all identical
3. **ComputeGraph structure unchanged**: nodes dict, edges list, metadata dict
4. **TrainingSchedule structure unchanged**: events, deps, metadata
5. **SimulationResult structure unchanged**: compute_graph, schedule, comm_events, fsdp_events, pp_events, memory_events, metadata
6. **Cost model input unchanged**: same OpNode fields, same TensorMeta shapes
7. **Export formats unchanged**: JSON, DOT, Chrome trace, HTML — all produce identical output
8. **Gloo backend mode unchanged**: still uses CPU tensors, still uses `patch_device_type_to_cpu()`
9. **API backward compatibility**: `Simulator.simulate_fx()`, `Simulator.simulate_runtime()`, `Simulator.simulate_pp_schedule()` all still work; `Simulator.simulate_all()` uses unified path internally