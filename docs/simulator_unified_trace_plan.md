# 模拟器：统一 Dispatch Trace + Meta Device Patching

## 问题陈述

模拟器架构中存在两个系统性缺陷：

### 缺陷1：三级 Compute Graph Trace 机制冗余

当前 `compute_graph` 通过 **三个互相重叠的机制** 填充：

| 机制 | 入口 | 捕获内容 | OpNode ID 前缀 | Phase 粒度 |
|------|------|----------|-----------------|------------|
| FX tracing | `capture_forward_fx` / `capture_joint_fx` | 通过 `make_fx` + `FakeTensorMode` 捕获 ATen ops | `fx_` | `"forward"` 或 `"joint"`（无 backward 分离） |
| Runtime capture (dispatch) | `OpCaptureMode` + `CommRecorder` + `FSDPEventRecorder` | 拦截 dispatched ops + monkey-patch comm + FSDP hooks | `op_` / `comm_` | `"forward"` / `"backward"` / `"optimizer"` |
| 合成通信注入 | `_inject_synthetic_comm_events` | 基于 model param numel 的启发式 FSDP/TP comm nodes | `comm_syn_` | `"forward"` / `"backward"` |

**冗余点：**
1. 算子分类逻辑重复：`_classify_fx_node()` (fx_capture.py) vs `_categorize_op()` (dispatch_interceptor.py) — 几乎相同的 marker 列表但有细微差异（`"broadcast_"` vs `"broadcast"`，FX 缺少 `"aten.rand"`）。
2. Comm ops 通过三种重叠方式捕获：(a) 作为 dispatched c10d_functional ATen ops 通过 `OpCaptureMode`，(b) 作为 monkey-patched `dist.*` calls 通过 `CommRecorder`，(c) 作为合成启发式 nodes。
3. 在 `trainer_runner.py` 中，FX forward/joint graphs 在 runtime capture 之后捕获，但存储在 `result.metadata["fx_forward_graph"]` — 一个未与主 compute_graph 合合的第二个完整 graph。
4. FX 路径不产生 backward-only ops；runtime 路径产生。joint FX 路径将所有 ops 标为 `"joint"`。
5. 在 fake_backend 模式下，`OpCaptureMode` 在 CPU tensors 上捕获 compute ops（有真实内存分配），而 `_inject_synthetic_comm_events` 基于启发式模型结构分析创建 comm nodes。这两个概念上是一步操作，应该合在一起。

### 缺陷2：CPU Device Patching 内存压力

模拟器将设备从 GPU → **CPU** patch，避免 GPU 依赖。但：
- 大模型（Llama 3 70B: ~70B params × 2 bytes = ~140GB）超出 CPU RAM
- 即使 debug 小模型也分配真实 tensors，浪费内存和拖慢 capture
- FX 路径已经使用 `FakeTensorMode`（仅形状，无分配），但 runtime 路径分配真实 CPU tensors

**机会：** 将设备 patch 到 **meta** 而不是 CPU。Meta tensors 具有：
- `.shape`, `.dtype`, `.device` — 我们需要的所有 `TensorMeta` 元数据
- 无数据分配（0 bytes 内存）
- PyTorch 的 `torch.device("meta")` context manager 用于模型构建
- TorchTitan core 已经使用 `with torch.device("meta"):` 进行模型 init

**约束：** `compute_graph` 和通信算子信息不能发生大的改变。`OpNode` 字段（op_name, op_type, phase, inputs, outputs, comm_op 等）必须保持一致。

---

## 解决方案设计

### A 部分：统一 Dispatch-Based Trace 模型

**核心思想：** 用 **单一 dispatch-based trace** 替换三级机制，使用 `TorchDispatchMode` + `FakeTensorMode` 一次捕获所有需要的信息。

#### A1：统一算子分类

创建单一 `op_classification.py` 模块，共享 marker 列表 `_COMM_MARKERS`, `_P2P_MARKERS`, `_DATA_MOVE_MARKERS`, `_MEMORY_MARKERS`, `TRIVIAL_TARGETS`, 和 `COMM_OP_MAP`。FX 和 dispatch 路径都调用同一 `_classify_op(target: str) -> (op_type, comm_op)` 函数。

```python
# torchtitan/experiments/simulator/op_classification.py
_COMM_MARKERS = ("_c10d_functional", "c10d_functional", "all_reduce", "all_gather", "reduce_scatter", "all_to_all", "broadcast", "wait_tensor", "barrier")
_P2P_MARKERS = ("_send", "_recv", ".send", ".recv")
_DATA_MOVE_MARKERS = ("_to_copy", "copy_", ".to.")
_MEMORY_MARKERS = ("aten.empty", "aten.zeros", "aten.ones", "aten.full", "aten.arange", "aten.rand")
_TRIVIAL_TARGETS = frozenset([...])
_COMM_OP_MAP = [("reduce_scatter", "reduce_scatter"), ...]

def classify_op(target: str) -> tuple[str, str | None]:
    """对任何 op target string 返回 (op_type, comm_op_or_None)。"""
    ...
```

`fx_capture.py` 和 `dispatch_interceptor.py` 都从此模块导入。

#### A2：统一 UnifiedTraceMode

创建新的 `UnifiedTraceMode`，将 `FakeTensorMode` + `TorchDispatchMode` 合入一个 context manager。此 mode：
- 内部使用 `FakeTensorMode` 使所有 tensors 为仅形状（无内存分配）
- 通过 `TorchDispatchMode.__torch_dispatch__` 拦截每个 dispatched op
- 将每个 op 记录为 `OpNode`，附带 phase, pp_stage, microbatch context
- 通过 tensor identity 跟踪 data-flow edges（同 `OpRecorder._tensor_producer` 机制）
- 通过 autograd hooks 检测 backward phase（同 `torch.Tensor.backward` monkey-patch 机制）

```python
# torchtitan/experiments/simulator/unified_trace.py
class UnifiedTraceMode(TorchDispatchMode):
    def __init__(self, recorder: TraceRecorder):
        self.recorder = recorder

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 1. 在 FakeTensorMode 下执行（仅形状，无内存）
        # 2. 通过统一 classify_op() 分类 op
        # 3. 记录为 OpNode，TensorMeta 来自 FakeTensor metadata
        # 4. 跟踪 data-flow edges
        ...
```

**核心收益：** 此单一 mode 产生与当前 runtime 路径 **相同** 的 compute_graph，但不分配真实 tensors。它同时替换 `OpCaptureMode` + `FakeTensorMode` 作为独立 contexts。

#### A3：将合成通信注入合入 Dispatch Capture

在 fake_backend 模式下，不再使用后处理 `_inject_synthetic_comm_events()` 创建启发式 comm nodes，`UnifiedTraceMode` 在正确并行化的模型 dispatch FSDP/TP/DP comm ops 时自然捕获。两种策略：

**策略1（首选）：** 在 `FakeTensorMode` 下运行模型，使用 **真实 parallelize 函数**（FSDP2, TP）。由于 FSDP2/TP 在 meta/FakeTensors 上正确工作（TorchTitan core 已这样做），`all_gather`, `reduce_scatter`, `all_reduce` ops 自然出现在 dispatch trace 中。无需合成注入。

**策略2（回退）：** 如果策略1对某些模型配置失败，保留 `_inject_synthetic_comm_events()` 作为回退，但在 capture context 内部调用而不是作为后处理步骤。

#### A4：将 FX Capture 作为独立路径移除

FX 路径 (`capture_forward_fx`, `capture_joint_fx`) 成为 **legacy/可选** mode。统一 dispatch 路径产生等同（或更好）的数据：
- 它捕获带有正确 phase labels 的 backward ops（FX joint 路径将所有 ops 标为 `"joint"`）
- 它捕获 FSDP lifecycle events（FX 路径不捕获）
- 它捕获 PP stage/microbatch context（FX 路径不捕获）
- 它捕获 DTensor placement info（FX 路径不捕获）

FX 路径仅保留用于：
1. 静态 graph export（产生 `fx.GraphModule` 用于下游编译工具）
2. 验证 dispatch trace 与静态 FX graph 是否匹配（交叉验证）

这些用例将 FX graph 存储在 `result.metadata` 中，不在主 `compute_graph` 中。

#### A5：文件变更汇总

| 文件 | 变更 |
|------|------|
| `op_classification.py` | **新增**：共享 `_COMM_MARKERS`, `classify_op()`, `TRIVIAL_TARGETS`, `COMM_OP_MAP` |
| `unified_trace.py` | **新增**：`UnifiedTraceMode`, `TraceRecorder`（合合 OpRecorder + phase tracking） |
| `dispatch_interceptor.py` | **修改**：从 `op_classification` 导入 `classify_op`，删除本地 `_categorize_op` 和 marker 列表 |
| `fx_capture.py` | **修改**：从 `op_classification` 导入 `classify_op`，删除本地 `_classify_fx_node` 和 marker 列表 |
| `runtime_capture.py` | **修改**：使用 `UnifiedTraceMode` 替代独立的 `OpCaptureMode` + `CommRecorder`；fake_backend 使用策略1 |
| `trainer_runner.py` | **修改**：拆分为 `_run_gloo_capture`（保留 RuntimeCapture）和 `_run_unified_capture`（TraceRecorder + unified_trace）；删除 fake_backend 的 `_inject_synthetic_comm_events()` 调用 |
| `trainer.py` | **修改**：fake_backend → `patch_device_type_to_meta()` + `torch.device("meta")`；gloo → 保留 CPU |
| `graph_assembler.py` | **无变更**：通过 `fx_graph_to_compute_graph` 委托，其已使用统一 `classify_op` |
| `nodes.py` | **无变更**：OpNode, TensorMeta, ComputeGraph 保持一致 |

---

### B 部分：Meta Device Patching

**核心思想：** 将 CPU device patching 替换为 meta device patching，使模拟器为模型参数和激活分配 **零 bytes**。

#### B1：Meta Tensors 的挑战

Meta tensors (`torch.device("meta")`) 具有 shape/dtype/device 元数据但 **无数据存储**。这意味着：
- `tensor.shape` 可用 → `TensorMeta.shape` 可用
- `tensor.dtype` 可用 → `TensorMeta.dtype` 可用
- `tensor.device` 可用 → `TensorMeta.device` = `"meta"`（需要在输出中映射到 `"cpu"`）
- `tensor.numel()` 可用 → cost_model FLOPs 计算可用
- `tensor.requires_grad` 可用 → phase tracking 可用

**但是：**
- **Meta tensors 上的操作崩溃**：`aten.mm.default(meta, meta)` 抛出 RuntimeError 因为没有数据可计算
- **FakeTensorMode 解决此问题**：FakeTensors 携带 shape/dtype 元数据并 **模拟** 操作输出而不计算。这正是 FX 路径已经做的。
- **A 部分的 `UnifiedTraceMode` 内部使用 FakeTensorMode**：所以 B 部分自然由 A 部分的统一 dispatch mode 解决。

#### B2：Patch 策略

将 `patch_device_type_to_cpu()` 替换为 `patch_device_type_to_meta()`：

```python
# torchtitan/experiments/simulator/meta_env.py
def patch_device_type_to_meta() -> None:
    """Monkey-patch torchtitan device helpers 为 'meta'。"""
    # 同 patch_device_type_to_cpu() 的模式，但：
    # - tt_utils.device_type = "meta"
    # - tt_utils.device_module = _make_meta_device_module()
    # - _PATCHED_MODULES 重绑定为 "meta"
    # - torch.cuda patches 保留（meta models 不需要 CUDA）

def _make_meta_device_module():
    """类似 torch.cuda 的 namespace 但报告 meta device。"""
    return types.SimpleNamespace(
        set_device=lambda device: None,
        current_device=lambda: 0,
        device_count=lambda: 0,  # 无真实设备
        ...
    )
```

#### B3：Meta 上的模型构建

将 `with torch.device("cpu"):` 替换为 `with torch.device("meta"):`：

```python
# 在 run_simulate.py / trainer.py
with torch.device("meta"):
    model = model_cls.from_model_args(model_config)
```

此创建所有参数为 meta tensors（仅形状，0 bytes）。然后：
1. **在 meta 上并行化**：TorchTitan 的并行化函数（FSDP2, TP）设计为在 meta tensors 上工作。`fully_shard` 和 TP wrapping 产生正确的分片元数据而不分配任何数据。
2. **无 `to_empty()` 或 `init_weights()`**：我们永远不物化模型。我们只需要 shape/dtype/device 元数据用于 compute_graph。
3. **在 `UnifiedTraceMode` 下运行**：`FakeTensorMode` 在 `UnifiedTraceMode` 内部将 meta tensors 转换为 FakeTensors 在第一个 op dispatch 时。所有后续 ops 被符号追踪。

#### B4：Meta/FakeTensor 模式下的通信

- **Fake_backend**：FSDP2 在 FakeTensors 上的 `all_gather` / `reduce_scatter` 产生具有正确形状的 FakeTensor 输出。这些 ops 自然出现在 dispatch trace 中。`UnifiedTraceMode` 将它们记录为 `OpNode(op_type="comm_collective", comm_op="all_gather")`。无需合成注入。
- **Gloo backend**：在 FakeTensors 上进行真实 `torch.distributed` comm **不可能**（FakeTensors 无数据可发送）。对于 gloo mode，我们必须将 tensors **物化**到 CPU 仅用于 comm ops。两种方法：
  1. **选择性物化**：在 `UnifiedTraceMode.__torch_dispatch__` 中检测 comm ops，仅将 comm 输入 tensors 物化到 CPU 进行真实 gloo 操作，然后将输出转换回 FakeTensor 元数据。
  2. **混合模式**：对于 gloo，回退到当前 CPU 路径（所有 ops 使用真实 CPU tensors）。Meta patching 仅用于 fake_backend mode。

**决策：** 使用方法2（混合）简化。Meta patching 仅在 `comm_backend=""` (fake_backend) 时应用。当 `comm_backend="gloo"` 时，保留 CPU patching 如原。

#### B5：TensorMeta.device 归一化

Meta tensors 具有 `device="meta"`。在输出 `TensorMeta` 中，我们需要 `device="cpu"` 用于与 cost_model 和 export formats 的向后兼容。在 `UnifiedTraceMode` 中添加归一化步骤：

```python
def _normalize_device(device_str: str) -> str:
    """将 'meta' → 'cpu' 用于输出 TensorMeta 兼容性。"""
    if device_str == "meta":
        return "cpu"
    return device_str
```

此在 `_collect_tensor_metas()` 中记录 OpNode inputs/outputs 时应用。compute_graph 和所有下游工具看到 `"cpu"` devices，保持向后兼容性。

#### B6：Meta/FakeTensors 上的 Phase Transition Detection

当前 backward-phase detection 使用 `torch.Tensor.backward` monkey-patch。在 FakeTensors 上：
- `backward()` 仍然工作（它通过 `__torch_dispatch__` dispatch）
- `UnifiedTraceMode` 可以拦截 backward call 并设置 phase

替代方案：使用 `torch.autograd.graph.Node` hooks 替代 monkey-patching `torch.Tensor.backward`。这在真实和 FakeTensors 上都更干净。

#### B7：Meta 上的内存估算

Meta tensors 具有 `tensor.numel()` 和 `tensor.dtype` 但 `tensor.element_size()` 可能不工作。memory estimator (`memory_estimator.py`) 需要更新：
- 使用 `dtype_size(dtype_str)`（已存在）替代 `tensor.element_size()`
- 计算内存为 `numel * dtype_size`（仅形状计算，无 tensor 数据需要）
- 这实际上比依赖真实 tensor 分配的当前方法 **更正确**

#### B8：文件变更汇总

| 文件 | 变更 |
|------|------|
| `meta_env.py` | **新增**：`patch_device_type_to_meta()`, `_make_meta_device_module()`, `_normalize_device()` |
| `cpu_env.py` | **无变更**：保留用于 gloo backend 回退 |
| `unified_trace.py` | **新增**：`UnifiedTraceMode` 带 FakeTensorMode 集成, device 归一化 |
| `trainer.py` | **修改**：fake_backend → `patch_device_type_to_meta()` + `torch.device("meta")`；gloo → 保留 CPU |
| `run_simulate.py` | **修改**：`with torch.device("meta"):` 用于 fake_backend 模型构建 |
| `memory_estimator.py` | **修改**：使用 `dtype_size()` + `numel` 替代 `tensor.element_size()` |
| `nodes.py` | **无变更**：TensorMeta 保持不变（输出中 device="cpu"） |
| `cost_model.py` | **无变更**：`_numel()` 已经在 tuple shapes 上工作；`_estimate_comm_bytes()` 已经使用 `numel * dtype_size` |

---

## 实施阶段

### 阶段1：统一算子分类（低风险，即时价值） ✅ 已完成

1. 创建 `op_classification.py` 带共享 markers 和 `classify_op()`
2. 更新 `fx_capture.py` 从 `op_classification` 导入
3. 更新 `dispatch_interceptor.py` 从 `op_classification` 导入
4. 测试：69 个现有测试不变，新增 `TestOpClassification` 7 个测试

### 阶段2：创建 UnifiedTraceMode（核心变更） ✅ 已完成

1. 创建 `unified_trace.py` 带 `UnifiedTraceMode(TorchDispatchMode)` + `TraceRecorder`
2. `UnifiedTraceMode.__torch_dispatch__`：
   - Enter `FakeTensorMode` 如果尚未激活
   - 执行 op（仅形状输出）
   - 通过统一 `classify_op()` 记录 OpNode
   - 跟踪 data-flow edges
   - 在 TensorMeta 中归一化 `device="meta"` → `"cpu"`
3. Phase detection：`torch.Tensor.backward` monkey-patch（同当前，在 FakeTensors 上工作）
4. 测试：`TestUnifiedTrace` — 追踪小模型，验证 compute_graph 与当前 runtime 路径输出匹配

### 阶段3：Meta Device Patching（B 部分） ✅ 已完成

1. 创建 `meta_env.py` 带 `patch_device_type_to_meta()`
2. 添加 `device_mode` 到 `SimulationConfig`（值：`"cpu"`，`"meta"`）
3. 更新 `trainer.py`：fake_backend → `patch_device_type_to_meta()` + `torch.device("meta")`；gloo → 保留 CPU
4. 更新 `simulator.py`：新增 `Simulator.simulate_unified()` 方法（device_mode='meta'|'cpu'）
5. 测试：`TestMetaDevicePatch` + `TestSimulatorUnified` — 验证 meta 模型有 is_meta 参数，TensorMeta 有 `"cpu"` device，compute_graph 匹配

### 阶段4：集成到 trainer 路径 ✅ 已完成

1. 在 `trainer_runner.py` 中拆分 `run_trainer_simulation`：
   - `_run_gloo_capture()`：保留 RuntimeCapture 路径（真实 CPU tensors）
   - `_run_unified_capture()`：TraceRecorder + unified_trace 路径（FakeTensorMode 零内存 capture）
2. Gloo 路径不变，fake_backend 使用 meta device + unified dispatch
3. 保留 `_inject_synthetic_comm_events()` 作为 fake_backend 回退（策略1 需要真实 parallelize 函数支持）

### 阶段5：清理和验证 ✅ 已完成

1. 移除 `fx_capture.py` 和 `dispatch_interceptor.py` 中的重复 `_classify_fx_node` 和 `_categorize_op`
2. 验证 `graph_assembler.py` 已通过 `fx_graph_to_compute_graph` 使用统一分类
3. 运行完整 91 测试套件 + pre-commit

---

## 风险评估

| 风险 | 缓解 |
|------|------|
| FSDP2 在 meta/FakeTensor 上可能不 dispatch all_gather/reduce_scatter 为可见 ATen ops | 保留 `_inject_synthetic_comm_events()` 作为回退；先在 llama3_debugmodel 上测试 |
| `UnifiedTraceMode` (FakeTensorMode + TorchDispatchMode) 可能与嵌套 mode handling 冲突 | PyTorch 支持嵌套 dispatch modes；在现有模型上测试 |
| Meta tensor backward 可能对某些 op types 不工作 | FakeTensorMode 处理此问题；如果特定 ops 失败，添加到 `_TRIVIAL_TARGETS` |
| `param.numel()` 在 meta tensors 上返回正确值但 `param.data_ptr()` 无效 | 在 simulator 代码中永远不调用 `data_ptr()`；仅使用 `numel()` |
| 现有 gloo mode 必须继续工作 | 保留 CPU 路径用于 gloo；meta 仅用于 fake_backend |
| FakeTensorMode 不接受混合 device 输入（CPU input + meta 参数） | 在 meta 模式下，输入也必须为 meta device；或在 FakeTensorMode 内将所有输入转换为 FakeTensor |

---

## 预期成果

| 指标 | 当前 | 之后 |
|------|------|------|
| Llama 3 8B debug model 内存 | ~16GB CPU RAM（真实 params + activations） | ~0 bytes（meta params, FakeTensor activations） |
| Llama 3 70B 内存 | ~140GB CPU RAM（大多数机器不可能） | ~0 bytes（任何机器可行） |
| Capture 速度（fake_backend） | 秒级（真实 CPU 计算） | 毫秒级（仅形状 dispatch） |
| 算子分类代码 | 2 份（fx_capture + dispatch_interceptor） | 1 份（op_classification） |
| 合成通信注入 | 启发式后处理 | 自然 dispatch capture（或回退） |
| FX graph 重复 | 存储在 metadata AND 作为独立 capture | 仅在 metadata（可选） |
| backward phase labels | runtime 正确，FX `"joint"` | 所有路径正确（统一 dispatch） |
| DTensor placement info | FX 路径缺失 | 通过 `TensorMeta.from_tensor()` 在 FakeTensors 上捕获 |

---

## 兼容性保证

1. **OpNode 字段不变**：node_id, op_name, op_type, phase, inputs, outputs, attrs, comm_op, comm_group_size, pp_stage, microbatch_idx, perf_result — 全部一致
2. **TensorMeta 字段不变**：shape, dtype, device（输出中始终 `"cpu"`），is_dtensor, placements, requires_grad — 全部一致
3. **ComputeGraph 结构不变**：nodes dict, edges list, metadata dict
4. **TrainingSchedule 结构不变**：events, deps, metadata
5. **SimulationResult 结构不变**：compute_graph, schedule, comm_events, fsdp_events, pp_events, memory_events, metadata
6. **Cost model 输入不变**：相同 OpNode 字段，相同 TensorMeta shapes
7. **Export formats 不变**：JSON, DOT, Chrome trace, HTML — 产生一致输出
8. **Gloo backend mode 不变**：仍使用 CPU tensors，仍使用 `patch_device_type_to_cpu()`
9. **API 向后兼容**：`Simulator.simulate_fx()`, `Simulator.simulate_runtime()`, `Simulator.simulate_pp_schedule()` 全部仍工作；`Simulator.simulate_all()` 内部使用统一路径；新增 `Simulator.simulate_unified()`

---

## 实施状态

| 阶段 | 状态 | 提交 | Tests |
|------|------|------|-------|
| 阶段1：统一算子分类 | ✅ 完成 | `0d3b4efc` | 7 TestOpClassification |
| 阶段2：创建 UnifiedTraceMode | ✅ 完成 | `0d3b4efc` | 7 TestUnifiedTrace |
| 阶段3：Meta Device Patching | ✅ 完成 | `0d3b4efc` | 4 TestMetaDevicePatch + 4 TestSimulatorUnified |
| 阶段4：集成到 trainer 路径 | ✅ 完成 | `e40d0c33` | — |
| 阶段5：清理和验证 | ✅ 完成 | — | 91 tests total |

**总测试数：91**（69 原始 + 7 TestOpClassification + 7 TestUnifiedTrace + 4 TestMetaDevicePatch + 4 TestSimulatorUnified）
