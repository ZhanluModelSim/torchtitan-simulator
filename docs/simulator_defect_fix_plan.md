# TorchTitan Simulator — 缺陷修复实施计划

本文档基于 `simulator_plan.md` 的缺陷分析与代码审计结果，给出分阶段的具体修复方案，包含代码定位、修复策略和测试要求。

---

## 缺陷概览与严重性分级

| # | 缺陷 | 严重性 | 影响模块 | 关键文件 |
|---|------|--------|----------|----------|
| D1 | `dp_shard` 自动推导恒等于 1 | P0 | 通信注入 | `trainer.py:280-283`, `trainer_runner.py:205-207` |
| D2 | 合成通信节点无图边连接 | P0 | DES 仿真 | `trainer_runner.py:264-599` |
| D3 | `shape=bytes` 代替 `shape=numel` | P0 | 通信字节量 | `trainer_runner.py:264-350` |
| D4 | FP32 dtype 硬编码 | P0 | 精度模拟 | `trainer_runner.py:264-599` |
| D5 | PP Schedule 提取器失效 | P0 | PP 仿真 | `pp_schedule_extractor.py`, `trainer.py:234-240` |
| D6 | 内存追踪全零 | P1 | 内存估算 | `meta_env.py:72-79`, `memory_estimator.py` |
| D7 | Cost Model 算子覆盖不足 | P1 | 算力评估 | `cost_model.py:68-157` |
| D8 | matmul FLOPs 双重计算 | P1 | 算力评估 | `cost_model.py:68-157` |
| D9 | 关键字匹配过度宽泛 | P1 | 算力评估 | `cost_model.py:68-157` |
| D10 | reduce_scatter/all_reduce 通信量偏差 | P1 | 通信字节量 | `cost_model.py:196-208` |
| D11 | 动态维度 fallback=1024 | P1 | 算力评估 | `cost_model.py` |
| D12 | Monkey patch 不可逆 | P2 | 稳定性 | `trainer_runner.py:645-727` |
| D13 | `_local_scalar_dense` patch 泄漏 | P2 | 稳定性 | `trainer_runner.py:656-658` |
| D14 | `_chunk_cat` patch 缺失 | P2 | 稳定性 | `test_full_native.py:19-22` |

---

## Phase 1: 通信拓扑与合成通信注入修复 (P0)

Phase 1 是所有后续工作的基础——拓扑错误会导致 DES 仿真结果完全不可信。

### 1.1 修复 `dp_shard` 自动推导逻辑 (D1)

**根因**: `_inject_synthetic_comm_events` (`trainer_runner.py:205-207`) 中：

```python
if ds < 0:
    world_size = pp * tp * cp * dr
    ds = max(1, world_size // (pp * tp * cp * dr))  # 恒等于 1
```

`world_size` 等于分母本身，导致 `ds = 1`，跳过所有 FSDP 通信注入。

`SimulationTrainer.__init__` (`trainer.py:280-283`) 中同样的逻辑：`ds = 1`。

**已有正确实现**:
- `ParallelDims._validate()` (`parallel_dims.py:75`): `dp_shard = world_size // (dp_replicate * cp * tp * pp)`
- `_set_fake_world_size()` (`trainer.py:244-266`): EP-aware 推导 `dp_shard = max(1, -(-ep // (cp * tp)))`

**修复方案**:

1. 创建共享辅助函数 `_resolve_dp_shard(parallelism_config, world_size_or_env)`，统一推导逻辑：
   - 如果 `dp_shard >= 1`，直接使用。
   - 如果 `dp_shard = -1`（Auto），从 `ParallelDims._validate()` 的公式推导：`ds = world_size // (dp_replicate * cp * tp * pp)`。
   - EP-aware 模型（DeepSeek V4）使用专家并行感知公式。
2. `SimulationTrainer.__init__` 和 `_inject_synthetic_comm_events` 均调用此函数，消除各自硬编码的推导。
3. `world_size` 应从 `os.environ["NGPU"]` 或 `os.environ["WORLD_SIZE"]` 获取（由 `_set_fake_world_size` 设置），而非从 `pp * tp * cp * dr` 局部计算。

**涉及文件**: `trainer.py:280-283`, `trainer_runner.py:205-207`

### 1.2 合成通信节点缺少图边连接 (D2)

**根因**: 注入的 FSDP/TP/PP 通信 OpNode 与计算图中的 anchor 节点之间仅用 `"sequential"` 类型边连接（弱依赖），无 `"data"` 边。DES 关键路径分析因此忽略通信节点，利用率指标失真。

**修复方案**:

1. FSDP `all_gather`: 添加 `"data"` 边从 shard 参数持有节点（前一层 compute）到 all_gather 节点。
2. FSDP `reduce_scatter`: 添加 `"data"` 边从 reduce_scatter 节点到下一层 compute 节点。
3. TP `all_reduce`: 添加 `"data"` 边连接前后 compute 节点。
4. PP send/recv: 创建跨 rank 的 `"data"` 边，连接发送 rank 的 compute 与接收 rank 的 compute。
5. 在 `ScheduleEvent` 中填充 `op_node_ids` 字段，建立 schedule → compute graph 的双向链接。

**涉及文件**: `trainer_runner.py:264-599`, `nodes.py` (ScheduleEvent.op_node_ids)

### 1.3 修复 shape=bytes vs numel (D3)

**根因**: 合成通信事件的 tensor metadata 使用 `shape=bytes`（字节量直接作为 shape 维度），而 cost model 中 `tensor_nbytes = prod(shape) * dtype_size()` 会导致 FP32 张量字节量被放大 4 倍。

**修复方案**: 将合成通信事件的 `shape` 字段改为存储 `numel`（元素数量），字节量由 cost model 的 `tensor_nbytes = prod(shape) * dtype_size()` 自动计算。

示例修复：
```python
# 当前（错误）
per_layer_numel = shard_numel // max(num_layers, 1)
tensor_meta = TensorMeta(shape=(per_layer_numel * 4,), dtype=torch.float32)
# 修复后
tensor_meta = TensorMeta(shape=(per_layer_numel,), dtype=torch.float32)
```

**涉及文件**: `trainer_runner.py:264-350` (FSDP), `353-433` (TP), `436-599` (PP)

### 1.4 修复 FP32 dtype 硬编码 (D4)

**根因**: 所有合成通信事件硬编码 `dtype=torch.float32`，忽略了 BF16/FP8 混合精度训练场景。

**修复方案**:
- Forward activation 张量 dtype 从 `config.model_spec.dtype` 读取（BF16 训练时为 `torch.bfloat16`）。
- Gradient/optimizer state 张量 dtype 使用 `torch.float32`（优化器始终 FP32）。
- All-gather 输出 dtype = 参数 dtype（与 gather 前一致）。
- Reduce-scatter 输出 dtype = 参数 dtype。

**涉及文件**: `trainer_runner.py:264-599`

### 1.5 PP Schedule 提取器修复 (D5)

**根因**: `PPScheduleExtractor` 依赖 `_actions`/`_compute_clock_cycles`（当前 PyTorch 已移除），三种 fallback 均失败。`MockSchedule` (`trainer.py:234-240`) 完全 stub 了 PP 分发，无生命周期事件。

**修复方案**:

1. 创建 `extract_schedule_from_pytorch()` 函数，直接读取 `pipeline_order_with_comms`（PyTorch PP 模块的真实 schedule 数据结构）。
2. 使用 `MockPipelineStage`（真实 schedule topology，fake 执行）进行单进程 schedule 提取：
   - 构建 `PipelineStage` 对象（model_part + stage_index）。
   - 调用 `ScheduleInterleaved1F1B` 的 `step()` 方法（不执行真实计算，只获取 action 序列）。
   - 从 `pipeline_order_with_comms` 提取每 rank 的 forward/backward/comms 事件时序。
3. fake_backend 模式：程序化提取 schedule topology，注入合成 PP send/recv 事件。
4. gloo 模式：通过实际 PP 执行捕获生命周期事件。

**涉及文件**: `trainer.py:234-240` (MockSchedule → MockPipelineStage), `pp_schedule_extractor.py` (重写), 新文件 `schedule_extractor.py`

---

## Phase 2: 内存仿真器适配 (P1)

### 2.1 基于 UnifiedTraceMode 拦截工厂方法 (D6)

**当前状态**: `UnifiedTraceMode` 已将 `aten.empty/zeros/ones/full/arange/rand` 分类为 `"memory"` ops 并记录 `TensorMeta`。但 `memory_estimator.py` 未使用这些节点，仍依赖 `torch.cuda.memory_allocated`（在 meta 模式下返回 0）。

**根因**: `meta_env.py:72-79` 和 `cpu_env.py:125-128` 将 `torch.cuda.memory_allocated` patch 为恒返回 0。无替代追踪机制。

**修复方案**:

1. 重写 `memory_estimator.py`，从 compute graph 的 `"memory"` 类型 OpNode 推导内存：
   - 每个 `"memory"` 节点：`bytes = prod(output_tensor.shape) * dtype_size(output_tensor.dtype)`
   - 计算张量生命周期：tensor 在工厂 op 处"出生"，在最后一个消费者 op 处"死亡"。
   - 使用图边拓扑计算 live range。
   - Peak dynamic memory = 所有 timestep 中 `sum(live_tensor_bytes)` 的最大值。
2. Static memory 保持现有逻辑：`sum(param.numel() * param.element_size()) / shard_factor`。
3. 在 `run_trainer_simulation` 中，trace 捕获后调用 `estimate_graph_memory(result.compute_graph)` 填充 `result.memory_events`。

**涉及文件**: `memory_estimator.py`（主要重写）, `nodes.py`（添加 `lifetime` 字段到 OpNode 或 TensorMeta）

### 2.2 内存事件收集

**修复方案**: 每个 `MemoryEvent` 记录：
- `timestamp_us`: 工厂 op 在图中的序号（用于 DES 时间线映射）
- `action`: `"alloc"` 或 `"free"`
- `tensor_name`: 关联的 OpNode name
- `numel`: tensor 元素数
- `dtype`: tensor 数据类型
- `bytes`: `numel * dtype_size`

在 `estimate_graph_memory` 中按图拓扑排序遍历 `"memory"` 节点，生成 alloc 事件；按最后消费者位置生成 free 事件。

**涉及文件**: `memory_estimator.py`, `nodes.py` (MemoryEvent)

### 2.3 DES 内存时间线验证

**修复方案**: `compute_des_memory_timeline()` (`des_engine.py`) 应使用新 `MemoryEvent` 列表（从工厂 op 拦截填充），而非依赖 `torch.cuda.memory_allocated`。

时间线生成：
- 按 `timestamp_us` 排序所有 alloc/free 事件。
- 累计当前 live bytes，记录每个 DES 时间点的内存水位。
- 在 HTML trace (`export.py`) 中渲染内存时间线图。

**涉及文件**: `des_engine.py`, `export.py` (summary + HTML trace 内存时间线)

---

## Phase 3: Cost Model 精度提升 (P1)

### 3.1 修复 matmul FLOPs 双重计算 (D8)

**根因**: `linear` op 被分解为 `addmm`（包含 matmul + bias add），但 `addmm` 和 `mm`/`matmul` 的 FLOPs 分别被独立计算，导致同一层线性变换被双重计费。

**修复方案**:
- 当检测到 `addmm` op 时，跳过其直接前驱中的 `mm`/`matmul` op（共享输入 tensor 的场景）。
- 或者：仅保留 `addmm` 的 FLOPs 计算（`2*M*N*K`），对 `mm`/`matmul` 做去重标记。
- 实现方式：在 `_estimate_flops` 中维护 `dedup_set`，记录已通过 `addmm` 计算过的 (input_tensor_id, weight_tensor_id) 对。

**涉及文件**: `cost_model.py:68-157` (_estimate_flops)

### 3.2 修复关键字匹配过度宽泛 (D9)

**根因**: `"add" in op_name` 匹配了 `addmm`, `baddbmm` 等非 element-wise add op，导致错误分类。

**修复方案**: 使用精确 op name 匹配或锚定 regex，与 `op_classification.py` 的 `classify_op` 保持一致：

```python
# 替换 substring 匹配
if "add" in op  # 过宽

# 改为精确匹配
_ELEMENTWISE_ADD_OPS = frozenset(["aten.add.Tensor", "aten.add.Scalar"])
if op in _ELEMENTWISE_ADD_OPS  # 精确
```

**涉及文件**: `cost_model.py:68-157`

### 3.3 DeepSeek MoE 算子支持 (D7)

**当前缺失**: `MockCostModel._estimate_flops` 不识别：
- `torch.ops.deepep.dispatch` / `torch.ops.deepep.combine`（MoE 算子）→ FLOPs = 0
- `aten._scaled_dot_product_flash_attention`（FlashAttention ATen kernel）→ 仅匹配 `"flash_attention"` 字串，不匹配实际 ATen 名称
- MoE gating/routing → 完全不可见

**修复方案**: 扩展 `_estimate_flops` 或创建 `DeepSeekCostModel` 子类：

| 算子 | FLOPs 公式 |
|------|-----------|
| `torch.ops.deepep.dispatch` | `top_k * hidden_dim * num_experts * batch_tokens * 2` |
| `torch.ops.deepep.combine` | 同上（反向路径） |
| `aten._scaled_dot_provider_flash_attention` | `2 * seq_len^2 * head_dim * num_heads` |
| MoE gating (softmax + top-k) | `num_tokens * num_experts * 2` |

在 `deepseek_v4/config_registry.py` 中注册 `DeepSeekCostModel`。

**涉及文件**: `cost_model.py`（新增 DeepSeek op handlers）, `deepseek_v4/config_registry.py`（注册 CostModel）

### 3.4 动态维度 fallback 修复 (D11)

**根因**: 动态 tensor 维度（shape 包含 `-1` 或 `0`）fallback 到硬编码 `1024`，导致 FLOPs 大幅偏差。

**修复方案**: 创建 `DimResolver` 类，从模型 config 映射符号维度到实际值：

```python
class DimResolver:
    def __init__(self, model_config):
        self.dim_map = {
            "hidden_dim": model_config.hidden_dim,
            "seq_len": model_config.seq_len,
            "num_heads": model_config.num_heads,
            "num_experts": model_config.num_experts,
            "vocab_size": model_config.vocab_size,
        }

    def resolve(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(self.dim_map.get(d, d) if d < 0 else d for d in shape)
```

各模型 config_registry 在注册时提供 `DimResolver` 配置。

**涉及文件**: `cost_model.py` (DimResolver), 各模型 config_registry

### 3.5 reduce_scatter/all_reduce 通信量修复 (D10)

**根因**:
- reduce_scatter 字节量 = `full_numel * dtype_size`（应为 `shard_numel * dtype_size * 2`：发送 shard + 接收聚合 shard）
- all_reduce ring factor 硬编码 `(world_size - 1) / world_size`
- critical-path overlap heuristic 硬编码 0.5

**修复方案**:
- reduce_scatter transferred bytes = `shard_numel * dtype_size * 2`
- all_reduce bytes (ring algorithm) = `2 * full_numel * dtype_size * (world_size - 1) / world_size`
- 移除 hardcoded overlap heuristic — DES engine 正确处理 overlap

**涉及文件**: `cost_model.py:196-208` (_numel), 通信字节量估算函数

---

## Phase 4: 解耦可观测性组件 (P2)

### 4.1 SimulationObservabilityGuard (D12, D13)

**根因**: 7+ 个 monkey patch 在 `trainer_runner.py:645-727` 防止 `.item()` 在 meta/FakeTensor 上崩溃。关键脆弱点：
- `aten._local_scalar_dense` (line 656-658): **未在 finally 中恢复** — 泄漏到全局状态
- `FakeTensor.__format__` (line 661-662): 修改核心 PyTorch 类
- `sl.log_trace_scalar` (line 684-696): 替换整个 logging 函数

**修复方案**: 创建 `SimulationObservabilityGuard`，在入口层包装而非修改内部：

1. Intercept `MetricsProcessor.log()` 接受 `float | None`，对 meta/FakeTensor 参数提供符号化默认值 (loss=0.0, grad_norm=0.0)。
2. Intercept `structured_logger.log_trace_scalar` 在 API boundary（已有 `safe_log`，改为 context-managed）。
3. 将 `aten._local_scalar_dense` patch 改为 `FakeTensorMode` 级别 dispatch override（PyTorch 正规机制，非全局 dict mutation）：
   ```python
   # 当前（全局 mutation，不可逆）
   torch._subclasses.fake_impls.op_implementations_dict[
       torch.ops.aten._local_scalar_dense.default
   ] = _mock_local_scalar_dense

   # 修复（FakeTensorMode 内注册）
   fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
   # 在 FakeTensorMode 内部处理 _local_scalar_dense
   ```
4. 移除 `FakeTensor.__format__` patch，确保无代码路径对 FakeTensor 使用非 trivial format spec。

**涉及文件**: 新文件 `observability_guard.py`, `trainer_runner.py`（移除 patch，使用 guard）, `meta_env.py`（集成 `_local_scalar_dense` 为 FakeTensorMode 正规 dispatch）

### 4.2 符号化预估替代动态评估

**修复方案**: 对数据依赖控制流（token 统计、loss 标量化）提供 `SymbolicValue`：

```python
class SymbolicValue:
    """代表一个无具体值的标量，用于 meta/FakeTensor 仿真中的控制流。"""
    POSITIVE_INT = 1   # int(local_valid_tokens) → 1
    NONNEG_FLOAT = 0.0 # float(loss.detach().item()) → 0.0
```

- `int(local_valid_tokens)` → `SymbolicValue.POSITIVE_INT` → 控制流中视为 `1`
- `float(loss.detach().item())` → `SymbolicValue.NONNEG_FLOAT` → metrics 中视为 `0.0`

**涉及文件**: 新文件 `symbolic_value.py`, `trainer_runner.py`

### 4.3 补丁清理与可逆性 (D12, D13, D14)

**修复方案**:

1. 所有 patch 在 `run_trainer_simulation` 的 `finally` 块中恢复：
   - 添加 `_local_scalar_dense` 恢复（当前缺失）
   - 添加 `torch._chunk_cat` patch 到主 runner（当前仅在 test_full_native.py）
2. 使用 `contextlib.ExitStack` 管理 patch lifecycle：
   ```python
   with contextlib.ExitStack() as stack:
       stack.callback(lambda: setattr(FakeTensor, "__format__", orig_format))
       stack.callback(lambda: setattr(sl, "log_trace_scalar", orig_log))
       # ... 所有 patch 的恢复回调
       run_simulation_inner(...)
   ```
3. 确保 `torch._subclasses.fake_impls.op_implementations_dict` 的修改在 ExitStack 中恢复。

**涉及文件**: `trainer_runner.py:645-727`（重构 patch 管理）

---

## 测试补充计划

Phase 合入前必须补充的测试用例：

| Phase | 缺失测试 | 优先级 | 测试文件 |
|-------|---------|--------|----------|
| Phase 1 | `_inject_synthetic_comm_events` with `dp_shard=-1` (Auto) | P0 | `test_simulator.py` (TestSyntheticCommInjection) |
| Phase 1 | 合成通信节点拥有 `"data"` 图边 | P0 | `test_simulator.py` |
| Phase 1 | `shape=numel` vs `shape=bytes` 字节量正确性 | P0 | `test_simulator.py` |
| Phase 1 | dtype 从 config 读取（BF16/FP32） | P0 | `test_simulator.py` |
| Phase 1 | `extract_schedule_from_pytorch` 生成 Interleaved1F1B | P0 | `test_simulator.py` |
| Phase 2 | `estimate_graph_memory` 从 `"memory"` OpNode 推导 | P1 | `test_simulator.py` (TestMemoryEstimator) |
| Phase 2 | MemoryEvent alloc/free 时序正确 | P1 | `test_simulator.py` |
| Phase 3 | `deepep.dispatch`/`combine` FLOPs | P1 | `test_simulator.py` (TestCostModel) |
| Phase 3 | `aten._scaled_dot_provider_flash_attention` FLOPs | P1 | `test_simulator.py` |
| Phase 3 | matmul deduplication（无双重计算） | P1 | `test_simulator.py` |
| Phase 3 | reduce_scatter 通信字节量（shard 而非 full） | P1 | `test_simulator.py` |
| Phase 3 | `DimResolver` 动态维度映射 | P1 | `test_simulator.py` |
| Phase 4 | Patch 在 `run_trainer_simulation` 后完整恢复 | P2 | `test_simulator.py` |
| Phase 4 | `_local_scalar_dense` patch 不泄漏 | P2 | `test_simulator.py` |

---

## 优先级与依赖关系

```
Phase 1 (P0) ───→ Phase 2 (P1) ───→ Phase 3 (P1) ───→ Phase 4 (P2)
      │                │                │
      │                │                └─ Cost model
      │                └─ Memory events    feeds accurate
      │                   populate          FLOPs to DES
      └─ Correct topology
         & comm events
         → DES produces
           meaningful results
```

- **Phase 1 是基础**：无正确拓扑和通信事件，DES 仿真产生无意义结果，无论 memory/cost model 多精确。
- **Phase 2 与 Phase 3 可并行推进**（Phase 1 之后）。
- **Phase 4 是稳定性保障**，不影响仿真精度但防止未来版本升级导致的崩溃。

---

## 与已有文档的关系

| 文档 | 内容 | 与本计划的关系 |
|------|------|---------------|
| `simulator_plan.md` | 4 类缺陷宏观分析 | 本计划的具体实施方案 |
| `docs/cost_model_integration.md` | CostModel 接入指南 | Phase 3 需遵循其基类接口 |
| `docs/simulator_unified_trace_plan.md` | Unified Trace 架构（已完成） | Phase 2/3 基于 UnifiedTraceMode |
| `simulator_plan.md` Phase 1-4 | 宏观行动方向 | 本计划细化每步的代码定位与修复策略 |
