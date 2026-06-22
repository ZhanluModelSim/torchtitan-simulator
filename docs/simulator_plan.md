# TorchTitan Simulator - Platform Defect Analysis & Refactoring Plan

## 1. 缺陷分析 (Defect Analysis)
根据最新的 `deepseek_v4_pro` E2E 仿真运行结果 (`summary.txt`)，模拟器在纯 Meta 追踪模式下存在以下核心缺陷：

### 1.1 内存追踪失效 (Memory Estimates = 0 B)
* **表现**：`Total memory events: 0`，`Static/Peak dynamic memory: 0 B`。
* **原因**：目前采用 `FakeTensorMode` + `meta` Device，框架原有的 `MemoryEstimator` 强依赖于 CUDA 的显存分配器钩子 (`torch.cuda.memory_allocated`)。由于没有实际的物理显存分配，导致完全丢失了张量维度的内存统计。

### 1.2 FSDP/PP 通信与状态机事件丢失 (0 FSDP/PP Events)
* **表现**：FSDP Events: 0, PP Events: 0。且在 Synthetic 模式下未注入 `all_gather` / `reduce_scatter` 等 FSDP 通信。
* **原因 1 (Synthetic逻辑错误)**：在 `_inject_synthetic_comm_events` 中，对于 `dp_shard = -1` (Auto) 的解析逻辑错误。由于模拟器处于单进程运行，其将 `ds` 错误地推导为了 `1` (`world_size // (pp * tp * cp * dr)` 其中 `world_size` 被设为了分母本身)，直接跳过了所有 FSDP 的通信注入。
* **原因 2 (Native事件丢失)**：我们通过 Mock Schedule (`MockSchedule`) 绕过了原生的 Pipeline 分发，并且 FSDP2 的执行在 FakeTensor 阶段缺乏显式的状态捕获，导致生命周期事件（Lifecycle events）未能注册进 Trace 中。

### 1.3 算力代价评估失真 (Cost Model Fidelity)
* **表现**：CP Step time (Compute Path) 仅为 `2.573 ms`，与预期的 E2E Step time (`1.200 s`) 存在极大的偏差，导致 Compute Utilization 被严重低估为 `15.3%`。
* **原因**：当前的 `MockCostModel` 仅仅粗略匹配了部分标准的 `matmul` / `bmm`。对于 DeepSeek-V4 中特有的 `torch.ops.deepep` (MoE) 算子、FlashAttention 以及各类 Sparse 操作未能识别，导致关键路径的 FLOPs 被评估为 0。

### 1.4 过度依赖 Monkey Patch (Fragile Patches)
* **表现**：为了让原生 Trainer 在 `FakeTensorMode` 下顺畅运行，我们强行注入了 `sl.log_trace_scalar`，`FakeTensor.__format__`，甚至拦截了 `torch._chunk_cat` 与 `aten._local_scalar_dense`。
* **原因**：TorchTitan 的可观测性组件与部分控制流依赖于张量内的标量值（如 Token 统计、Loss 标量化），这在基于 Meta 的仿真机制中十分脆弱，稍微更新就会引发崩溃。

---

## 2. 演进与修复规划 (Action Plan)

### Phase 1: 修复通信拓扑与合成通信注入 (Synthetic Comm Fix)
* **目标**：修正单进程模拟时的拓扑计算，使仿真器能够正确生成并串联 FSDP2 的 All-Gather / Reduce-Scatter 事件。
* **行动**：
  1. 修复 `trainer_runner.py::_inject_synthetic_comm_events` 中的 `dp_shard` 自动推导逻辑，利用全局或解析自 `ConfigManager` 的 Fake World Size，而非局部推导。
  2. 显式在 Trace Graph 中连接 DP 和 FSDP 的 DataEdge，以恢复正确的依赖（Schedule Deps）。

### Phase 2: 内存仿真器适配 (Meta Memory Estimator)
* **目标**：在 0 物理显存的状态下，精准推演模型的峰值内存。
* **行动**：
  1. 重写 `memory_estimator.py`。
  2. 利用 `UnifiedTraceMode` (`__torch_dispatch__`) 拦截所有的 `aten.empty/zeros/ones` 等工厂方法。
  3. 基于 `tensor.numel() * dtype_size()` 以及算子的生命周期手动计算峰值 Dynamic Memory（激活值）与 Static Memory（参数+优化器状态），而不是调用 CUDA API。

### Phase 3: Cost Model 精度提升 (Cost Model Fidelity)
* **目标**：使 `CP Step Time` 的评估符合模型真实 FLOPs，恢复利用率数据的指导价值。
* **行动**：
  1. 扩充 `cost_model.py`，专门引入 `DeepSeekCostModel` 及对新算子的支持。
  2. 对 `torch.ops.deepep.*` 进行解析，根据 `num_experts`、`top_k` 以及 `hidden_dim` 的形状手动注入对应的运算时长。
  3. 为 `FlashAttention` / `FlexAttention` 匹配对应的复杂度公式 $O(N^2 \cdot d)$。

### Phase 4: 解耦可观测性组件 (Decoupling Observability)
* **目标**：消除由于日志组件试图 `item()` Meta Tensor 带来的脆弱补丁。
* **行动**：
  1. 在 `Trainer` 外层构建更安全的 Logging Hook（如接管 `MetricsProcessor` 与 `structured_logger`）。
  2. 针对包含 `DataDependentOutputException` 的控制流（如 Dynamic Token Count），提供符号化（Symbolic）的固定预估值替代动态评估，移除对 PyTorch 底层派发机制的强行侵入。
