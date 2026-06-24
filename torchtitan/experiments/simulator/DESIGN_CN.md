# TorchTitan Simulator -- 架构设计文档

## 1. 概述

TorchTitan Simulator 是一个纯 CPU 训练 trace/仿真系统，作为 side-loaded experiment 构建在 torchtitan LLM 训练平台之上。它能够在**无任何 GPU 硬件**的环境下，捕获前向/反向计算图、通信模式和训练调度（PP、FSDP、TP、DP），从而实现：

- **任意规模的训练步分析**（如 1024 GPU 拓扑）在单台 CPU 机器上完成
- **性能预测**：基于 Cost Model 和离散事件仿真（DES）
- **并行策略探索**：PP/TP/DP/FSDP 各种度数组合，无需真实硬件
- **工作负载图导出**：供下游硬件仿真器（ZhanluModelSim 等）使用

### 设计原则

1. **Side-loaded 实验** -- `train.py` 保持不变；所有仿真器代码位于 `torchtitan/experiments/simulator/`
2. **捕获忠实** -- 调度和计算图从*捕获数据*派生，而非重新实现训练逻辑
3. **PyTorch 原生** -- 复用上游 PyTorch 调度对象（`PipelineSchedule`）、FX tracing 和 `TorchDispatchMode`
4. **模型无关的核心** -- 模型特定代码隔离在各模型子目录（`llama3/`、`deepseek_v4/`）

---

## 2. 目录结构

```
simulator/
  __init__.py              # 公开 API，模块别名（向后兼容）
  trainer.py               # SimulationTrainer（继承 Trainer），SimulationConfig
  trainer_runner.py        # run_trainer_simulation() -- 主执行编排器
  nodes.py                 # 核心数据模型（OpNode, ComputeGraph, TrainingSchedule, ...）
  export.py                # 多格式导出（JSON, DOT, Chrome Trace, HTML, CSV, Text）
  cost_model.py            # CostModel ABC + MockCostModel（roofline 风格估算）
  des_engine.py            # 基于 salabim 的离散事件仿真引擎
  memory_estimator.py      # 激活值 / 模型状态 / 通信缓冲区内存估算
  op_classification.py     # 统一的算子分类（compute/comm/data_move/memory）
  cpu_env.py               # CPU 设备补丁（monkey-patch torch.cuda -> CPU 桩）
  meta_env.py              # Meta 设备补丁（0 字节张量，用于大模型仿真）
  synthetic_dataloader.py  # SyntheticTokenDataLoader（随机 token 生成）
  extension_hooks.py       # NPU/其他 side-load 的扩展点

  capture/                 # Trace 捕获（单一统一模块）
    unified_trace.py       # FakeTensorMode + TorchDispatchMode + CommRecorder + FSDP hooks

  schedule/                # 训练调度提取与生成
    schedule_extract.py    # 从真实 PyTorch PipelineSchedule 对象提取调度
    schedule_generator.py  # 从配置生成语义 Interleaved1F1B 调度
    pp_schedule_extractor.py # PPScheduleExtractor 类（读取 pipeline_order 表）

  ir/                      # 分层 IR（L0-L3）用于工作负载图导出
    op_node.py             # L0: SpecOpNode 投影
    step_graph.py          # L1: StepGraph（每阶段 DAG 模板）
    schedule_graph.py      # L2: ScheduleGraph（编排：实例、数据传递）
    workload_graph.py      # L3: WorkloadGraph（迭代语义 + 数据流）
    builder.py             # IR 投影顶层编排器

  llama3/                  # Llama3 特定仿真配置
    config_registry.py     # llama3_sim_debugmodel, llama3_sim_1024gpu, ...

  deepseek_v4/             # DeepSeek V4 特定仿真配置
    config_registry.py     # deepseek_v4_sim_smoketest, deepseek_v4_pro_sim_smoketest

  tests/                   # 单元测试
    test_simulator.py
    test_ir.py
```

---

## 3. 入口点

### 3.1 SimulationTrainer（通过 `run_train.sh`）

端到端仿真的主要入口。`SimulationTrainer` 继承上游 `Trainer`：

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

**`__init__` 中的关键行为：**
- 强制 `config.comm.mode = "fake_backend"`（单进程无需真实 NCCL/gloo rendezvous）
- 替换 `parallelize_fn` 为 CPU 桩（`_cpu_noop_parallelize` 或 `_cpu_gloo_parallelize_*`）
- 替换 `pipelining_fn` 为 CPU 桩（`_cpu_noop_pipeline` 或 `_cpu_semantic_pipeline`）
- 通过 `_set_fake_world_size()` 从并行配置设置虚拟 world size
- 根据 `comm_backend` 应用设备补丁（meta 或 CPU）
- `super().__init__()` 之后，可选择性地为 gloo 通信捕获用 FSDP1 包装模型

**`SimulationConfig`**（`trainer.py` 中的 dataclass）：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `output_dir` | `"./simulator_output"` | 导出目录 |
| `output_formats` | `["json","dot","chrome_trace","html","text","csv"]` | 导出格式 |
| `mode` | `"all"` | `"all"`、`"runtime"` 或 `"schedule"` |
| `capture_joint_fx` | `False` | 联合 fwd+bwd FX 捕获 |
| `semantic_schedule` | `False` | 从配置生成完整 PP/TP/DP 调度 |
| `cost_model` | `False` | 对计算图运行 Cost Model |
| `cost_model_class` | `""` | 自定义 CostModel 类路径（空 = MockCostModel） |
| `comm_backend` | `""` | `""`（fake）或 `"gloo"`（真实 CPU 通信） |
| `device_mode` | `""` | `""`（自动）、`"meta"` 或 `"cpu"` |

### 3.2 编程式 API

捕获层可以直接用于编程式追踪：

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

## 4. 捕获模式

仿真器使用通过 `unified_trace()` 的单一统一捕获模式。此上下文管理器在一次遍历中组合 `FakeTensorMode` 与 `TorchDispatchMode`，并可选择性地激活通信拦截和 FSDP 生命周期 hooks：

```
unified_trace() 上下文管理器
  |
  +-- FakeTensorMode（仅形状，0 字节）-- 当 use_fake_mode=True 时
  |     |
  |     +-- 每个算子产生 FakeTensor 输出（仅 shape/dtype 元数据）
  |
  +-- UnifiedTraceMode（TorchDispatchMode）
  |     |
  |     +-- 每个被 dispatch 的算子记录为 TraceRecorder 中的 OpNode
  |     +-- 通过 id(tensor) -> node_id 映射追踪张量生产者-消费者
  |     +-- 自动构建数据流边
  |     +-- 阶段追踪（forward/backward/optimizer）
  |
  +-- [可选] CommRecorder -- 当 capture_comm=True 时（gloo 模式）
  |     |
  |     +-- Monkey-patch torch.distributed 集合通信和 _functional_collectives
  |
  +-- [可选] FSDPEventRecorder -- 当 capture_fsdp=True 时（gloo 模式）
        |
        +-- 为 FSDP allgather/reshard/reduce-scatter 生命周期附加模块 hooks
```

**TraceRecorder** 是统一的记录器，累积：
- `nodes: list[OpNode]` -- 每个被 dispatch 的算子一个
- `edges: list[(src, dst, type)]` -- 数据流依赖
- `comm_events`、`fsdp_events`、`pp_events` -- 专用事件列表
- 通过可变字段追踪 Phase/PP-stage/microbatch 上下文

**`build_result()`** 将所有内容组装为带有已填充 `ComputeGraph` 的 `SimulationResult`。

**CommRecorder**（内联到 `unified_trace.py`）拦截 `torch.distributed` 集合通信和 P2P 操作，记录张量元数据、组大小和来自活跃 `TraceRecorder` 的源节点引用。

**FSDPEventRecorder**（内联到 `unified_trace.py`）为 FSDP 包装的模块附加 PyTorch 模块 hooks，捕获参数生命周期（allgather/reshard/reduce-scatter）。

---

## 5. 核心数据模型（`nodes.py`）

```
SimulationResult
  |
  +-- compute_graph: ComputeGraph
  |     +-- nodes: dict[str, OpNode]     # 按插入顺序排列
  |     +-- edges: list[DataEdge]         # 数据流 / 控制 / 阶段边界
  |     +-- metadata: dict
  |
  +-- schedule: TrainingSchedule | None
  |     +-- events: list[ScheduleEvent]   # 粗粒度调度事件
  |     +-- deps: list[ScheduleDep]       # 事件间依赖
  |
  +-- comm_events: list[dict]             # 原始通信事件
  +-- fsdp_events: list[dict]             # 原始 FSDP 生命周期事件
  +-- pp_events: list[dict]               # 原始 PP 事件
  +-- memory_events: list[MemoryEvent]    # 内存分配/驻留估算
  +-- metadata: dict
```

**OpNode** -- 计算图中的单个算子：
- `op_type`：`"compute"` | `"comm_collective"` | `"comm_p2p"` | `"data_move"` | `"memory"`
- `phase`：`"forward"` | `"backward"` | `"optimizer"`
- `pp_stage`、`pp_rank`、`microbatch_idx`：并行上下文
- `comm_op`、`comm_group_size`：用于通信算子
- `perf_result: PerfResult | None`：Cost Model 输出（compute_time、comm_time、FLOPs、bytes）
- `des_start_time_us`、`des_finish_time_us`：DES 引擎时间戳

**ComputeGraph** 提供：
- `fix_comm_phase_labels()`：修正标记错误的通信节点阶段
- `add_phase_boundary_edges()`：插入哨兵节点以强制 forward->backward->optimizer 顺序
- `summary()`：算子类型计数

---

## 6. 执行流程（`trainer_runner.py`）

`run_trainer_simulation()` 是主编排器：

```
1. 设置补丁
   - trainer.device 设为 meta
   - Mock _local_scalar_dense, FakeTensor.__format__
   - No-op clip_grad_norm_, dist_sum, dist_max
   - No-op optimizer.step(), lr_schedulers.step()
   - No-op parallel_dims mesh 访问

2. 捕获
   - 从 dataloader 预取 batch（在 FakeTensorMode 之外）
   - unified_trace() 上下文 + TraceRecorder
   - _patch_backward_phase() 将 recorder 阶段设为 "backward"
   - Trainer.train_step() 执行完整一步

3. 后处理
   - result = recorder.build_result()
   - [仅 fake_backend] 注入合成通信事件：
     - FSDP2 all_gather / reduce_scatter（每层）
     - TP all_reduce（每层）
     - PP send/recv（每 microbatch）
   - [semantic_schedule] 从配置注入完整 PP/TP/DP 调度
   - [cost_model] 应用 CostModel（MockCostModel 或自定义）
   - 内存估算（图 + 通信 + 模型状态）
   - 扩展钩子（postprocess_extension_result）

4. 导出
   - _export_result() -> JSON, DOT, Chrome Trace, HTML, Text, CSV
   - _export_workload_graph() -> workload_graph.json（L0-L3 IR）
```

---

## 7. 设备环境层

两种补丁模式实现无 GPU 执行：

### 7.1 Meta 模式（`meta_env.py`）

- 用于 `fake_backend`（无真实通信）
- `patch_device_type_to_meta()` -- 将 `device_type` 重定向到 `"meta"`，创建 0 字节张量
- 能够以极少 RAM 仿真任意大模型（如 1T+ 参数）
- 补丁 `torchtitan.tools.utils.device_type`、`device_module` 及下游模块
- 补丁 `torch.cuda.*` 入口点为 meta 桩

### 7.2 CPU 模式（`cpu_env.py`）

- 用于 `gloo` 后端（真实 CPU 通信捕获）
- `patch_device_type_to_cpu()` -- 将 `device_type` 重定向到 `"cpu"`
- 创建真实 CPU 张量（gloo 张量交换所需）
- `init_cpu_distributed()` -- 为单进程仿真设置 gloo 进程组
- 补丁 `torch.cuda.*` 入口点为 CPU 桩

### 设备模式选择逻辑

```
comm_backend == "gloo"  -->  device_mode = "cpu"    # 需要真实张量
comm_backend == ""      -->  device_mode = "meta"   # 0 字节，支持超大模型
（可通过 SimulationConfig.device_mode 覆盖）
```

---

## 8. 捕获层（`capture/`）

捕获层是单一模块（`unified_trace.py`），整合了所有 trace 捕获功能：

### 8.1 TraceRecorder

中心记录器，累积：
- `nodes: list[OpNode]` -- 带元数据的被 dispatch 算子
- `edges: list[(src, dst, type)]` -- 通过张量生产者-消费者追踪的数据流依赖
- `comm_events`、`fsdp_events`、`pp_events` -- 专用事件列表
- `build_result()` -- 将所有内容组装为 `SimulationResult`

### 8.2 UnifiedTraceMode

`TorchDispatchMode` 子类，拦截每个被 dispatch 的算子：
- 记录输入/输出张量元数据
- 通过 `id(tensor) -> node_id` 追踪张量生产者-消费者
- 通过 `op_classification.classify_op()` 分类算子
- 过滤琐碎算子（detach、alias、view 等）

### 8.3 CommRecorder

拦截 `torch.distributed` 函数：
- `all_reduce`、`all_gather`、`reduce_scatter`、`all_to_all`、`send`、`recv`、`broadcast`、`barrier`
- 同时补丁 `torch.distributed._functional_collectives`（FSDP2 和 DTensor 使用）
- 记录张量元数据、组大小、PP stage、microbatch
- 通过 `get_current_recorder()` 解析活跃 `TraceRecorder` 的源节点引用

### 8.4 FSDPEventRecorder

为 FSDP 包装的模块附加 PyTorch 模块 hooks：
- `forward_pre_hook` -> allgather 事件
- `forward_hook` -> reshard 事件
- `backward_pre_hook` -> allgather 事件
- `backward_hook` -> reduce-scatter 事件

### 8.5 unified_trace() 上下文管理器

编排所有捕获组件：

```python
with unified_trace(recorder, use_fake_mode=True, capture_comm=False):
    output = model(*inputs)
    recorder.current_phase = "backward"
    output.sum().backward()
```

- 激活 `FakeTensorMode`（若 `use_fake_mode=True`）进行仅形状计算
- 激活 `UnifiedTraceMode` 进行算子级捕获
- 可选择性地激活 `CommRecorder`（用于 gloo 后端模式）
- 可选择性地激活 `FSDPEventRecorder`（用于 gloo 后端模式）
- 通过 `recorder.current_phase` 追踪阶段（可变，由 `_patch_backward_phase` 设置）

---

## 9. 调度层（`schedule/`）

### 9.1 调度提取（`schedule_extract.py`）

**核心策略**：使用 `MockPipelineStage` 实例构建真实的 PyTorch `_PipelineSchedule` 对象，并读取其 `pipeline_order_with_comms` action 表。

这保证了提取的调度与上游 PyTorch 行为完全一致，且自动适配调度算法变更，无需修改仿真器代码。

`MockPipelineStage` -- 鸭子类型 mock，满足 `_PipelineSchedule.__init__` 的属性读取，不调用 `dist.get_rank()` / `dist.get_world_size()`。

支持的调度类型：`torch.distributed.pipelining.schedules` 中注册的所有调度 -- 1F1B、GPipe、Interleaved1F1B、LoopedBFS、InterleavedZeroBubble、ZBVZeroBubble、DualPipeV 以及 CSV 驱动的运行时调度。

Action 类型映射（`_ComputationType` -> `event_type`）：

| Action | 事件类型 |
|--------|----------|
| `F` | `pp_forward` |
| `B` | `pp_backward` |
| `UNSHARD` | `fsdp2_all_gather` |
| `RESHARD` | `fsdp2_reduce_scatter` |
| `SEND_F` | `pp_send_activation` |
| `RECV_F` | `pp_recv_activation` |
| `SEND_B` | `pp_send_gradient` |
| `RECV_B` | `pp_recv_gradient` |
| `REDUCE_GRAD` | `dp_gradient_sync` |

### 9.2 调度生成器（`schedule_generator.py`）

从并行配置生成语义 Interleaved1F1B 调度，无需任何真实调度对象。当 `semantic_schedule=True` 时使用。

产生完整的多 rank 拓扑：
- PP send/recv 对（跨 rank）
- FSDP2 all-gather/reduce-scatter（每 DP 组）
- TP all-reduce（每 TP 组）
- DP 梯度同步

### 9.3 PP 调度提取器（`pp_schedule_extractor.py`）

`PPScheduleExtractor` 类，从已有的 `_PipelineSchedule` 实例提取。主路径委托给 `schedule_extract._convert_pipeline_order_to_training_schedule()`。回退方案：启发式 1F1B 重建。

---

## 10. Cost Model（`cost_model.py`）

### 架构

```
CostModel (ABC)
  |
  +-- estimate_node(OpNode) -> PerfResult
  +-- estimate_graph(ComputeGraph) -- 就地标注每个节点
  +-- predict_step_time_us(ComputeGraph) -- 关键路径分析
  |
  +-- MockCostModel (具体实现)
        - compute_time = flops / (tflops * 1e6)
        - comm_time = latency + bytes / (comm_gb_per_s * 1e3)
        - 当算术强度 < 阈值时应用 memory-bound 上限
        - 高斯噪声增加真实性
```

### FLOPs 估算

按算子类别的启发式规则：
- **矩阵乘法类**（`mm`、`matmul`、`bmm`、`addmm`）：`2 * batch * M * K * N`
- **注意力**（`scaled_dot_product`、`flash_attention`）：`2 * numel(Q) * K_seq + 2 * numel(Q) * V_dim`
- **激活函数**（gelu、silu、sigmoid 等）：每个输出元素 1-5 FLOPs
- **归一化**（rms_norm、layer_norm）：每个元素 5 FLOPs
- **DeepEP/MoE**（`deepep.dispatch`、`deepep.combine`）：粗略近似

### 通信字节估算

Ring 算法缩放：
- `all_gather`：`input_bytes * (P-1)/P`
- `reduce_scatter`：`input_bytes * (P-1)/P`
- `all_reduce`：`output_bytes * 2*(P-1)/P`
- `send`/`recv`：原始张量字节数

### `apply_cost_model()`

便捷函数：
1. 运行 `cost_model.estimate_result(result)` 标注所有节点
2. 计算每阶段时间汇总
3. 运行 `_critical_path_time_us()` 预测步时间
4. 将结果存储到 `result.metadata["cost_model"]`

---

## 11. DES 引擎（`des_engine.py`）

### 架构

使用 **salabim** 离散事件仿真库建模资源竞争：

```
每个 rank 的资源：
  - compute（容量=1）：处理 compute、data_move、memory 算子
  - comm   （容量=1）：处理 comm_collective、comm_p2p 算子
```

不同引擎上的算子可以重叠（模拟 GPU 计算/通信并行）。
同一引擎上的算子被串行化（资源竞争）。
DAG 依赖通过 salabim `State` 信号建模。

### 两个仿真层级

1. **单 rank DES**（`simulate_single_rank_des`）：
   - 计算图的拓扑排序
   - 每个节点成为带资源请求的 `_OpComponent`
   - 返回所有节点的最大完成时间

2. **多 rank DES**（`simulate_multi_rank_des`）：
   - 使用 `TrainingSchedule` 事件和依赖
   - 每个事件成为 `_ScheduleEventComponent`
   - 持续时间通过 `link_schedule_to_graph()` 从关联的 `OpNode.perf_result` 派生
   - 每个 rank 独立的 compute/comm 资源
   - 跨 rank 依赖（PP send/recv）创建 rank 间同步

### `DESEngine` 类

```python
engine = DESEngine()
step_time = engine.predict_step_time_us(result, cost_model)
engine.annotate(result)  # 写入 result.metadata["des_engine"]
```

### 利用率分析

`compute_des_utilization()` 计算：
- `e2e_step_time_us`：总仿真步时间
- `compute_busy_pct` / `comm_busy_pct`：引擎利用率
- `overlap_pct`：计算/通信重叠
- `contention_count`：因资源竞争而延迟的算子数
- `des_vs_cp_ratio`：DES 时间 vs 关键路径时间（量化竞争影响）

---

## 12. 内存估算（`memory_estimator.py`）

三个估算来源：

1. **图内存**（`estimate_graph_memory`）：
   - 来自节点输出的激活值/输出内存
   - 生命周期 = 生产者节点索引到最后消费者节点索引
   - 类别：`activation`、`comm_buffer`、`allocation`、`data_move`

2. **通信内存**（`estimate_comm_memory`）：
   - 来自通信事件的通信缓冲区内存
   - 类别：`comm_event_buffer`

3. **模型状态内存**（`estimate_model_state_memory`）：
   - 来自 `model.named_parameters()` 的参数内存
   - 优化器状态内存（Adam：2x 参数大小用于动量 + 方差）
   - 考虑 FSDP 分片（除以 `dp_shard` 度数）

---

## 13. 分层 IR（`ir/`）

四层中间表示，将捕获数据投影为面向下游硬件仿真器的规范对齐格式：

```
L0: SpecOpNode       -- 带代价和邻接关系的单个算子
      |
L1: StepGraph        -- 每阶段 DAG 模板（forward/backward/optimizer）
      |
L2: ScheduleGraph    -- StepInstance 编排、DataPass、并行度数
      |
L3: WorkloadGraph    -- 迭代语义、数据流、跨迭代传递
```

### 投影流水线

```
SimulationResult
  |
  +-- build_step_graphs()     --> dict[str, StepGraph]       (L1)
  +-- build_schedule_graph()  --> ScheduleGraph              (L2)
  +-- build_workload_graph()  --> WorkloadGraph              (L3)
```

所有投影都是**只读**的 -- 从不修改捕获的图。

### L0: SpecOpNode

- `flops`、`peak_mem`、`param_mem`、`comm_bytes` -- 代价字段
- `predecessors`、`successors` -- 邻接关系（仅数据边）

### L1: StepGraph

- 按阶段划分捕获图
- 计算 `entry_nodes`、`exit_nodes`、`tensor_lifetimes`
- 通过 Kahn 算法验证无环性

### L2: ScheduleGraph

- `StepInstance` -- (step_ref, micro_batch_idx, pipeline_stage, device_ids, dp_group)
- `DataPass` -- 步骤实例间的张量流（带 `comm_primitive`）
- 并行度数：pp、tp、dp、ep、cp

### L3: WorkloadGraph

- `IterationSpec` -- 用 microbatch 数包装 ScheduleGraph
- `DataFlow` -- 数据加载器输入/输出描述
- `cross_iter_passes` -- 跨迭代的张量传递（如 optimizer -> 下一个 forward）

---

## 14. 导出系统（`export.py`）

| 格式 | 文件 | 说明 |
|------|------|------|
| JSON | `simulation_result.json` | 完整结构化转储，>10K 节点时使用紧凑格式 |
| DOT | `compute_graph.dot` | Graphviz 格式，按算子类型着色 |
| Chrome Trace | `trace.json` | `chrome://tracing` 兼容时间线 |
| HTML | `trace.html` | 自包含交互式可视化 |
| Text | `summary.txt` | 人类可读的统计信息 |
| CSV | `kernel_summary.csv` | 每算子 kernel trace（nsys/msprof 兼容） |
| Workload Graph | `workload_graph.json` | L0-L3 IR 投影 |

### HTML 可视化

HTML trace 自包含（无 CDN 依赖），包含：
- 泳道调度视图（按阶段、按策略：PP/FSDP/TP/DP/Optimizer）
- 可展开详情的算子 DAG
- 阶段边界标记
- Cost Model 激活时的 DES 时间叠加

---

## 15. 通信捕获策略

### 15.1 fake_backend 模式（默认）

无真实分布式通信。合成通信事件在捕获后**注入**：

```
_inject_synthetic_comm_events()
  |
  +-- FSDP2 all_gather / reduce_scatter（每层，每 PP stage）
  |     shape: shard_numel / num_layers（每个分片）
  |     group_size: dp_shard 度数
  |
  +-- TP all_reduce（每层 2 次，forward + backward）
  |     shape: batch * seq_len * hidden
  |     group_size: tp 度数
  |
  +-- PP send/recv（每 microbatch，每 stage 边界）
        shape: batch * seq_len * hidden
        group_size: 2（相邻 stage）
```

### 15.2 gloo 模式

真实 CPU 通信捕获：
- 通过 `_apply_fsdp1_on_cpu()` 在 init 之后应用 FSDP1 包装
- `CommRecorder` 拦截真实的 all-gather/reduce-scatter 调用
- 需要 `init_cpu_distributed()` 建立进程组
- 单进程即可（`init_distributed` 使用 `FakeProcessGroup`）

---

## 16. CPU 上的流水线并行

### 语义流水线（`_cpu_semantic_pipeline`）

用于 fake_backend 模式下 PP > 1：
1. 使用上游 `_generate_llm_fqn_per_model_part` + `_split_module` 将模型拆分为 PP stage
2. 立即将部件移至 meta 设备（避免 1T+ 模型的 OOM）
3. 返回 `MockSchedule`（no-op step）和模型部件列表
4. 真实的 PP 调度稍后通过 `_inject_semantic_schedule()` 提取

### Gloo 流水线

用于 gloo 模式下 PP > 1：
- `_cpu_pp_module_split` 在 CPU 上将模型拆分为 stage
- 使用真实 gloo 通信的上游调度对象

---

## 17. 模型配置注册表

每个模型有自己的 `config_registry.py`，返回 `SimulationTrainer.Config`：

### Llama3

| 配置 | 拓扑 | 说明 |
|------|------|------|
| `llama3_sim_debugmodel` | 1 GPU | 小模型，gloo 通信，cost model |
| `llama3_sim_1024gpu` | PP=4, TP=8, DP_shard=4, DP_repl=8 | 1024 GPU 语义仿真 |

### DeepSeek V4

| 配置 | 拓扑 | 说明 |
|------|------|------|
| `deepseek_v4_sim_smoketest` | PP=2, TP=2, DP=2 | 2 层冒烟测试（8 rank） |
| `deepseek_v4_pro_sim_smoketest` | PP=8, TP=8, EP=192 | 61 层全并行 |

---

## 18. 扩展系统（`extension_hooks.py`）

两个钩子供外部 side-load 使用（如 torchtitan-npu）：

1. `collect_extension_metadata(trainer, capture)` -- 如果实现了则调用 `trainer.collect_simulation_metadata(capture)`
2. `postprocess_extension_result(result, trainer, sim_opts)` -- 如果实现了则调用 `trainer.postprocess_simulation_result(result, sim_opts)`

使用鸭子类型避免核心仿真器代码的导入依赖。

---

## 19. 关键设计决策

### 19.1 为什么只保留单一入口（`SimulationTrainer`）而不是 `Simulator` 类？

- **一致性**：所有仿真运行走相同的 `Trainer` 初始化路径，确保模型构建、并行化和配置与真实训练完全一致
- **配置驱动**：`SimulationConfig` 是与 torchtitan 配置系统集成的 dataclass，支持 CLI 覆盖和配置注册表
- **无 API 漂移**：单一入口点防止两个 API 随时间分化

### 19.2 为什么只保留单一捕获模块（`unified_trace.py`）？

- **无循环依赖**：`CommRecorder` 和 `FSDPEventRecorder` 内联到与 `TraceRecorder` 相同的文件中，消除了跨模块导入循环
- **单一真实来源**：所有捕获逻辑（dispatch 拦截、通信记录、FSDP hooks）集中在一个地方
- **简化维护**：无需在多个捕获模块间同步接口

### 19.3 为什么使用 `FakeTensorMode` + `TorchDispatchMode`？

- **零内存**：仅形状的张量使得仿真 1T+ 参数模型成为可能
- **单次遍历**：所有捕获组件组合在一个上下文管理器中
- **Dispatch 层**：捕获所有 ATen 算子，包括被 autograd/FSDP 隐藏的

### 19.4 为什么 Mock PyTorch 调度对象？

- 精确复用上游调度算法实现
- 自动适配新调度类型（ZeroBubble、DualPipeV 等）
- 避免脆弱地重新实现复杂调度逻辑

### 19.5 为什么合成通信注入？

- `fake_backend` 模式避免多进程复杂性
- 通信形状/大小从实际模型参数派生
- 无需真实数据交换即可满足性能估算需求

### 19.6 为什么使用 salabim DES？

- 建模 compute/comm 资源竞争（GPU 有独立引擎）
- 处理跨 rank 同步（PP send/recv 依赖）
- 产生考虑了重叠和竞争的真实步时间预测

### 19.7 为什么使用分层 IR（L0-L3）？

- 关注点分离：算子级 -> 步骤级 -> 调度级 -> 工作负载级
- 面向下游硬件仿真器的规范对齐
- 所有投影从捕获数据派生，不重新实现逻辑

---

## 20. 第四阶段 — 自然通信捕获

### 20.1 问题陈述

当前的合成通信注入（`_inject_synthetic_comm_events()`）使用启发式假设来生成通信算子：
- 假设层均匀分布
- 手动计算张量大小
- 硬编码通信模式（例如，每层 2 次 TP all_reduce）

这种方法不是捕获忠实的，可能产生不准确的通信模式。

### 20.2 实验结果

一系列实验测试了 DTensor/FSDP2 在 meta device 上使用 FakeProcessGroup 运行时是否能自然发射通信算子：

| 实验 | 方法 | 结果 |
|------|------|------|
| 1a | FSDP2 on meta | ❌ `_validate_no_meta_params` 拒绝 meta 参数 |
| 1b | DTensor TP on meta (ws=1) | ✅ 成功但 0 个通信算子（ws=1 无需通信） |
| 1b | DTensor TP on meta (ws=4) | ✅ **自然发射 2 个通信算子**（`all_reduce`、`wait_tensor`） |
| 1b | FSDP2 on CPU (ws=4) | ✅ **自然发射 10 个通信算子**（all_gather、reduce_scatter） |
| 1c | FSDP2 on meta + patch 验证 | ❌ Meta/cpu 设备传播不匹配 |
| 1d | 多种 patching 方法 | ❌ 全部因深层不兼容而失败 |
| 1e | Patch `FakeTensor._find_common_device` | ❌ 设备不匹配错误 |
| 1f | Patch `wrap_meta_outputs_with_default_device_logic` | ✅ **FSDP2 on meta 自然发射 10 个通信算子** |

### 20.3 关键发现

**DTensor TP 在 meta device + FakeProcessGroup(world_size>1) 下自然发射通信算子：**
- `all_reduce` 用于张量并行归约
- `wait_tensor` 用于异步操作同步
- 通信形状匹配实际张量维度
- 无需启发式代码

**FSDP2 在 meta device 上可通过三个针对性 patch 实现：**
1. **Patch `FSDPParamGroup._validate_no_meta_params`** — 跳过拒绝 meta 参数的验证
2. **Patch `FakeTensor._find_common_device`** — 通过优先选择 meta 设备允许 FSDP 算子使用混合 meta/cpu 张量
3. **Patch `FakeTensorMode.wrap_meta_outputs_with_default_device_logic`** — 将 CPU 张量转换为 meta 张量用于 FSDP 内部缓冲区

**FSDP2 自然发射所有通信算子：**
- 前向：`fsdp.all_gather_copy_in` + `c10d._allgather_base_`（每个 FSDP 组）
- 反向：all_gather（用于梯度计算）+ `c10d._reduce_scatter_base_`（用于梯度归约）
- 通信形状从实际模型参数维度派生

### 20.4 设计变更

#### 20.4.1 新模块：`meta_device_patches.py`

封装三个 FSDP2 meta device patches：

```python
def apply_meta_device_patches():
    """启用 FSDP2 在 meta device 上运行以进行仿真。"""
    # Patch 1: 跳过 meta 参数验证
    FSDPParamGroup._validate_no_meta_params = lambda self: None
    
    # Patch 2: 允许 FSDP 算子使用混合 meta/cpu 张量
    FakeTensor._find_common_device = _patched_find_common_device
    
    # Patch 3: 将 CPU 张量转换为 meta 张量用于 FSDP 内部缓冲区
    FakeTensorMode.wrap_meta_outputs_with_default_device_logic = _patched_wrap

def restore_meta_device_patches():
    """恢复原始 FSDP2 行为。"""
    # ... 恢复原始函数
```

#### 20.4.2 修改 `trainer.py`

用真实并行化替换 `_cpu_noop_parallelize` 桩：

```python
def _meta_parallelize_llama(model: Any, parallel_dims: Any, **kwargs: Any) -> Any:
    """在 meta device 上应用真实 TP/EP，跳过 FSDP。"""
    # 应用 TP（DTensor placement）— 在 meta 上有效
    if parallel_dims.tp_enabled:
        apply_non_moe_tp(model, parallel_dims.get_mesh("tp"), ...)
    
    # 应用 EP（DTensor placement）— 在 meta 上有效
    if parallel_dims.ep_enabled:
        apply_moe_ep_tp(model, ..., ep_mesh=parallel_dims.get_optional_mesh("ep"))
    
    # 跳过 FSDP（fully_shard）— 稍后如需应用
    # FSDP 需要真实张量，但我们将通过自然发射捕获通信算子
    return model
```

#### 20.4.3 修改 `trainer_runner.py`

删除合成 FSDP/TP 通信注入：

```python
def run_trainer_simulation(trainer: Any, sim_opts: Any) -> None:
    # ... 现有代码 ...
    
    # 应用 meta device patches
    from .meta_device_patches import apply_meta_device_patches
    apply_meta_device_patches()
    
    # 运行 unified_trace — TP/FSDP 通信算子自然发射
    with unified_trace(recorder, model_parts, ...):
        output = trainer.train_step()
        output.backward()
    
    # 恢复 patches
    from .meta_device_patches import restore_meta_device_patches
    restore_meta_device_patches()
    
    # 删除：_inject_synthetic_comm_events(...) — TP/FSDP 不再需要
    
    # 保留：PP send/recv 注入（PP 使用 dist.send/recv，不走 DTensor）
    if parallel_dims.pp > 1:
        _inject_pp_send_recv(result, trainer.config)
```

### 20.5 收益

| 方面 | 当前（合成） | 第四阶段（自然） |
|------|-------------|-----------------|
| **代码** | ~400 行注入逻辑 | ~50 行 meta patches |
| **TP 精度** | 启发式（每层 2 次 all_reduce） | 精确（来自 DTensor dispatch） |
| **FSDP 精度** | 手动形状计算 | 精确（来自 FSDP2 内部） |
| **EP 支持** | 未实现 | 自动（基于 DTensor） |
| **维护** | 为新并行类型更新 | 自动（PyTorch 处理） |

### 20.6 限制

- **PP send/recv 仍需注入** — 流水线并行直接使用 `dist.send()`/`dist.recv()`，不走 DTensor dispatch
- **Meta device patches 脆弱** — 依赖可能变化的 PyTorch 内部实现细节
- **需要 FakeProcessGroup(world_size>1)** — 无法用 world_size=1 模拟通信

### 20.7 实施计划

1. **创建 `meta_device_patches.py`** — 封装三个 patches
2. **修改 `trainer.py`** — 用 `_meta_parallelize_*` 替换 `_cpu_noop_parallelize`
3. **修改 `trainer_runner.py`** — 应用 patches，删除 TP/FSDP 注入
4. **更新测试** — 验证自然发射产生正确的通信算子
5. **E2E 验证** — 运行 DeepSeek V4 Pro 仿真，与当前结果比较
