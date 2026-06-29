# Simulator 分层 IR 重构设计（L0–L3）

对齐 `ZhanluModelSim/workload-model-platform` 的 `spec/` 四层 IR，重构本仓
`torchtitan/experiments/simulator/` 对 forward/backward/optimizer 子图结构与
Trainer 调度流水的建模方式。

## 1. 背景与约束

### 1.1 目标
- 把 forward/backward/optimizer **子图结构**按 spec 的 **L1 StepGraph** 重新组织
  （当前是单个 `ComputeGraph` 用 `phase` 字段混在一起）。
- 把 **Trainer 调度流水**按 spec 的 **L2 ScheduleGraph / L3 WorkloadGraph** 抽象
  （当前 `TrainingSchedule` 是另一套较粗的事件模型，缺少 StepInstance/DataPass/
  迭代语义）。

### 1.2 关键约束（用户明确）
> 训练过程的抽象主要靠对 torchtitan 原生 workflow 管线的**捕获或 hook 获取**，
> **不允许采用代码复刻方式**进行，避免复刻失真。

因此新 IR 必须是对**已捕获数据的投影 / 派生**，不得重新实现 torchtitan 的训练/
并行逻辑：
- 阶段边界（fwd/bwd/opt）来自已有的 autograd backward hook + optimizer.step
  包装（`_patch_backward_phase` 等），不是手写规则。
- PP 结构来自对 torchtitan 真实 PP schedule 对象的提取
  （`schedule/pp_schedule_extractor.py` 读取 `pipeline_order`）。
- 并行度（dp/tp/pp/ep/cp）来自 `config.parallelism` 声明值，不复刻推导逻辑。

## 2. 现状 → spec 映射

| spec 层 | spec 结构 | 当前实现 | 差距 |
|---|---|---|---|
| L0 | `OpNode`(flops/peak_mem/param_mem/comm_bytes/preds/succs) | `nodes.OpNode` + `PerfResult` + `DataEdge` | 字段为 side-object，需投影 |
| L1 | `StepGraph`(每步 DAG 模板 + entry/exit/lifetimes/totals) | 单 `ComputeGraph`，phase 混合 | **缺失**，需切分 |
| L2 | `ScheduleGraph`(StepInstance/DataPass/并行度) | `TrainingSchedule`(Event/Dep) | 模型不同，需新建 |
| L3 | `WorkloadGraph`(迭代语义 + DataFlow) | `SimulationResult`(扁平容器) | **缺失**，需新建 |

## 3. 方案：新增 `ir/` 子包（投影层），不破坏现有捕获链路

### 3.1 取舍
- **命名**：采用 spec 命名，置于新子包 `simulator/ir/`，避免污染既有 `nodes.py`，
  保持向后兼容（现有导出/可视化不变）。
- **构建方式**：纯投影。输入是已捕获的 `SimulationResult`（含 `ComputeGraph`、
  `schedule`、`pp_events`、`metadata`）+ `config`，输出 spec 四层结构。
- **synthetic 注入**：保留为**显式标注的 fallback**（`attrs.synthetic=True`），
  投影层会标记其来源，不把 fallback 当作真实捕获。

### 3.2 文件结构
```
simulator/ir/
  __init__.py          # 导出 build_workload_graph + 四层 dataclass
  op_node.py           # L0 投影：from_captured(OpNode, edges) -> SpecOpNode
  step_graph.py        # L1：StepGraph + StepBuilder.from_compute_graph(...)
  schedule_graph.py    # L2：StepInstance/TensorSlot/DataPass/ScheduleGraph + ScheduleBuilder
  workload_graph.py    # L3：DataFlow/IterationSpec/WorkloadGraph + WorkloadBuilder
  builder.py           # 顶层 orchestrator：build_workload_graph(result, config)
```

### 3.3 L1 StepBuilder（capture-derived）
输入：`ComputeGraph`（已捕获）。
步骤：
1. 按 `node.phase ∈ {forward, backward, optimizer}` 分区（phase 来自 hook 捕获）。
2. 每个分区构建一个 `StepGraph`：
   - `nodes`：该 phase 的 OpNode（投影为 L0）。
   - `entry_nodes`：分区内入度为 0 的节点。
   - `exit_nodes`：分区内出度为 0 的节点。
   - `tensor_lifetimes`：由 producer→last-consumer 的拓扑序号区间派生。
   - `total_flops/param_mem/comm_volume/peak_active_mem`：聚合自 `PerfResult` +
     comm 元数据。
   - `is_acyclic`：Kahn 拓扑排序校验。
3. 仅使用分区内部边；跨 phase 边留给 L2/CTRL。

### 3.4 L2 ScheduleBuilder（capture-derived）
输入：L1 三个 StepGraph 模板 + 已捕获 `TrainingSchedule`/`pp_events` + `config.parallelism`。
步骤：
1. `step_templates = {forward, backward, optimizer}`。
2. `instances`：从捕获的 PP schedule 事件（fwd/bwd per microbatch per stage）生成
   `StepInstance`（micro_batch_idx/pipeline_stage/device 来自捕获事件）。无 PP 时退化为
   单实例 fwd→bwd→opt。
3. `data_passes`：从已捕获的跨 stage p2p 边 / pp_events 派生
   （PP 相邻 stage、PP 反向、Bwd→Opt）。
4. 并行度字段直接取 `config.parallelism` 声明值。
5. `pipeline_schedule` 取 `config.parallelism.pipeline_parallel_schedule`。

### 3.5 L3 WorkloadBuilder（capture-derived）
输入：L2 ScheduleGraph + `config` + 捕获到的 Trainer 元数据。
步骤：
1. `workload_type="train"`。
2. `iteration=IterationSpec(schedule=..., microbatch_count=梯度累积步数)`。
   梯度累积步数优先取自捕获的 `result.metadata["gradient_accumulation_steps"]`
   （由 `trainer.gradient_accumulation_steps` 注入，capture-faithful），
   缺失时回退到 `config.training`。
3. `num_iterations=config.training.steps`，`warmup`/dataloader 取 config。
4. `data_inputs`：从 dataloader 配置派生 `DataFlow`（seq_len/batch/vocab）。
5. `cross_iter_passes`：optimizer exit(param) → 下轮 forward entry(param)。

### 3.6 L0 投影
`SpecOpNode` 字段从现有数据派生：
- `flops=perf_result.flops`，`comm_bytes` 来自 comm tensor 体积，
  `param_mem` 来自 is_parameter 输入，`peak_mem` 来自输出张量字节。
- `predecessors/successors` 从 `edges` 反查。
- 不改动捕获层 `OpNode`，仅新增只读投影。

## 4. 接入点
- `trainer_runner.py`：在 `apply_cost_model` 之后、导出之前，调用
  `build_workload_graph(result, trainer.config)`，把 L1/L2/L3 投影写为
  `workload_graph.json`（新增产物，不改既有 json/html/dot）。
- 现有 `export.py` 可视化与统计**不变**，保证回归无损。

## 5. 测试策略（TDD）
- `StepBuilder`：构造小型混合-phase ComputeGraph，断言切出 3 个 StepGraph、
  entry/exit、Kahn `is_acyclic`、totals 聚合正确。
- `ScheduleBuilder`：用合成 PP 事件 + 并行度 config，断言 StepInstance 数量、
  DataPass 生成、并行度透传。
- `WorkloadBuilder`：断言 iteration/num_iterations/data_inputs 来自 config。
- L0 投影：断言 preds/succs/flops/comm_bytes 派生正确。
- E2E：DeepSeek-v4-Pro fake_backend 跑通并产出 `workload_graph.json`，
  L1 三步存在、L2 instance 数与 pp×microbatch 一致。

## 6. 非目标（YAGNI）
- 不重写捕获链路（unified_trace/dispatch/comm）。
- 不改既有 HTML/DOT/Chrome trace 输出格式。
- 不实现 inference/rag/recommendation 形态（仅 train）。
- 不移除 synthetic fallback（仅显式标注来源）。

## 7. 向后兼容
- 现有公共 API（`Simulator`、`SimulationResult`、`export_*`）保持不变。
- 新增 `simulator.ir` 命名空间与一个新产物文件；旧产物不变。
