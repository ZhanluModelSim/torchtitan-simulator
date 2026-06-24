# TorchTitan Simulator -- 架构设计文档

> **状态说明：** 本文档反映截至最新提交的**实际实现（as-built）**状态。各主要阶段的
> 演进历史内嵌在相应章节中，便于评审者理解系统如何演进。第 [21](#21-技术债--已知差异)
> 节列出了已知的代码与文档漂移，以驱动评审讨论。

## 1. 概述

TorchTitan Simulator 是一个纯 CPU 训练 trace/仿真系统，作为 side-loaded experiment 构建在
torchtitan LLM 训练平台之上。它能够在**无任何 GPU 硬件**的环境下，捕获前向/反向计算图、
通信模式和训练调度（PP、FSDP、TP、DP），从而实现：

- **任意规模的训练步分析**（如 1024 GPU 拓扑）在单台 CPU 机器上完成
- **性能预测**：基于 Cost Model 和离散事件仿真（DES）
- **并行策略探索**：PP/TP/DP/FSDP 各种度数组合，无需真实硬件
- **工作负载图导出**：供下游硬件仿真器（ZhanluModelSim 等）使用

### 设计原则

1. **Side-loaded 实验** -- `train.py` 保持不变；所有仿真器代码位于 `torchtitan/experiments/simulator/`
2. **捕获忠实** -- 调度和计算图从*捕获数据*（或真实 PyTorch 调度对象）派生，而非重新实现训练逻辑
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
  meta_device_patches.py   # Phase 4：3 个 FSDP2 补丁，启用 meta 上的自然通信
  synthetic_dataloader.py  # SyntheticTokenDataLoader（随机 token 生成）
  extension_hooks.py       # NPU/其他 side-load 的扩展点

  capture/                 # Trace 捕获（单一统一模块）
    __init__.py
    unified_trace.py       # FakeTensorMode + TorchDispatchMode + CommRecorder + FSDP hooks

  schedule/                # 训练调度提取与生成
    __init__.py            # 导出 PPScheduleExtractor, extract_schedule_from_pytorch
    schedule_extract.py    # 从真实 PyTorch PipelineSchedule 对象提取调度
    schedule_generator.py  # [已废弃] 语义 Interleaved1F1B 生成器（见 §9.2）
    pp_schedule_extractor.py # PPScheduleExtractor 类（读取 pipeline_order 表）

  ir/                      # 分层 IR（L0-L3）用于工作负载图导出
    __init__.py
    op_node.py             # L0: SpecOpNode 投影
    step_graph.py          # L1: StepGraph（每阶段 DAG 模板）
    schedule_graph.py      # L2: ScheduleGraph（编排：实例、数据传递）
    workload_graph.py      # L3: WorkloadGraph（迭代语义 + 数据流）
    builder.py             # IR 投影顶层编排器

  llama3/                  # Llama3 特定仿真配置
    __init__.py
    config_registry.py     # llama3_sim_debugmodel, llama3_sim_1024gpu

  deepseek_v4/             # DeepSeek V4 特定仿真配置
    __init__.py
    config_registry.py     # deepseek_v4_sim_smoketest, deepseek_v4_pro_sim_smoketest

  tests/                   # 单元测试
    __init__.py
    test_simulator.py      # 核心仿真器 + 捕获 + DES + 导出测试
    test_ir.py             # 分层 IR（L0-L3）投影测试
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

**`__init__` 中的关键行为（`trainer.py`）：**

1. 读取 `sim_opts = config.simulation` 并解析 `comm_backend`。
2. 读取并行度数（`pp`、`tp`、`dp_shard`、`dp_replicate`）；自动解析 `dp_shard=-1` -> 1。
3. 若拓扑 `pp*tp*dp_shard*dp_replicate > 1` -> 调用 `_set_fake_world_size(config)`。
4. **强制** `config.comm.mode = "fake_backend"`，使 `init_distributed` 使用 fake 进程组
   （无需 NCCL/gloo rendezvous，无需 torchrun）。
5. 若实际 CLI `comm.mode == "fake_backend"`，覆盖 `comm_backend = ""`。
6. 解析 `device_mode`：空则自动选 `"meta"`，除非 `comm_backend == "gloo"`。
7. 据此调用 `patch_device_type_to_meta()` 或 `patch_device_type_to_cpu()`。
8. meta 模式下，若未设则默认 `config.debug.seed = 42`。
9. **并行化函数选择**（见 §15、§16）：
   - `comm_backend == "gloo"` -> `_cpu_gloo_parallelize_llama` / `_cpu_gloo_parallelize_dsv4`
     （基于模型名），init 后再 FSDP1 包装。
   - 否则（默认）-> `_meta_parallelize_with_skip_fsdp(real_parallelize_fn)` --
     Phase 4 包装器，在 meta 设备上应用*真实*并行化（TP/EP/CP/FSDP2），由
     `apply_meta_device_patches()` 启用。
10. **流水线函数选择**：
    - `pp > 1` 且非 gloo -> `partial(_cpu_semantic_pipeline, ...)`（meta PP 切分）。
    - 否则 -> `_cpu_noop_pipeline`（单 stage）。
11. 调用 `super().__init__(config)`（真实 `Trainer`）。
12. 若 `_cpu_semantic_pipeline` 填充了 `self._pp_model_parts`，覆盖 `self.model_parts`。
13. 若 `comm_backend == "gloo"`：对每个 model part 应用 `_apply_fsdp1_on_cpu`。

`train()` 重新打设备补丁，然后调用 `run_trainer_simulation(self, sim_opts)`。

**`SimulationConfig`**（`trainer.py` 中的 `@dataclass(kw_only=True, slots=True)`）：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `output_dir` | `"./simulator_output"` | 导出目录 |
| `output_formats` | `["json","dot","chrome_trace","html","text","csv"]` | 导出格式 |
| `mode` | `"all"` | `"all"`、`"runtime"` 或 `"schedule"` |
| `max_seq_len` | `128` | 用于动态维度解析的序列长度 |
| `batch_size` | `2` | 用于动态维度解析的 batch size |
| `capture_joint_fx` | `False` | 联合 fwd+bwd FX 捕获 |
| `semantic_schedule` | `False` | 从配置生成完整 PP/TP/DP 调度 |
| `cost_model` | `False` | 对计算图运行 Cost Model |
| `cost_model_class` | `""` | 自定义 CostModel 类/工厂路径（空 = MockCostModel） |
| `cost_model_kwargs` | `""` | CostModel 的 JSON 字符串（CLI）或 dict（注册表） |
| `comm_backend` | `""` | `""`（fake）或 `"gloo"`（真实 CPU 通信） |
| `device_mode` | `""` | `""`（自动）、`"meta"` 或 `"cpu"` |
| `operator_swimlane_comm_scope` | `"model_only"` | `"model_only"` 在算子泳道中隐藏合成 PP/DP/FSDP 通信；`"all"` 显示全部 |

### 3.2 编程式捕获 API（实际实现）

捕获层可以直接用于编程式追踪。这是目前唯一实现的编程式 API：

```python
from torchtitan.experiments.simulator import TraceRecorder, unified_trace

recorder = TraceRecorder(rank=0)
with unified_trace(recorder, use_fake_mode=True, phase="forward"):
    output = model(*inputs)
    recorder.current_phase = "backward"
    output.sum().backward()
result = recorder.build_result()
```

> **路线图：** 原始设计设想的更高级 `Simulator` 类，含
> `simulate_fx` / `simulate_runtime` / `simulate_pp_schedule` 三种模式（`AGENTS.md` 中有引用）。
> 此类**尚未实现**。见 §22 路线图。

---

## 4. 捕获模式

仿真器使用通过 `unified_trace()` 的单一统一捕获模式。此上下文管理器在一次遍历中组合
`FakeTensorMode` 与 `TorchDispatchMode`，并可选地激活通信拦截和 FSDP 生命周期 hooks：

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
  +-- [可选] FSDPEventRecorder -- 当 capture_fsdp=True 且提供 model_parts 时
        |
        +-- 为 FSDP allgather/reshard/reduce-scatter 生命周期附加模块 hooks
```

**签名：**
```python
def unified_trace(
    recorder: TraceRecorder,
    model: torch.nn.Module | None = None,
    example_inputs: tuple | None = None,
    use_fake_mode: bool = True,
    phase: str = "forward",
    capture_comm: bool = False,
    capture_fsdp: bool = True,
    model_parts: list[torch.nn.Module] | None = None,
) -> Generator[TraceRecorder, None, None]
```
（`model` / `example_inputs` 为与 FX tracing API 对称而接受，但上下文管理器内部不使用；
模型在调用方的 `with` 体中执行。）

**TraceRecorder** 是统一记录器，累积：
- `nodes: list[OpNode]` -- 每个被 dispatch 的算子一个
- `edges: list[(src, dst, type)]` -- 数据流依赖（每次调用内去重）
- `_tensor_producer: dict[int, str]` -- `id(tensor) -> node_id`，生产者追踪核心
- `comm_events`、`fsdp_events`、`pp_events` -- 专用事件列表
- 通过可变字段追踪 Phase/PP-stage/microbatch 上下文

**`build_result()`** 将所有内容组装为带有已填充 `ComputeGraph` 的 `SimulationResult`。若无显式
边记录，则在每个 `(phase, pp_stage, microbatch_idx)` 组内推断顺序边。将 `comm_events` 合并为
`OpNode` 条目，并从 `fsdp_events`/`pp_events` 构建 `TrainingSchedule`。

模块级记录器栈（`_RECORDER_STACK` / `get_current_recorder()`）将 `CommRecorder`/
`FSDPEventRecorder` 与活跃 `TraceRecorder` 解耦，使通信事件能通过 `get_producer`/
`set_producer` 形成回到计算节点的数据流边。

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
  |     +-- events: list[ScheduleEvent]
  |     +-- deps: list[ScheduleDep]
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

**PerfResult** -- Cost Model 输出（所有时间单位 µs）：
- `compute_time_us`、`comm_time_us`、`total_time_us`、`flops`、`bytes_read`、`bytes_written`、`metadata`

**DataEdge** -- `src_node_id`、`dst_node_id`、`edge_type`（`"data"` / `"control"` / `"pp_p2p"`）、`tensor_meta`。

**ScheduleEvent** -- 粗粒度调度事件：`event_id`、`event_type`、`rank`、`pp_stage`、`pp_rank`、
`microbatch_idx`、`logical_clock`、`op_node_ids`（由 `link_schedule_to_graph` 填充）、
`des_start_time_us`、`des_finish_time_us`。

**ScheduleDep** -- `from_event_id`、`to_event_id`、`dep_type`（`"data"` / `"control"` /
`"pp_comm"` / `"fsdp_comm"`）。

**MemoryEvent** -- `event_id`、`category`、`bytes`、`phase`、`device`、`dtype`、`shape`、
`node_id`、`lifetime_start`、`lifetime_end`、`metadata`。

**ComputeGraph** 提供：
- `fix_comm_phase_labels()`：修正标记错误的通信节点阶段（如反向后触发但时间上记为 forward 的
  FSDP `reduce_scatter`）。仅当通信节点有唯一与自身不同的前驱阶段时才重标。**必须在
  `add_phase_boundary_edges` 之前运行。**
- `add_phase_boundary_edges()`：插入 `phase_end_{phase}` 哨兵节点作为 fan-in/fan-out 汇合点，
  强制 forward->backward->optimizer 顺序（捕获图仅有数据流边，否则反向节点可能在单个前向
  前驱完成时即启动）。
- `summary()`：算子类型计数

---

## 6. 执行流程（`trainer_runner.py`）

`run_trainer_simulation()` 是主编排器：

```
1. 设置补丁（捕获前）
   - trainer.device = torch.device("meta")
   - Mock _local_scalar_dense -> 0，FakeTensor.__format__ -> "0.0"
   - No-op clip_grad_norm_ -> meta tensor(0.0)，dist_sum/dist_max -> identity
   - No-op parallel_dims get_mesh/get_optional_mesh -> None
   - No-op optimizer.step()、lr_schedulers.step()
   - 包装 sl.log_trace_scalar 将张量值强制为 0

2. 捕获
   - 预取 gradient_accumulation_steps 个 batch（在 FakeTensorMode 之外）
   - 构建 mock_data_iterator() 生成器
   - use_fake = comm_backend != "gloo"
   - [use_fake] apply_meta_device_patches()   # Phase 4
   - unified_trace(recorder, use_fake_mode=use_fake,
                   capture_comm=not use_fake, capture_fsdp=not use_fake)
     + _patch_backward_phase(recorder)   # 标记反向算子
     + Trainer.train_step(mock_data_iterator())  # 完整一步
   - [finally] restore_meta_device_patches()；恢复所有补丁

3. 后处理
   - result = recorder.build_result()
   - 设置 metadata（operator_swimlane_comm_scope、gradient_accumulation_steps）
   - [use_fake] _inject_synthetic_compute_anchors(result, trainer)  # Cube/Vec 泳道
   - [use_fake] _inject_pp_send_recv(result, trainer)              # PP 仍为合成
   - [semantic_schedule] _inject_semantic_schedule(result, config)  # 真实 PyTorch 调度
   - [cost_model] apply_cost_model(result, cm)
   - 内存估算（build_runtime_memory + attach_model_state_memory）
   - postprocess_extension_result(result, trainer, sim_opts)        # 鸭子类型 hook

4. 导出
   - _export_result() -> JSON, DOT, Chrome Trace, HTML, Text, CSV（仅 rank 0）
   - _export_workload_graph() -> workload_graph.json（L0-L3 IR）
```

> **注：** `_inject_synthetic_comm_events()`（旧的约 490 行 FSDP/TP 启发式注入器）在
> `trainer_runner.py` 中**仍然定义着**，但生产路径中**不再调用** -- 仅在
> `tests/test_simulator.py` 中存活。Phase 4 自然捕获取代了它。见 §15 和 §21。

---

## 7. 设备环境层

两种补丁模式实现无 GPU 执行：

### 7.1 Meta 模式（`meta_env.py`）

- 用于 `fake_backend`（默认）。全局、基本不可逆的 monkey-patch。
- `patch_device_type_to_meta()` -- 将 `device_type` 重定向到 `"meta"`，创建 0 字节张量。
- 能够以极少 RAM 仿真任意大模型（如 1T+ 参数）。
- 补丁 `torchtitan.tools.utils.device_type`、`device_module` 及下游模块
  （`metrics`、`parallel_dims`、`distributed.utils`）。
- 为 meta 设备补丁 FSDP2 内部：`_get_device_from_mesh` -> 当 `mesh.device_type == "meta"`
  时返回 `torch.device("meta")`；重导出到 `_fsdp_init`、`_fully_shard`、`_fsdp_param_group`、
  `_fsdp_state`、`_fsdp_collectives`；补丁 `_get_device_handle`；补丁
  `FSDPParamGroup._validate_no_meta_params` -> no-op。
- 补丁 `Decoder._init_self_buffers`（meta 时 buffer_device=None）。
- 补丁 `torch.cuda.*` 入口点为 meta 桩（0 设备，报告 80GB）。

### 7.2 CPU 模式（`cpu_env.py`）

- 用于 `gloo` 后端（真实 CPU 通信捕获）。
- `patch_device_type_to_cpu()` -- 将 `device_type` 重定向到 `"cpu"`，创建真实 CPU 张量
  （gloo 张量交换所需）。
- `init_cpu_distributed()` -- 为单进程仿真设置 gloo 进程组（环境变量 `master_addr`/
  `master_port`，`dist.init_process_group(backend="gloo")`）。
- 补丁 `torch.cuda.*` 入口点为 CPU 桩（1 设备，`CPU_Simulator`）。

### 设备模式选择逻辑

```
comm_backend == "gloo"  -->  device_mode = "cpu"     # 需要真实张量
comm_backend == ""      -->  device_mode = "meta"    # 0 字节，支持超大模型
（可通过 SimulationConfig.device_mode 覆盖）
```

---

## 8. 捕获层（`capture/unified_trace.py`）

单一模块整合所有 trace 捕获。组件：

### 8.1 TraceRecorder

中心记录器（见 §4）。`record()` 构建 `OpNode`，从输入张量的生产者构造 "data" 边，并为每个
输出张量更新 `_tensor_producer`。

### 8.2 UnifiedTraceMode

`TorchDispatchMode` 子类，拦截每个被 dispatch 的算子：
- **短路** `TRIVIAL_TARGETS` 中的算子（detach、alias、view、as_strided、unsafe_view、lift、
  lift_fresh_copy、t）-- 不记录直接返回。
- 运行算子，然后记录。
- 将标量 Python 参数捕获为 `arg_{i}` 属性。
- 通过 `op_classification.classify_op()` 分类算子。

### 8.3 CommRecorder

线程安全（`threading.Lock`）。拦截 `torch.distributed` 函数：
- `all_reduce`、`all_gather`、`all_gather_into_tensor`、`reduce_scatter`、`reduce_scatter_tensor`、
  `all_to_all`、`all_to_all_single`、`send`、`recv`、`isend`、`irecv`、`broadcast`、`barrier`
  （通过 `capture_comms` 上下文的 13 个函数）。
- 同时补丁 `torch.distributed._functional_collectives`（FSDP2/DTensor 使用）：
  `all_reduce`、`all_gather_tensor`、`reduce_scatter_tensor`、`all_to_all_single`、
  `broadcast`、`wait_tensor`。
- 记录张量元数据、组大小、PP stage、microbatch。
- 通过 `get_current_recorder()` 解析活跃 `TraceRecorder` 的源节点引用。
- `all_gather_into_tensor`、`reduce_scatter` 等调用 `set_producer(output, event_id)` 将通信
  事件链接进数据流图。

### 8.4 FSDPEventRecorder

线程安全。为 FSDP 包装的模块附加 PyTorch 模块 hooks：
- `forward_pre_hook` -> `fsdp_allgather_pre_fwd`（allgather_params）
- `forward_hook` -> `fsdp_reshard_post_fwd`（reshard_params）
- `backward_pre_hook` -> `fsdp_allgather_pre_bwd`（allgather_params_for_bwd）
- `backward_hook` -> `fsdp_reduce_scatter_post_bwd`（reduce_scatter_grads）

### 8.5 unified_trace() 上下文管理器

编排所有组件（见 §4 签名）。条件性进入 `FakeTensorMode` + `UnifiedTraceMode`，然后可选
`CommRecorder`（gloo）和每个 model part 的 `FSDPEventRecorder`。退出后将 `comm_events`/
`fsdp_events` 转移到 recorder。

---

## 9. 调度层（`schedule/`）

### 9.1 调度提取（`schedule_extract.py`）-- 运行时路径

**核心策略**：使用 `MockPipelineStage` 实例构建真实的 PyTorch `_PipelineSchedule`，读取其
`pipeline_order_with_comms` action 表。

这保证提取的调度与上游 PyTorch 行为完全一致，且自动适配调度算法变更，无需修改仿真器代码。
这是 `semantic_schedule=True` 时 `_inject_semantic_schedule()` 实际使用的路径。

`MockPipelineStage` -- 鸭子类型 mock，满足 `_PipelineSchedule.__init__` 的属性读取，不调用
`dist.get_rank()` / `dist.get_world_size()`。

支持的调度类型：`Schedule1F1B`、`ScheduleGPipe`、`ScheduleInterleaved1F1B`、
`ScheduleLoopedBFS`、`ScheduleInterleavedZeroBubble`、`ScheduleZBVZeroBubble`、
`ScheduleDualPayV` 以及 CSV 驱动的 `_PipelineScheduleRuntime` -- `torch.distributed.pipelining.schedules`
中注册的所有调度。

Action 类型映射（`_ComputationType` -> `event_type`）：

| Action | 事件类型 |
|--------|----------|
| `F` | `pp_forward` |
| `B` | `pp_backward` |
| `I` | `pp_backward_input` |
| `W` | `pp_backward_weight` |
| `UNSHARD` | `fsdp2_all_gather` |
| `RESHARD` | `fsdp2_reduce_scatter` |
| `SEND_F` | `pp_send_activation` |
| `RECV_F` | `pp_recv_activation` |
| `SEND_B` | `pp_send_gradient` |
| `RECV_B` | `pp_recv_gradient` |
| `REDUCE_GRAD` | `dp_gradient_sync` |
| `OVERLAP_F_B` | `overlap_forward_backward` |

`_convert_pipeline_order_to_training_schedule()` 是共享的转换核心（两个提取入口都导入它）。
其逻辑：将 action 转为 `ScheduleEvent`；在每个 PP rank 内添加顺序 `control` 依赖；添加跨 rank
PP 通信依赖（`SEND_F(stage=s)->RECV_F(stage=s+1)`、`SEND_B(stage=s)->RECV_B(stage=s-1)`）；
将 PP 组事件复制到 TP/DP 兄弟 rank；为每个 rank 追加 `optimizer_step` 事件。

### 9.2 调度生成器（`schedule_generator.py`）-- 已废弃

从并行配置生成语义 Interleaved1F1B 调度，*无需*任何真实 PyTorch 调度对象。**此模块未从
`schedule/__init__.py` 导出，也不在运行时路径上**（仅通过向后兼容的模块别名可达）。运行时
使用 `extract_schedule_from_pytorch`（§9.1）。它作为参考实现保留。见 §21。

### 9.3 PP 调度提取器（`pp_schedule_extractor.py`）

`PPScheduleExtractor` 类，从已有的 `_PipelineSchedule` 实例提取。主路径委托给
`schedule_extract._convert_pipeline_order_to_training_schedule()`（`tp_degree=1, dp_degree=1`）。
回退：启发式 1F1B 重建。注意启发式回退产生与主路径**不同的事件类型分类法**
（`fwd`/`bwd`/`send_fwd` vs `pp_forward`/`pp_backward`/`pp_send_activation`）-- 见 §21。

---

## 10. Cost Model（`cost_model.py`）

### 架构

```
CostModel (ABC)
  +-- estimate_node(OpNode) -> PerfResult            # 抽象
  +-- estimate_graph(ComputeGraph)                    # 就地标注每个节点
  +-- predict_step_time_us(ComputeGraph)              # 关键路径分析（O(V+E)）
  +-- estimate_result(SimulationResult)               # compute_graph 上的便捷封装
  |
  +-- MockCostModel (具体实现)
        - compute_time = flops / (tflops * 1e6)
        - comm_time = latency + bytes / (comm_gb_per_s * 1e3)
        - 当算术强度 < 阈值时应用 memory-bound 上限
        - 高斯噪声增加真实性
        - 可选 OverlapStrategy
```

### `MockCostModel` 构造参数

`tflops=10.0`（FP16/BF16）、`gb_per_s=100.0`（HBM）、`comm_gb_per_s=50.0`、
`comm_latency_us=5.0`、`arithmetic_intensity_threshold=10.0`、`noise_std=0.05`、`seed=42`、
`default_seq_len=4096`、`overlap_strategy: OverlapStrategy | None = None`。

### OverlapStrategy

- `OverlapStrategy`（ABC）：`overlap_factor(compute_us, comm_us) -> float`。
- `NoOverlap`：求和（`compute + comm`）。
- `FixedOverlap(factor=0.5)`：`compute + max(0, comm - compute*factor)`。

### FLOPs 估算（`_estimate_flops`）

按算子类别的启发式规则：
- **矩阵乘法类**（`mm`、`matmul`、`bmm`、`addmm`、`linear`）：`2 * batch * M * K * N`
- **注意力**（`scaled_dot_product_attention`）：`2 * numel(Q) * K_seq + 2 * numel(Q) * V_dim`
- **DeepEP/MoE**（`deepep.dispatch`、`deepep.combine`）：粗略近似
- **激活函数**（gelu、silu、sigmoid 等）：每个输出元素 1-5 FLOPs
- **归一化**（rms_norm、layer_norm）：每个元素 5 FLOPs

### 通信字节估算（`_estimate_comm_bytes`）

Ring 算法缩放：
- `all_gather`：`input_bytes * (P-1)/P`
- `reduce_scatter`：`input_bytes * (P-1)/P`
- `all_reduce`：`output_bytes * 2*(P-1)/P`
- `send`/`recv`：原始张量字节数

### DimResolver

通过映射（`hidden_dim`、`seq_len`、`num_heads`、`num_experts`、`vocab_size`，从模型配置提取）
将符号化 shape 维度解析为具体值。使动态 shape（如 `-1` batch/seq 维度）仍能产生具体 FLOPs/bytes。

### 调度链接

- **`link_schedule_to_graph(result)`**：通过匹配 `(phase, pp_stage, microbatch_idx)` 填充
  `ScheduleEvent.op_node_ids`。对单 rank trace（节点缺少 stage/mb 标签），赋空 `op_node_ids`
  以避免跨事件重复计时。
- **`predict_multi_rank_step_time_us(result, cost_model)`**：惰性导入
  `simulate_multi_rank_des`；若无调度事件则回退到单 rank 关键路径。

### `apply_cost_model(result, cost_model=None)`

`trainer_runner.py` 调用的函数：
1. 若 `cost_model is None` -> `MockCostModel()`。
2. `cost_model.estimate_graph(result.compute_graph)`。
3. `single_rank_step` = 关键路径时间。
4. `e2e_step` = `predict_multi_rank_step_time_us`（有调度时用多 rank DES）。
5. 构建每阶段细分（`compute_time_us`、`comm_time_us`、`total_time_us`）。
6. 返回 dict：`e2e_step_time_us`、`single_rank_step_time_us`、`total_compute_time_us`、
   `total_comm_time_us`、`per_phase`。

---

## 11. DES 引擎（`des_engine.py`）

使用 **salabim** 离散事件仿真库建模资源竞争：

```
每个 rank 的资源（各容量=1）：
  - compute：处理 compute、data_move、memory 算子
  - comm：处理 comm_collective、comm_p2p 算子
```

不同引擎上的算子可以重叠（模拟 GPU 计算/通信并行）。同一引擎上的算子被串行化（资源竞争）。
DAG 依赖通过 salabim `State` 信号建模。

### 两个仿真层级

1. **单 rank DES**（`simulate_single_rank_des`）：
   - 计算图拓扑排序；每个节点成为带资源请求的 `_OpComponent`。返回所有节点的最大完成时间。

2. **多 rank DES**（`simulate_multi_rank_des`）：
   - 使用 `TrainingSchedule` 事件和依赖。每个事件成为 `_ScheduleEventComponent`。
   - 持续时间通过 `link_schedule_to_graph()` 从关联的 `OpNode.perf_result` 派生：先对匹配节点求和，
     再除以 `events_per_key`（按 microbatch 比例分摊）。
   - 每个 rank 独立的 compute/comm 资源。
   - 跨 rank 依赖（PP send/recv）创建 rank 间同步。

`_event_engine_type(event_type)` 将 PP/FSDP2/DP 事件类型映射到 `"comm"`
（`pp_send_activation`、`pp_recv_activation`、`pp_send_gradient`、`pp_recv_gradient`、
`fsdp2_all_gather`、`fsdp2_reduce_scatter`、`dp_gradient_sync`）；其余 -> `"compute"`。

### `DESEngine` 类

```python
engine = DESEngine()
step_time = engine.predict_step_time_us(result, cost_model)  # 有调度时用多 rank
engine.annotate(result)  # 写入 result.metadata["des_engine"]["e2e_step_time_us"]
```

### 利用率分析（`compute_des_utilization`）

从已标注的节点/事件计算（由 `export.py` 的 `_populate_des_metadata` 惰性缓存到
`result.metadata["des_engine"]`）：
- `e2e_step_time_us`、`single_rank_step_time_us`
- `compute_busy_pct` / `comm_busy_pct`：引擎利用率
- `overlap_pct`：计算/通信重叠（通过区间合并）
- `contention_count`：实际持续时间超过 perf 持续时间 > 0.1µs 的算子数
- `des_vs_cp_ratio`：DES 时间 vs 关键路径时间（量化竞争影响）

### DES 内存时间线（`compute_des_memory_timeline`）

将 `MemoryEvent` 生命周期映射到 DES 壁钟时间戳（缓存到
`result.metadata["des_memory"]`）。构建 alloc/free 事件的扫掠线时间线；计算
`peak_dynamic_bytes`、`peak_total_bytes` 和 `phase_peak`（每阶段峰值及类别细分）。供 HTML
内存 trace 图表使用。

---

## 12. 内存估算（`memory_estimator.py`）

三个估算来源：

1. **图内存**（`estimate_graph_memory`）：
   - 来自节点输出的激活值/输出内存。生命周期以节点顺序近似（生产者索引到最后消费者索引）。
   - 类别：`activation`、`comm_buffer`、`allocation`、`data_move`。

2. **通信内存**（`estimate_comm_memory`）：
   - 来自通信事件的通信缓冲区内存。
   - 类别：`comm_event_buffer`。

3. **模型状态内存**（`estimate_model_state_memory`）：
   - 来自 `model.named_parameters()` 的参数内存。
   - 优化器状态内存（Adam：2x 参数大小用于动量 + 方差）。
   - 考虑分片：`shard_factor = max(1, tp_degree * fsdp_degree)`；每 GPU 参数事件 =
     `nbytes // shard_factor`。
   - **EP/MoE 专家分布显式不建模。**

`build_runtime_memory()` 合并图 + 通信内存；`attach_model_state_memory()` 扩展
`result.memory_events` 并最终化 `result.metadata["memory"]`。

---

## 13. 分层 IR（`ir/`）

四层中间表示，将捕获数据投影为面向下游硬件仿真器的规范对齐格式（镜像
`ZhanluModelSim/workload-model-platform`）。**所有层都是只读投影** -- 从不修改捕获图，从不
重新实现 torchtitan 逻辑。

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
  +-- build_step_graphs(result)     --> dict[str, StepGraph]       (L1)
  +-- build_schedule_graph(result)  --> ScheduleGraph              (L2)
  +-- build_workload_graph(result)  --> WorkloadGraph              (L3，完整流水线)
```

`_gradient_accumulation(result, config)` 捕获忠实：`result.metadata["gradient_accumulation_steps"]`
中捕获的值优先于声明的配置值。

### L0: SpecOpNode（`ir/op_node.py`）

- `flops`、`peak_mem`（输出张量体积）、`param_mem`（需梯度的输入）、`comm_bytes`
  （通信算子的输出或输入体积）。
- `predecessors`、`successors` -- 仅从 DATA 边构建邻接。
- `project_op_nodes(graph)` 构建完整 L0 投影。

### L1: StepGraph（`ir/step_graph.py`）

- `StepBuilder.from_compute_graph` 按阶段划分捕获图。
- 计算 `entry_nodes`（indeg=0）、`exit_nodes`（outdeg=0）、`tensor_lifetimes`
  （以捕获顺序作为拓扑代理的跨度指标）、汇总（`total_flops`、`peak_active_mem`、`param_mem`、
  `comm_volume`）。
- 通过 Kahn 算法验证无环性（`_kahn_is_acyclic`）。

### L2: ScheduleGraph（`ir/schedule_graph.py`）

- `ScheduleBuilder.from_capture(step_templates, schedule, parallelism, ...)`。
- `StepInstance` -- (step_ref, micro_batch_idx, pipeline_stage, device_ids, dp_group)。
- `DataPass` -- 实例间的张量流（`TensorSlot`、`comm_primitive`）。
- 若有捕获的 PP 调度：从 fwd/bwd 事件构建实例，在相邻 stage 间连接
  `p2p_send_recv` 数据传递。回退：每个梯度累积 microbatch 一条 fwd->bwd->opt 链。
- 并行度数：pp、tp、dp、ep、cp。

### L3: WorkloadGraph（`ir/workload_graph.py`）

- `WorkloadBuilder.from_capture(schedule_graph, step_templates, training)`。
- `IterationSpec` 用 microbatch 数包装 ScheduleGraph。
- `DataFlow` 用于数据加载器（体积 = `batch * seq_len * dtype_bytes * ga`）。
- `cross_iter_passes` -- 如 optimizer -> 下一个 forward 的首实例。

---

## 14. 导出系统（`export.py`）

| 格式 | 文件 | 说明 |
|------|------|------|
| JSON | `simulation_result.json` | 完整结构化转储；<=10K 节点美化，否则紧凑 |
| DOT | `compute_graph.dot` | Graphviz，按算子类型着色 |
| Chrome Trace | `trace.json` | `chrome://tracing` 兼容时间线（有 DES 时双轨） |
| HTML | `trace.html` | 交互式可视化（数据内联；**ECharts 从 CDN 加载**） |
| Text | `summary.txt` | 人类可读统计信息 |
| CSV | `kernel_summary.csv` | 每算子 kernel trace（nsys/msprof 兼容） |
| Workload Graph | `workload_graph.json` | L0-L3 IR 投影 |

> **CDN 注意：** HTML trace 将所有结果数据内联嵌入，但使用 **从 `cdn.jsdelivr.net` 加载的
> ECharts 5.4.3** 渲染图表。因此并非完全离线/自包含。`export_html` docstring 中提到的
> "AntV G6" 已过时 -- 未使用 G6。

### Chrome Trace pid 布局

0=OpNodes、1=FSDP、2=PP、3=FSDP sched、4=TP sched、5=DP sched、6=Optimizer、7=聚合阶段块。
有 DES 计时时，算子拆分为 `compute_engine` / `comm_engine` 双轨。

### HTML 可视化

生成的 HTML 包含：
- 汇总指标卡片（节点/边/调度计数、内存、步时间；有计时时额外 DES 卡片）。
- 并行度行（TP、FSDP、shard_factor）。
- 内存 trace 时间线图表（每 GPU total/static、whole-model static）。
- 每个训练步的分区：
  - **PP/FSDP2/TP/DP/comm 调度泳道**（ECharts Gantt，按 rank）。
  - **每阶段算子泳道（Cube / Vec / Communication）**，实现**浏览器内双引擎 DES 调度器**
    （Cube+Vec 共享串行 Compute 引擎；Communication 在独立 Comm 引擎；Kahn 拓扑排序）。
    除非 `operator_swimlane_comm_scope == "all"`，否则过滤掉合成的集群并行通信节点。

`export_text_summary` 返回字符串（runner 写入 `summary.txt`）。

---

## 15. 通信捕获策略

仿真结果中的通信分**三层**，"自然度"递增：

### 15.1 自然捕获（默认，Phase 4）-- TP / EP / FSDP2

默认 `fake_backend` / meta 路径运行**真实**并行化（`_meta_parallelize_with_skip_fsdp`）
在 meta 设备上。在 `apply_meta_device_patches()` 激活（7 个补丁 -- 见 §20）下，
DTensor TP 和 FSDP2 通过 dispatcher **自然发射**其通信算子，`unified_trace` 精确捕获：

- **DTensor TP** 在 meta + `FakeProcessGroup(world_size>1)` 上：每次 TP 归约发射 `all_reduce` +
  `wait_tensor`。
- **FSDP2** 在 meta 上：发射 `all_gather`（前向 unshard）和 `reduce_scatter`（反向梯度归约），
  形状来自实际参数维度。
- **EP** all-to-all：由 `AllToAllTokenDispatcher` 自然发射（meta 上使用强制负载均衡 mock -- 见 §20.6）。

模型在 trace 期间通过 `SequentialPipelineSchedule`（§16）**实际运行**，该调度顺序执行所有模型部件
并调用 `loss.backward()`。这触发 FSDP2 生命周期钩子（unshard/reshard/reduce-scatter）和
DTensor TP 归约，在捕获图中产生真实通信算子。

在 DeepSeek V4 smoketest（PP=2, TP=2, DP=2）上验证：**223 个自然通信算子**
（all_gather×85, reduce_scatter×14, all_reduce×19, wait×105），bwd/fwd = 2.15。

### 15.2 合成 PP 注入（总是，PP > 1 时）

流水线并行直接使用 `dist.send()`/`dist.recv()`（不走 DTensor dispatch），因此 PP send/recv
无法自然发射。`_inject_pp_send_recv(result, trainer)`（在后处理步骤调用）创建
`pp_send_activation` / `pp_recv_activation`（前向）和 `pp_send_gradient` / `pp_recv_gradient`
（反向）`OpNode` 对，带 `comm_op="send"/"recv"`、`comm_group_size=2`、
`attrs={"synthetic": True, "pp": True}`。

### 15.3 合成计算锚点（fake_backend）

`_inject_synthetic_compute_anchors(result, trainer)` 确保每个阶段有足够的 "Cube"
（矩阵乘法类：`mm`、`matmul`、`bmm`、`addmm`、`linear`、`conv`、`gemm`、`dot`）和
"Vec"（`aten.add.Tensor`）泳道信号供 HTML 算子泳道使用。泳道目标 =
`max(pp*num_layers, pp)`。注入合成 `aten.mm.default`（cube）/ `aten.add.Tensor`（vec）节点，
带 `attrs={"synthetic_compute_anchor": True, ...}`。

### 15.4 gloo 模式

真实 CPU 通信捕获：
- 通过 `_apply_fsdp1_on_cpu()` 在 init 之后应用 FSDP1 包装。
- `CommRecorder` 拦截真实的 all-gather/reduce-scatter 调用。
- 需要 `init_cpu_distributed()` 建立进程组。
- 单进程即可（`init_distributed` 使用 `FakeProcessGroup`）。

> **注：** 当前从 `run_train.sh` 实际无法触达 gloo 模式，因为 `SimulationTrainer.__init__`
> 强制 `config.comm.mode = "fake_backend"`，随后覆盖 `comm_backend = ""`。见 §21。

---

## 16. CPU 上的流水线并行

### 16.1 语义流水线（`_cpu_semantic_pipeline`）

用于 fake_backend 模式下 PP > 1：
1. 通过 `_meta_parallelize_with_skip_fsdp` 对**完整**模型应用真实并行化（TP/EP/FSDP2）。
   **不**进行 PP 切分 -- 切分使用 `copy.deepcopy`，会剥离 FSDP2 wrapper 类并破坏 DTensor mesh 对齐。
2. 返回 `SequentialPipelineSchedule`，顺序运行所有模型部件（扁平化 PP：
   `part_0(input) → part_1(out) → … → loss.backward()`）。这触发每个部件上的 FSDP2 生命周期钩子，
   在 trace 期间自然发射 `all_gather` / `reduce_scatter`。
3. 真实 PP 调度通过 `_inject_semantic_schedule()` 单独提取（使用 `extract_schedule_from_pytorch`
   和 mock pipeline stage -- 无需模型执行）。

**关键设计选择**：跳过 PP 切分以保留 FSDP2 的 wrapper 类和 DTensor mesh 对齐。
用于可视化/DES 的 PP 调度来自 `extract_schedule_from_pytorch`，而非实际 PP 执行。

### Gloo 流水线

用于 gloo 模式下 PP > 1：`_cpu_pp_module_split` 在 CPU 上将模型拆分为 stage；使用真实 gloo 通信的
上游调度对象。

---

## 17. 模型配置注册表

每个模型有自己的 `config_registry.py`，返回 `SimulationTrainer.Config`。两者都复用上游模型代码
（`torchtitan.models.*`）；仿真器模型目录中无 `parallelize.py` / 架构文件。

### Llama3

| 配置 | 拓扑 | 说明 |
|------|------|------|
| `llama3_sim_debugmodel` | 1 GPU（默认并行） | 小模型，gloo 通信，cost model，seq=64，bs=1 |
| `llama3_sim_1024gpu` | PP=4, TP=8, dp_shard=4, dp_replicate=8（1024 GPU），Interleaved1F1B，vpp=2 | semantic_schedule=True，seq=64，bs=8，8 microbatch |

### DeepSeek V4

| 配置 | 拓扑 | 说明 |
|------|------|------|
| `deepseek_v4_sim_smoketest` | PP=2, TP=2, dp_shard=2, dp_replicate=1（8 rank），Interleaved1F1B | 2 层冒烟测试，vocab=129280，seq=128，bs=4 |
| `deepseek_v4_pro_sim_smoketest` | PP=8, TP=8, dp_shard=-1(auto), EP=192, Interleaved1F1B | DeepSeek V4 Pro 61 层，seq=4096，bs=1 |

所有配置注册全部六种输出格式并启用 `cost_model=True`。

---

## 18. 扩展系统（`extension_hooks.py`）

两个鸭子类型钩子供外部 side-load 使用（如 torchtitan-npu）：

1. `collect_extension_metadata(trainer, capture)` -- 若实现了则调用
   `trainer.collect_simulation_metadata(capture)`。缺失/None/非 dict 时返回 `{}`。
   **已定义但 `trainer_runner.py` 中未调用** -- 供扩展包使用的工具。
2. `postprocess_extension_result(result, trainer, sim_opts)` -- 若实现了则调用
   `trainer.postprocess_simulation_result(result, sim_opts)`。在后处理步骤中**被调用**
   （导出前）。

---

## 19. 关键设计决策

### 19.1 为什么只保留单一入口（`SimulationTrainer`）？

- **一致性**：所有仿真运行走相同的 `Trainer` 初始化路径，确保模型构建、并行化和配置与真实训练一致。
- **配置驱动**：`SimulationConfig` 与 torchtitan 配置系统集成。
- **无 API 漂移**：单一入口防止分化。

### 19.2 为什么只保留单一捕获模块（`unified_trace.py`）？

- **无循环依赖**：`CommRecorder` 和 `FSDPEventRecorder` 与 `TraceRecorder` 内联。
- **单一真实来源**：所有捕获逻辑集中一处。

### 19.3 为什么使用 `FakeTensorMode` + `TorchDispatchMode`？

- **零内存**：仅形状的张量使得仿真 1T+ 参数模型成为可能。
- **Dispatch 层**：捕获所有 ATen 算子，包括被 autograd/FSDP 隐藏的。

### 19.4 为什么 Mock PyTorch 调度对象？

- 精确复用上游调度算法；自动适配新调度类型（ZeroBubble、DualPipeV 等）。

### 19.5 为什么自然捕获而非合成注入？

- TP/FSDP 通信形状精确（来自 DTensor/FSDP2 dispatch），非启发式。
- 自动支持 EP/CP，无需改仿真器代码。
- ~50 行 meta 补丁替代 ~400 行注入逻辑。

### 19.6 为什么使用 salabim DES？

- 建模 compute/comm 资源竞争。
- 处理跨 rank 同步（PP send/recv 依赖）。

### 19.7 为什么使用分层 IR（L0-L3）？

- 关注点分离：算子 -> 步骤 -> 调度 -> 工作负载。
- 面向下游硬件仿真器的规范对齐。

---

## 20. Phase 4 -- 自然通信捕获（已完成并验证）

> 本节记录已建成的 Phase 4 工作。代码已集成、在默认路径中激活，并通过 E2E smoketest **验证**
> （223 个自然通信算子被捕获）。

### 20.1 它解决的问题

原始的 `_inject_synthetic_comm_events()` 使用启发式假设：层均匀分布、手动张量大小计算、
硬编码模式（如每层 2 次 TP all_reduce）。这不捕获忠实。

### 20.2 Meta 设备补丁（`meta_device_patches.py`）— 7 个补丁

| # | 补丁 | 效果 |
|---|------|------|
| 1 | `FSDPParamGroup._validate_no_meta_params = lambda self: None` | 跳过拒绝 meta 参数的验证 |
| 2 | `FakeTensor._find_common_device = _patched_find_common_device` | 为 FSDP 混合 meta/cpu 算子优先 meta 设备 |
| 3 | `FakeTensorMode.wrap_meta_outputs_with_default_device_logic = _patched_wrap` | 将 CPU 张量转为 meta 用于 FSDP 内部缓冲区 |
| 4 | `foreach_reduce` dtype 强制器 | 将混合 dtype 梯度强制为统一 dtype（避免 meta 上 FSDP2 reduce-scatter dtype 断言） |
| 5 | `_unimplemented_deepcopy` → `_fsdp_meta_deepcopy` | 允许 FSDP 模块 deepcopy（绕过 `__deepcopy__` 阻断 + `disable_fsdp_module_new_init` + identity-copy `ProcessGroup`/`DeviceMesh`） |
| 6 | `nn.Module.to_empty` 在 meta 上 no-op | 保留 FSDP2 分片状态（to_empty 创建新张量，丢弃 FSDP2 的 DTensor placements） |
| 7 | `repeat_interleave` fake impl 覆盖 | 返回占位形状而非抛出 `DynamicOutputShapeException`（MoE dispatch 使用动态形状 `repeat_interleave`） |

### 20.3 E2E 自然捕获的额外修复

除 7 个 meta 补丁外，以下修复使模型在 trace 期间**实际运行**（触发 FSDP2/TP 钩子）：

| 修复 | 文件 | 效果 |
|------|------|------|
| **从 llama4 导入 `apply_fsdp`** | `models/deepseek_v4/parallelize.py` | 从 `llama4` 导入 EP 感知 `apply_fsdp`（支持 `edp_mesh`/`ep_degree`）而非 llama3（不支持）。过滤 `gradient_divide_factor`。 |
| **`SequentialPipelineSchedule`** | `trainer.py` | 替代 `MockSchedule`（no-op）。顺序运行所有模型部件 + `loss.backward()`，触发 FSDP2 钩子。 |
| **跳过 PP 切分** | `trainer.py` | 不 `deepcopy`/切分模型 -- 保留 FSDP2 wrapper 类和 DTensor mesh 对齐。PP 调度单独来自 `extract_schedule_from_pytorch`。 |
| **`mixed_precision_param=fp32`** | `trainer.py` | 在 meta 上强制 fp32（FSDP2 的 bfloat16 转换导致 meta 上 DTensor dtype 不匹配）。 |
| **禁用激活检查点** | `trainer.py` | AC 的变更检查在 FakeTensor 上抛出；AC 在 meta 上无内存收益（0 字节张量）。 |
| **`DTensor.__format__` 补丁** | `trainer_runner.py` | 防止日志/指标格式化 DTensor 时崩溃。 |
| **EP 专家 padding** | `models/common/token_dispatcher.py` | 当 `num_experts` 不能被 `ep_size` 整除时（如 256 专家，EP=192），pad 到可整除大小。 |
| **强制负载均衡 MoE dispatch** | `models/common/token_dispatcher.py` | 在 meta 上（检测到 FakeTensor），绕过动态 `bincount`/`all_to_all`/`.tolist()`（均返回零），使用均匀 token 分布。给出具体非零 split 列表，使所有下游形状静态。恒等排列（均匀 = 无需重排）。 |

### 20.4 集成点

- **`trainer.py`** -- `_meta_parallelize_with_skip_fsdp.wrapper`：在调用真实并行化函数前应用 meta 补丁，
  强制 fp32 + 禁用 AC，在 `finally` 中恢复。
- **`trainer_runner.py`** -- `run_trainer_simulation`：在 `unified_trace(...)` + `Trainer.train_step(...)`
  块周围应用 meta 补丁，补丁 `DTensor.__format__`，在 `finally` 中恢复。
- **`models/deepseek_v4/parallelize.py`** -- 从 `llama4` 导入 `apply_fsdp`（非 `llama3`），
  启用 EP 感知 FSDP2 的 per-param mesh 路由。
- **`models/common/token_dispatcher.py`** -- `AllToAllTokenDispatcher.dispatch/combine`
  有 `_is_fake` 分支，在 meta 上使用强制负载均衡。

### 20.5 实际通信分类法

Phase 4 后，仿真结果的通信来自：
- **自然**（TP/FSDP2/EP）：来自 DTensor/FSDP2 dispatch 的精确形状。验证：smoketest 上 223 个算子
  （all_gather, reduce_scatter, all_reduce, wait_tensor）。
- **合成 PP**（`_inject_pp_send_recv`）：仍为启发式，因为 PP 直接使用 `dist.send/recv`。
- **合成计算锚点**（`_inject_synthetic_compute_anchors`）：用于可视化的 Cube/Vec 泳道填充。

### 20.6 强制负载均衡 MoE dispatch（仅 meta）

在 meta 设备上，FakeTensor 不携带真实值。MoE EP dispatch 链
（`bincount` → `all_to_all` → `.tolist()` → 动态 splits）产生全零，
导致零大小张量和形状崩溃。强制负载均衡 mock 通过从已知静态量计算**均匀** token 计数来绕过此问题：

```
total_routed = num_tokens * top_k           （静态）
tokens_per_rank = total_routed // ep_size   （均匀）
tokens_per_expert = tokens_per_rank // num_local_experts
```

Split 列表构造为 Python int（非 FakeTensor 上的 `.tolist()`）。
`_permute` 替换为恒等排列（均匀分布 = 无需重排）。`combine` 中的 `_unpermute` 跳过（恒等）。

这是**仿真近似** -- 真实训练有非均匀 token 路由。但对通信算子捕获（all_to_all、
FSDP2 all_gather/reduce_scatter），确切 token 计数不影响通信形状（通信形状来自参数维度，
非 token 计数）。

### 20.7 剩余限制

- **PP send/recv 仍需注入** -- PP 使用 `dist.send()`/`dist.recv()`。
- **Meta 设备补丁脆弱** -- 依赖 PyTorch 内部实现。
- **强制负载均衡是近似** -- 真实 MoE 路由是非均匀的。
- **Pro 61 层 E2E 较慢**（~8 分钟/步）-- 使用 4 层配置可加速测试。
- **`repeat_interleave` fake impl 返回占位形状** -- 可能并非所有情况下都与下游预期匹配。

---

## 21. 技术债 & 已知差异

为评审讨论标记的事项：

1. **`_inject_synthetic_comm_events` 是死代码。** 定义于 `trainer_runner.py` 但生产中**不调用**
   -- 仅 `tests/test_simulator.py` 使用。Phase 4 自然捕获取代了它。待遗留测试迁移后可移除。

2. **`schedule_generator.py` 已废弃。** 未从 `schedule/__init__.py` 导出，也不在运行时路径；
   运行时使用 `extract_schedule_from_pytorch`。

3. **事件类型分类法不一致**，跨调度文件：
   - `schedule_extract.py`（主路径）：`pp_forward`、`pp_send_activation`...
   - `pp_schedule_extractor.py`（启发式回退）：`fwd`、`bwd`、`send_fwd`...
   - `schedule_generator.py`（废弃）：`pp_forward`、`tp_all_reduce`...
   建议：统一到 `pp_*` / `fsdp2_*` 分类法。

4. **`Simulator` 编程式 API 不存在。** `AGENTS.md` 引用了带
   `simulate_fx`/`simulate_runtime`/`simulate_pp_schedule` 的 `Simulator` 类。见 §22 路线图。

5. **HTML trace 非完全自包含。** ECharts 5.4.3 从 `cdn.jsdelivr.net` 加载。要么 vendor ECharts，
   要么更正文档。

6. **gloo 模式从 `run_train.sh` 不可达。** `SimulationTrainer.__init__` 强制
   `config.comm.mode = "fake_backend"`，随后覆盖 `comm_backend = ""`。

7. **两个 `_DTYPE_BYTES` 表**，键约定不同：`ir/op_node.py` 用 `"torch.float32"` 键（默认 4）；
   `ir/workload_graph.py` 用 `"float32"` 键（默认 8）。有静默字节计算错误风险。

8. **强制负载均衡是仿真近似。** meta 上的 MoE EP dispatch 使用均匀 token 分布（§20.6）。
   真实训练有非均匀路由。这影响 token 计数相关形状，但不影响通信算子形状
   （通信形状来自参数维度）。

9. **`mixed_precision_param` 在 meta 上强制为 fp32。** FSDP2 的 bfloat16 转换导致 meta 上
   DTensor dtype 不匹配。强制 fp32 意味着仿真不捕获混合精度通信模式（如 bfloat16 all_gather）。

10. **激活检查点在 meta 上禁用。** AC 的变更检查在 FakeTensor 上抛出。禁用 AC 意味着仿真
    不捕获 AC 的内存节省或其对计算图的影响（更少的保存激活值）。

11. **`apply_fsdp` 从 llama4 导入。** deepseek_v4 parallelize 从 `llama4.parallelize`
    导入 `apply_fsdp`（EP 感知）而非 `llama3.parallelize`。这是跨模型依赖，如果 llama4
    被上游移除可能中断。

12. **`_reapply_fsdp2_to_parts` 是死代码。** 定义于 `trainer_runner.py` 但不再调用
    （PP 切分已移除 -- FSDP2 在并行化期间应用于完整模型）。可移除。

---

## 22. 路线图：编程式 `Simulator` API

更高级的编程式 API 已规划但**尚未实现**。设想设计（`AGENTS.md` 中引用）为含三种模式的
`Simulator` 类：

| 模式 | 目的 | 基础 |
|------|------|------|
| `simulate_fx` | 静态 FX trace 捕获 | 模型的 `torch.fx` 符号 trace |
| `simulate_runtime` | 动态 1 步捕获 | 当前 `unified_trace` + train-step 路径 |
| `simulate_pp_schedule` | 仅调度提取 | `extract_schedule_from_pytorch` |

```python
from torchtitan.experiments.simulator import Simulator

result = Simulator(model, config).simulate_runtime()   # 设想
```

目前，等价物是捕获用的 `TraceRecorder` + `unified_trace` 直接使用（§3.2），或端到端运行的
`SimulationTrainer`（§3.1）。在这些原语之上构建 `Simulator` facade 是建议的下一步。
