# Simulator 使用指南

## 概述

CPU simulator 是 TorchTitan 的一个 side-loaded experiment，可在无 GPU 环境下捕获训练计算图、通信事件、FSDP/PP 调度语义以及内存估算。它不修改 `torchtitan/train.py` 入口，而是通过 config_registry 替换 Trainer 类型。

## 两种调用方式

### 方式一：SimulationTrainer（通过 run_train.sh 侧加载）

这是最接近真实训练流程的方式——模型构建、分布式初始化、并行策略全部走 TorchTitan 标准路径，仅将 `trainer.train()` 替换为一步模拟捕获。

```bash
MODULE=simulator.llama3 CONFIG=llama3_sim_debugmodel ./run_train.sh
```

输出目录和格式由 `SimulationConfig` 控制，默认写到 `./simulator_output`。可通过 CLI override 更改：

```bash
MODULE=simulator.llama3 CONFIG=llama3_sim_debugmodel \
  ./run_train.sh --simulation.output_dir ./sim_out_torchtitan_memory_trace
```

调试模式（不需要 GPU）：

```bash
# 仅验证配置，不执行
NGPU=32 COMM_MODE="fake_backend" ./run_train.sh

# 单 GPU 调试模拟多 GPU 行为
NGPU=32 COMM_MODE="local_tensor" ./run_train.sh
```

### 方式二：run_simulate.py（独立 CLI）

直接构建模型并运行模拟，不依赖 TorchTitan Trainer 完整构建流程。适合快速测试或自定义模型。

```bash
# 单进程模拟
python -m torchtitan.experiments.simulator.run_simulate \
  --job.config_file ./train_configs/llama3_8b.toml \
  --simulate.mode all \
  --simulate.output_dir ./sim_out \
  --simulate.output_format json,dot,chrome_trace,html,text,csv

# 多进程 PP 模拟（torchrun）
torchrun --nproc_per_node 4 \
  -m torchtitan.experiments.simulator.run_simulate \
  --job.config_file ./train_configs/llama3_8b.toml \
  --training.pipeline_parallel_degree 4 \
  --simulate.mode all
```

CLI 参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--simulate.mode` | `all` | `fx`（静态图）、`runtime`（运行时捕获）、`schedule`（PP 调度提取）、`all` |
| `--simulate.output_dir` | `./simulator_output` | 输出目录 |
| `--simulate.output_format` | `json,dot,chrome_trace,html,text,csv` | 输出格式列表 |
| `--simulate.max_seq_len` | `128` | 输入序列长度 |
| `--simulate.batch_size` | `2` | 输入 batch size |

## 三种模拟模式

| 模式 | 方法 | 说明 |
| --- | --- | --- |
| `fx` | `simulate_fx()` | 静态图捕获：使用 `make_fx` + `FakeTensorMode`，不执行真实前向，通过符号推断 tensor shape |
| `runtime` | `simulate_runtime()` | 动态运行时捕获：在 CPU(gloo) 上执行一步真实训练，拦截每个算子、通信和 FSDP 事件 |
| `schedule` | `simulate_pp_schedule()` | 纯 PP 调度提取：不运行模型，仅读取或推断 PP schedule 的 action 表 |

`all` 模式依次执行 fx + runtime + schedule（如有 PP），合并结果后导出。

## 输出文件

所有输出写到 `simulation.output_dir` 或 `--simulate.output_dir` 指定的目录：

| 文件 | 说明 |
| --- | --- |
| `simulation_result.json` | 完整结构化数据：compute_graph、schedule、comm/fsdp/pp/memory events、metadata |
| `compute_graph.dot` | Graphviz DOT 格式，节点按 op_type 颜色编码（蓝=compute，黄=comm，紫=memory） |
| `trace.json` | Chrome Trace 格式（`chrome://tracing`），按 phase 分 thread，时间轴为逻辑顺序 |
| `trace.html` | 自包含交互式 HTML：训练步骤层级、PP/FSDP/TP/DP 调度泳道、前向/反向算子 DAG、内存 timeline |
| `summary.txt` | 人类可读的文本摘要：op 计数、通信统计、内存估算峰值 |
| `kernel_summary.csv` | 逐算子 trace（forward+backward+optimizer），格式对齐 CUDA/CANN kernel_summary：算子名/顺序/起止时间/耗时/breakdown |
| `workload_graph.json` | spec 分层 IR（L0 OpNode / L1 StepGraph / L2 ScheduleGraph / L3 WorkloadGraph）的投影结果 |

### kernel_summary.csv 列说明

逐算子一行，覆盖 forward + backward + optimizer step；有 DES 调度时按 DES 起始时间排序并取其 start/end，否则按捕获顺序在串行累积时间轴上铺排。

| 列 | 含义 |
| --- | --- |
| `Order` | 算子顺序（0 起的序号） |
| `Name` / `Type` / `Phase` | 算子名 / 算子类别 / 所属阶段 |
| `PPStage` / `MicrobatchIdx` | PP stage 与微批次（如有） |
| `Start(us)` / `End(us)` / `Duration(us)` | 起始时间 / 结束时间 / 耗时 |
| `ComputeTime(us)` / `CommTime(us)` | breakdown：计算/通信耗时拆解 |
| `FLOPs` / `BytesRead` / `BytesWritten` | breakdown：算力与访存量 |

> 注：fake_backend 路径下 optimizer.step 被 patch 为 no-op，因此该步算子可能不被捕获（CSV 仅在捕获到 optimizer 算子时输出对应行）。

### 分层 IR（L0–L3）

对齐 `workload-model-platform/spec`，`simulator/ir/` 子包把已捕获的
`SimulationResult` **投影**为四层 IR（不复刻 torchtitan 逻辑，全部派生自捕获数据 +
声明式 config）：

| 层 | 结构 | 来源 |
| --- | --- | --- |
| L0 | `SpecOpNode` | 捕获的 `OpNode` + 数据边，补 flops/comm_bytes/preds/succs |
| L1 | `StepGraph` | 按捕获 `phase`（autograd/optimizer hook 派生）切出 fwd/bwd/opt 模板 |
| L2 | `ScheduleGraph` | 捕获的 PP schedule 事件 + `config.parallelism` 声明并行度 |
| L3 | `WorkloadGraph` | 迭代语义（steps/warmup/梯度累积）+ dataloader 数据流 |

程序化使用：

```python
from torchtitan.experiments.simulator.ir import build_workload_graph
workload = build_workload_graph(result, trainer.config)
workload.to_dict()  # JSON-serializable 四层结构
```

`trace.html` 中前向/反向 operator swimlane 的通信口径由
`simulation.operator_swimlane_comm_scope` 控制：

- `model_only`（默认）：隐藏 synthetic 的 FSDP/PP/DP 调度通信，保留 TP/CP/EP 等模型内通信
- `all`：显示所有通信（包含 synthetic 并行通信）

## 核心组件与数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│  SimulationTrainer / run_simulate.py                                │
│                                                                     │
│  1. patch_device_type_to_cpu()  → 强制所有设备为 CPU                 │
│  2. 构建 model + 并行策略         → FSDP2/TP/PP 均在 CPU gloo 上    │
│  3. RuntimeCapture.activate()   → 同时启动所有拦截器                 │
│     ├─ OpRecorder               → 捕获每个算子及 tensor metadata     │
│     ├─ CommRecorder             → 拦截 torch.distributed 通信        │
│     ├─ FSDPEventRecorder        → 拦截 FSDP allgather/reduce_scatter │
│     └─ PP hooks (optional)      → 拦截 PipelineStage forward/bwd    │
│  4. 执行一步训练                  → forward/backward/optimizer       │
│  5. capture.build_result()      → 组装 SimulationResult             │
│     ├─ GraphAssembler           → 从 op/comm 记录构建 ComputeGraph   │
│     ├─ build_runtime_memory()   → 从 graph + comm 估算内存           │
│     ├─ attach_model_state_memory│ → 从 model 参数估算 model state    │
│     └─ PPScheduleExtractor      → 从 PP schedule 提取调度事件        │
│  6. _export_result()            → 写出 JSON/DOT/Trace/HTML/Text     │
└─────────────────────────────────────────────────────────────────────┘
```

关键源文件（均在 `torchtitan/experiments/simulator/` 下）：

| 文件 | 角色 |
| --- | --- |
| `trainer.py` | SimulationTrainer：继承 Trainer，patch CPU，调用 trainer_runner |
| `trainer_runner.py` | 用已构建的 Trainer 执行一步模拟捕获并导出 |
| `run_simulate.py` | 独立 CLI 入口：手动构建模型、配置 CPU env、调用 Simulator |
| `simulator.py` | Simulator 类：fx/runtime/schedule 三模式的顶层 API |
| `cpu_env.py` | 强制 device_type=CPU，初始化 gloo 分布式 |
| `capture/dispatch_interceptor.py` | 拦截 PyTorch dispatch，记录算子和 tensor 数据边 |
| `capture/comm_interceptor.py` | Monkey-patch torch.distributed，记录通信事件 |
| `capture/runtime_capture.py` | 统一管理所有拦截器的 context manager，组装 SimulationResult |
| `capture/graph_assembler.py` | 从 op 记录构建 ComputeGraph，合并通信事件 |
| `memory_estimator.py` | 估算 activation lifetime、comm buffer、model state 内存 |
| `schedule/pp_schedule_extractor.py` | 从 PP schedule 提取语义事件和依赖 |
| `capture/fx_capture.py` | 使用 make_fx + FakeTensorMode 静态捕获前向/联合图 |
| `export.py` | 导出 JSON/DOT/Chrome Trace/HTML/Text |
| `ir/` | spec 分层 IR 投影：op_node(L0)/step_graph(L1)/schedule_graph(L2)/workload_graph(L3)/builder |
| `extension_hooks.py` | Duck-typed 钩子：collect_simulation_metadata / postprocess_simulation_result |
| `nodes.py` | 数据模型：OpNode、DataEdge、ComputeGraph、MemoryEvent、SimulationResult 等 |

## 内存估算模型

内存追踪是确定性估算而非分配器实测：

- **activation/data_move/comm_buffer**：graph 输出 tensor 转为 lifetimed `MemoryEvent`，lifetime 从 producer 到 last consumer
- **graph peak**：scanline peak over lifetimed events
- **model state**：从实际 model parameters 估算：parameters + gradients + Adam/AdamW optimizer state (exp_avg + exp_avg_sq)
- **resident baseline**：parameters/gradients/optimizer_state 无 graph lifetime，在 HTML timeline 中显示为恒定基线

导出 `metadata["memory"]` 字段包含：`peak_live_bytes`、`parameter_bytes`、`gradient_bytes`、`optimizer_state_bytes`、`model_state_total_bytes`，以及 `by_category`/`by_phase`/`by_device` 分组。

## 模拟器单元测试

```bash
pytest torchtitan/experiments/simulator/tests/test_simulator.py -v
```

覆盖：数据模型、op/comm/FSDP/PP 拦截、导出器、HTML 生成、graph 组装、extension hooks。

## 配置示例

`torchtitan/experiments/simulator/llama3/config_registry.py` 中的 `llama3_sim_debugmodel()`：

```python
SimulationTrainer.Config(
    simulation=SimulationConfig(
        output_dir="./simulator_output",
        output_formats=["json", "dot", "chrome_trace", "html", "text", "csv"],
        capture_joint_fx=False,
        operator_swimlane_comm_scope="model_only",
    ),
    parallelism=ParallelismConfig(
        pipeline_parallel_schedule="Interleaved1F1B",
    ),
    training=TrainingConfig(local_batch_size=1, seq_len=64, steps=1),
    dataloader=SyntheticTokenDataLoader.Config(vocab_size=2048, seed=42),
)
```

通过 CLI override 调整输出目录和并行度：

```bash
MODULE=simulator.llama3 CONFIG=llama3_sim_debugmodel \
  ./run_train.sh \
  --simulation.output_dir ./sim_out_torchtitan_memory_trace \
  --training.pipeline_parallel_degree 2 \
  --training.tensor_parallel_degree 2 \
  --training.data_parallel_shard_degree 2 \
  --training.steps 2
```

## 限制

- 运行时捕获仅观察当前进程，多 rank trace 需多进程执行或后处理聚合
- CPU 模拟不复现 GPU/NPU kernel 性能或真实设备内存压力
- 部分算子级 aliasing 和 in-place 行为通过 tensor producer tracking 近似
- 并行调度在当前环境无法运行时可使用语义模式
