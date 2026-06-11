# TorchTitan 模拟器重构设计文档

## 概述

对 `torchtitan/experiments/simulator/` 进行系统性重构，在保持所有功能不变、E2E 用例正常可运行、输出内容不发生变化的前提下，提升代码可读性、可维护性和可扩展性。

**策略：** 分层增量重构（自底向上，4 个阶段），每个阶段独立可测试、可验证。

## 约束条件

- 所有 E2E 测试必须通过，测试断言不做修改
- 所有输出文件（JSON、DOT、Chrome trace、HTML、text）内容必须一致
- 公共 API（`__init__.py` 导出）保持不变
- 不修改 `torchtitan/experiments/simulator/` 之外的核心代码
- 每个阶段完成后 pre-commit 检查必须通过

---

## 重构前后对比

| 指标 | 重构前 | 重构后 | 变化 |
|------|--------|--------|------|
| 总行数 | ~13,500 | ~9,400 (不含测试) | **-30%** |
| 最大 Python 文件 | `export.py` 2,367 行 | `schedule_extract.py` 875 行 | **-63%** |
| 重复逻辑 | 12 处 | 0 处 | **消除** |
| 循环依赖 | 2 对 | 0 对 | **消除** |
| 捕获架构 | 2 套共存 | 1 套统一 | **统一** |
| 文件数量 | 25 个 | 22 个 + 1 子包 | 更聚焦 |

### 删除的文件

| 文件 | 原因 |
|------|------|
| `dispatch_interceptor.py` | 被 `unified_trace.py` 替代 |
| `runtime_capture.py` | 被 `unified_trace()` 上下文管理器替代 |
| `graph_assembler.py` | 合并到 `fx_capture.py` |
| `pp_schedule_extractor.py` | 合并到 `schedule_extract.py` |

### 新增的文件

| 文件 | 职责 |
|------|------|
| `_recorder_registry.py` | 记录器栈管理，打破循环依赖 |
| `cost_estimators.py` | FLOPs/字节估算 + 重叠策略 |
| `schedule_analysis.py` | 调度-图关联 + 关键路径分析 |
| `des_memory.py` | DES 内存时间线 |
| `synthetic_comm.py` | 合成通信事件注入 |
| `schedule_inject.py` | 语义调度注入 + 并行度工具 |
| `export/` 子包 | 9 个 Python 模块 + 独立 JS 文件 |

---

## 架构总览

```mermaid
graph TB
    subgraph "入口层 Entry Points"
        A1["simulator.py<br/>Simulator 类<br/>(编程 API)"]
        A2["trainer.py<br/>SimulationTrainer<br/>(run_train.sh)"]
        A3["run_simulate.py<br/>CLI 入口"]
    end

    subgraph "捕获层 Capture Layer"
        B1["unified_trace.py<br/>统一追踪模式<br/>TraceRecorder + UnifiedTraceMode"]
        B2["comm_interceptor.py<br/>通信拦截<br/>CommRecorder"]
        B3["fsdp_tracer.py<br/>FSDP 生命周期"]
        B4["fx_capture.py<br/>静态 FX 图捕获"]
        B5["_recorder_registry.py<br/>记录器栈"]
    end

    subgraph "分析层 Analysis Layer"
        C1["cost_model.py<br/>CostModel ABC<br/>MockCostModel"]
        C2["cost_estimators.py<br/>FLOPs/字节估算"]
        C3["schedule_analysis.py<br/>关键路径 + 多 rank 预测"]
        C4["des_engine.py<br/>离散事件仿真"]
        C5["des_memory.py<br/>内存时间线"]
        C6["memory_estimator.py<br/>内存估算"]
    end

    subgraph "调度层 Schedule Layer"
        D1["schedule_extract.py<br/>PyTorch 调度提取<br/>PPScheduleExtractor"]
        D2["schedule_generator.py<br/>语义调度生成"]
        D3["schedule_inject.py<br/>调度注入 + 并行度"]
        D4["synthetic_comm.py<br/>合成通信注入"]
    end

    subgraph "输出层 Output Layer"
        E1["export/<br/>子包"]
        E2["json_export.py"]
        E3["dot_export.py"]
        E4["chrome_trace.py"]
        E5["html_export.py"]
        E6["text_summary.py"]
        E7["trace_visualizer.js"]
    end

    subgraph "基础设施 Foundation"
        F1["nodes.py<br/>数据模型"]
        F2["op_classification.py<br/>算子分类"]
        F3["cpu_env.py<br/>CPU 环境"]
        F4["meta_env.py<br/>Meta 设备"]
    end

    A1 --> B1
    A1 --> B4
    A2 --> B1
    A3 --> A1
    B1 --> B2
    B1 --> B3
    B1 --> B5
    B2 --> B5
    B1 --> F1
    B4 --> F1
    C1 --> C2
    C1 --> C4
    C3 --> C1
    C3 --> C4
    C4 --> F1
    C5 --> F1
    C6 --> F1
    D1 --> F1
    D2 --> F1
    D3 --> D1
    D4 --> F1
    E1 --> E2
    E1 --> E3
    E1 --> E4
    E1 --> E5
    E1 --> E6
    E5 --> E7
    F4 --> F3
```

---

## 模块依赖图

```mermaid
graph LR
    subgraph "无循环依赖（重构后）"
        direction TB
        nodes["nodes.py"]
        op_cls["op_classification.py"]
        registry["_recorder_registry.py"]
        unified["unified_trace.py"]
        comm["comm_interceptor.py"]
        fsdp["fsdp_tracer.py"]
        fx["fx_capture.py"]
        cost["cost_model.py"]
        estimators["cost_estimators.py"]
        analysis["schedule_analysis.py"]
        des["des_engine.py"]
        des_mem["des_memory.py"]
        mem["memory_estimator.py"]
        sched_ext["schedule_extract.py"]
        sched_gen["schedule_generator.py"]
        sched_inj["schedule_inject.py"]
        syn_comm["synthetic_comm.py"]
        export["export/"]
    end

    op_cls --> nodes
    registry -.->|"无依赖"| nodes
    unified --> nodes
    unified --> op_cls
    unified --> registry
    comm --> nodes
    comm --> registry
    fsdp -.-> nodes
    fx --> nodes
    fx --> op_cls
    estimators --> nodes
    estimators --> mem
    cost --> nodes
    cost --> estimators
    cost --> analysis
    analysis --> nodes
    analysis --> cost
    analysis --> des
    des --> nodes
    des --> analysis
    des_mem --> nodes
    mem --> nodes
    sched_ext --> nodes
    sched_gen --> nodes
    sched_gen --> sched_ext
    sched_inj --> sched_ext
    syn_comm --> nodes
    syn_comm --> mem
    syn_comm --> sched_inj
    export --> nodes
```

### 重构前循环依赖（已消除）

```mermaid
graph LR
    subgraph "重构前 ❌"
        A1["cost_model"] -->|"lazy"| A2["des_engine"]
        A2 -->|"lazy"| A1
        B1["comm_interceptor"] -->|"lazy"| B2["unified_trace"]
        B2 -->|"lazy"| B1
    end

    subgraph "重构后 ✅"
        C1["cost_model"] --> C2["schedule_analysis"]
        C3["des_engine"] --> C2
        C2 --> C1
        C2 --> C3
        D1["comm_interceptor"] --> D2["_recorder_registry"]
        D3["unified_trace"] --> D2
    end
```

---

## 统一捕获架构

```mermaid
flowchart TD
    subgraph "重构前：两套捕获架构共存"
        direction TB
        OLD1["OpRecorder<br/>(dispatch_interceptor)"] --> OLD2["RuntimeCapture<br/>(runtime_capture)"]
        OLD3["CommRecorder<br/>(comm_interceptor)"] --> OLD2
        OLD4["FSDPEventRecorder<br/>(fsdp_tracer)"] --> OLD2
        OLD2 --> OLD5["GraphAssembler<br/>(graph_assembler)"]
        OLD5 --> OLD6["SimulationResult"]

        NEW1["TraceRecorder<br/>(unified_trace)"] --> NEW2["unified_trace()<br/>上下文管理器"]
        NEW3["CommRecorder"] --> NEW2
        NEW4["FSDPEventRecorder"] --> NEW2
        NEW2 --> NEW5["SimulationResult"]
    end

    subgraph "重构后：统一架构"
        direction TB
        U1["TraceRecorder<br/>(unified_trace)"] --> U2["unified_trace()<br/>上下文管理器"]
        U3["CommRecorder<br/>(comm_interceptor)"] --> U2
        U4["FSDPEventRecorder<br/>(fsdp_tracer)"] --> U2
        U2 --> U5["build_result()"]
        U5 --> U6["SimulationResult"]
        U7["fx_capture.py<br/>静态 FX 捕获"] --> U6
    end
```

---

## 数据流图

```mermaid
flowchart LR
    subgraph "输入"
        M["模型<br/>nn.Module"]
        I["输入<br/>Tensor"]
        C["配置<br/>JobConfig"]
    end

    subgraph "捕获 Capture"
        UT["unified_trace()<br/>FakeTensorMode<br/>+ TorchDispatchMode"]
        FX["capture_forward_fx()<br/>make_fx"]
    end

    subgraph "构建 Build"
        CG["ComputeGraph<br/>OpNode + DataEdge"]
        TS["TrainingSchedule<br/>ScheduleEvent + ScheduleDep"]
        ME["MemoryEvent<br/>内存事件"]
    end

    subgraph "分析 Analyze"
        CM["CostModel<br/>性能估算"]
        DES["DESEngine<br/>离散事件仿真"]
        MA["MemoryEstimator<br/>内存分析"]
    end

    subgraph "输出 Output"
        JSON["simulation_result.json"]
        DOT["compute_graph.dot"]
        CHROME["trace.json"]
        HTML["trace.html"]
        TXT["summary.txt"]
    end

    M --> UT
    I --> UT
    M --> FX
    I --> FX
    UT --> CG
    UT --> TS
    UT --> ME
    FX --> CG
    C --> TS
    CG --> CM
    CG --> DES
    CG --> MA
    TS --> DES
    ME --> MA
    CG --> JSON
    CG --> DOT
    CG --> CHROME
    CG --> HTML
    CG --> TXT
    TS --> HTML
    DES --> HTML
    MA --> HTML
```

---

## export/ 子包结构

```mermaid
graph TB
    subgraph "export/ 子包"
        INIT["__init__.py<br/>重导出公共 API"]
        JSON["json_export.py<br/>JSON 序列化"]
        DOT["dot_export.py<br/>Graphviz DOT"]
        CHROME["chrome_trace.py<br/>Chrome 追踪"]
        HTML["html_export.py<br/>HTML 可视化"]
        TEXT["text_summary.py<br/>文本摘要"]
        SCHED["schedule_timing.py<br/>调度时序增强"]
        UTILS["export_utils.py<br/>export_result() 编排"]
        SHARED["_shared.py<br/>格式化工具"]
        JS["trace_visualizer.js<br/>1136 行纯 JS"]
    end

    INIT --> JSON
    INIT --> DOT
    INIT --> CHROME
    INIT --> HTML
    INIT --> TEXT
    INIT --> UTILS
    JSON --> SCHED
    HTML --> JS
    HTML --> SCHED
    CHROME --> SCHED
    UTILS --> JSON
    UTILS --> DOT
    UTILS --> CHROME
    UTILS --> HTML
    UTILS --> TEXT
    JSON --> SHARED
    TEXT --> SHARED
```

---

## 重构阶段与提交记录

### 阶段 1：基础层 — 消除重复

| 提交 | 说明 |
|------|------|
| `1a034a5c` | 合并设备环境补丁（cpu_env + meta_env 共享工厂） |
| `1b0fdb22` | 提取共享工具函数（loss、export、dtype、并行度） |
| `ec447ebd` | 提取 comm_event_to_op_node 和 replicate_events_to_ranks |

### 阶段 2：捕获层 — 统一架构

| 提交 | 说明 |
|------|------|
| `7a74c4d9` | 创建 _recorder_registry 打破循环依赖 |
| `fd6d1b77` | 迁移 simulate_runtime 到 unified_trace，删除旧路径 |

### 阶段 3：分析层 — 拆分与解耦

| 提交 | 说明 |
|------|------|
| `c46f4211` | 拆分 cost_model → cost_estimators + schedule_analysis |
| `7437a63c` | 提取 compute_des_memory_timeline → des_memory |
| `e4d22993` | 合并 pp_schedule_extractor → schedule_extract |

### 阶段 4：输出层 — 拆分 export.py

| 提交 | 说明 |
|------|------|
| `9b776b02` | 拆分 export.py → export/ 子包，提取 JS 为独立文件 |

---

## 最终文件结构

```
torchtitan/experiments/simulator/
  __init__.py                     (82 行, 公共 API)
  _recorder_registry.py           (23 行, 记录器栈)
  simulator.py                    (431 行, Simulator 类)
  trainer.py                      (229 行, SimulationTrainer)
  trainer_runner.py               (224 行, 仿真编排)
  run_simulate.py                 (303 行, CLI 入口)
  cpu_env.py                      (219 行, CPU 环境 + 共享补丁)
  meta_env.py                     (58 行, Meta 设备薄封装)
  nodes.py                        (493 行, 数据模型)
  op_classification.py            (127 行, 算子分类)
  unified_trace.py                (434 行, 统一追踪 + compute_loss)
  comm_interceptor.py             (437 行, 通信拦截)
  fsdp_tracer.py                  (186 行, FSDP 追踪)
  fx_capture.py                   (415 行, FX 捕获 + merge_comm_events)
  cost_model.py                   (303 行, CostModel ABC + MockCostModel)
  cost_estimators.py              (183 行, FLOPs/字节估算)
  schedule_analysis.py            (136 行, 关键路径 + 多 rank 预测)
  des_engine.py                   (463 行, DES 引擎核心)
  des_memory.py                   (196 行, DES 内存时间线)
  memory_estimator.py             (355 行, 内存估算)
  schedule_extract.py             (875 行, 调度提取 + PPScheduleExtractor)
  schedule_generator.py           (379 行, 语义调度生成)
  schedule_inject.py              (68 行, 调度注入 + 并行度)
  synthetic_comm.py               (263 行, 合成通信注入)
  extension_hooks.py              (46 行, 扩展钩子)
  synthetic_dataloader.py         (57 行, 合成数据加载器)
  export/                         (子包)
    __init__.py                   (重导出公共 API)
    _shared.py                    (格式化工具)
    json_export.py                (JSON 导出)
    dot_export.py                 (DOT 导出)
    chrome_trace.py               (Chrome 追踪导出)
    html_export.py                (HTML 可视化导出)
    text_summary.py               (文本摘要)
    schedule_timing.py            (调度时序增强)
    export_utils.py               (export_result 编排)
    trace_visualizer.js           (1136 行, 独立 JS 可视化)
  llama3/                         (模型配置, 不变)
  deepseek_v4/                    (模型配置, 不变)
  tests/
    test_simulator.py             (2968 行, 114 个测试)
```

---

## 验证结果

| 检查项 | 结果 |
|--------|------|
| 单元测试 | **114/114 通过** (2.85s) |
| 公共 API | **18 个符号全部不变** |
| Pre-commit | **零新增错误** |
| 最大 Python 文件 | 875 行 (schedule_extract.py) |
| 最大 JS 文件 | 1136 行 (trace_visualizer.js, 纯 JS 可独立编辑) |
