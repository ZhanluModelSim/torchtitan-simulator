# Simulator 导出逻辑审计清单

> 审计日期：2026-06-25  
> 审计范围：`torchtitan/experiments/simulator/export.py`、`trainer_runner.py` 中的导出编排、`des_engine.py` 中导出依赖的 DES 汇总，以及当前 `simulator_output/` 输出件。

## 结论

当前导出链路**基本能完整生成多格式输出，且 JSON/CSV/summary 的核心计数在当前样例中自洽**；但从设计原则看，它只能算“部分符合”：

| 设计原则 | 结论 | 主要原因 |
|---|---|---|
| Side-loaded 实验 | 基本符合 | 导出逻辑位于 `torchtitan/experiments/simulator/`，没有修改 `train.py`；但导出函数会反向修改 `SimulationResult.metadata`，使导出不再是纯输出动作。 |
| 捕获忠实 | 部分符合 | 主体依赖捕获到的 `SimulationResult`，但 HTML/Chrome trace 的 schedule timing fallback 会按 phase 均分时间；PP comm 和合成 compute anchor 也会把非捕获信息写入最终输出。 |
| PyTorch 原生 | 基本符合 | 导出使用 PyTorch 捕获结果和调度对象派生数据，没有重新实现训练主逻辑；HTML 前端依赖 ECharts CDN，不影响 PyTorch 原生训练原则，但影响离线交付。 |
| 模型无关核心 | 部分违反 | 导出前处理会访问 `trainer.model_parts` 并通过模型结构猜测 layer/hidden dim，核心导出管线暴露了模型反射逻辑。 |

## 当前输出件的自洽性

当前 `simulator_output/` 文件完整，但体积较大：

| 文件 | 大小 | 说明 |
|---|---:|---|
| `workload_graph.json` | 391 MB | L1-L3 workload graph，当前最大输出件。 |
| `simulation_result.json` | 240 MB | 完整结构化结果。 |
| `trace.json` | 113 MB | Chrome trace。 |
| `compute_graph.dot` | 42 MB | Graphviz DOT。 |
| `kernel_summary.csv` | 31 MB | 逐 op 性能明细。 |
| `trace.html` | 4.2 MB | 交互式 HTML，可视化数据经过压缩/截断。 |
| `summary.txt` | 8 KB | 文本汇总。 |

已校验的当前样例：

| 项目 | 结果 |
|---|---:|
| JSON compute nodes | 309247 |
| JSON edges | 285018 |
| CSV rows | 309247 |
| schedule events | 6336 |
| schedule deps | 13624 |
| `result.comm_events` | 48 |
| graph `comm_collective` | 4554 |
| graph `comm_p2p` | 48 |
| memory events | 121814 |
| CSV 负 duration | 0 |
| CSV `ComputeTime > Duration` | 0 |
| CSV `CommTime > Duration` | 0 |
| CSV `Duration != End-Start` | 0 |

这说明当前输出的基础结构没有明显损坏；主要问题集中在**指标口径、导出副作用、HTML 展示和大图可扩展性**。

## 问题清单

### P0. HTML 的 Activation peak 当前显示错误

**现象**

当前 `trace.html` 卡片显示：

```text
Activation peak: 0 B
```

但 `summary.txt` 和 `simulation_result.json` 中存在顶层内存峰值：

```text
peak_live_bytes: 257551912868
graph_peak_live_bytes: 257551912868
```

**证据**

- `export.py:1037-1040`：HTML 从 `result.metadata["memory"]` 读取 `peak_live_bytes` / `graph_peak_live_bytes`。
- `summary.txt:103-104`：当前样例的 `peak_live_bytes` 和 `graph_peak_live_bytes` 位于 metadata 顶层。
- `trace.html:36`：当前页面实际渲染为 `Activation peak = 0 B`。

**影响**

HTML 首页最重要的内存峰值卡片直接误导用户，会让用户以为激活峰值不存在。

**设计原则判断**

违反“捕获忠实”：已捕获并导出的内存峰值没有被正确展示。

### P0. 文本 summary 和 HTML 的通信事件口径容易误导

**现象**

`summary.txt` 的 “Communication Events” 只显示 48 个 send/recv：

```text
Total comm events: 48
send: 24
recv: 24
```

但同一个 compute graph 中还有：

```text
comm_collective: 4554
comm_p2p: 48
```

**证据**

- `export.py:1847-1854`：`export_text_summary()` 的 “Communication Events” 只统计 `result.comm_events`。
- `export.py:1139`：HTML 卡片 `Communication events` 也只显示 `len(result.comm_events)`。
- `summary.txt:7-10`：graph 类型计数中存在 4554 个 `comm_collective` 和 48 个 `comm_p2p`。
- `summary.txt:18-23`：通信事件章节只显示 48。

**影响**

用户会误判“通信很少”，而实际图中存在大量 collective 通信节点。该问题尤其影响性能瓶颈判断和硬件仿真输入检查。

**设计原则判断**

违反“捕获忠实”的展示层含义：捕获到的通信算子存在，但汇总卡片只展示了另一套窄口径事件。

### P0. 导出结果依赖 output format 顺序，`text` 单独导出可能缺 DES 汇总

**现象**

DES metadata 不是在仿真结束时统一生成，而是在部分导出函数内部懒加载写入：

- `export_json()` 调用 `_populate_des_metadata()`。
- `export_kernel_summary_csv()` 调用 `_populate_des_metadata()`。
- `export_html()` 通过 `_json_script_payload()` 调用 `_populate_des_metadata()`。
- `export_text_summary()` 自己不调用 `_populate_des_metadata()`。

当前默认顺序中 JSON/HTML 在 text 之前，所以 `summary.txt` 有 DES 汇总；但如果用户只配置 `output_formats=["text"]`，或配置 `["text", "csv"]`，文本汇总可能不会包含 DES Engine / DES Memory 章节。

**证据**

- `trainer_runner.py:114-128`：导出顺序固定为 json、dot、chrome_trace、html、text、csv。
- `export.py:643-666`：`_populate_des_metadata()` 直接写入 `result.metadata`。
- `export.py:1802-2007`：`export_text_summary()` 只读取已有 metadata，不主动补齐 DES metadata。

**影响**

同一次仿真，改变 `output_formats` 会改变 `summary.txt` 内容。这不符合用户对“导出格式只影响输出文件，不影响指标计算”的预期。

**设计原则判断**

违反 Side-loaded/关注点分离：导出函数带有计算和状态修改副作用；也削弱导出结果可复现性。

### P1. HTML 的 Predicted step time 与 DES step time 展示逻辑不一致

**现象**

当前 `trace.html` 首页显示：

```text
Predicted step time: 0.0 µs
```

但 `simulation_result.json` / `summary.txt` 中 DES 结果为：

```text
E2E step time (DES): 40.810 s
```

**证据**

- `export.py:1047-1048`：HTML 的 `Predicted step time` 只读 `metadata["cost_model"]["e2e_step_time_us"]`。
- `export.py:1051-1072`：DES 卡片只在任意 node 有 `des_start_time_us` 时显示。
- `des_engine.py:337-442`：DES 汇总支持从 schedule events 计算，即使 node 没有 DES 时间。
- `trace.html:38`：当前页面显示 `Predicted step time = 0.0 µs`。
- `summary.txt:69-80`：当前文本汇总有 DES step time 40.810 s。

**影响**

首页性能指标与文本/JSON 不一致。用户打开 HTML 时会以为性能预测没有运行。

**设计原则判断**

违反“捕获忠实”的展示层一致性：DES 已经产出，但 HTML 没有按同一口径展示。

### P1. HTML 的 Schedule events 数量混合了不同来源

**现象**

`summary.txt` 显示训练调度事件为 6336；HTML 首页卡片显示 `Schedule events = 6384`。

**证据**

- `export.py:971-990`：`_schedule_events_for_html()` 将 `result.schedule.events`、`fsdp_events`、`pp_events`、`comm_events` 合并为 HTML schedule events。
- `export.py:1138`：HTML 卡片直接显示合并后的 `len(schedule_events)`。
- `summary.txt:37`：训练调度事件为 6336。
- `trace.html:32`：HTML 显示 6384，正好是 6336 + 48 个 `result.comm_events`。

**影响**

“Schedule events” 标签不准确。它实际是“可视化泳道事件数”，不是纯训练调度事件数。

**设计原则判断**

属于导出语义问题：不是计算错误，但输出标签和数据来源不一致。

### P1. 大图 HTML 数据存在截断，但标记不完整

**现象**

当 graph node 数超过 10000 时，HTML payload 会走压缩路径：

- schedule event 的 `op_node_ids` 超过 100 会截断，并设置 `op_node_ids_truncated=True`。
- `memory_events` 只保留前 1000 条，但没有同等的 `truncated` / `original_count` 标记。
- compute graph 会按 PP stage 相似性压缩。

**证据**

- `export.py:795-820`：大图 HTML payload 压缩、截断 schedule op ids、只导出前 1000 个 memory events。
- 当前样例：JSON 有 121814 个 memory events，但 HTML payload 最多只含 1000 个。

**影响**

HTML 作为可视化摘要是合理的，但页面和 metadata 没有清楚说明“哪些数据被截断、原始数量是多少”。下游用户如果误把 HTML 内嵌 JSON 当完整数据，会得出错误结论。

**设计原则判断**

不直接违反捕获忠实，但违反导出透明性。应明确区分“完整输出件”和“可视化摘要输出件”。

### P1. DES Memory 的全局 peak 与 per-phase peak 不一致

**现象**

当前 DES memory 汇总：

```text
Peak dynamic memory: 1.8 GiB
Per-phase forward dynamic: 0 B
Per-phase backward dynamic: 0 B
```

**证据**

- `des_engine.py:652-689`：per-phase peak 通过 `phase_ranges` 中的时间区间筛选 timeline sample。
- `summary.txt:85-92`：全局 dynamic peak 非零，但 forward/backward dynamic 都为 0。
- 当前 JSON：`peak_dynamic_bytes=1879048192`，`phase_peak.forward.peak_dynamic_bytes=0`，`phase_peak.backward.peak_dynamic_bytes=0`。

**影响**

用户无法判断动态峰值属于哪个阶段。当前结果可能表示动态通信 buffer 的生命周期落在 phase range 外，也可能是 phase range 构造没有覆盖 schedule-derived 时间。

**设计原则判断**

部分违反捕获忠实：全局数据被捕获/推导出来了，但按 phase 展示时丢失归属。

### P1. 导出前处理访问模型对象，核心导出管线不够模型无关

**现象**

导出前的后处理阶段会访问模型对象：

- `_project_pp_comm_from_schedule()` 通过 `_guess_hidden_dim(trainer.model_parts[0])` 推断 PP 激活 shape。
- `_inject_synthetic_compute_anchors()` 通过 `_infer_num_layers(model_parts)` 和 `_guess_hidden_dim()` 注入合成 compute。
- `attach_model_state_memory()` 直接使用 `trainer.model_parts` 做模型状态内存估算。

**证据**

- `trainer_runner.py:250-261`：PP comm projection 使用 batch/seq/hidden，其中 hidden 来自模型反射。
- `trainer_runner.py:335-451`：合成 compute anchor 注入。
- `trainer_runner.py:453-494`：通过 `config.n_layers`、`layers`、参数名前缀、首个 `nn.Linear` 猜测模型结构。
- `trainer_runner.py:722-729`：导出前调用 `attach_model_state_memory(result, trainer.model_parts, ...)`。

**影响**

导出结果中部分 shape/内存/compute lane 数据不再纯粹来自捕获图或声明 config，而依赖模型对象结构启发式。跨模型迁移时容易出现“能导出但含义偏差”的问题。

**设计原则判断**

违反“模型无关的核心”。`export.py` 本身较模型无关，但导出管线中的结果构造/补全逻辑已经暴露模型反射。

### P2. Chrome trace 和 HTML 的 fallback schedule timing 是启发式估算

**现象**

若 schedule event 没有 DES start/finish，`_inject_schedule_timing()` 会把同一 phase 的总时间均分给该 phase 下的 schedule events，并按累计值生成起点。

**证据**

- `export.py:830-900`：无 DES event timing 时，使用 `per_event = phase_total / count` 和 `cumulative_per_phase`。
- `export.py:903-925`：event type 到 phase 的映射也包含默认 forward fallback。

**影响**

这种 fallback 对可视化有用，但不应被解释为真实调度时序。当前函数注释只写“using DES results when available”，没有显式说明非 DES 情况是 phase-level 均分。

**设计原则判断**

部分违反“捕获忠实”的文档表达：代码实际含启发式时序，应在输出中标记来源。

### P2. HTML 不是完全离线自包含

**现象**

HTML 数据内联，但图表库通过 CDN 加载：

```html
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
```

**证据**

- `export.py:1024-1029`：docstring 已说明 ECharts 从 CDN 加载，页面不是 fully offline。
- `trace.html:7`：实际输出引用 jsdelivr CDN。

**影响**

离线环境、内网环境或归档报告打开时，交互图表不可用。

**设计原则判断**

不直接违反四条核心训练原则，但影响输出件可交付性；如果设计文档描述“HTML 自包含”，则文档需要改成“数据自包含，渲染库非自包含”。

### P2. 输出体积大，缺少分层/摘要输出策略

**现象**

当前单次输出约 821 MB，其中 `workload_graph.json`、`simulation_result.json`、`trace.json`、`compute_graph.dot` 合计约 786 MB。

**影响**

对 CI artifact、远程传输、浏览器加载、人工审阅都偏重。当前 HTML 做了压缩，但完整 JSON/DOT/trace 仍按全量输出。

**设计原则判断**

不违反核心原则，但影响“任意规模训练步分析”的可用性。大规模仿真需要明确 full / compact / summary-only 三类输出模式。

## 建议整改优先级

| 优先级 | 建议 | 目标 |
|---|---|---|
| P0 | 在仿真后处理阶段统一生成 DES metadata，不放在具体 exporter 内懒加载。 | 消除 format 顺序依赖；让 `text` 单独导出也完整。 |
| P0 | 修复 HTML Activation peak：从正确 metadata 层级读取，或在 metadata 中统一内存 summary schema。 | 避免首页内存指标错误。 |
| P0 | 拆分通信指标：`Graph communication ops`、`Projected/raw comm events`、`Schedule visualization events` 分别展示。 | 避免通信数量误读。 |
| P1 | HTML 首页优先展示 DES E2E step time；当 cost model summary 为 0 但 DES metadata 存在时，不显示 0.0 µs。 | 保持 HTML 与 summary/JSON 一致。 |
| P1 | 为 HTML compact payload 增加 `truncation` metadata：原始数量、保留数量、截断字段。 | 避免把摘要数据误认为完整数据。 |
| P1 | 修正或解释 DES Memory per-phase dynamic 为 0 的原因。 | 让内存峰值可归因。 |
| P1 | 把 `_infer_num_layers()`、`_guess_hidden_dim()`、合成 compute anchor 的模型相关逻辑移出通用导出管线，改为模型子目录提供 metadata 或 adapter。 | 恢复模型无关核心边界。 |
| P2 | 增加 `summary-only` / `compact-json` / gzip 输出选项，或默认压缩大型 JSON/DOT。 | 改善大规模输出可用性。 |
| P2 | HTML 支持离线模式：内嵌 ECharts、使用本地 asset，或输出无网络 fallback 提示。 | 提升报告可交付性。 |

## 审计判断

导出逻辑不是“不可用”，当前输出的主干数据也是可校验的；问题在于**导出层混合了承载真实数据、补全推导、可视化摘要和指标计算副作用**。这会导致同一份结果在 JSON、summary、HTML 中出现不同口径，尤其是通信、性能时间和内存峰值。

下一步应优先把“仿真后处理/指标计算”和“格式化输出”分离：前者只运行一次并写入标准 metadata schema，后者只读取 schema 并明确标注完整输出或摘要输出。这样最符合 `DESIGN_CN.md` 中的捕获忠实、side-loaded 和模型无关核心原则。
