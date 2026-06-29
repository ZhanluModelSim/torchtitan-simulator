# Simulator 导出逻辑二次审计清单

> 审计日期：2026-06-25  
> 审计依据：当前 `simulator_output/` 输出件、`DESIGN_PROPOSAL_EXPORT_FIX_CN.md`、当前 `export.py` / `trainer_runner.py` / `des_engine.py` 实现。
> 本文覆盖上一版 `EXPORT_LOGIC_AUDIT_CN.md`。

## 结论

本轮整改已经修复了上一版审计中的多项展示口径问题：HTML 首页的 Activation peak、Predicted step time、通信统计标签、Swimlane events 标签已经与当前输出件对齐；文本 summary 也已拆分 graph comm ops 与 PP projected events。

但当前实现仍有三个需要优先处理的问题：

1. **Schedule swimlane 前四个 rank 与后四个 rank 明显不对称，根因在导出前的 PP comm projection 只处理 `rank == 0`。** 当前 JSON 中 rank 0-3 的 PP comm event 都绑定了 `op_node_ids`，duration 约 1-3 us；rank 4-7 同类 event 没有 `op_node_ids`，DES 回退到 phase 均分，duration 约 199 us / 442 us。HTML 只是忠实渲染了这组异常数据。
2. **“后处理与导出分离”只做到部分落地。** `trainer_runner.py` 已经在 `_export_result()` 前统一生成 DES metadata 和 schedule timing；但 `export_json()`、`export_kernel_summary_csv()`、`_json_script_payload()` 内仍保留 `_populate_des_metadata()`，导出函数仍不是纯读取。
3. **新增 `_schedule_events_enriched` metadata 造成 summary 和 JSON metadata 膨胀。** 当前 `summary.txt` 达到 276 KB，其中大部分来自把 696 条 enriched schedule events 作为 metadata 原样打印；这与“summary 应为人类可读摘要”的目标冲突。

## 当前输出件概况

当前 `simulator_output/` 已不再是上一轮的 800MB 级输出，输出体积明显下降：

| 文件 | 当前大小 |
|---|---:|
| `summary.txt` | 276 KB |
| `kernel_summary.csv` | 552 KB |
| `compute_graph.dot` | 836 KB |
| `trace.json` | 2.5 MB |
| `trace.html` | 5.9 MB |
| `workload_graph.json` | 7.9 MB |
| `simulation_result.json` | 9.7 MB |

当前核心计数：

| 项目 | 当前值 |
|---|---:|
| compute graph nodes | 6275 |
| compute graph edges | 4930 |
| schedule events | 696 |
| schedule deps | 1390 |
| HTML swimlane events | 744 |
| graph comm ops | 219 |
| PP projected events / `result.comm_events` | 48 |
| memory events | 2321 |
| CSV rows | 6275 |
| CSV 负 duration | 0 |
| CSV `ComputeTime > Duration` | 0 |
| CSV `CommTime > Duration` | 0 |

## 整改方案落地情况

| 方案项 | 当前状态 | 证据 | 审计结论 |
|---|---|---|---|
| DES metadata 前移到导出前统一生成 | 部分完成 | `trainer_runner.py:734-746` | 已前移，但 exporter 内仍保留懒加载调用。 |
| memory summary 同时写入 `metadata["memory"]` | 完成 | `trainer_runner.py:720-723`；`trace.html:37` | HTML Activation peak 已显示 261.2 MiB。 |
| 通信指标拆分 | 完成 | `summary.txt:18-29`；`trace.html:33-34` | Graph comm ops=219，PP projected events=48。 |
| Predicted step time 使用 DES fallback | 完成 | `export.py:1056-1062`；`trace.html:39` | HTML 显示 36.194 ms，不再是 0。 |
| Schedule events 标签改为 Swimlane events | 完成 | `export.py:1152`；`trace.html:32` | 标签与“schedule + comm 可视化事件”语义一致。 |
| compact HTML truncation metadata | 代码实现，当前输出未触发 | `export.py:814-826` | 当前 graph 6275 nodes，小于 10000，因此没有 `_truncation` 是合理的。 |
| `timing_source` 标注 | 完成 | `export.py:884-897`；当前 HTML 中出现 1392 次 `timing_source` | schedule timing 来源可见。 |
| 移除 exporter 内 `_populate_des_metadata()` | 未完成 | `export.py:68`、`export.py:183`、`export.py:797` | 仍存在导出副作用入口，虽有 early-return 幂等保护。 |
| 模型反射从导出管线移出 | 未完成，方案也标为后续 | `trainer_runner.py:256-260`、`trainer_runner.py:348-371`、`trainer_runner.py:453-494` | `_guess_hidden_dim()` / `_infer_num_layers()` 仍在通用 runner。 |

## 已修复项确认

### 1. HTML Activation peak 已修复

当前 HTML 首页：

```text
Activation peak: 261.2 MiB
```

对应 summary / JSON：

```text
peak_live_bytes: 273906808
graph_peak_live_bytes: 273906808
```

这说明 `metadata["memory"]` 嵌套写入已生效。

### 2. HTML Predicted step time 已修复

当前 HTML 首页：

```text
Predicted step time: 36.194 ms
```

对应 DES summary：

```text
E2E step time (DES): 36.194 ms
```

`cost_model` metadata 为空时回退到 `metadata["des_engine"]["e2e_step_time_us"]` 的逻辑已生效。

### 3. 通信统计口径已拆分

当前 summary：

```text
Graph comm ops: 219
  all_gather: 61
  all_reduce: 19
  recv: 24
  reduce_scatter: 14
  send: 24
  wait: 77
PP projected events: 48
  recv: 24
  send: 24
```

当前 HTML 首页也拆分为：

```text
Graph comm ops: 219
PP projected events: 48
```

这比上一版只显示 `Communication events: 48` 更符合捕获数据。

## 重点问题：schedule swimlane 前四个 rank 异常

### 现象

当前 `simulation_result.json` 中 8 个 rank 的 schedule event 总数是对称的：

```text
rank 0-7: 每个 rank 87 events
```

但 PP comm event 的绑定和 duration 不对称：

| event type | rank 0-3 | rank 4-7 |
|---|---:|---:|
| `pp_send_activation` | 16/event rank，全部有 `op_node_ids`，平均 2.015 us | 8/event rank，全部无 `op_node_ids`，平均 199.114 us |
| `pp_recv_activation` | 8/event rank，全部有 `op_node_ids`，平均 2.658 us | 16/event rank，全部无 `op_node_ids`，平均 199.113 us |
| `pp_send_gradient` | 8/event rank，全部有 `op_node_ids`，平均 2.329 us | 16/event rank，全部无 `op_node_ids`，平均 441.857 us |
| `pp_recv_gradient` | 16/event rank，全部有 `op_node_ids`，平均 1.810 us | 8/event rank，全部无 `op_node_ids`，平均 441.857 us |

因此用户在 HTML schedule swimlane 中看到的“前四个 rank 异常”，本质是**前四个 rank 与后四个 rank 使用了不同 duration 来源**。

### 根因链路

1. `trainer_runner.py` 只为 `rank == 0` 的 PP comm event 投影 `comm_p2p` OpNode：

```python
pp_events = [
    e for e in schedule.events
    if e.event_type in _PP_EVENT_MAP and e.rank == 0
]
```

2. 当前 compute graph 中只有 48 个 `comm_p2p` 节点，全部属于 `pp_rank=0`，且只覆盖 `pp_stage=0` 和 `pp_stage=2`：

```text
comm_p2p nodes: 48
by_pp_rank: {0: 48}
by_stage: {0: 16, 2: 32}
```

3. `cost_model.link_schedule_to_graph()` 只按 `(phase, pp_stage, microbatch_idx)` / `(phase, pp_stage, None)` 绑定节点，不区分 rank。结果是：

- rank 0-3 共享 pp_rank 0 的 stage 0/2 节点，因此有 `op_node_ids`。
- rank 4-7 的 pp_rank 1 / stage 1 没有对应 `comm_p2p` 节点，因此 `op_node_ids=[]`。

4. `des_engine.simulate_multi_rank_des()` 对两种 event 采用不同 duration 计算：

- 有 `op_node_ids`：累计节点 `perf_result.total_time_us` 后按 `(event_type, rank, pp_stage)` 事件数拆分。
- 无 `op_node_ids`：回退到 `phase_total / event_type_count`。

这造成 rank 0-3 的 PP comm 是微秒级，而 rank 4-7 是百微秒级。

5. HTML schedule swimlane 渲染逻辑只读取 `perf_cumulative_start_us` 和 `perf_total_time_us`：

```javascript
const start = ev.perf_cumulative_start_us || 0;
const duration = ev.perf_total_time_us || 0;
```

因此 HTML 不是根因，只是把 JSON 中已经异常的 duration 画出来。

### 设计原则判断

该问题违反“捕获忠实”：

- PP schedule 本身是多 rank 的；
- 但导出前投影只为 `rank == 0` 建立 `comm_p2p` 节点；
- 后续 DES 和 HTML 用不同 fallback 填补缺失 rank，导致可视化和性能指标不再忠实反映同一套调度数据。

### 修复方向

优先修复 `_project_pp_comm_from_schedule()` 的 rank 过滤逻辑：

- 不应只过滤 `rank == 0`。
- 应按 schedule event 的真实 `rank` / `pp_rank` / `pp_stage` 为所有 PP comm event 投影节点，或明确只投影代表 rank 并在 schedule linking / DES duration 中也按代表 rank 对称复用。
- 修复后应重新生成输出，检查 rank 0-7 的同类 PP comm event 是否使用一致的 duration 来源。

## 仍需处理的问题

### P0. Exporter 内仍有 DES metadata 懒加载调用

当前 `trainer_runner.py` 已在导出前执行：

```python
_populate_des_metadata(result)
_inject_schedule_timing(_result_dict, result)
```

但 exporter 内仍有重复入口：

- `export.py:68`：`export_json()` 调用 `_populate_des_metadata(result)`。
- `export.py:183`：`export_kernel_summary_csv()` 调用 `_populate_des_metadata(result)`。
- `export.py:797`：`_json_script_payload()` 调用 `_populate_des_metadata(result)`。

虽然 `_populate_des_metadata()` 有：

```python
if "des_engine" in result.metadata:
    return
```

但从设计上看 exporter 仍不是纯读取，和 `DESIGN_PROPOSAL_EXPORT_FIX_CN.md` 的“格式化输出纯读取”目标不完全一致。

### P0. Summary 被 `_schedule_events_enriched` 撑大

当前 `trainer_runner.py` 将 enriched schedule events 写入 metadata：

```python
result.metadata["_schedule_events_enriched"] = _result_dict["schedule"]["events"]
```

当前输出中：

```text
_schedule_events_enriched length: 696
serialized bytes: 275776
summary.txt size: 276 KB
```

`export_text_summary()` 的 Metadata 章节会打印除 `cost_model`、`des_engine`、`des_memory` 外的所有 metadata，因此整份 schedule event 列表被原样写入 `summary.txt`。这不是摘要信息，影响人工阅读，也让 summary 变成半结构化 dump。

建议：

- 不要把 `_schedule_events_enriched` 放进通用 metadata；
- 或在 `export_text_summary()` 的 Metadata 章节过滤 `_schedule_events_enriched`；
- 如果 HTML 需要 enriched schedule，应在 HTML payload 中局部生成，不污染全局 metadata。

### P1. DES cards 仍未出现在 HTML 首页

当前 `trace.html` 首页有 `Predicted step time: 36.194 ms`，但没有 DES cards：

```text
DES step time: 0 occurrences
Compute utilization: 0 occurrences
Peak DES memory: 0 occurrences
```

原因是 `export_html()` 的 `has_des` 只检查 compute graph nodes：

```python
has_des = any(
    n.des_start_time_us is not None for n in result.compute_graph.nodes.values()
)
```

当前 DES 时间主要在 schedule events 上，compute graph nodes 没有 DES 时间：

```text
node_des_phase_ranges: {}
schedule_phase_ranges.forward: [0.0, 36193.561]
schedule_phase_ranges.backward: [4048.165, 36193.561]
```

建议让 HTML DES cards 与 `_populate_des_metadata()` 一样同时检查 schedule DES event，或直接以 `metadata["des_engine"]` 是否存在为准。

### P1. DES Memory per-phase dynamic peak 仍为 0

当前 DES memory：

```text
Peak dynamic memory: 1.0 MiB
Per-phase forward dynamic: 0 B
Per-phase backward dynamic: 0 B
```

根因仍在 `compute_des_memory_timeline()` 的 phase range 构造：它只从 node DES time 推导 phase range；当前 node 没有 DES time，而 schedule events 有 DES time。因此全局 timeline 能看到动态 comm buffer 峰值，但 per-phase peak 取样不到对应时间区间。

当前最大 dynamic sample：

```text
time_us: 3493.957
dynamic_bytes: 1048576
by_category: {"comm_buffer": 1048576}
```

但 `phase_peak.forward.dynamic=0`、`phase_peak.backward.dynamic=0`。

建议 `compute_des_memory_timeline()` 在 node DES time 缺失时，使用 schedule event DES ranges 构造 phase ranges。

### P1. `single_rank_step_time_us` 仍与 `e2e_step_time_us` 相同

当前 `des_engine.compute_des_utilization()` 返回：

```python
"e2e_step_time_us": round(e2e_step, 3),
"single_rank_step_time_us": round(e2e_step, 3),
```

该字段名仍有歧义：在 multi-rank DES 下它并不是独立计算的单 rank step time，而是复制 E2E step time。建议改名或删除，避免输出误导。

### P2. HTML 仍依赖 CDN

当前 `trace.html` 仍使用：

```html
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
```

这不影响本次 P0/P1 修复，但仍不适合离线归档或内网环境。

### P2. 模型反射仍在通用 runner

以下逻辑仍在通用 `trainer_runner.py`：

- `_project_pp_comm_from_schedule()` 用 `_guess_hidden_dim(trainer.model_parts[0])` 推断 activation shape。
- `_inject_synthetic_compute_anchors()` 用 `_infer_num_layers()` / `_guess_hidden_dim()` 注入 compute anchor。
- `attach_model_state_memory()` 仍直接读取 `trainer.model_parts`。

该问题已经在方案中标为后续较大重构；本轮不应阻塞导出展示修复，但仍不完全符合“模型无关核心”原则。

## 建议整改优先级

| 优先级 | 建议 | 验收方式 |
|---|---|---|
| P0 | 修复 `_project_pp_comm_from_schedule()` 的 `rank == 0` 过滤或同步修正 schedule linking / DES duration 对代表 rank 的处理。 | 重新生成输出后，rank 0-7 同类 PP comm event 的 `op_node_ids` 绑定策略和 duration 来源一致。 |
| P0 | 从 exporter 中移除 `_populate_des_metadata()` 懒加载调用。 | `export_json()`、`export_kernel_summary_csv()`、`_json_script_payload()` 不再修改 `SimulationResult`。 |
| P0 | 不在文本 summary 中打印 `_schedule_events_enriched`。 | `summary.txt` 回到 KB 级摘要，不包含 696 条 event dump。 |
| P1 | HTML DES cards 的 `has_des` 改为基于 `metadata["des_engine"]` 或 schedule DES event。 | HTML 首页出现 DES step time / utilization / peak DES memory 卡片。 |
| P1 | DES memory phase peak 使用 schedule DES ranges 兜底。 | `Peak dynamic memory` 非零时，对应 phase 的 dynamic peak 不再全部为 0。 |
| P1 | 澄清或移除 `single_rank_step_time_us`。 | 多 rank DES 输出中不再出现与 E2E 完全重复且命名误导的字段。 |
| P2 | 保留 CDN 问题为可交付性优化。 | 离线打开 HTML 时有本地 ECharts 或明确 fallback。 |
| P2 | 模型结构信息改由 metadata / adapter 提供。 | 通用 runner 不再通过模型反射猜测 hidden_dim / num_layers。 |

## 二次审计判断

当前整改方向正确，且已修复上一轮最明显的展示错误；但 rank swimlane 异常暴露出更深一层的问题：**PP schedule 是多 rank 的，PP comm projection 却只为 rank 0 构造图节点**。这使 DES 和 HTML 在不同 rank 上混用了“节点真实耗时”和“phase fallback 均分耗时”，是当前最优先的准确性问题。

在修复该问题前，当前 schedule swimlane 不应作为多 rank PP 通信时序的可靠依据；可以用于查看事件顺序，但不适合用来判断不同 rank 的 PP 通信耗时差异。
