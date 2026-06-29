# 方案设计：导出逻辑审计整改

> 依据：`EXPORT_LOGIC_AUDIT_CN.md` 审计清单。
> 核心问题：导出层混合了真实数据、启发式补全、可视化摘要和指标计算副作用。
> 设计目标：**分离"后处理/指标计算"与"格式化输出"**。

## 1. 架构整改：后处理与导出分离

### 现状

```
run_trainer_simulation:
  build_result → schedule → PP comm → compute_anchors → cost_model → memory
  → _export_result()
      → export_json()        → _populate_des_metadata() ← 副作用：写 metadata
      → export_dot()
      → export_chrome_trace()
      → export_html()        → _json_script_payload() → _populate_des_metadata() ← 副作用
      → export_text_summary() ← 不调用 _populate_des_metadata，可能缺 DES
      → export_kernel_csv()  → _populate_des_metadata() ← 副作用
```

**问题**：DES metadata 在导出函数内懒加载，导致 `text` 单独导出时缺失；导出函数有写副作用。

### 改后

```
run_trainer_simulation:
  build_result → schedule → PP comm → compute_anchors → cost_model → memory
  → _populate_des_metadata(result)     ← 新增：统一在此处生成
  → _populate_schedule_timing(result)  ← 新增：统一在此处生成 fallback timing
  → _export_result()
      → export_json()        ← 纯读取
      → export_dot()         ← 纯读取
      → export_chrome_trace() ← 纯读取
      → export_html()        ← 纯读取
      → export_text_summary() ← 纯读取（现在有 DES 数据）
      → export_kernel_csv()  ← 纯读取
```

**改动点**：
1. `trainer_runner.py`：在 `_export_result` 调用前，显式调用 `_populate_des_metadata(result)` 和 `_inject_schedule_timing(result.to_dict(), result)`（将 timing 写回 result.metadata）。
2. `export.py`：`_populate_des_metadata()` 保持幂等（已有 early-return），但不再由各 exporter 调用。`_json_script_payload` 和 `export_json`/`export_kernel_csv` 中的调用删除。
3. `export_text_summary`：现在能读到 DES metadata，不再依赖 format 顺序。

## 2. P0 问题修复

### P0-1: HTML Activation peak 显示 0

**根因**：`build_runtime_memory()` 通过 `result.metadata.update(memory_summary)` 将 `peak_live_bytes` 写到 `result.metadata` 顶层，但 HTML 从 `result.metadata["memory"]["peak_live_bytes"]` 读取（嵌套层级）。

**修复**：在 `trainer_runner.py` 后处理中，将 memory summary 同时写入 `result.metadata["memory"]` 子 dict：
```python
result.metadata.update(memory_summary)
result.metadata.setdefault("memory", {}).update(memory_summary)
```

### P0-2: 通信事件口径误导

**根因**：`result.comm_events` 只有 48 个 PP 事件；4554 个 collective comm ops 在计算图中是 `OpNode`，不在 `comm_events` 列表。

**修复**：在 `export_text_summary` 和 HTML 卡片中拆分展示：
- **Graph communication ops**：从 `compute_graph.nodes` 统计 `comm_collective` + `comm_p2p`（= 4602）
- **PP projected events**：`len(result.comm_events)`（= 48）
- HTML 卡片改标签为 "Graph comm ops" 和 "PP projected events"

### P0-3: DES metadata 依赖 format 顺序

**根因**：`_populate_des_metadata()` 在 exporter 内懒加载。

**修复**：已在 §1 架构整改中解决 — 统一在后处理阶段调用。

## 3. P1 问题修复

### P1-1: HTML Predicted step time 显示 0.0 µs

**根因**：HTML 读 `metadata["cost_model"]["e2e_step_time_us"]`（=0，因为 cost_model metadata 为空），但实际值在 `metadata["des_engine"]["e2e_step_time_us"]`（=40.8s）。

**修复**：HTML 读取逻辑增加 fallback：
```python
perf_grand_total_us = (
    cost_summary.get("e2e_step_time_us", 0)
    or result.metadata.get("des_engine", {}).get("e2e_step_time_us", 0)
)
```

### P1-2: Schedule events 数量混合来源

**根因**：HTML 的 `_schedule_events_for_html()` 合并了 schedule + fsdp + pp + comm 事件（6384），但标签写 "Schedule events"（暗示纯调度事件 = 6336）。

**修复**：HTML 卡片标签改为 "Swimlane events"（准确描述合并后的可视化事件数）。或在卡片旁加 `(6336 schedule + 48 comm)`。

### P1-3: HTML 截断标记不完整

**根因**：大图（>10000 节点）截断 memory_events 到 1000，无 truncation 标记。

**修复**：在 `_json_script_payload` 的 compact 分支中，添加截断元数据：
```python
compact["_truncation"] = {
    "memory_events_original": len(result.memory_events),
    "memory_events_kept": 1000,
    "graph_compressed": True,
    "op_node_ids_max": 100,
}
```

### P1-4: DES Memory per-phase peak = 0

**根因**：`compute_des_memory_timeline` 的 `phase_ranges` 基于 DES 时间戳，但 schedule-derived PP 事件没有 DES 时间（只有 fallback timing），导致 phase range 不覆盖动态内存事件。

**修复**：在 `_populate_schedule_timing` 写回 result 后，确保 phase_ranges 使用 fallback timing 的时间区间。或在后处理阶段统一注入 schedule timing（§1），让 DES memory timeline 能读到。

### P1-5: 导出管线中的模型反射

**根因**：`_guess_hidden_dim`、`_infer_num_layers` 访问模型对象。

**修复**：将模型结构信息（hidden_dim, num_layers）在 `SimulationTrainer.__init__` 中提取并写入 `result.metadata["model_info"]`，后处理和导出只读 metadata。**此为较大重构，列为后续工作。**

## 4. P2 问题（低优先级）

| 问题 | 修复方案 | 优先级 |
|------|----------|--------|
| Fallback timing 是启发式 | 在 event 中标注 `timing_source: "des"` 或 `"phase_even_split"` | P2 |
| HTML 非离线 | Vendor ECharts 或加离线提示 | P2 |
| 输出体积大 | 增加 `summary-only`/`compact` 输出模式 | P2 |

## 5. 实施计划

| 步骤 | 文件 | 改动 | 优先级 |
|------|------|------|--------|
| 1 | `trainer_runner.py` | 在 `_export_result` 前调用 `_populate_des_metadata` + `_inject_schedule_timing` | P0 |
| 2 | `trainer_runner.py` | 修复 memory summary 写入 `metadata["memory"]` 子 dict | P0 |
| 3 | `export.py` | `export_text_summary` + HTML 卡片：拆分通信指标展示 | P0 |
| 4 | `export.py` | HTML Predicted step time：增加 DES fallback | P1 |
| 5 | `export.py` | HTML Schedule events 标签改为 "Swimlane events" | P1 |
| 6 | `export.py` | `_json_script_payload` compact 分支：添加 truncation 元数据 | P1 |
| 7 | `export.py` | 移除各 exporter 中的 `_populate_des_metadata` 调用 | P0 |
| 8 | `export.py` | `_inject_schedule_timing` 标注 timing_source | P2 |
| 9 | 测试 + E2E 验证 | 运行 smoketest + Pro，检查输出一致性 | — |
