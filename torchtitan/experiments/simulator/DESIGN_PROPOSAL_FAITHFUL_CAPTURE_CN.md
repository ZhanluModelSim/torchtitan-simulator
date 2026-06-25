# 方案设计：消除合成注入，实现完全捕获忠实

> 依据：`DESIGN_PRINCIPLES_CHECKLIST_CN.md` 审查结论 — "捕获忠实"部分符合。
> 目标：消除两个合成注入路径，使主路径完全忠实。

## 1. 现状

```
后处理流程 (trainer_runner.py:1350-1356):
  1. _inject_synthetic_compute_anchors(result, trainer)   ← 合成计算锚点
  2. _inject_pp_send_recv(result, trainer)                ← 合成 PP 通信
  3. _inject_semantic_schedule(result, config)            ← 语义调度 (忠实)
```

Phase 4 之后模型实际运行（`SequentialPipelineSchedule`），捕获图从 110 节点增长到 8169 节点。
两个合成注入的存在理由已大幅削弱：

| 注入函数 | 原存在理由 | Phase 4 后状态 |
|----------|-----------|---------------|
| `_inject_synthetic_compute_anchors` | MockSchedule 不运行模型，捕获图为空 | **不再需要** — 自然捕获有 2484 个计算节点 |
| `_inject_pp_send_recv` | PP 用 `dist.send/recv`，dispatch 捕获不到 | **可替代** — `extract_schedule_from_pytorch` 已从真实 `_PipelineSchedule` 提取 PP 事件 |
| `_inject_synthetic_comm_events` (死代码) | Phase 4 前的 FSDP/TP 合成注入 | **可删除** — 已被自然捕获取代 |

## 2. 方案

### 2.1 PP 通信：从启发式注入 → 调度投影

**当前** `_inject_pp_send_recv`（~240 行）：
- 启发式假设：均匀层分布、手动计算 `batch * seq_len * hidden`
- 为每个 microbatch × stage boundary 创建 send/recv 对
- `attrs={"synthetic": True, "pp": True}`

**改为** `_project_pp_comm_from_schedule`（~80 行）：
- 从 `result.schedule`（由 `extract_schedule_from_pytorch` 产生）读取 PP 事件
- 事件来源：真实 `_PipelineSchedule.pipeline_order_with_comms` action 表
- 事件已有正确的 `pp_stage`、`pp_rank`、`microbatch_idx`
- 将 `ScheduleEvent`（`pp_send_activation` / `pp_recv_activation` / `pp_send_gradient` /
  `pp_recv_gradient`）投影为 `OpNode`（`comm_p2p`，`comm_op="send"/"recv"`）
- 张量形状仍从 `batch * seq_len * hidden` 计算（与当前相同），但与调度事件绑定
- `attrs={"schedule_derived": True}` — 标注来源为真实调度，非启发式

**忠实性论证**：设计原则说"调度和计算图应来自捕获数据**或真实 PyTorch 调度对象**"。
PP 通信来自 `extract_schedule_from_pytorch`（基于真实 `_PipelineSchedule`），符合原则。

### 2.2 计算锚点：从总是注入 → 条件注入

**当前** `_inject_synthetic_compute_anchors`：检查 Cube/Vec 缺口，有缺口就注入。

**改为**：增加前置检查 — 如果自然捕获的计算节点数已超过阈值（`pp * num_layers`），
直接返回，不注入任何合成节点。

```python
def _inject_synthetic_compute_anchors(result, trainer):
    # ... 计算 lane_target ...
    # Phase 4: 自然捕获已产生足够计算节点时，跳过注入
    total_compute = sum(1 for n in graph.nodes.values()
                        if n.op_type == "compute" and n.phase in ("forward", "backward"))
    if total_compute >= lane_target * 2:  # Cube + Vec 各达标
        return  # 自然捕获足够，无需合成
    # ... 原有注入逻辑 ...
```

Phase 4 smoketest：2484 个计算节点 >> `2 * 2 * 2 = 8` 阈值 → 跳过注入。

### 2.3 死代码清理

- 删除 `_inject_synthetic_comm_events`（~490 行）
- 迁移 `tests/test_simulator.py` 中引用它的 2 个测试用例
  （`TestSyntheticCommInjection` 类 → 改用自然捕获验证）

### 2.4 后处理流程重排

```
改后流程:
  1. result = recorder.build_result()
  2. _inject_semantic_schedule(result, config)        ← 先产生调度（忠实）
  3. _project_pp_comm_from_schedule(result, trainer)   ← 从调度投影 PP 通信（忠实）
  4. _inject_synthetic_compute_anchors(result, trainer) ← 条件注入（通常跳过）
```

关键变化：`_inject_semantic_schedule` 提前到 PP 投影之前，因为 PP 投影依赖调度事件。

## 3. 实现步骤

| 步骤 | 文件 | 内容 |
|------|------|------|
| 1 | `trainer_runner.py` | 新增 `_project_pp_comm_from_schedule()` 函数 |
| 2 | `trainer_runner.py` | 后处理流程重排（§2.4） |
| 3 | `trainer_runner.py` | `_inject_synthetic_compute_anchors` 增加前置检查 |
| 4 | `trainer_runner.py` | 删除 `_inject_synthetic_comm_events` |
| 5 | `tests/test_simulator.py` | 迁移 `TestSyntheticCommInjection` 测试 |
| 6 | `DESIGN.md` / `DESIGN_CN.md` | 更新 §15/§21 — PP 标注为"调度投影"，锚点标注为"条件注入" |
| 7 | `DESIGN_PRINCIPLES_CHECKLIST_CN.md` | 更新审查结论 — "捕获忠实"→符合 |

## 4. 预期结果

改后通信来源分类：

| 来源 | 算子 | 忠实性 |
|------|------|--------|
| **自然捕获** | TP all_reduce, FSDP2 all_gather/reduce_scatter, EP all_to_all, wait_tensor | 来自 dispatch（真实执行） |
| **调度投影** | PP send/recv | 来自 `extract_schedule_from_pytorch`（真实 `_PipelineSchedule`） |
| ~~合成注入~~ | ~~PP send/recv~~ | ~~已删除~~ |
| ~~合成计算~~ | ~~Cube/Vec 锚点~~ | ~~条件注入，Phase 4 下通常不触发~~ |

审查清单结论从"部分符合"→"符合"。
