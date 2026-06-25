# TorchTitan Simulator 设计原则审查清单

> 依据：`torchtitan/experiments/simulator/DESIGN_CN.md` 的“设计原则”章节。
> 目标：快速判断当前 simulator 实现是否违背这些原则，并给出证据位置。

## 审查结论

- [x] **Side-loaded 实验**：符合
- [x] **捕获忠实**：符合（PP 通信改为调度投影，合成注入已消除）
- [x] **PyTorch 原生**：符合

## 逐项清单

### 1. Side-loaded 实验：`train.py` 保持不变，代码只在 simulator 目录内

- [x] `train.py` 未被 simulator 逻辑侵入
- [x] simulator 主要实现位于 `torchtitan/experiments/simulator/`
- [x] 入口仍是 `SimulationTrainer` / `run_train.sh`

**证据：**
- `torchtitan/experiments/simulator/DESIGN_CN.md:20-23`
- `torchtitan/experiments/simulator/__init__.py:20-43`
- `torchtitan/experiments/simulator/trainer.py:392-483`

**结论：** 符合。

### 2. 捕获忠实：调度和计算图应来自捕获数据或真实 PyTorch 调度对象

- [x] 主路径使用 `unified_trace()` 捕获真实 dispatch
- [x] 语义调度使用 `extract_schedule_from_pytorch()`
- [x] PP 通信从语义调度投影（`_project_pp_comm_from_schedule`），来源为真实 `_PipelineSchedule`
- [x] 计算锚点为条件注入（Phase 4 自然捕获充足时自动跳过）

**证据：**
- `torchtitan/experiments/simulator/trainer_runner.py` — `_project_pp_comm_from_schedule()` 从 `result.schedule` 投影 PP 事件
- `torchtitan/experiments/simulator/trainer_runner.py` — `_inject_synthetic_compute_anchors()` 有 gap 检查，Phase 4 下注入 0 个
- smoketest E2E 验证：171 自然 + 48 调度投影 + 0 合成 = 219 通信算子

**结论：** 符合。合成注入已消除，PP 通信来自真实调度投影。

### 3. PyTorch 原生：复用上游 `PipelineSchedule`、FX tracing、`TorchDispatchMode`

- [x] `unified_trace()` 基于 `TorchDispatchMode`
- [x] `extract_schedule_from_pytorch()` 基于真实 `_PipelineSchedule`
- [x] `SimulationTrainer` 仍沿用上游 `Trainer` 初始化路径

**证据：**
- `torchtitan/experiments/simulator/capture/unified_trace.py`
- `torchtitan/experiments/simulator/schedule/schedule_extract.py`
- `torchtitan/experiments/simulator/trainer.py:463-483`

**结论：** 符合。

## 最终判定

- **符合**：Side-loaded 实验、捕获忠实、PyTorch 原生

## 建议关注点

- PP 调度投影的张量形状仍从 `batch * seq_len * hidden` 计算（非捕获），可考虑从模型参数维度推导
- 强制负载均衡 MoE dispatch（§20.6）是仿真近似，真实路由非均匀
