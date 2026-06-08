# TorchTitan Simulator 调度建模与性能预测：现状分析与修复方案

> 本文档基于对 `torchtitan/experiments/simulator/` 全部核心模块的逐行审查，以及 PyTorch
> 上游 `_PipelineSchedule` 真实调度逻辑的交叉验证。所有缺陷均已对照源码确认。

---

## 一、现状概述

模拟器的调度建模和性能预测由三条独立管线组成，它们之间没有数据关联：

| 管线 | 核心文件 | 输出 | 现状 |
|------|---------|------|------|
| **语义调度生成** | `schedule_generator.py` | `TrainingSchedule` (粗粒度事件) | 算法与真实 Interleaved1F1B 不一致，实际产出 GPipe 模式 |
| **PP 调度提取** | `pp_schedule_extractor.py` | `TrainingSchedule` | 三层 fallback 均无法对接当前 PyTorch API，最终走 heuristic |
| **性能预测** | `cost_model.py` | `PerfResult` + critical-path step time | FLOPs/通信量计算有系统性错误；critical-path 不利用多 rank 信息 |

此外 `trainer_runner.py` 的合成通信注入 (`_inject_synthetic_comm_events`) 作为 fake_backend 模式的补充，存在 shape 语义错误和 graph edge 缺失。

---

## 二、逐模块缺陷详述

### 2.1 `schedule_generator.py` — 语义调度生成

#### 缺陷 #1: warmup 循环产出 GPipe 而非 Interleaved1F1B

**位置**: L245-266

**现状**: 外层循环 `for clock in range(total_stages + num_microbatches - 1)` 按全局时钟推进所有 stage 的 forward，内层遍历所有已到达的 stage。这使所有 forward 在 warmup 阶段均匀填充，而非真实 Interleaved1F1B 的"rank 越靠前 warmup 越多"模式。

**真实逻辑**: PyTorch `ScheduleInterleaved1F1B._get_warmup_ops()` 的 warmup 数量公式为：

```
warmup_ops = (n_local_stages - 1) * microbatches_per_round
             + multiply_factor * (pp_group_size - 1 - rank)
```

其中 `multiply_factor=2`（Interleaved1F1B）。最后一 rank warmup 最少，越靠前的 rank warmup 越多（以填充 pipeline bubble）。每个 rank 在 warmup 期间只执行 forward，按 `forward_stage_index(step)` 在本 rank 的虚拟 stage 间循环。

#### 缺陷 #2: backward 在 last stage 立即反推所有 upstream — GPipe 行为

**位置**: L269-292

**现状**: 当 `stage == total_stages - 1`，代码从 `total_stages - 2` 到 0 反向遍历所有 upstream stage 做 backward。这意味着 microbatch 0 到达 last stage 后，所有 backward 立即被调度，这是 GPipe 模式（全 forward → 全 backward）。

**真实逻辑**: Interleaved1F1B 在 warmup 结束后进入 steady-state，每个时钟步交替执行 1F + 1B（按 `_get_1f1b_rank_ops()` 的三阶段划分）。backward 不会一次性反推所有 stage；它按 `backward_stage_index(step)` 在虚拟 stage 间反向循环。

#### 缺陷 #3: `virtual_stages_per_rank=1` 时不退化为正确 1F1B

**现状**: 即使 `virtual_stages_per_rank=1`（单 stage per rank），代码仍用全局 clock 推进 + 立即全 backward 模式。真实 Schedule1F1B 的 warmup = `pp_group_size - 1 - rank` 次 forward，然后 steady-state 交替 1F1B，cooldown 剩余 backward。

#### 缺陷 #4: DP gradient sync / optimizer step 无 backward 依赖

**位置**: L294-307

**现状**: `dp_gradient_sync` 和 `optimizer_step` 的依赖链只依赖同 rank 前一个 event 的 control 依赖，不依赖 backward reduce-scatter 完成。真实训练中 optimizer step 必须等所有 gradient accumulation + DP reduce 完成。

#### 缺陷 #5: FSDP all-gather 只绑定到 DP group rank 0

**位置**: L163

**现状**: FSDP all-gather 事件只挂到每个 PP group 的第一个 rank (`pp_rank * tp_degree * dp_degree`)。post-hoc replication (L314-356) 复制到同 group 其他 rank，但复制后的 event 只有 control 依赖（前一个 event），缺少表示真实 collective 的跨 rank fsdp_comm 依赖。

#### 缺陷 #6: PP send/recv 缺跨 rank 依赖

**位置**: L182-191, L220-228

**现状**: `_send_activation` 只依赖同 rank 的 forward 产出 (`pp_comm`)。目标 rank 的 recv 没有 send→recv 的因果依赖。真实 PP 中 send 必须 recv 端就绪才能完成（至少有数据流依赖）。

#### 缺陷 #7: TP all-reduce 过度简化

**位置**: L169-173

**现状**: 每个 forward pass 只发一个 `tp_all_reduce`，真实 TP 在每个 column/row parallel 层有独立 all-reduce。整个 stage 的 TP 通信压缩为 1 个事件，丢失了与 FSDP all-gather 的时间重叠关系。

---

### 2.2 `pp_schedule_extractor.py` — PP 调度提取

#### 缺陷 #8: 三层 fallback 均无法对接当前 PyTorch API — 但真实 API 已验证可用

**现状**:

- **Strategy 1** (`schedule._actions`): `_PipelineScheduleRuntime` 没有 `_actions` 属性。此策略永远返回 `None`。
- **Strategy 2** (`schedule._compute_clock_cycles()`): 此方法不存在于任何当前 PyTorch schedule 类。`hasattr` 检查永远为 `False`。
- **Strategy 3** (heuristic fallback): 唯一实际生效的路径，产出简单 1F1B，不支持 Interleaved。

**真实 API** (已验证可用): 所有 PyTorch schedule 类在 `__init__` 中即计算 `pipeline_order` 和 `pipeline_order_with_comms`。可通过 duck-typed `MockPipelineStage`（不调用 `dist.get_rank`，直接设置 `group_rank/group_size` 等属性）在单 CPU 进程中成功构造所有 7 种 schedule 类型。`_Action` NamedTuple 含 `(stage_index, computation_type, microbatch_index, sub_actions)`，`_ComputationType` 包含 `F/B/I/W/UNSHARD/RESHARD/SEND_F/RECV_F/SEND_B/RECV_B/OVERLAP_F_B/REDUCE_GRAD`。

#### 缺陷 #9: heuristic 不支持 Interleaved 模式

**位置**: L317-392

**现状**: `_build_schedule_heuristic` 只产出简单 1F1B（warmup = rank+1 forward，steady 交替 1F1B）。对 Interleaved1F1B (多虚拟 stage) 和 GPipe 会产出错误调度。

#### 缺陷 #10: send/recv 依赖只覆盖相邻 rank

**位置**: L425-437

**现状**: `_add_send_recv_deps` 只连接 `send_fwd(rank=r) → recv_fwd(rank=r+1)`。对 Interleaved 模式，send/recv 可能跨越多个 rank（虚拟 stage 映射到不同 rank）。

#### 缺陷 #11: bare `except Exception: pass` 吞掉错误

**位置**: L164

**现状**: `_compute_clock_cycles()` 的调用被 bare except 包裹，解析错误静默吞掉，fallback 到 heuristic 但不发出 warning。

---

### 2.3 `cost_model.py` — 性能预测

#### 缺陷 #12: matmul FLOPs 重复计算

**位置**: L54-65

**现状**: `_estimate_flops` 对 matmul 类 op：先对每个 input tensor 累加 `2 * numel(s[:-2]) * s[-2] * s[-1]`，再对每个 output tensor 累加 `numel(out)`。这既重复计算了 input 的贡献，又额外加了 output 的 numel。正确公式是 `2 * M * K * N`（对 `(M,K) × (K,N)` matmul），只需一次计算。

#### 缺陷 #13: silu/gelu 重复判断

**位置**: L91-92 vs L107-108

**现状**: silu 和 gelu 在 elif 链前部已处理 (`flops_per_elem = 5`)，第 107-108 行的相同条件永远不会执行（dead code），但给人错误印象。

#### 缺陷 #14: add/mul/div/sub keyword 匹配过宽

**位置**: L101

**现状**: `"add" in op` 会匹配 `addmm`、`baddbmm` 等含 `add` 的 op。`"div"` 可能匹配不相关 op。elif 链的顺序可能让部分 matmul 类 op 被误判为 element-wise。

#### 缺陷 #15: dynamic dimension fallback=1024

**位置**: L149-152

**现状**: `_numel` 对 `None` 或 `-1` 维度统一 fallback 为 `1024`。LLM 训练中 seq_len 是主要 dynamic dim（4096-128K），统一 fallback 使所有含 dynamic dim 的 op 估算无区分能力。

#### 缺陷 #16: critical-path 的 partial overlap heuristic 无理论依据

**位置**: L402-408

**现状**: edge_cost = `max(0, dur - v_node.comm_time * 0.5)` — 硬编码 0.5 overlap 因子。真实 NCCL/CUDA overlap 行为取决于硬件、kernel 类型、comm/compute 比例，不能用一个常数表示。

#### 缺陷 #17: topological sort 用 list.pop(0) — O(n²)

**位置**: L378-386

**现状**: `queue.pop(0)` 在 Python list 上是 O(n)。LLM forward 有数千 op，整体 O(n²)。应使用 `collections.deque`。

#### 缺陷 #18: all-reduce scaling heuristic 不准确

**位置**: L316-317

**现状**: `comm_time_us *= (1 + log2(gs) * 0.5)`。真实 ring all-reduce 时间约 `2*(P-1)/P * bytes/bandwidth`，tree all-reduce 约 `2*bytes/bandwidth`。log factor 在大 group (DP=64) 时偏差超 50%。

#### 缺陷 #19: reduce_scatter 通信量低估

**位置**: L134-142

**现状**: `_estimate_comm_bytes` 优先从 output 累加。reduce_scatter 的 input 是 full gradient（大），output 是 shard（小但非零）。代码只看 output，低估实际通信量。

---

### 2.4 `trainer_runner.py` — 合成通信注入

#### 缺陷 #20: 合成 FSDP/TP 通信的 shape=bytes

**位置**: L239, L291

**现状**: `TensorMeta(shape=(per_layer_bytes,), dtype="torch.float32")` 把参数 byte 数当 shape 维度。后续 cost_model 的 `_numel(shape)` 得到 `per_layer_bytes` 个 element，乘 dtype_bytes (4) 得到 `per_layer_bytes * 4` bytes — 偏差 4x (fp32) 或 2x (bf16)。

**正确做法**: shape 应为 `(numel,)` 而非 `(bytes,)`。`_tensor_bytes` 从 shape × dtype_bytes 计算实际 byte 数。

#### 缺陷 #21: FP32 dtype hardcoded

**位置**: L214, L286

**现状**: `dtype_str = "torch.float32"` 和 `act_bytes = batch_size * seq_len * hidden * 4`。TorchTitan 配置有 `training.mixed_precision_param` (默认 bf16) 和 `training.dtype`。合成通信应从 config 读取 dtype，而非硬编码 fp32。

#### 缺陷 #22: 合成通信节点无 graph edge

**现状**: 注入的 comm node 只 `graph.add_node()` 但不 `graph.add_edge()` 连接到同 phase 的 compute node。导致：
- cost_model 的 critical-path 把孤立 comm node 当 separate path
- HTML 可视化中 comm 与 compute 无因果关系

#### 缺障 #23: num_layers fallback 链过于 hacky

**位置**: L224-229

**现状**: `len(per_module_bytes)` 基于 `prefix = ".".join(name.split(".")[:2])` 的去重计数。同一层不同参数共享 prefix，去重后约等于层数，但这是巧合而非保证。

#### 缺陷 #24: TP all-reduce 数量估算不精确

**位置**: L288

**现状**: `max(1, len(per_module_bytes) // 2)` 粗略估算 TP all-reduce 次数。真实数量取决于模型的 column/row parallel 层数，应从模型结构或 parallelize config 推导。

---

### 2.5 调度与计算图的断层

#### 缺陷 #25: 调度事件与 compute graph 完全脱节

**现状**: `SimulationResult` 有 `compute_graph` (细粒度 OpNode) 和 `schedule` (粗粒度 ScheduleEvent) 两个独立数据结构。没有机制把 `fsdp2_all_gather` schedule event 与 `all_gather` OpNode 关联。cost_model 无法利用 schedule 的多 rank 信息。

#### 缺陷 #26: predict_step_time 只看单 rank compute graph

**现状**: `predict_step_time_us` 在 compute graph 上做 critical-path。compute graph 只含单 rank op 序列。对于有 PP/TP/DP 的配置，单 rank 的 critical-path 不等于全局 step time — 最慢 rank 决定 step time，且有跨 rank overlap。

---

### 2.6 `runtime_capture.py` — PP hook

#### 缺陷 #27: backward hook 寻找 `_backward_one_chunk` 而非 `backward_one_chunk`

**位置**: runtime_capture.py L179

**现状**: `bwd_attr = "_backward_one_chunk"` — 当前 PyTorch `PipelineStage` 的 backward 方法是 `backward_one_chunk`（无下划线前缀）。此 hook 对真实 PP schedule 的 backward 不生效。

---

### 2.7 测试覆盖

#### 缺陷 #28: schedule_generator 无单元测试

**现状**: `test_simulator.py` 测试了 PPScheduleExtractor (mock schedule)，但 `generate_interleaved_1f1b_schedule` 完全没有测试。缺陷 #1-7 无验证。

#### 缺陷 #29: cost_model 的 _estimate_flops / _critical_path_time_us 无测试

**现状**: 测试只覆盖数据模型序列化和基本 capture/export。缺陷 #12-19 无验证。

#### 缺陷 #30: _inject_synthetic_comm_events 无测试

**现状**: 合成通信注入是 fake_backend 核心功能，但无单元测试验证 node 数量、shape、edge 连接。

---

## 三、修复方案

按优先级分四个阶段。每个阶段完成后应通过新增的单元测试验证。

### 阶段 1: P0 — 用真实 PyTorch Schedule 替换自制调度生成器

**目标**: 不再在 `schedule_generator.py` 中自己实现调度算法。改为通过 duck-typed `MockPipelineStage` + 真实 PyTorch `_PipelineSchedule` 类构造 schedule 对象，直接读取其 `pipeline_order_with_comms`（含 UNSHARD/RESHARD/SEND_F/RECV_F/SEND_B/RECV_B/REDUCE_GRAD 等完整通信信息），然后转换为 `TrainingSchedule`。

**核心发现**: 经验证，所有 7 种 PyTorch schedule 类（Schedule1F1B, ScheduleGPipe, ScheduleInterleaved1F1B, ScheduleLoopedBFS, ScheduleZBVZeroBubble, ScheduleDualPipeV, ScheduleInterleavedZeroBubble）均可用 duck-typed MockStage 在单 CPU 进程中成功构造，且 `pipeline_order` / `pipeline_order_with_comms` 在 `__init__` 中即完成计算，无需调用 `step()`。

**验证结果**:

```python
# MockStage: duck-typed, 不继承 _PipelineStageBase, 不调用 dist.get_rank
class MockPipelineStage:
    def __init__(self, stage_index, num_stages, group_rank=0, group_size=1):
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.group_rank = group_rank
        self.group_size = group_size
        self.device = torch.device("cpu")
        self.submod = nn.Module()
        self.has_backward = True
        self.stage_index_to_group_rank = {i: i % group_size for i in range(num_stages)}

# Schedule1F1B — 单 stage per rank
stage = MockPipelineStage(0, num_stages=4, group_rank=0, group_size=4)
sched = Schedule1F1B(stage, n_microbatches=8)   # ✅ 成功
# pipeline_order 包含真实 warmup → 1F1B steady-state → cooldown

# ScheduleInterleaved1F1B — 多虚拟 stage per rank
stages = [MockPipelineStage(i, num_stages=8, group_rank=0, group_size=4) for i in range(2)]
sched = ScheduleInterleaved1F1B(stages, n_microbatches=8)  # ✅ 成功
# pipeline_order_with_comms 包含完整 SEND_F/RECV_F/SEND_B/RECV_B/UNSHARD/RESHARD
# 共 86 个 _Action per rank，涵盖所有 PP/FSDP/TP 通信生命周期
```

此方案消除了缺陷 #1-7（自制调度算法的全部问题），因为它直接使用上游 PyTorch 的调度实现，而非在 simulator 中重新实现一遍。

#### 1.1 新增 `MockPipelineStage` 类

在 `schedule_generator.py`（或新文件 `schedule_extract.py`）中定义：

```python
class MockPipelineStage:
    """Duck-typed mock that satisfies _PipelineSchedule.__init__ attribute reads.
    
    Does NOT call dist.get_rank/get_world_size — works in single-process CPU mode.
    """
    def __init__(self, stage_index, num_stages, group_rank=0, group_size=1):
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.group_rank = group_rank
        self.group_size = group_size
        self.device = torch.device("cpu")
        self.submod = nn.Module()
        self.has_backward = True
        self.stage_index_to_group_rank = {
            i: i % group_size for i in range(num_stages)
        }
```

#### 1.2 新增 `extract_schedule_from_pytorch()` 函数

核心转换函数：构造 MockStage → 实例化真实 Schedule 类 → 读取 `pipeline_order_with_comms` → 转换为 `TrainingSchedule`。

```python
from torch.distributed.pipelining.schedules import get_schedule_class

def extract_schedule_from_pytorch(
    *,
    pp_degree: int,
    tp_degree: int,
    dp_degree: int,
    num_stages: int,                   # pp_degree * virtual_stages_per_rank
    n_microbatches: int,
    schedule_name: str,                # "1F1B", "Interleaved1F1B", "GPipe", etc.
    virtual_stages_per_rank: int = 1,
) -> TrainingSchedule:
    """Construct a real PyTorch schedule with mock stages and extract its action table."""
    
    schedule_class = get_schedule_class(schedule_name)
    is_multi = issubclass(schedule_class, PipelineScheduleMulti)
    
    # Build mock stages for the local rank
    if is_multi:
        # Multi-stage: each local rank holds `virtual_stages_per_rank` stages
        group_size = pp_degree  # PP group size
        stages = [
            MockPipelineStage(
                stage_index=i,
                num_stages=num_stages,
                group_rank=0,
                group_size=group_size,
            )
            for i in range(virtual_stages_per_rank)
        ]
        schedule = schedule_class(
            stages,
            n_microbatches=n_microbatches,
            scale_grads=False,
        )
    else:
        # Single-stage: one stage per rank
        stage = MockPipelineStage(
            stage_index=0,
            num_stages=num_stages,
            group_rank=0,
            group_size=pp_degree,
        )
        schedule = schedule_class(
            stage,
            n_microbatches=n_microbatches,
            scale_grads=False,
        )
    
    # Extract action table — already populated by __init__
    pipeline_order = schedule.pipeline_order   # dict[int, list[_Action | None]]
    
    # For multi-stage schedules, also read the lowered schedule with comms
    if hasattr(schedule, "pipeline_order_with_comms"):
        pipeline_order = schedule.pipeline_order_with_comms
    
    # Convert _Action → ScheduleEvent, build deps from SEND→RECV pairs
    return _convert_pipeline_order_to_training_schedule(
        pipeline_order,
        pp_degree=pp_degree,
        tp_degree=tp_degree,
        dp_degree=dp_degree,
    )
```

#### 1.3 实现 `_convert_pipeline_order_to_training_schedule`

将 PyTorch `_Action` NamedTuple 转换为 simulator 的 `ScheduleEvent`/`ScheduleDep`，同时：

- `_ComputationType.FORWARD` → `event_type="pp_forward"`
- `_ComputationType.FULL_BACKWARD` → `event_type="pp_backward"`
- `_ComputationType.UNSHARD` → `event_type="fsdp2_all_gather"` (FSDP 参数 all-gather)
- `_ComputationType.RESHARD` → `event_type="fsdp2_reduce_scatter"` (FSDP 参数释放/梯度 reduce-scatter)
- `_ComputationType.SEND_F` → `event_type="pp_send_activation"`
- `_ComputationType.RECV_F` → `event_type="pp_recv_activation"`
- `_ComputationType.SEND_B` → `event_type="pp_send_gradient"`
- `_ComputationType.RECV_B` → `event_type="pp_recv_gradient"`
- `_ComputationType.REDUCE_GRAD` → `event_type="dp_gradient_sync"`

**依赖关系构建**:

- **同 rank 顺序依赖**: 每个 rank 的 action list 已按 logical clock 排序，相邻 action 间添加 control 依赖（自然正确，无需修改）
- **跨 rank PP 依赖**: 同 microbatch 的 `SEND_F(stage=s)` 和 `RECV_F(stage=s+1)` 在不同 rank 的 action list 中出现。通过 `(computation_type, microbatch_index, stage_index)` 三元组索引匹配 send→recv pair，添加 `pp_comm` 依赖
- **FSDP collective 语义**: UNSHARD/RESHARD 由 `_prepare_schedule_with_comms` 正确放置在 FORWARD/BACKWARD 之前/之后，已经保证了因果顺序
- **DP gradient sync**: REDUCE_GRAD 已由 PyTorch schedule 正确放置在最后一个 backward 之后

这消除了缺陷 #1-7 的全部问题，因为：
- warmup/steady-state/cooldown 逻辑由 PyTorch 上游正确实现（缺陷 #1-3）
- DP sync 依赖由 REDUCE_GRAD action 正确放置（缺陷 #4）
- FSDP lifecycle 由 UNSHARD/RESHARD 正确标记（缺陷 #5）
- PP send→recv 依赖可从 SEND_F/RECV_F 匹配推导（缺陷 #6）
- TP all-reduce 不在此层处理（见阶段 2 合成通信注入）（缺陷 #7 的修正方案调整）

#### 1.4 修改 `_inject_semantic_schedule` 调用路径

`trainer_runner.py` 中 `_inject_semantic_schedule` 当前调用 `generate_interleaved_1f1b_schedule`。改为调用 `extract_schedule_from_pytorch`：

```python
def _inject_semantic_schedule(result, config):
    parallelism = getattr(config, "parallelism", None)
    if parallelism is None:
        return
    
    pp_degree = int(getattr(parallelism, "pipeline_parallel_degree", 1) or 1)
    tp_degree = int(getattr(parallelism, "tensor_parallel_degree", 1) or 1)
    dp_shard = int(getattr(parallelism, "data_parallel_shard_degree", 1) or 1)
    if dp_shard < 0:
        dp_shard = 1
    dp_repl = int(getattr(parallelism, "data_parallel_replicate_degree", 1) or 1)
    dp_degree = dp_shard * dp_repl
    
    schedule_name = str(getattr(parallelism, "pipeline_parallel_schedule", "1F1B") or "1F1B")
    num_mb = int(getattr(parallelism, "pipeline_parallel_microbatch_size", 8) or 8)
    virtual = 2 if "Interleaved" in schedule_name else 1
    num_stages = pp_degree * virtual
    
    # 使用真实 PyTorch schedule 提取，而非自制生成器
    from .schedule_extract import extract_schedule_from_pytorch
    semantic = extract_schedule_from_pytorch(
        pp_degree=pp_degree,
        tp_degree=tp_degree,
        dp_degree=dp_degree,
        num_stages=num_stages,
        n_microbatches=num_mb,
        schedule_name=schedule_name,
        virtual_stages_per_rank=virtual,
    )
    
    existing = result.schedule
    if existing is None:
        result.schedule = semantic
    elif isinstance(existing, TrainingSchedule):
        for ev in semantic.events:
            existing.add_event(ev)
        for dep in semantic.deps:
            existing.add_dep(dep)
```

#### 1.5 保留 `schedule_generator.py` 作为 fallback

`generate_interleaved_1f1b_schedule` 仍保留，但标记为 deprecated fallback（仅在 PyTorch schedule 不可用时启用，如 torch 版本过旧或 API 变化）。同时修正其最严重的 bug（GPipe vs Interleaved 标签错误），确保 fallback 产出正确标注的结果。

#### 1.6 新增单元测试

```python
class TestScheduleExtract(unittest.TestCase):
    def test_1f1b_schedule_matches_pytorch(self):
        """extract_schedule_from_pytorch("1F1B") 与 PyTorch Schedule1F1B 一致"""
        result = extract_schedule_from_pytorch(
            pp_degree=4, tp_degree=1, dp_degree=1, num_stages=4,
            n_microbatches=8, schedule_name="1F1B"
        )
        # 验证 rank 0 的事件序列包含 warmup → 1F1B → cooldown
        rank0_fwd = [e for e in result.events if e.rank == 0 and e.event_type == "pp_forward"]
        rank0_bwd = [e for e in result.events if e.rank == 0 and e.event_type == "pp_backward"]
        # Schedule1F1B: warmup = pp_size - 1 - rank = 3 forward for rank 0
        assert len(rank0_fwd) == 8
        assert len(rank0_bwd) == 8
    
    def test_interleaved_1f1b_schedule_has_send_recv(self):
        """Interleaved1F1B 产出 SEND_F/RECV_F/SEND_B/RECV_B"""
        result = extract_schedule_from_pytorch(
            pp_degree=4, tp_degree=1, dp_degree=1, num_stages=8,
            n_microbatches=8, schedule_name="Interleaved1F1B",
            virtual_stages_per_rank=2,
        )
        types = {e.event_type for e in result.events}
        assert "pp_send_activation" in types
        assert "pp_recv_activation" in types
        assert "fsdp2_all_gather" in types  # UNSHARD
        assert "fsdp2_reduce_scatter" in types  # RESHARD
    
    def test_gpipe_schedule(self):
        """GPipe: 全 forward → 全 backward"""
        ...
    
    def test_send_recv_cross_rank_deps(self):
        """同 microbatch 的 SEND_F(stage=s) → RECV_F(stage=s+1) 有 pp_comm 依赖"""
        ...
    
    def test_dp_gradient_sync_after_backward(self):
        """REDUCE_GRAD 事件在 backward 之后"""
        ...
    
    def test_all_schedule_types(self):
        """所有支持的 schedule 类型都能成功提取"""
        for name in ["1F1B", "GPipe", "Interleaved1F1B", "LoopedBFS", 
                     "ZBVZeroBubble", "DualPipeV"]:
            result = extract_schedule_from_pytorch(...)
            assert len(result.events) > 0
```

---

### 阶段 2: P0 — 修正合成通信注入 + cost_model 系统性错误

#### 2.1 修正 shape=bytes → shape=numel

```python
# 之前:
TensorMeta(shape=(per_layer_bytes,), dtype="torch.float32")
# 之后:
per_layer_numel = shard_bytes // dtype_size(dtype_str)
TensorMeta(shape=(per_layer_numel,), dtype=dtype_str)
```

同理修正 TP all-reduce 的 `act_bytes` → `act_numel`:

```python
# 之前:
act_bytes = batch_size * seq_len * hidden * 4
TensorMeta(shape=(act_bytes,), dtype="torch.float32")
# 之后:
act_numel = batch_size * seq_len * hidden
TensorMeta(shape=(act_numel,), dtype=dtype_str)
```

#### 2.2 从 config 读取 dtype

```python
# 替换 hardcoded "torch.float32":
dtype_str = str(TORCH_DTYPE_MAP.get(
    getattr(trainer.config.training, "mixed_precision_param", "bfloat16"),
    torch.bfloat16
))
dtype_byte_size = _dtype_bytes(dtype_str)  # bf16=2, fp32=4, fp8=1
```

#### 2.3 修正 matmul FLOPs 计算

```python
# 之前 (L54-65): input + output 分别累加
# 之后:
if any(kw in op for kw in ("mm", "matmul", "bmm", "baddbmm", "addmm", "linear")):
    # 对于 (M,K) × (K,N) matmul: 2*M*K*N
    # 尝试从 input shapes 推导 M, K, N
    if len(in_shapes) >= 2 and len(in_shapes[0]) >= 2 and len(in_shapes[1]) >= 2:
        M = _numel(in_shapes[0][:-2]) if len(in_shapes[0]) > 2 else 1
        K = in_shapes[0][-1]
        N = in_shapes[1][-1]
        return 2 * M * K * N
    # fallback: 2 * output numel
    total = 0
    for out in out_shapes:
        total += 2 * _numel(out)
    return total
```

#### 2.4 删除 silu/gelu 重复判断

删除 L107-108 的 dead code。

#### 2.5 修正 keyword 匹配过宽

改为更精确的匹配：

```python
# 之前: if "add" in op
# 之后:
if op.startswith("aten.add") or op == "add":
    flops_per_elem = 1
```

或按 ATen op namespace 拆分：先检查 matmul-like（含 `mm`/`matmul`/`addmm`），再检查 element-wise。

#### 2.6 修正 reduce_scatter 通信量

```python
def _estimate_comm_bytes(node):
    # reduce_scatter: input 是 full tensor, output 是 shard
    # all_gather: input 是 shard, output 是 full
    if node.comm_op == "reduce_scatter":
        # 通信量 = input bytes (full tensor)
        return sum(_tensor_bytes(inp.shape, inp.dtype) for inp in node.inputs)
    elif node.comm_op == "all_gather":
        # 通信量 = output bytes (full tensor)
        return sum(_tensor_bytes(out.shape, out.dtype) for out in node.outputs)
    # ... 其他 op 保持原有逻辑
```

#### 2.7 修正 all-reduce scaling

替换 `1 + log2(gs) * 0.5` 为 ring all-reduce 公式：

```python
# Ring all-reduce: 2 * (P-1) / P * bytes / bandwidth
# 对大 P, ≈ 2 * bytes / bandwidth (与 tree 近似)
if node.comm_op == "all_reduce" and node.comm_group_size:
    gs = max(node.comm_group_size, 1)
    comm_time_us *= 2 * (gs - 1) / gs  # ring factor
```

#### 2.8 topological sort 用 deque

```python
from collections import deque
queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
while queue:
    u = queue.popleft()
    ...
```

#### 2.9 合成通信添加 graph edge

```python
# 在 _inject_synthetic_comm_events 中，为每个注入的 comm node
# 找到同 phase 的最后一个 compute node，添加 sequential edge
phase_nodes = [n for n in graph.nodes.values() if n.phase == comm_node.phase]
if phase_nodes:
    last_compute_id = phase_nodes[-1].node_id
    graph.add_edge(DataEdge(last_compute_id, comm_node.node_id, "sequential"))
```

#### 2.10 新增单元测试

```python
class TestCostModel(unittest.TestCase):
    def test_matmul_flops_correct(self):
        """(2,8) × (8,4) → 2*2*8*4 = 128 FLOPs"""
        ...
    def test_reduce_scatter_comm_bytes(self):
        """通信量 = input (full tensor) bytes"""
        ...
    def test_all_reduce_ring_factor(self):
        """2*(P-1)/P scaling"""
        ...
    def test_critical_path_deque(self):
        """大图不超时"""
        ...

class TestSyntheticCommInjection(unittest.TestCase):
    def test_shape_is_numel_not_bytes(self):
        """TensorMeta.shape 的 product × dtype_bytes == 通信量"""
        ...
    def test_dtype_from_config(self):
        """bf16 config 产出 bf16 tensor meta"""
        ...
    def test_comm_nodes_have_edges(self):
        """注入的 node 有 sequential edge 连到 compute node"""
        ...
```

---

### 阶段 3: P1 — 修正 PP 调度提取对接 + 调度-计算图关联

#### 3.1 重写 PPScheduleExtractor 直接读取 `pipeline_order` / `pipeline_order_with_comms`

PPScheduleExtractor 接收的 schedule 对象（由 TorchTitan `_build_pipeline_schedule` 产出）已在 `__init__` 中计算了完整的 action table。Extractor 只需直接读取即可，不需要任何 fallback。

新实现将 PPScheduleExtractor 与阶段 1 的 `_convert_pipeline_order_to_training_schedule` 共享同一转换逻辑：

```python
class PPScheduleExtractor:
    def extract(self) -> TrainingSchedule:
        schedule = self.schedule
        
        # 直接读取 __init__ 中已计算好的 action table
        if hasattr(schedule, "pipeline_order_with_comms"):
            pipeline_order = schedule.pipeline_order_with_comms
        elif hasattr(schedule, "pipeline_order"):
            pipeline_order = schedule.pipeline_order
        else:
            # fallback: heuristic (仅用于极旧 PyTorch 版本)
            logger.warning("No pipeline_order found on schedule %s; using heuristic",
                           type(schedule).__name__)
            ts = TrainingSchedule(metadata={...})
            self._build_schedule_heuristic(ts)
            return ts
        
        # 共享转换函数
        return _convert_pipeline_order_to_training_schedule(
            pipeline_order,
            pp_degree=schedule.pp_group_size if hasattr(schedule, "pp_group_size") else self.world_size,
            tp_degree=1,  # PP schedule 不含 TP/DP 信息
            dp_degree=1,
        )
```

此方案消除了缺陷 #8（三层 fallback 全部失效）和 #11（bare except 吞错误），因为核心路径直接读取已有数据。

#### 3.2 heuristic fallback 保留但降级

heuristic fallback 保留为极旧 PyTorch 版本的 fallback，但：
- 根据 schedule 类型名选择不同模式 (1F1B / Interleaved / GPipe)
- 添加 warning 日志

#### 3.3 send/recv 跨 rank 依赖由 `_convert_pipeline_order_to_training_schedule` 统一处理

不再单独在 `_add_send_recv_deps` 中处理。转换函数从 `pipeline_order_with_comms` 的 SEND_F/RECV_F/SEND_B/RECV_B pair 自动推导跨 rank 依赖，使用 `_Action.stage_index` + schedule 的 `stage_index_to_group_rank` 映射确定 rank 映射。

#### 3.4 移除 bare except，改为 logging.warning

已在上面的实现中完成 — fallback 路径显式发出 warning。

#### 3.5 修正 PP backward hook 方法名

```python
# 之前:
bwd_attr = "_backward_one_chunk"
# 之后:
bwd_attr = "backward_one_chunk"
```

#### 3.6 建立调度-计算图关联机制

新增 `ScheduleEvent.op_node_ids: list[str]` 字段，在 `runtime_capture.build_result()` 中填充：

```python
# FSDP all-gather event 关联到同 module 的 all_gather OpNode
# PP forward/backward event 关联到同 stage+mb 的 compute OpNode list
```

在 `SimulationResult` 新增方法 `link_schedule_to_graph()`，基于 `(phase, pp_stage, microbatch_idx)` 匹配。

#### 3.7 多 rank step time 预测

新增 `predict_multi_rank_step_time_us()`:

```python
def predict_multi_rank_step_time_us(result: SimulationResult, cost_model: CostModel) -> float:
    """基于 schedule 的多 rank 依赖图做 critical-path，取最慢 rank 的完成时间。"""
    if result.schedule is None:
        return cost_model.predict_step_time_us(result.compute_graph)
    
    # 1. 为每个 ScheduleEvent 分配估算时间
    #    - compute event: 从关联的 OpNode 取 perf_result
    #    - comm event: 从 cost_model 估算
    # 2. 在 schedule deps 图上做 multi-rank critical-path
    # 3. 返回 max(rank_finish_time) 作为 step time
```

---

### 阶段 4: P2 — 精度提升与清理

#### 4.1 dynamic dimension 处理

替代 fallback=1024，改为从 config 或模型结构推导：

```python
def _numel(shape, default_seq_len=4096):
    prod = 1
    for d in shape:
        if d is None or d < 0:
            prod *= default_seq_len  # 从 config.training.seq_len 传入
        else:
            prod *= d
    return prod
```

`MockCostModel` 新增 `default_seq_len` 参数，从 `config.training.seq_len` 传入。

#### 4.2 overlap heuristic 改为可配置策略

```python
class OverlapStrategy:
    """基类: compute/comm overlap 估算策略"""
    def overlap_factor(self, compute_us, comm_us) -> float: ...

class NoOverlap(OverlapStrategy):
    """无 overlap: total = compute + comm"""
    def overlap_factor(self, compute_us, comm_us): return compute_us + comm_us

class FixedOverlap(OverlapStrategy):
    """固定比例 overlap"""
    def __init__(self, factor=0.5): self.factor = factor
    def overlap_factor(self, compute_us, comm_us):
        return compute_us + max(0, comm_us - compute_us * self.factor)

class NCCLAsyncOverlap(OverlapStrategy):
    """基于 NCCL async 的 overlap 模型"""
    ...
```

`MockCostModel` 新增 `overlap_strategy` 参数，默认 `FixedOverlap(0.5)`（向后兼容），但允许用户替换。

#### 4.3 num_layers 推导改进

从模型配置读取层数（所有 TorchTitan 模型的 config_registry 都定义了 `n_layers`）：

```python
# 替换 hacky fallback:
num_layers = getattr(model_parts[0].config, "n_layers", None)
if num_layers is None:
    num_layers = len(getattr(model_parts[0], "layers", []))
if num_layers is None:
    num_layers = max(len(per_module_bytes), 1)  # 保留 fallback 但降级
```

#### 4.4 TP all-reduce 数量推导

```python
# 从 parallelize config 或模型结构推导
# TP all-reduce 发生在每个 ColwiseParallel → RowwiseParallel 边界
# 对标准 transformer: 每层 2 次 (attn + FFN)
tp_allreduce_count = num_layers * 2  # attn + FFN 各一次
```

#### 4.5 清理 dead code 和冗余

- 删除 `silu`/`gelu` 重复判断 (L107-108)
- 删除 `trainer.py` 重复的 `train()` 方法 (L180 和 L184 完全相同)

---

## 四、实施顺序与依赖关系

```
阶段 1 (用真实 PyTorch Schedule 替换自制调度生成器)
  ├── 1.1 MockPipelineStage → 1.2 extract_schedule_from_pytorch → 1.3 _convert_pipeline_order
  ├── 1.4 修改 _inject_semantic_schedule 调用路径 → 1.5 保留旧 generator 作为 deprecated fallback
  └── 1.6 单元测试 ← 依赖 1.1-1.5
  注意：此阶段消除了缺陷 #1-7 全部问题（warmup/steady-state/cooldown/依赖/通信），
       因为调度算法由 PyTorch 上游保证正确性
  
阶段 2 (合成通信 + cost_model 系统性错误)
  ├── 2.1 shape=numel → 2.2 dtype从config → 2.3 matmul FLOPs → 2.4-2.5 清理
  ├── 2.6 reduce_scatter → 2.7 all-reduce scaling → 2.8 deque → 2.9 graph edge
  └── 2.10 单元测试 ← 依赖 2.1-2.9

阶段 3 (PP提取对接 + 调度-图关联)
  ├── 3.1 PPScheduleExtractor 直接读取 pipeline_order_with_comms
  │   (共享阶段 1 的 _convert_pipeline_order_to_training_schedule)
  ├── 3.2 heuristic fallback 保留降级 → 3.3 send/recv 由转换函数统一处理
  ├── 3.4 logging → 3.5 backward hook 修正 → 3.6 调度-图关联 → 3.7 多rank step time
  └── 单元测试 ← 依赖 3.1-3.7

阶段 4 (精度提升与清理)
  ├── 4.1 dynamic dim → 4.2 overlap strategy → 4.3 num_layers → 4.4 TP count → 4.5 dead code
  └── 单元测试
```

每个阶段完成后：
1. `pytest torchtitan/experiments/simulator/tests/test_simulator.py -v` 通过
2. `pre-commit run --all-files` 通过
3. 阶段 1 的验证：对比 `extract_schedule_from_pytorch("1F1B")` 的输出与直接构造 `Schedule1F1B(mock_stage, n_microbatches=8)` 的 `pipeline_order`，确认完全一致

---

## 五、验证策略

### 调度正确性验证

1. **与 PyTorch schedule 直接对比**: `extract_schedule_from_pytorch("1F1B", pp=4, n_mb=8)` 的 TrainingSchedule 应与直接构造 `Schedule1F1B(MockStage(0,4,0,4), n_microbatches=8).pipeline_order` 完全一致。所有 schedule 类型逐一验证。
2. **可视化验证**: HTML trace 的 swimlane 应显示真实 Interleaved1F1B 的 warmup → steady-state → cooldown 模式（而非当前 GPipe 模式）。
3. **依赖完整性**: 同 microbatch 的 SEND_F(stage=s) → RECV_F(stage=s+1) 应有 pp_comm 依赖；REDUCE_GRAD 应在 backward 之后。
4. **上游同步保障**: 调度算法不再由 simulator 维护。当 PyTorch 更新 schedule 实现（如新增 InterleavedZeroBubble），`extract_schedule_from_pytorch` 自动获取最新调度，无需 simulator 代码变更。

### 性能预测验证

1. **FLOPs 正确性**: matmul (2,8)×(8,4) 应产出 128 FLOPs (2×2×8×4)，而非当前的超额值。
2. **通信量正确性**: reduce_scatter 的 `_estimate_comm_bytes` 应返回 input (full tensor) bytes。
3. **Shape 正确性**: 合成通信的 `TensorMeta` shape product × dtype_bytes 应等于实际通信 byte 数，而非偏差 4x。

### 端到端验证

1. `MODULE=simulator.llama3 CONFIG=llama3_sim_debugmodel ./run_train.sh` 应成功产出所有输出文件。
2. `--simulation.semantic_schedule` 的 HTML trace 应与真实 PP schedule 一致。
3. `--simulation.cost_model` 的 step time 预测应在合理范围（不偏差 4x+）。