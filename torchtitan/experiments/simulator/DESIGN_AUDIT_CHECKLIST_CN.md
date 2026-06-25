# TorchTitan Simulator 设计原则与实现审计清单

> 依据：`DESIGN_CN.md` 的设计原则、当前 `torchtitan/experiments/simulator/` 实现，以及 simulator 直接依赖/修改的核心模型文件。
> 审计范围：设计原则符合性、文档冗余/过期内容、文档声明但代码未实现/不可达/死路径。

## 结论总览

| 类别 | 结论 | 主要问题 |
|------|------|----------|
| 设计原则 | 部分违背 | Side-loaded 与模型无关核心被核心模型文件改动、DeepSeek 特化逻辑、跨模型依赖削弱 |
| 文档冗余/过期 | 已修复 | patch 数量、PP 通信注入名称、meta_device_patches docstring、HTML 自包含描述已统一 |
| 文档有但代码未实现 | 已标注 | `mode`/`capture_joint_fx`/`max_seq_len`/`batch_size` 已标注"Reserved"；`Simulator` API 未实现(路线图) |
| 代码死路径/不可达 | 已清理 | `_reapply_fsdp2_to_parts`、`schedule_generator.py`、`rewrite_runner.py` 已删除；gloo 不可达已记录 |

## 1. 设计原则符合性

### 1.1 Side-loaded 实验

**原则：** `train.py` 保持不变；所有仿真器代码位于 `torchtitan/experiments/simulator/`。

**判定：部分违背。**

- `train.py` 未被 simulator 入口侵入，主入口仍通过 `SimulationTrainer` side-load。
- 但当前实现包含 simulator 专用逻辑进入核心模型文件：
  - `torchtitan/models/common/token_dispatcher.py:249-258` 明确写入 “meta / fake-tensor simulation” 的强制负载均衡路径。
  - `torchtitan/models/common/token_dispatcher.py:283-297` 在 fake tensor 下构造均匀 EP split。
  - `torchtitan/models/common/token_dispatcher.py:338-341` 在 fake tensor 下跳过真实 expert-major permute。
  - `torchtitan/models/common/token_dispatcher.py:453-459` 在 fake tensor combine 路径跳过 `_unpermute`。
  - `torchtitan/models/deepseek_v4/parallelize.py:41-54` 为 simulator 需要复用 llama4 的 EP-aware `apply_fsdp` 包装。

**影响：** simulator 不再完全是 `torchtitan/experiments/simulator/` 内部的 side-loaded 实验；部分仿真约束泄漏到了核心模型实现。

### 1.2 捕获忠实

**原则：** 调度和计算图从捕获数据或真实 PyTorch 调度对象派生，而不是重新实现训练逻辑。

**判定：部分符合。**

- 符合项：
  - `run_trainer_simulation()` 通过 `TraceRecorder.build_result()` 使用捕获结果作为主 compute graph 来源：`trainer_runner.py:736`。
  - PP 通信已从旧的启发式注入改为从真实 PyTorch schedule 投影：`trainer_runner.py:266-280`。
  - `_inject_semantic_schedule()` 使用 `extract_schedule_from_pytorch()`：`trainer_runner.py:246-254`。
- 风险项：
  - `_project_pp_comm_from_schedule()` 的 PP 事件顺序来自真实 schedule，但 tensor shape 仍来自配置和 `_guess_hidden_dim()`：`trainer_runner.py:307-322`。
  - `_inject_synthetic_compute_anchors()` 仍会按 Cube/Vec 泳道需求补 `aten.mm.default` / `aten.add.Tensor` 节点：`trainer_runner.py:396-449`、调用点 `trainer_runner.py:750-752`。
  - MoE fake 路径强制均匀 token 分布是仿真近似，不是捕获真实路由：`token_dispatcher.py:249-258`、`token_dispatcher.py:283-297`。

**影响：** 主路径比旧版更接近“捕获忠实”，但仍存在为了可视化和 fake tensor 可执行性引入的非捕获节点/近似。

### 1.3 PyTorch 原生

**原则：** 复用上游 PyTorch 调度对象、FX tracing、`TorchDispatchMode`。

**判定：基本符合，但文档配置项有缺口。**

- `unified_trace` 基于 `TorchDispatchMode` / `FakeTensorMode`：`capture/unified_trace.py`。
- 调度提取基于真实 PyTorch `_PipelineSchedule`：`schedule/schedule_extract.py`。
- 但 `SimulationConfig.capture_joint_fx` 文档声称支持联合 fwd+bwd FX 捕获，当前代码未读取该字段：`trainer.py:31` 仅定义；全目录搜索只有配置注册表设置它。

**影响：** 主实现是 PyTorch-native；FX 捕获相关配置仍是未落地接口。

### 1.4 模型无关的核心

**原则：** 模型特定代码隔离在各模型子目录（`llama3/`、`deepseek_v4/`）。

**判定：违背。**

- `trainer.py` 的通用 wrapper 内有 DeepSeek 特化兼容逻辑：
  - 注入 `ParallelismConfig.expert_parallel_comm_backend`：`trainer.py:123-130`。
  - patch `ParallelDims.get_optional_mesh()` 以兼容 DeepSeek 的 `"etp"` 查询：`trainer.py:131-146`。
  - 注入 `ParallelDims.fsdp_gradient_divide_factor`：`trainer.py:148-151`。
- `SimulationTrainer.__init__` 通过模型名字符串判断 DeepSeek / Llama：`trainer.py:436-441`。
- DeepSeek V4 并行化依赖 llama4 的 `apply_fsdp`：`torchtitan/models/deepseek_v4/parallelize.py:41-54`。

**影响：** simulator core 和 DeepSeek/Llama 模型特化逻辑耦合，后续新增模型会继续扩大条件分支。

## 2. 文档冗余、过期或内部不一致

### 2.1 `meta_device_patches.py` patch 数量描述不一致

- `DESIGN_CN.md:40` 仍写 “3 个 FSDP2 补丁”。
- `DESIGN_CN.md:866-876` 正确列出 7 个补丁。
- `meta_device_patches.py:13-19` 文件 docstring 只列 4 类。
- `meta_device_patches.py:217-223` `apply_meta_device_patches()` docstring 只列 5 类。
- 实际 `_ORIGINAL_FUNCTIONS` 和 apply/restore 逻辑覆盖 7 类：`meta_device_patches.py:52-60`、`meta_device_patches.py:251-312`、`meta_device_patches.py:322-351`。

**建议：** 统一为 7 个，并同步文件级 docstring 与函数 docstring。

### 2.2 执行流程章节仍引用旧 PP 注入路径

- `DESIGN_CN.md:300-302` 仍写：
  - `[use_fake] _inject_synthetic_compute_anchors(...)`
  - `[use_fake] _inject_pp_send_recv(...)`
  - `[semantic_schedule] _inject_semantic_schedule(...)`
- 当前代码顺序是先按 `semantic_schedule` 注入 schedule，再在 fake path 调用 `_project_pp_comm_from_schedule()` 与 `_inject_synthetic_compute_anchors()`：`trainer_runner.py:744-752`。
- `_inject_pp_send_recv` 已不存在；当前函数是 `_project_pp_comm_from_schedule()`：`trainer_runner.py:266-280`。

**建议：** 更新第 6 节执行流程为当前顺序，并删除 `_inject_pp_send_recv`。

### 2.3 通信分类章节仍残留旧名称

- `DESIGN_CN.md:720-733` 已正确描述“调度投影 PP 通信”与 `_project_pp_comm_from_schedule()`。
- 但 `DESIGN_CN.md:904-910` 又写 “合成 PP（`_inject_pp_send_recv`）”，与当前实现和前文冲突。

**建议：** 将 `20.5` 改为 “调度投影 PP（`_project_pp_comm_from_schedule`）”，并说明 shape 仍由配置估计。

### 2.4 HTML 导出自包含描述不一致

- `DESIGN_CN.md:672-674` 明确说明 ECharts 从 CDN 加载，非完全离线/自包含。
- `export.py:16-17` 和 `export.py:1024-1026` 仍称 “self-contained HTML visualization”，且还提到 AntV G6。
- 实际 HTML 使用 CDN：`export.py:1113`。

**建议：** 更正 `export.py` docstring，删除 “self-contained” 和未使用的 “AntV G6”。

### 2.5 目录结构遗漏清理对象的解释边界

- `DESIGN_CN.md:48-51` 的目录结构不列 `schedule/schedule_generator.py`，但 `DESIGN_CN.md:942` 又把它列为已废弃技术债。
- 当前文件仍存在：`schedule/schedule_generator.py:27` 定义 `generate_interleaved_1f1b_schedule()`。
- `schedule/__init__.py:7-10` 不导出它，但顶层兼容 alias 仍会导入它：`__init__.py:56-58`。

**建议：** 若保留兼容 alias，应在目录结构或技术债中明确“文件仍存在，仅兼容旧 import”；否则删除 alias 与文件。

## 3. 文档声明但代码未实现、未消费或不可达

### 3.1 `SimulationConfig.mode` 未被读取

- 文档说明：`DESIGN_CN.md:128` 标注 `"all"`、`"runtime"`、`"schedule"`。
- 代码仅定义字段：`trainer.py:28`。
- 全 simulator 源码未读取 `sim_opts.mode` 或 `config.simulation.mode`。

**影响：** CLI/配置设置该字段不会改变行为。

### 3.2 `SimulationConfig.capture_joint_fx` 未实现

- 文档说明：`DESIGN_CN.md:131` “联合 fwd+bwd FX 捕获”。
- 代码仅定义字段：`trainer.py:31`，配置注册表设置为 False：`llama3/config_registry.py:56`、`llama3/config_registry.py:111`。
- 全 simulator 源码没有读取该字段，也没有 joint FX capture 路径。

**影响：** 文档中的 FX 捕获配置目前是空接口。

### 3.3 `SimulationConfig.max_seq_len` / `batch_size` 未被消费

- 文档说明：`DESIGN_CN.md:129-130` 用于动态维度解析。
- 代码仅定义字段：`trainer.py:29-30`。
- 当前动态 shape / PP tensor shape 使用 `trainer.config.training.seq_len` 与 `local_batch_size`：`trainer_runner.py:315-322`。

**影响：** 这两个字段不会影响捕获或导出，文档说明与实现不一致。

### 3.4 `Simulator` 高层编程式 API 未实现

- 文档明确列为路线图：`DESIGN_CN.md:957-974`。
- 全 simulator 源码没有 `class Simulator` 或 `simulate_fx` / `simulate_runtime` / `simulate_pp_schedule`。
- 当前可用 API 是 `TraceRecorder` + `unified_trace` 与 `SimulationTrainer`。

**影响：** 这是已知未实现，不应作为当前可用 API 写入其它使用文档。

### 3.5 gloo 路径从 `run_train.sh` 不可达

- `SimulationTrainer.__init__` 强制 `config.comm.mode = "fake_backend"`：`trainer.py:403-407`。
- 随后读取 `actual_comm_mode` 并把 `comm_backend` 覆盖为空：`trainer.py:412-415`。
- 因此 `comm_backend == "gloo"` 分支基本不可通过标准入口触达：`trainer.py:436-441`、`trainer.py:472-473`。
- 文档已记录此限制：`DESIGN_CN.md:946`。

**影响：** `_cpu_gloo_parallelize_llama()`、`_cpu_gloo_parallelize_dsv4()`、`_apply_fsdp1_on_cpu()` 是当前标准入口下的不可达路径。

### 3.6 `_reapply_fsdp2_to_parts` 是死代码

- 仅定义于 `trainer_runner.py:39`，无调用点。
- 文档已记录：`DESIGN_CN.md:952`。

**建议：** 删除或移入历史参考文档，避免后续误用。

### 3.7 `schedule_generator.py` 是废弃模块

- `schedule/schedule_generator.py:27` 定义 `generate_interleaved_1f1b_schedule()`。
- `schedule/__init__.py:7-10` 未导出。
- 顶层 `__init__.py:56-58` 仍保留旧 import path alias。
- 文档已记录：`DESIGN_CN.md:942`。

**建议：** 决定是删除兼容路径，还是明确其只为旧 import 兼容而存在。

### 3.8 `collect_extension_metadata()` 已定义但 runner 未调用

- hook 定义：`extension_hooks.py:12-32`。
- runner 只导入并调用 `postprocess_extension_result()`：`trainer_runner.py:31`、`trainer_runner.py:784`。
- 文档已记录：`DESIGN_CN.md:807-812`。

**影响：** 扩展包如果期望 capture 阶段收集 metadata，当前不会被主 runner 触发。

### 3.9 `rewrite_runner.py` 是破损的一次性脚本

- `rewrite_runner.py:89` 调用已删除的 `_inject_synthetic_comm_events()`。
- 当前 `trainer_runner.py` 无该函数，PP 已改为 `_project_pp_comm_from_schedule()`。
- 文档已记录：`DESIGN_CN.md:953`。

**建议：** 删除该脚本，或移到历史目录且明确不可执行。

## 4. 其它一致性风险

### 4.1 `_DTYPE_BYTES` 表重复且键格式不同

- `ir/op_node.py:25-46` 使用 `"torch.float32"` 风格。
- `ir/workload_graph.py:87-93` 使用 `"float32"` 风格。
- 文档已记录：`DESIGN_CN.md:947`。

**影响：** dtype 字符串来自不同来源时可能静默回退到默认字节数。

### 4.2 `comm_backend="gloo"` 出现在配置注册表但会被强制改为空

- Llama3 configs 设置 `comm_backend="gloo"`：`llama3/config_registry.py:58`、`llama3/config_registry.py:114`。
- DeepSeek V4 configs 设置 `comm_backend="gloo"`：`deepseek_v4/config_registry.py:46`、`deepseek_v4/config_registry.py:88`。
- `trainer.py:407-415` 会把标准入口下的 `comm_backend` 覆盖为空。

**影响：** 配置表和实际行为不一致，容易误导使用者以为 gloo capture 已启用。

## 5. 建议优先级

1. **先清理文档明显过期内容**：`DESIGN_CN.md:300-302`、`DESIGN_CN.md:904-910`、`DESIGN_CN.md:40`。
2. **明确 Side-loaded 边界**：要么把 core model 中的 fake/simulator 逻辑抽回 experiment 或公共 PyTorch-native helper，要么更新设计原则，承认需要小范围核心模型适配。
3. **收敛模型特化逻辑**：将 DeepSeek 专用兼容迁出 `trainer.py`，放到 `deepseek_v4/config_registry.py` 或模型适配层。
4. **删除死代码/脚本**：`_reapply_fsdp2_to_parts`、`rewrite_runner.py`、必要时 `schedule_generator.py` 与 alias。
5. **处理空配置项**：删除或实现 `mode`、`capture_joint_fx`、`max_seq_len`、`batch_size`。
6. **修正文档/API 注释**：`export.py` self-contained/G6 说明、`meta_device_patches.py` patch 数量说明。
