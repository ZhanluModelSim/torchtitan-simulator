# CostModel 接入指南

本文档说明如何将第三方 CostModel（计算 + 通信）接入 TorchTitan Simulator，实现端到端训练时长预测。

**无需修改 `trainer_runner.py`** — 只需配置 `cost_model_class` 即可。

## 架构概览

```
┌──────────────────────────────────────────────────────────────┐
│                   SimulationResult                           │
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │    ComputeGraph       │  │     TrainingSchedule          │ │
│  │  ┌──────┐ ┌──────┐   │  │  ┌─────────┐ ┌─────────┐    │ │
│  │  │OpNode│ │OpNode│…  │  │  │Schedule │ │Schedule │…   │ │
│  │  │.perf │ │.perf │   │  │  │ Event   │ │ Event   │    │ │
│  │  └──────┘ └──────┘   │  │  └─────────┘ └─────────┘    │ │
│  └──────────────────────┘  └──────────────────────────────┘ │
│                          ▲                                   │
│                          │ estimate_graph()                  │
│                    ┌─────┴──────┐                            │
│                    │  CostModel  │                            │
│                    │ (Your impl) │                            │
│                    └────────────┘                            │
└──────────────────────────────────────────────────────────────┘
```

Simulator 捕获 ComputeGraph（每个 OpNode 含 tensor shape、op type、phase 等元信息）后，CostModel 遍历所有节点，调用 `estimate_node()` 为每个 OpNode 填充 `PerfResult`（耗时、FLOPs、字节量）。填充后的数据自动反映到 Chrome Trace、HTML 泳道图和 text summary 中。

## CostModel 基类

**位置**: `torchtitan/experiments/simulator/cost_model.py`

```python
class CostModel:
    """性能预估抽象接口。"""

    def estimate_node(self, node: OpNode) -> PerfResult:
        """预估单个算子的性能。"""
        raise NotImplementedError

    def estimate_graph(self, graph: ComputeGraph) -> None:
        """遍历所有节点并填充 node.perf_result。"""
        for node in graph.nodes.values():
            node.perf_result = self.estimate_node(node)

    def estimate_result(self, result: SimulationResult) -> None:
        """对 SimulationResult 调用 estimate_graph。"""
        self.estimate_graph(result.compute_graph)

    def predict_step_time_us(self, graph: ComputeGraph) -> float:
        """基于 perf_result 预测端到端 step 耗时（关键路径分析）。"""
        return _critical_path_time_us(graph)
```

## PerfResult 数据类

**位置**: `torchtitan/experiments/simulator/nodes.py`

```python
@dataclass
class PerfResult:
    compute_time_us: float = 0.0   # 计算耗时（微秒）
    comm_time_us: float = 0.0      # 通信耗时（微秒）
    total_time_us: float = 0.0     # 总耗时（compute + comm）
    flops: int = 0                 # 浮点运算量
    bytes_read: int = 0            # 读取字节数
    bytes_written: int = 0         # 写入字节数
    metadata: dict[str, Any] = field(default_factory=dict)  # 扩展字段
```

## OpNode 关键字段

在 `estimate_node()` 中可以使用的字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `op_name` | `str` | 完整 ATen 函数名，如 `aten.mm.default`、`aten._scaled_dot_product_flash_attention.default` |
| `op_type` | `str` | `"compute"` / `"comm_collective"` / `"comm_p2p"` / `"data_move"` / `"memory"` |
| `phase` | `str` | `"forward"` / `"backward"` / `"optimizer"` |
| `inputs` | `list[TensorMeta]` | 输入 tensor 的 shape、dtype 等 |
| `outputs` | `list[TensorMeta]` | 输出 tensor 的 shape、dtype 等 |
| `comm_op` | `str \| None` | 通信类型：`"all_reduce"`、`"all_gather"` 等 |
| `comm_group_size` | `int \| None` | 通信组大小 |

## 接入步骤

### Step 1: 实现 CostModel 子类

在你的项目中创建一个 Python 文件，继承 `CostModel` 并实现 `estimate_node()`：

```python
# my_cost_model.py

from torchtitan.experiments.simulator.cost_model import CostModel
from torchtitan.experiments.simulator.nodes import OpNode, PerfResult, TensorMeta

class MyCostModel(CostModel):
    """接入第三方算力 + 带宽模型的 CostModel。"""

    def __init__(
        self,
        compute_tflops: float = 312.0,   # 你的硬件算力 (TFLOPS)
        hbm_gb_per_s: float = 2000.0,    # HBM 带宽 (GB/s)
        nvlink_gb_per_s: float = 600.0,  # 卡间通信带宽 (GB/s)
        comm_latency_us: float = 2.0,    # 通信固定延迟 (µs)
    ):
        self.compute_tflops = compute_tflops
        self.hbm_gb_per_s = hbm_gb_per_s
        self.nvlink_gb_per_s = nvlink_gb_per_s
        self.comm_latency_us = comm_latency_us

    def estimate_node(self, node: OpNode) -> PerfResult:
        # ── 1. 预估计算 ──────────────────────────────────
        if node.op_type == "compute":
            flops = self._estimate_flops(node)
            bytes_r, bytes_w = self._estimate_bytes(node)
            # 计算时间：FLOPs / 算力
            compute_us = flops / (self.compute_tflops * 1e6)
            # 内存受限检查：算术强度 < 阈值则按带宽计
            total_bytes = bytes_r + bytes_w
            if total_bytes > 0:
                ai = flops / total_bytes
                mem_us = total_bytes / (self.hbm_gb_per_s * 1e3)
                if ai < 50:  # 你的硬件 AI 阈值
                    compute_us = max(compute_us, mem_us)
            return PerfResult(
                compute_time_us=compute_us,
                total_time_us=compute_us,
                flops=flops,
                bytes_read=bytes_r,
                bytes_written=bytes_w,
            )

        # ── 2. 预估通信 ──────────────────────────────────
        if node.op_type in ("comm_collective", "comm_p2p"):
            comm_bytes = self._estimate_comm_bytes(node)
            # alpha-beta 模型：latency + bytes / bandwidth
            comm_us = self.comm_latency_us + comm_bytes / (self.nvlink_gb_per_s * 1e3)
            # all_reduce = 2 * (group_size - 1) / group_size * bytes （ring 算法）
            if node.comm_op == "all_reduce" and node.comm_group_size:
                gs = max(node.comm_group_size, 1)
                comm_us *= 2 * (gs - 1) / gs
            return PerfResult(
                comm_time_us=comm_us,
                total_time_us=comm_us,
                bytes_read=comm_bytes,
                bytes_written=comm_bytes,
            )

        # ── 3. 其他（data_move / memory）─────────────────
        bytes_r, bytes_w = self._estimate_bytes(node)
        total_bytes = bytes_r + bytes_w
        if total_bytes > 0:
            t = total_bytes / (self.hbm_gb_per_s * 1e3)
            return PerfResult(total_time_us=t, bytes_read=bytes_r, bytes_written=bytes_w)

        return PerfResult()

    # ── 辅助方法 ──────────────────────────────────────

    def _estimate_flops(self, node: OpNode) -> int:
        """你的 FLOPs 预估逻辑。参考 cost_model.py 中 _estimate_flops 的实现。"""
        # TODO: 接入你的计算 CostModel
        return 0

    def _estimate_bytes(self, node: OpNode) -> tuple[int, int]:
        """根据 input/output TensorMeta 计算字节数。"""
        bytes_r = sum(self._tensor_bytes(t.shape, t.dtype) for t in node.inputs)
        bytes_w = sum(self._tensor_bytes(t.shape, t.dtype) for t in node.outputs)
        return bytes_r, bytes_w

    def _estimate_comm_bytes(self, node: OpNode) -> int:
        """预估通信字节数。"""
        total = 0
        for o in node.outputs:
            total += self._tensor_bytes(o.shape, o.dtype)
        if total == 0:
            for i in node.inputs:
                total += self._tensor_bytes(i.shape, i.dtype)
        return total

    @staticmethod
    def _tensor_bytes(shape: tuple, dtype: str) -> int:
        """shape × dtype_size。"""
        dtype_sizes = {
            "torch.float32": 4, "torch.float16": 2, "torch.bfloat16": 2,
            "torch.int64": 8, "torch.int32": 4, "torch.bool": 1,
        }
        numel = 1
        for d in shape:
            numel *= max(d, 1) if d is not None else 1024
        return numel * dtype_sizes.get(dtype, 2)
```

### Step 2: 配置 `cost_model_class`

在 `config_registry.py` 中设置 `simulation.cost_model_class` 为你的类的完整路径：

```python
simulation=SimulationConfig(
    output_dir="./simulator_output",
    output_formats=["json", "dot", "chrome_trace", "html", "text"],
    cost_model=True,
    cost_model_class="my_package.my_cost_model.MyCostModel",  # ← 你的类路径
    semantic_schedule=True,
),
```

或通过 CLI override（无需改 config_registry）：

```bash
MODULE=simulator.deepseek_v4 CONFIG=deepseek_v4_sim_smoketest \
  NGPU=1 python3 -m torchtitan.train \
  --module simulator.deepseek_v4 --config deepseek_v4_sim_smoketest \
  --training.steps 1 --comm.mode=fake_backend \
  --simulation.cost_model True \
  --simulation.cost_model_class my_package.my_cost_model.MyCostModel
```

`trainer_runner.py` 会自动通过 `importlib` 动态加载你的类，实例化后调用 `estimate_graph()`。

> **无需修改 `trainer_runner.py` 或 simulator 的任何源码。**

### Step 3: 确保你的类在 Python path 中

确保包含 `MyCostModel` 的 Python 包在 `PYTHONPATH` 中：

```bash
export PYTHONPATH="/path/to/your/project:$PYTHONPATH"
```

或者在项目的 `setup.py` / `pyproject.toml` 中声明依赖。

### Step 4: 运行并查看结果

| 输出文件 | 包含的 CostModel 数据 |
|---------|----------------------|
| `simulation_result.json` | 每个节点的 `perf_result` 字段 |
| `trace.json` | Chrome Trace 事件使用 `perf_result` 的 `ts`/`dur` |
| `summary.txt` | "Performance Estimate" 段：step time、per-phase breakdown |
| `trace.html` | DAG 节点标注耗时；Chrome Trace 泳道图 |

### predict_step_time_us 说明

基类默认使用**拓扑最长路径**算法计算 step 耗时。如果你的 CostModel 需要考虑算子融合、通信重叠、wave-level 并行等更复杂的调度，可以 override：

```python
class MyCostModel(CostModel):
    def predict_step_time_us(self, graph: ComputeGraph) -> float:
        """你的端到端预估逻辑。"""
        # 方案 1: 在 estimate_graph 后，用 TrainingSchedule 的时序做模拟
        # 方案 2: 直接对外部调度器输出做预估
        return super().predict_step_time_us(graph)
```

## 参考

- `torchtitan/experiments/simulator/cost_model.py` — MockCostModel 完整实现（含 FLOPs 启发式预估、算力/带宽模型、关键路径分析）
- `torchtitan/experiments/simulator/nodes.py` — PerfResult 和 OpNode 定义
- `torchtitan/experiments/simulator/trainer_runner.py` — 集成点
- `docs/simulator_architecture.md` — Simulator 整体架构
