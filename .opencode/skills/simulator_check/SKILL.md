---
name: simulator_check
description: Run simulator unit tests, data-structure validation, schedule-extract smoketest, cost-model sanity checks, and E2E simulation before every commit. Use when committing simulator code or when asked to validate simulator changes.
---

# Simulator Pre-Commit Validation Skill

Run this skill **before every commit** that touches files under
`torchtitan/experiments/simulator/`. It exercises the full validation
chain: unit tests, data-structure integrity, schedule extraction,
cost-model sanity, and an end-to-end simulation.

## Step 1: Simulator unit tests

```bash
pytest torchtitan/experiments/simulator/tests/test_simulator.py -v
```

All 95 tests must pass. If any fail, **do not commit** — fix the failure
first.

## Step 2: Data-structure round-trip validation

Run inline and confirm all assertions pass:

```python
python3 -c "
from torchtitan.experiments.simulator.nodes import (
    TensorMeta, OpNode, PerfResult, DataEdge, ComputeGraph,
    ScheduleEvent, ScheduleDep, TrainingSchedule, SimulationResult,
)
import json

# TensorMeta round-trip
tm = TensorMeta(shape=(2, 16), dtype='torch.float32', device='cpu', requires_grad=True)
d = tm.to_dict()
tm2 = TensorMeta(shape=tuple(d['shape']), dtype=d['dtype'], device=d['device'], requires_grad=d['requires_grad'])
assert tm2.shape == tm.shape and tm2.dtype == tm.dtype

# ScheduleEvent has op_node_ids field
ev = ScheduleEvent('e1', 'pp_forward', rank=0, pp_stage=0, microbatch_idx=0, op_node_ids=['n1'])
assert ev.op_node_ids == ['n1']
d_ev = ev.to_dict()
assert 'op_node_ids' in d_ev
assert d_ev['op_node_ids'] == ['n1']

# SimulationResult JSON-serializable
g = ComputeGraph()
n1 = OpNode('n1', 'aten.mm.default', 'compute', 'forward', [], [])
g.add_node(n1)
r = SimulationResult(compute_graph=g)
json.dumps(r.to_dict())

print('Data-structure validation PASSED')
"
```

## Step 3: Schedule-extract smoketest (all 7 schedule types)

```python
python3 -c "
from torchtitan.experiments.simulator.schedule_extract import (
    extract_schedule_from_pytorch, MockPipelineStage
)

# MockPipelineStage attributes
s = MockPipelineStage(stage_index=0, num_stages=4, group_rank=0, group_size=4)
assert s.stage_index == 0 and s.num_stages == 4 and s.has_backward is True

# All 7 schedule types produce events
for name, pp, virtual in [
    ('1F1B', 4, 1),
    ('GPipe', 4, 1),
    ('Interleaved1F1B', 4, 2),
    ('LoopedBFS', 4, 2),
    ('ZBVZeroBubble', 4, 2),
    ('DualPipeV', 4, 2),
    ('InterleavedZeroBubble', 4, 2),
]:
    result = extract_schedule_from_pytorch(
        pp_degree=pp, tp_degree=1, dp_degree=1,
        num_stages=pp*virtual, n_microbatches=8,
        schedule_name=name, virtual_stages_per_rank=virtual,
    )
    assert len(result.events) > 0, f'{name} should produce events'

# Interleaved1F1B has pp_comm deps
r = extract_schedule_from_pytorch(
    pp_degree=4, tp_degree=1, dp_degree=1, num_stages=8,
    n_microbatches=8, schedule_name='Interleaved1F1B', virtual_stages_per_rank=2,
)
dep_types = {d.dep_type for d in r.deps}
assert 'pp_comm' in dep_types, 'Interleaved must have pp_comm deps'
assert 'control' in dep_types

# DeepSeek V4 config: PP=2 TP=2 DP=2 Interleaved1F1B
r = extract_schedule_from_pytorch(
    pp_degree=2, tp_degree=2, dp_degree=2, num_stages=4,
    n_microbatches=8, schedule_name='Interleaved1F1B', virtual_stages_per_rank=2,
)
ranks = sorted(set(e.rank for e in r.events))
assert len(ranks) == 8, f'Expected 8 ranks, got {len(ranks)}'
event_types = set(e.event_type for e in r.events)
for expected in ['pp_forward', 'pp_backward', 'pp_send_activation', 'pp_recv_activation',
                  'fsdp2_all_gather', 'fsdp2_reduce_scatter', 'dp_gradient_sync']:
    assert expected in event_types, f'{expected} missing from DeepSeek V4 schedule'

print('Schedule-extract smoketest PASSED')
"
```

## Step 4: Cost-model sanity checks

```python
python3 -c "
from torchtitan.experiments.simulator.cost_model import (
    _estimate_flops, _estimate_comm_bytes, _numel, _tensor_bytes,
    MockCostModel, NoOverlap, FixedOverlap, OverlapStrategy,
    _critical_path_time_us, predict_multi_rank_step_time_us,
    link_schedule_to_graph,
)
from torchtitan.experiments.simulator.nodes import (
    OpNode, TensorMeta, PerfResult, ComputeGraph, DataEdge,
    ScheduleEvent, ScheduleDep, TrainingSchedule, SimulationResult,
)

# Matmul FLOPs: (2,8) x (8,4) = 2*2*8*4 = 128
node = OpNode('n1', 'aten.mm.default', 'compute', 'forward',
    [TensorMeta((2,8),'torch.float32','cpu'), TensorMeta((8,4),'torch.float32','cpu')],
    [TensorMeta((2,4),'torch.float32','cpu')])
assert _estimate_flops(node) == 128

# addmm not matched as element-wise add
node2 = OpNode('n2', 'aten.addmm.default', 'compute', 'forward',
    [TensorMeta((2,4),'torch.float32','cpu'), TensorMeta((2,8),'torch.float32','cpu'), TensorMeta((8,4),'torch.float32','cpu')],
    [TensorMeta((2,4),'torch.float32','cpu')])
assert _estimate_flops(node2) == 128

# reduce_scatter reads input bytes
node3 = OpNode('n3', 'reduce_scatter', 'comm_collective', 'backward',
    [TensorMeta((4,128),'torch.float32','cpu')], [TensorMeta((4,32),'torch.float32','cpu')],
    comm_op='reduce_scatter', comm_group_size=4)
assert _estimate_comm_bytes(node3) == 4*128*4

# Dynamic dim uses default_seq_len
assert _numel((2, -1, 8), default_seq_len=4096) == 2*4096*8

# Overlap strategies
assert NoOverlap().overlap_factor(10.0, 5.0) == 15.0
strategy = FixedOverlap(0.5)
assert strategy.overlap_factor(10.0, 5.0) == 10.0 + max(0, 5.0 - 5.0)

# Critical path with chain
graph = ComputeGraph()
for i in range(10):
    n = OpNode(f'n{i}', f'op{i}', 'compute', 'forward', [], [],
               perf_result=PerfResult(total_time_us=1.0))
    graph.add_node(n)
    if i > 0:
        graph.add_edge(DataEdge(f'n{i-1}', f'n{i}', 'data'))
assert _critical_path_time_us(graph) == 10.0

# Multi-rank step time
graph2 = ComputeGraph()
fwd = OpNode('n1','aten.mm.default','compute','forward',
    pp_stage=0, microbatch_idx=0,
    inputs=[TensorMeta((2,8),'torch.float32','cpu')],
    outputs=[TensorMeta((2,4),'torch.float32','cpu')],
    perf_result=PerfResult(total_time_us=10.0))
bwd = OpNode('n2','aten.mm.default','compute','backward',
    pp_stage=0, microbatch_idx=0,
    inputs=[], outputs=[],
    perf_result=PerfResult(total_time_us=20.0))
graph2.add_node(fwd)
graph2.add_node(bwd)
sched = TrainingSchedule()
e0 = ScheduleEvent('e0','pp_forward',rank=0,pp_stage=0,microbatch_idx=0)
e1 = ScheduleEvent('e1','pp_backward',rank=0,pp_stage=0,microbatch_idx=0)
sched.add_event(e0); sched.add_event(e1)
sched.add_dep(ScheduleDep('e0','e1','control'))
result = SimulationResult(compute_graph=graph2, schedule=sched)
link_schedule_to_graph(result)
assert e0.op_node_ids == ['n1']
assert e1.op_node_ids == ['n2']
step_time = predict_multi_rank_step_time_us(result)
assert step_time == 30.0

# MockCostModel accepts overlap_strategy + default_seq_len
model = MockCostModel(noise_std=0.0, default_seq_len=2048, overlap_strategy=FixedOverlap(0.5))
assert model.default_seq_len == 2048
assert model.overlap_strategy is not None

print('Cost-model sanity checks PASSED')
"
```

## Step 5: End-to-end simulation (fx + runtime + export)

```python
python3 -c "
import os, tempfile
import torch, torch.nn as nn
from torchtitan.experiments.simulator import Simulator
import torch.distributed as dist

os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29507')
os.environ.setdefault('RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')
dist.init_process_group(backend='gloo', init_method='env://')

try:
    sim = Simulator(rank=0, verbose=False)
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
    inputs = (torch.randn(2, 16),)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = sim.simulate_all([model], inputs, output_dir=tmpdir)
        assert len(result.compute_graph.nodes) > 0
        for fname in ['simulation_result.json', 'summary.txt']:
            assert os.path.exists(os.path.join(tmpdir, fname))

    print('E2E simulation PASSED')
finally:
    dist.destroy_process_group()
"
```

## Step 6: Pre-commit

```bash
pre-commit run --files torchtitan/experiments/simulator/
```

flake8, ufmt, pydoclint, and codespell must all pass. pyrefly errors in
simulator files are pre-existing type-narrowing issues (duck-typed mocks,
dynamic attribute injection) — they do not block commits.

## Step 7: DeepSeek V4 smoketest config validation

```python
python3 -c "
from torchtitan.experiments.simulator.deepseek_v4.config_registry import deepseek_v4_sim_smoketest
config = deepseek_v4_sim_smoketest()
assert config.parallelism.pipeline_parallel_degree == 2
assert config.parallelism.tensor_parallel_degree == 2
assert config.parallelism.data_parallel_shard_degree == 2
assert config.parallelism.pipeline_parallel_schedule == 'Interleaved1F1B'
assert config.simulation.semantic_schedule is True
assert config.simulation.cost_model is True
assert config.simulation.comm_backend == 'gloo'

# Schedule extraction for this config
from torchtitan.experiments.simulator.schedule_extract import extract_schedule_from_pytorch
r = extract_schedule_from_pytorch(
    pp_degree=2, tp_degree=2, dp_degree=2, num_stages=4,
    n_microbatches=8, schedule_name='Interleaved1F1B', virtual_stages_per_rank=2,
)
assert len(r.events) > 0
ranks = sorted(set(e.rank for e in r.events))
assert len(ranks) == 8

print('DeepSeek V4 smoketest config validation PASSED')
"
```

## Decision rule

- If **any** step fails: **do not commit**. Fix the failure, re-run this skill.
- If all steps pass: proceed with the commit.

## Key data structures to verify on every change

| Structure | Key fields to check | File |
|-----------|---------------------|------|
| `ScheduleEvent` | `op_node_ids: list[str]` | `nodes.py` |
| `PerfResult` | `compute_time_us`, `comm_time_us`, `total_time_us`, `flops` | `nodes.py` |
| `OpNode` | `pp_stage`, `microbatch_idx`, `comm_op`, `perf_result` | `nodes.py` |
| `TrainingSchedule` | `events`, `deps` | `nodes.py` |
| `SimulationResult` | `compute_graph`, `schedule`, `comm_events` | `nodes.py` |
| `MockCostModel` | `default_seq_len`, `overlap_strategy`, `noise_std` | `cost_model.py` |
| `_numel` | `default_seq_len` parameter | `cost_model.py` |
| `_estimate_flops` | `default_seq_len` parameter, matmul formula | `cost_model.py` |
| `_estimate_comm_bytes` | `reduce_scatter` reads input, `all_gather` reads output | `cost_model.py` |
| `OverlapStrategy` | `NoOverlap`, `FixedOverlap(factor)` | `cost_model.py` |
| `MockPipelineStage` | `stage_index`, `num_stages`, `has_backward` | `schedule_extract.py` |
| `extract_schedule_from_pytorch` | 7 schedule types, `pp_comm` deps | `schedule_extract.py` |
| `_infer_num_layers` | config.n_layers → layers → prefix fallback | `trainer_runner.py` |