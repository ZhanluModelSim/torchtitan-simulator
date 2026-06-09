---
name: simulator_result_audit
description: Audit simulator output for result rationality after a simulation run. Checks op classification, CostModel, comm capture, graph topology, memory, schedule, and known anomalies. Use after running deepseek_v4 smoketest or any simulation.
---

# Simulator Result Rationality Audit

Run this skill **after any simulation** to verify output correctness.
Covers 8 dimensions. Any failure blocks the commit.

## CostModel Metrics Explained

| Metric | Meaning | Calculation |
|---|---|---|
| `step_time` | Critical-path longest-path duration (us) | `_critical_path_time_us()` topological longest-path |
| `total_compute` | Sum of `perf_result.compute_time_us` over all nodes | Arithmetic sum |
| `total_comm` | Sum of `perf_result.comm_time_us` over all nodes | Arithmetic sum |

**Relationship:** `step_time <= compute + comm` always holds because
parallel execution makes the critical path shorter than the serial sum.
Ratio `step_time / (compute+comm)` in 0.2-0.8 is reasonable.

Example: two parallel 10ms branches → critical path = 10ms, sum = 20ms.

## Step 1: Op classification correctness

No comm op should be classified as `compute`. Check:

```python
import json
from torchtitan.experiments.simulator.op_classification import classify_op
COMM_KW = ['all_gather','allgather','all_reduce','reduce_scatter',
            'broadcast','barrier','c10d.','c10d_functional']
with open('simulator_output/simulation_result.json') as f:
    r = json.load(f)
mis = [n for n in r['compute_graph']['nodes']
       if n['op_type']=='compute' and any(k in n['op_name'] for k in COMM_KW)]
if mis:
    for n in mis:
        print(f'BUG: {n["op_name"]} -> {n["op_type"]} expected {classify_op(n["op_name"])}')
    FAIL
else:
    print('OK: no comm op misclassified as compute')
```

## Step 2: CostModel sanity

```python
cm = r['metadata']['cost_model']
step = cm['step_time_us']
sum_ = cm['total_compute_time_us'] + cm['total_comm_time_us']
ratio = step / sum_ if sum_ > 0 else 0
if step > sum_:
    FAIL  # impossible
elif ratio < 0.15:
    WARN  # excessive parallelism or missing edges
elif ratio > 0.85:
    WARN  # nearly serial
else:
    OK

# Per-phase: forward comm should have all_gather, backward comm reduce_scatter
phases = cm['per_phase']
assert phases['forward']['comm_time_us'] > 0  # FSDP all_gather
assert phases['backward']['comm_time_us'] > 0  # FSDP reduce_scatter
```

## Step 3: Comm capture consistency

- `comm_events` from CommRecorder should have `group_size` populated
- `comm_events` should be non-synthetic for gloo mode
- dispatch-level `c10d.*` nodes are supplementary (may lack `group_size`)
- Check for double-counting: `c10d.allgather_` (dispatch) vs
  `all_gather` (CommRecorder) represent the same logical operation

```python
comm_events = r.get('comm_events', [])
assert len(comm_events) >= 2  # all_gather + reduce_scatter
assert all(e.get('group_size') is not None for e in comm_events)
assert all(not e.get('synthetic', False) for e in comm_events)  # gloo=real
```

## Step 4: Graph topology

- backward ops >= forward ops (bwd/fwd ratio 1.5-2.5)
- All TensorMeta device should be `cpu` or `unknown` (never `meta`)
- Edges should connect most nodes

```python
nodes = r['compute_graph']['nodes']
fwd = sum(1 for n in nodes if n['phase']=='forward')
bwd = sum(1 for n in nodes if n['phase']=='backward')
ratio = bwd / fwd if fwd else 0
assert ratio >= 1.0  # backward should have more ops
devices = set(tm['device'] for n in nodes
              for tm in n['inputs']+n['outputs'] if tm.get('device'))
assert 'meta' not in devices
```

## Step 5: Memory estimate

- `gradient_bytes == parameter_bytes` (1 gradient per param)
- `optimizer_bytes >= 2 * parameter_bytes` (Adam: m+v states)
- `parameter_bytes / total_bytes` in 15%-40%

```python
mem = r['metadata']['memory']
assert mem['gradient_bytes'] == mem['parameter_bytes']
assert mem['optimizer_state_bytes'] >= 2 * mem['parameter_bytes']
```

## Step 6: Schedule & FSDP events

- Schedule should reflect full multi-rank topology (8 ranks for PP=2 TP=2 DP=2)
- `semantic_schedule=True` should produce schedule events
- FSDP events may be 0 (FSDP1 hooks differ from FSDP2)

```python
sched = r.get('schedule', {})
ranks = sorted(set(e.get('rank',0) for e in sched.get('events',[])))
assert len(ranks) == 8  # for PP=2 TP=2 DP=2
```

## Step 7: Known anomalies

- `fx_forward_graph_error`: CPU/meta mismatch during FX capture.
  Acceptable — FX is optional, dispatch trace is primary.
- `c10d.*` events with `group_size=None`: dispatch capture lacks group_size.
  Acceptable — CommRecorder provides it.
- `reduce_scatter_tensor` phase="forward": `TraceRecorder.current_phase`
  not switched to "backward" before `loss.backward()`. Known bug.

## Step 8: Output file completeness

```python
import os
expected = ['simulation_result.json', 'compute_graph.dot',
            'trace.json', 'trace.html', 'summary.txt']
assert all(os.path.exists(f'simulator_output/{f}') for f in expected)
# trace.html should be self-contained (no CDN)
with open('simulator_output/trace.html') as f:
    assert 'cdn.' not in f.read().lower()
```

## Decision Rule

| Result | Action |
|---|---|
| Any FAIL | Do not commit. Fix root cause, re-run audit. |
| All PASS | Commit. |
| WARN only | Commit with note in message. |

## Expected Values for DeepSeek V4 Smoketest

| Metric | Expected | Reason |
|---|---|---|
| Nodes | ~1585 | fwd+bwd compute+comm+data_move+memory |
| fwd nodes | ~586 | forward phase ops |
| bwd nodes | ~999 | backward phase ops |
| bwd/fwd | ~1.7 | backward > forward |
| comm nodes | 4 | 2 dispatch c10d + 2 CommRecorder |
| comm_events | 2 | all_gather + reduce_scatter (CommRecorder) |
| schedule events | 696 | Interleaved1F1B PP=2 TP=2 DP=2 |
| schedule deps | 1247 | schedule dependency edges |
| FSDP events | 0 | FSDP1 hooks not attached |
| parameter_bytes | 16.8 MB | 33M params x 0.5 bytes (bfloat16) |
| gradient_bytes | 16.8 MB | == parameter_bytes |
| optimizer_bytes | 33.6 MB | Adam: 2x parameter_bytes |
| total_bytes | 67.3 MB | param+grad+opt |
| step_time | ~12 ms | critical path |
| compute+comm sum | ~43 ms | serial sum |
| step/sum ratio | 0.28 | parallel execution paths |
| fwd compute | ~16 ms | forward compute |
| fwd comm | ~5.5 ms | FSDP all_gather |
| bwd compute | ~18 ms | backward compute |
| bwd comm | ~3 ms | FSDP reduce_scatter |

## Root Causes for Common Anomalies

| Anomaly | Cause | Fix |
|---|---|---|
| Comm ops classified as compute | `_COMM_MARKERS` missing c10d variants | Add `allgather`, `_allgather_base`, `c10d.` to markers |
| step_time > compute+comm | Comm nodes have nonzero compute_time | Fix classify_op |
| group_size=None on c10d ops | UnifiedTraceMode lacks group_size | CommRecorder provides it |
| reduce_scatter phase=forward | TraceRecorder.current_phase not switched | Set phase="backward" before loss.backward() |
| No comm events (gloo) | FSDP1 not wrapping model | Fix _cpu_noop_pipeline to forward parallelize_fn; wrap after super().__init__() |
| fx_forward_graph_error | CPU/meta mismatch | Acceptable; FX is optional |