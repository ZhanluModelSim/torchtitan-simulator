---
name: simulator_e2e_smoketest
description: Use when asked to run an E2E simulator smoketest, validate DES output visually, or do a full end-to-end validation of the simulator with real multi-process execution. Runs the DeepSeek V4 smoketest with PP=2 TP=2 DP=2 Interleaved1F1B schedule using gloo backend.
---

# Simulator E2E Smoketest

Full end-to-end validation of the simulator with real multi-process execution.
Uses the DeepSeek V4 smoketest config: PP=2, TP=2, DP=2 (8 ranks),
Interleaved1F1B schedule, gloo comm backend, semantic_schedule=True, cost_model=True.

## Command

```bash
MODULE=simulator.deepseek_v4 CONFIG=deepseek_v4_sim_smoketest NGPU=8 bash run_train.sh
```

This runs torchrun with 8 processes, executes one instrumented training step,
produces DES simulation with real CPU communication capture.

## Expected Output

Output directory: `./simulator_output/`

| File | Content |
|------|---------|
| `simulation_result.json` | Full JSON with 696 schedule events, DES engine/memory metadata, all events have `perf_total_time_us`/`perf_cumulative_start_us` |
| `trace.html` | Interactive HTML with swimlanes, DAGs, memory trace, DES stats |
| `trace.json` | Chrome-trace format for `chrome://tracing` |
| `compute_graph.dot` | Graphviz topology |
| `summary.txt` | Human-readable text summary |

## Key Validation Points

After the run completes, verify:

1. **Schedule events** — 696 events, all have `perf_total_time_us` and `perf_cumulative_start_us`
2. **DES engine metadata** — `compute_busy_pct` > 0, `overlap_pct` > 0, `des_vs_cp_ratio` > 1
3. **OpNode DES timing** — All 1585 nodes have `des_start_time_us`/`des_finish_time_us`
4. **Swimlanes** — First phase (forward) and later phases (backward, optimizer) both render correctly with proportional bar widths, not squished
5. **No fallback timing** — Events use DES timing (not cumulative-per-phase approximation)

```python
python3 -c "
import json
data = json.load(open('simulator_output/simulation_result.json'))
events = data.get('schedule', {}).get('events', [])
des_engine = data.get('metadata', {}).get('des_engine', {})
nodes = data.get('compute_graph', {}).get('nodes', [])

assert len(events) > 0, 'Schedule events missing'
assert all('perf_total_time_us' in ev for ev in events), 'Events missing perf timing'
assert all('perf_cumulative_start_us' in ev for ev in events), 'Events missing cumulative timing'
assert des_engine.get('compute_busy_pct', 0) > 0, 'DES engine compute utilization missing'
assert des_engine.get('des_vs_cp_ratio', 0) > 1, 'DES vs CP ratio must be > 1 (DES accounts for contention)'
assert all(n.get('des_start_time_us') is not None for n in nodes), 'OpNode DES timing missing'

print(f'Events: {len(events)}, Nodes: {len(nodes)}, DES ratio: {des_engine[\"des_vs_cp_ratio\"]}')
print('E2E smoketest validation PASSED')
"
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No such file or directory: 'lspci'` | Expected on WSL — fallback to A100 peak FLOPs |
| `Pipeline Parallel loss is not visible` | Add rank 4 to LOG_RANK if you want to see loss |
| `warmup steps exceed total steps` | Expected — auto-adjusted |
| `DTensor random operators may not have complete support on cpu` | Expected on CPU — non-blocking |
| Process hangs | Check `MASTER_PORT` conflicts; kill stale torchrun processes |
| DES engine metadata empty | Bug: `_populate_des_metadata` not detecting ScheduleEvent DES timing — see recent fix |