# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..nodes import OpNode, SimulationResult
from .schedule_timing import _inject_schedule_timing


def _op_to_chrome_event(
    node: OpNode,
    pid: int = 0,
    tid: int = 0,
    ts_us: float = 0.0,
    dur_us: float = 1.0,
) -> dict[str, Any]:
    return {
        "ph": "X",
        "pid": pid,
        "tid": tid,
        "ts": ts_us / 1000.0,
        "dur": dur_us / 1000.0,
        "name": node.op_name,
        "cat": node.op_type,
        "args": {
            "node_id": node.node_id,
            "phase": node.phase,
            "pp_stage": node.pp_stage,
            "microbatch": node.microbatch_idx,
            "outputs": [str(o.shape) for o in node.outputs],
            "comm_op": node.comm_op,
        },
    }


def export_chrome_trace(
    result: SimulationResult,
    path: str | os.PathLike,
    us_per_op: float = 1.0,
) -> None:
    """
    Write a ``chrome://tracing``-compatible JSON trace file.

    Each op becomes a duration event (``"ph": "X"``).  Events are laid out
    sequentially per phase on separate *threads* (tid).  When
    :attr:`OpNode.perf_result` is available the duration reflects the
    estimated compute / communication time; otherwise events fall back to
    *us_per_op* microsecond slots.

    Args:
        result: The simulation result to render.
        path: Output JSON file path.
        us_per_op: Duration in microseconds to assign each op slot when no
            :class:`PerfResult` is available.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    phase_tid: dict[str, int] = {}
    tid_counter = [0]

    def _get_tid(phase: str) -> int:
        if phase not in phase_tid:
            phase_tid[phase] = tid_counter[0]
            tid_counter[0] += 1
        return phase_tid[phase]

    phase_ts: dict[str, float] = {}

    def _node_dur_us(node: OpNode) -> float:
        if node.perf_result is not None and node.perf_result.total_time_us > 0:
            return node.perf_result.total_time_us
        return us_per_op

    has_des = any(
        n.des_start_time_us is not None for n in result.compute_graph.nodes.values()
    )

    events: list[dict[str, Any]] = []
    if has_des:
        for node in result.compute_graph.nodes.values():
            if node.op_type in ("comm_collective", "comm_p2p"):
                engine = "comm_engine"
            else:
                engine = "compute_engine"
            tid = _get_tid(engine)
            ts_us = (
                node.des_start_time_us if node.des_start_time_us is not None else 0.0
            )
            if (
                node.des_finish_time_us is not None
                and node.des_start_time_us is not None
            ):
                dur_us = node.des_finish_time_us - node.des_start_time_us
            else:
                dur_us = _node_dur_us(node)
            events.append(
                _op_to_chrome_event(node, pid=0, tid=tid, ts_us=ts_us, dur_us=dur_us)
            )

        phase_starts: dict[str, float] = {}
        for node in result.compute_graph.nodes.values():
            phase = node.phase or "unknown"
            if node.des_start_time_us is not None:
                if (
                    phase not in phase_starts
                    or node.des_start_time_us < phase_starts[phase]
                ):
                    phase_starts[phase] = node.des_start_time_us
        for phase, start_us in sorted(phase_starts.items()):
            events.append(
                {
                    "ph": "i",
                    "pid": 0,
                    "tid": _get_tid("phase_markers"),
                    "ts": start_us / 1000.0,
                    "name": f"{phase} phase start",
                    "cat": "phase_boundary",
                    "s": "g",
                }
            )
    else:
        for node in result.compute_graph.nodes.values():
            phase = node.phase or "unknown"
            tid = _get_tid(phase)
            ts = phase_ts.get(phase, 0.0)
            dur = _node_dur_us(node)
            events.append(
                _op_to_chrome_event(node, pid=0, tid=tid, ts_us=ts, dur_us=dur)
            )
            phase_ts[phase] = ts + dur

    for ev in result.fsdp_events:
        phase = ev.get("phase", "unknown")
        ts = phase_ts.get(f"fsdp_{phase}", 0.0)
        events.append(
            {
                "ph": "X",
                "pid": 1,
                "tid": _get_tid(f"fsdp_{phase}"),
                "ts": ts / 1000.0,
                "dur": us_per_op / 1000.0,
                "name": ev.get("event_type", "fsdp_event"),
                "cat": "fsdp",
                "args": ev,
            }
        )
        phase_ts[f"fsdp_{phase}"] = ts + us_per_op

    if result.schedule is not None:
        _add_schedule_trace_events(events, result, _get_tid)

    for phase, tid in phase_tid.items():
        name = phase
        if has_des:
            if phase == "compute_engine":
                name = "Compute Engine"
            elif phase == "comm_engine":
                name = "Comm Engine"
            elif phase == "phase_markers":
                name = "Phase Markers"
        events.append(
            {
                "ph": "M",
                "pid": 0,
                "tid": tid,
                "name": "thread_name",
                "args": {"name": name},
            }
        )

    trace = {"traceEvents": events, "displayTimeUnit": "ms"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, default=str)


def _add_schedule_trace_events(
    events: list[dict[str, Any]],
    result: SimulationResult,
    _get_tid: Any,
) -> None:
    _add_aggregated_phase_events(events, result)

    if result.schedule is None:
        return

    des_event_map: dict[str, dict[str, Any]] = {}
    schedule_data: dict[str, Any] = {
        "events": [ev.to_dict() for ev in result.schedule.events],
    }
    _inject_schedule_timing({"schedule": schedule_data}, result)
    for ev_dict in schedule_data["events"]:
        eid = ev_dict.get("event_id", "")
        if "perf_cumulative_start_us" in ev_dict or "perf_total_time_us" in ev_dict:
            des_event_map[eid] = ev_dict

    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for ev in result.schedule.events:
        d = ev.to_dict()
        strategy = d.get("metadata", {}).get("strategy", "")
        et = d.get("event_type", "")
        if et.startswith("pp_") or et.startswith("loss"):
            strategy = "pp"
        elif et.startswith("fsdp2_"):
            strategy = "fsdp"
        elif et.startswith("tp_"):
            strategy = "tp"
        elif et.startswith("dp_"):
            strategy = "dp"
        elif et.startswith("optimizer"):
            strategy = "optim"
        by_strategy.setdefault(strategy, []).append(d)

    pid_map = {"pp": 2, "fsdp": 3, "tp": 4, "dp": 5, "optim": 6}

    for strategy, ev_list in by_strategy.items():
        pid = pid_map.get(strategy, 7)
        tid_map: dict[str, int] = {}
        tid_counter = [0]
        for ev in ev_list:
            lane = _schedule_event_lane_for_trace(ev, strategy)
            if lane not in tid_map:
                tid_map[lane] = tid_counter[0]
                tid_counter[0] += 1
            tid = pid * 100 + tid_map[lane]
            eid = ev.get("event_id", "")
            enriched = des_event_map.get(eid, ev)
            ts = enriched.get("perf_cumulative_start_us", ev.get("logical_clock", 0))
            dur = enriched.get("perf_total_time_us", ev.get("perf_total_time_us", 1.0))
            if dur <= 0:
                dur = 1.0
            events.append(
                {
                    "ph": "X",
                    "pid": pid,
                    "tid": tid,
                    "ts": ts / 1000.0,
                    "dur": dur / 1000.0,
                    "name": ev.get("event_type", "event"),
                    "cat": strategy,
                    "args": {
                        "pp_stage": ev.get("pp_stage"),
                        "mb": ev.get("microbatch_idx"),
                        "rank": ev.get("rank"),
                    },
                }
            )

    strategy_names = {
        "pp": "PP Schedule",
        "fsdp": "FSDP",
        "tp": "TP",
        "dp": "DP",
        "optim": "Optimizer",
    }
    for strategy, pid in pid_map.items():
        events.append(
            {
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "name": "process_name",
                "args": {"name": strategy_names.get(strategy, strategy)},
            }
        )


def _schedule_event_lane_for_trace(ev: dict[str, Any], strategy: str) -> str:
    return f"Rank {ev.get('rank', 0)}"


def _add_aggregated_phase_events(
    events: list[dict[str, Any]],
    result: SimulationResult,
) -> None:
    graph = result.compute_graph

    groups: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    group_indices: dict[tuple[str, int, int], list[int]] = defaultdict(list)

    node_list = list(graph.nodes.values())
    for idx, node in enumerate(node_list):
        if node.perf_result is None:
            continue
        key = (node.phase or "unknown", node.pp_stage or 0, node.microbatch_idx or 0)
        groups[key].append(node.perf_result.total_time_us)
        group_indices[key].append(idx)

    phase_order = ["forward", "backward", "optimizer"]

    pid = 7
    tid = 0
    cumulative_ts_us = 0.0

    for phase in phase_order:
        phase_groups = [
            (k, sum(times), min(group_indices[k]))
            for k, times in groups.items()
            if k[0] == phase
        ]
        if not phase_groups:
            continue

        for key, total_us, _ in phase_groups:
            pp_stage, mb = key[1], key[2]
            name = phase
            if pp_stage or mb:
                name += f" (s{pp_stage} mb{mb})"

            events.append(
                {
                    "ph": "X",
                    "pid": pid,
                    "tid": tid,
                    "ts": cumulative_ts_us / 1000.0,
                    "dur": total_us / 1000.0,
                    "name": name,
                    "cat": "aggregated",
                    "args": {
                        "phase": phase,
                        "pp_stage": pp_stage,
                        "microbatch": mb,
                        "op_count": len(groups[key]),
                        "total_us": round(total_us, 3),
                    },
                }
            )
            cumulative_ts_us += total_us
            tid += 1

    events.append(
        {
            "ph": "M",
            "pid": pid,
            "tid": 0,
            "name": "process_name",
            "args": {"name": "Aggregated Phases"},
        }
    )
