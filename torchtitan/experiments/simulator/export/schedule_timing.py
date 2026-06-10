# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

from ..nodes import SimulationResult


def _populate_des_metadata(result: SimulationResult) -> None:
    has_des_nodes = any(
        n.des_start_time_us is not None for n in result.compute_graph.nodes.values()
    )
    has_des_events = result.schedule is not None and any(
        ev.des_start_time_us is not None for ev in result.schedule.events
    )
    if not has_des_nodes and not has_des_events:
        return
    from ..des_engine import compute_des_memory_timeline, compute_des_utilization

    util = compute_des_utilization(result)
    result.metadata.setdefault("des_engine", {}).update(util)
    mem = compute_des_memory_timeline(result)
    result.metadata["des_memory"] = {
        "static_memory_bytes": mem["static_memory_bytes"],
        "peak_dynamic_bytes": mem["peak_dynamic_bytes"],
        "peak_total_bytes": mem["peak_total_bytes"],
        "timeline": mem["timeline"],
        "phase_peak": mem["phase_peak"],
    }


def _inject_schedule_timing(data: dict[str, Any], result: SimulationResult) -> None:
    graph = result.compute_graph

    phase_totals: dict[str, float] = {}
    for node in graph.nodes.values():
        if node.perf_result is None:
            continue
        phase = node.phase or "unknown"
        phase_totals[phase] = (
            phase_totals.get(phase, 0.0) + node.perf_result.total_time_us
        )
    grand_total = sum(phase_totals.values())

    event_counts: dict[str, int] = {}
    schedule = data.get("schedule")
    if schedule and schedule.get("events"):
        for ev in schedule["events"]:
            ev_type = ev.get("event_type", "")
            metadata = ev.get("metadata", {}) or {}
            strategy = metadata.get("strategy", "")
            phase = _schedule_event_to_phase(ev_type, strategy)
            event_counts[phase] = event_counts.get(phase, 0) + 1

    enriched_events: list[dict[str, Any]] = []
    if schedule and schedule.get("events"):
        des_event_map: dict[str, tuple[float, float]] = {}
        if result.schedule is not None:
            for ev in result.schedule.events:
                if (
                    ev.des_start_time_us is not None
                    and ev.des_finish_time_us is not None
                ):
                    des_event_map[ev.event_id] = (
                        ev.des_start_time_us,
                        ev.des_finish_time_us,
                    )

        cumulative_per_phase: dict[str, float] = {}
        for ev in schedule["events"]:
            ev_type = ev.get("event_type", "")
            metadata = ev.get("metadata", {}) or {}
            strategy = metadata.get("strategy", "")
            phase = _schedule_event_to_phase(ev_type, strategy)
            eid = ev.get("event_id", "")

            ev_copy = dict(ev)
            if eid in des_event_map:
                start, finish = des_event_map[eid]
                ev_copy["perf_total_time_us"] = round(finish - start, 3)
                ev_copy["perf_cumulative_start_us"] = round(start, 3)
            else:
                count = event_counts.get(phase, 1)
                phase_total = phase_totals.get(phase, 0.0)
                per_event = phase_total / max(count, 1)
                ev_copy["perf_total_time_us"] = round(per_event, 3)
                ev_copy["perf_cumulative_start_us"] = round(
                    cumulative_per_phase.get(phase, 0.0), 3
                )
                cumulative_per_phase[phase] = (
                    cumulative_per_phase.get(phase, 0.0) + per_event
                )
            enriched_events.append(ev_copy)

        schedule["events"] = enriched_events
        schedule["perf_grand_total_us"] = round(grand_total, 3)

    data["perf_schedule"] = {
        "grand_total_us": round(grand_total, 3),
        "phase_totals": {p: round(t, 3) for p, t in sorted(phase_totals.items())},
    }


def _schedule_event_to_phase(event_type: str, strategy: str) -> str:
    et = event_type.lower()
    strategy_lower = (strategy or "").lower()
    if "bwd" in et or "backward" in et or "backward" in strategy_lower:
        return "backward"
    if "fwd" in et or "forward" in et:
        return "forward"
    if "optim" in et:
        return "optimizer"
    if strategy_lower in ("pp", "compute"):
        return "forward"
    if "reduce" in et or "gradient" in et:
        return "backward"
    if strategy_lower in ("fsdp2", "tp", "dp"):
        return "forward"
    if "loss" in et:
        return "forward"
    return "forward"


def _event_lane(ev: dict[str, Any]) -> str:
    event_type = str(ev.get("event_type", ""))
    metadata = ev.get("metadata", {}) or {}
    strategy = str(metadata.get("strategy", "")).lower()

    if event_type.startswith("dp_") or strategy == "dp":
        return f"DP rank {ev.get('rank', 0)}"
    if event_type.startswith("optimizer") or "step" in event_type:
        return f"Optim rank {ev.get('rank', 0)}"
    if event_type.startswith("tp_") or strategy == "tp":
        return f"TP rank {ev.get('rank', 0)}"
    if event_type.startswith("fsdp_") or strategy in ("fsdp2", "fsdp"):
        return f"FSDP rank {ev.get('rank', 0)}"
    if event_type.startswith("pp_") or ev.get("pp_stage") is not None:
        pp_rank = ev.get("pp_rank", 0)
        pp_stage = ev.get("pp_stage")
        return f"PP stage {pp_stage or pp_rank} (rank {pp_rank})"
    if event_type.startswith("loss") or strategy == "compute":
        return f"Loss (pp rank {ev.get('pp_rank', 0)})"
    if ev.get("op"):
        return f"Comm rank {ev.get('rank', 0)}"
    return f"Rank {ev.get('rank', 0)}"


def _event_step(ev: dict[str, Any]) -> int:
    metadata = ev.get("metadata", {}) or {}
    try:
        return int(metadata.get("step", ev.get("step", 0)))
    except (TypeError, ValueError):
        return 0
