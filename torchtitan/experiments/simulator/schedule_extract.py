# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Extract training schedules from real PyTorch ``_PipelineSchedule`` objects
using duck-typed ``MockPipelineStage`` instances, without requiring
multi-process distributed execution.

Instead of reimplementing PP scheduling algorithms (1F1B, Interleaved,
GPipe, etc.) in the simulator, this module constructs a real PyTorch
schedule object with mock stages and reads the ``pipeline_order`` /
``pipeline_order_with_comms`` action tables that are computed during
``__init__``.  This guarantees the extracted schedule matches upstream
PyTorch behaviour exactly, and automatically picks up upstream schedule
changes without simulator code modifications.

Supported schedule types
-------------------------
All schedules registered in ``torch.distributed.pipelining.schedules``:
Schedule1F1B, ScheduleGPipe, ScheduleInterleaved1F1B, ScheduleLoopedBFS,
ScheduleInterleavedZeroBubble, ScheduleZBVZeroBubble, ScheduleDualPipeV,
and ``_PipelineScheduleRuntime`` (CSV-driven).

Usage::

    from torchtitan.experiments.simulator.schedule_extract import (
        extract_schedule_from_pytorch,
    )

    schedule = extract_schedule_from_pytorch(
        pp_degree=4,
        tp_degree=1,
        dp_degree=1,
        num_stages=4,
        n_microbatches=8,
        schedule_name="1F1B",
    )
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from .nodes import ScheduleDep, ScheduleEvent, TrainingSchedule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# _ComputationType → event_type mapping
# ---------------------------------------------------------------------------

_COMP_TYPE_TO_EVENT_TYPE: dict[str, str] = {
    "F": "pp_forward",
    "B": "pp_backward",
    "I": "pp_backward_input",
    "W": "pp_backward_weight",
    "UNSHARD": "fsdp2_all_gather",
    "RESHARD": "fsdp2_reduce_scatter",
    "SEND_F": "pp_send_activation",
    "RECV_F": "pp_recv_activation",
    "SEND_B": "pp_send_gradient",
    "RECV_B": "pp_recv_gradient",
    "REDUCE_GRAD": "dp_gradient_sync",
    "OVERLAP_F_B": "overlap_forward_backward",
}


def _comp_type_to_event_type(comp_type: Any) -> str:
    value = str(comp_type.value) if hasattr(comp_type, "value") else str(comp_type)
    return _COMP_TYPE_TO_EVENT_TYPE.get(value, value.lower())


# ---------------------------------------------------------------------------
# MockPipelineStage
# ---------------------------------------------------------------------------


class MockPipelineStage:
    """Duck-typed mock that satisfies ``_PipelineSchedule.__init__`` attribute reads.

    Does **not** call ``dist.get_rank`` / ``dist.get_world_size`` — works in
    single-process CPU mode.  The real ``PipelineStage.__init__`` calls those
    functions unconditionally, which would fail without ``dist.init_process_group``.
    This mock bypasses that by setting the attributes directly.

    All PyTorch schedule ``__init__`` methods only read integer attributes
    (``stage_index``, ``num_stages``, ``group_rank``, ``group_size``) and never
    perform ``isinstance`` checks on the stage object during construction.
    """

    def __init__(
        self,
        stage_index: int,
        num_stages: int,
        group_rank: int = 0,
        group_size: int = 1,
    ) -> None:
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.group_rank = group_rank
        self.group_size = group_size
        self.device = torch.device("cpu")
        self.submod = nn.Module()
        self.has_backward = True
        self.stage_index_to_group_rank: dict[int, int] = {
            i: i % group_size for i in range(num_stages)
        }


# ---------------------------------------------------------------------------
# Schedule construction from config
# ---------------------------------------------------------------------------


def _build_mock_stages_and_schedule(
    schedule_name: str,
    pp_degree: int,
    num_stages: int,
    n_microbatches: int,
    virtual_stages_per_rank: int,
) -> Any:
    """Construct a real PyTorch schedule object with mock stages."""
    from torch.distributed.pipelining.schedules import (
        PipelineScheduleMulti,
        PipelineScheduleSingle,
        get_schedule_class,
    )

    schedule_class = get_schedule_class(schedule_name)
    is_multi = issubclass(schedule_class, PipelineScheduleMulti)

    if is_multi:
        stages = [
            MockPipelineStage(
                stage_index=i,
                num_stages=num_stages,
                group_rank=0,
                group_size=pp_degree,
            )
            for i in range(virtual_stages_per_rank)
        ]
        schedule = schedule_class(
            stages,
            n_microbatches=n_microbatches,
            scale_grads=False,
        )
    else:
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

    return schedule


def extract_schedule_from_pytorch(
    *,
    pp_degree: int,
    tp_degree: int,
    dp_degree: int,
    num_stages: int,
    n_microbatches: int,
    schedule_name: str,
    virtual_stages_per_rank: int = 1,
) -> TrainingSchedule:
    """Construct a real PyTorch schedule with mock stages and extract its action table.

    The PyTorch schedule computes ``pipeline_order`` (compute-only) and
    ``pipeline_order_with_comms`` (with FSDP/PP communication actions) during
    ``__init__``.  This function reads those tables and converts them into a
    simulator ``TrainingSchedule``, preserving the exact warmup/steady-state/
    cooldown ordering, FSDP all-gather/reduce-scatter lifecycle, PP send/recv
    pairs, and DP gradient sync placement.

    Args:
        pp_degree: Number of pipeline-parallel ranks.
        tp_degree: Tensor-parallel degree (used for rank mapping).
        dp_degree: Data-parallel degree (dp_shard × dp_replicate).
        num_stages: Total pipeline stages (pp_degree × virtual_stages_per_rank).
        n_microbatches: Microbatches per training step.
        schedule_name: Schedule type string (``"1F1B"``, ``"Interleaved1F1B"``,
            ``"GPipe"``, etc.).
        virtual_stages_per_rank: Virtual pipeline stages per rank.

    Returns:
        A populated :class:`TrainingSchedule`.
    """
    schedule = _build_mock_stages_and_schedule(
        schedule_name=schedule_name,
        pp_degree=pp_degree,
        num_stages=num_stages,
        n_microbatches=n_microbatches,
        virtual_stages_per_rank=virtual_stages_per_rank,
    )

    pipeline_order: dict[int, list[Any]] | None = None

    if hasattr(schedule, "pipeline_order_with_comms"):
        pipeline_order = schedule.pipeline_order_with_comms
    elif hasattr(schedule, "pipeline_order"):
        pipeline_order = schedule.pipeline_order

    if pipeline_order is None:
        logger.warning(
            "Schedule %s has no pipeline_order; returning empty TrainingSchedule",
            type(schedule).__name__,
        )
        return TrainingSchedule(
            metadata={
                "schedule_type": type(schedule).__name__,
                "pp_degree": pp_degree,
                "tp_degree": tp_degree,
                "dp_degree": dp_degree,
                "n_microbatches": n_microbatches,
                "virtual_stages_per_rank": virtual_stages_per_rank,
                "extraction_method": "none",
            }
        )

    return _convert_pipeline_order_to_training_schedule(
        pipeline_order,
        schedule=schedule,
        pp_degree=pp_degree,
        tp_degree=tp_degree,
        dp_degree=dp_degree,
        n_microbatches=n_microbatches,
        schedule_name=schedule_name,
        virtual_stages_per_rank=virtual_stages_per_rank,
    )


# ---------------------------------------------------------------------------
# Core conversion: _Action list → TrainingSchedule
# ---------------------------------------------------------------------------

# _Action is a NamedTuple: (stage_index, computation_type, microbatch_index, sub_actions)


def _action_to_event_type(action: Any) -> str:
    comp_type = action.computation_type
    return _comp_type_to_event_type(comp_type)


def _stage_to_pp_rank(
    stage_index: int, schedule: Any, pp_degree: int
) -> int:
    if hasattr(schedule, "stage_index_to_group_rank"):
        mapping = schedule.stage_index_to_group_rank
        if mapping and stage_index in mapping:
            return mapping[stage_index]
    return stage_index // pp_degree if pp_degree > 0 else 0


def _convert_pipeline_order_to_training_schedule(
    pipeline_order: dict[int, list[Any]],
    *,
    schedule: Any,
    pp_degree: int,
    tp_degree: int,
    dp_degree: int,
    n_microbatches: int,
    schedule_name: str,
    virtual_stages_per_rank: int,
) -> TrainingSchedule:
    """Convert a PyTorch ``pipeline_order`` dict to a ``TrainingSchedule``.

    Each ``_Action`` in the pipeline_order becomes a ``ScheduleEvent``.
    Dependencies are built from:
    - Sequential ordering within each rank (control deps)
    - SEND_F ↔ RECV_F and SEND_B ↔ RECV_B pairs across ranks (pp_comm deps)
    """
    total_ranks = pp_degree * tp_degree * dp_degree
    ts = TrainingSchedule(
        metadata={
            "schedule_type": schedule_name,
            "pp_degree": pp_degree,
            "tp_degree": tp_degree,
            "dp_degree": dp_degree,
            "n_microbatches": n_microbatches,
            "virtual_stages_per_rank": virtual_stages_per_rank,
            "total_ranks": total_ranks,
            "extraction_method": "pytorch_pipeline_order",
        }
    )

    counter = [0]

    def _next_id(prefix: str) -> str:
        counter[0] += 1
        return f"{prefix}_{counter[0]:07d}"

    # ── Build ScheduleEvents from _Action list ────────────────────────
    # PyTorch pipeline_order is keyed by *PP rank*, not global rank.
    # Each PP rank covers tp_degree × dp_degree global ranks.

    # Track events per (event_type, microbatch_idx, stage_index, pp_rank)
    # for cross-rank dependency matching.
    event_index: dict[tuple[str, int | None, int, int], ScheduleEvent] = {}

    per_rank_prev: dict[int, str] = {}

    for pp_rank_key, action_list in pipeline_order.items():
        logical_clock = 0
        for action in action_list:
            if action is None:
                logical_clock += 1
                continue

            event_type = _action_to_event_type(action)
            stage_index = action.stage_index
            mb_index = action.microbatch_index

            # pipeline_order keys are PP ranks; use the key directly as pp_rank.
            # For PipelineScheduleSingle (1F1B, GPipe), each key is a PP rank
            # and the actions list describes what that PP rank does.
            actual_pp_rank = pp_rank_key

            # Base global rank for this PP rank
            base_global_rank = actual_pp_rank * tp_degree * dp_degree

            eid = _next_id(event_type)
            ev = ScheduleEvent(
                event_id=eid,
                event_type=event_type,
                rank=base_global_rank,
                pp_rank=actual_pp_rank,
                pp_stage=stage_index,
                microbatch_idx=mb_index,
                logical_clock=logical_clock,
                metadata={
                    "strategy": "pp",
                    "step": 0,
                    "mb": mb_index,
                },
            )
            ts.add_event(ev)

            # Sequential dependency within same PP rank
            if pp_rank_key in per_rank_prev:
                ts.add_dep(ScheduleDep(per_rank_prev[pp_rank_key], eid, "control"))
            per_rank_prev[pp_rank_key] = eid

            # Index for cross-rank dependency matching
            key = (event_type, mb_index, stage_index, pp_rank_key)
            event_index[key] = ev

            logical_clock += 1

    # ── Cross-rank PP communication dependencies ─────────────────────
    # In pipeline_order_with_comms:
    #   SEND_F(stage=s, mb=m) on the rank that owns stage s
    #   RECV_F(stage=s+1, mb=m) on the rank that owns stage s+1  (or same rank
    #     if both stages are on the same rank, e.g. Interleaved)
    # Similarly for SEND_B/RECV_B but in the backward direction.
    #
    # We match SEND_F → RECV_F by (microbatch, next_stage) where
    # next_stage = stage_of_SEND + 1 for the same microbatch.

    # Build lookup: (event_type, microbatch, stage) → (pp_rank, event)
    send_recv_lookup: dict[tuple[str, int | None, int], tuple[int, ScheduleEvent]] = {}
    for key, ev in list(event_index.items()):
        etype, mb, stage, pp_r = key
        if etype in ("pp_send_activation", "pp_recv_activation",
                     "pp_send_gradient", "pp_recv_gradient"):
            send_recv_lookup[(etype, mb, stage)] = (pp_r, ev)

    # SEND_F(stage=s) → RECV_F(stage=s+1) for same microbatch
    for (etype, mb, stage), (pp_r, ev) in list(send_recv_lookup.items()):
        if etype == "pp_send_activation":
            recv_stage = stage + 1
            recv_key = ("pp_recv_activation", mb, recv_stage)
            if recv_key in send_recv_lookup:
                recv_pp_r, recv_ev = send_recv_lookup[recv_key]
                if recv_pp_r != pp_r:
                    ts.add_dep(
                        ScheduleDep(ev.event_id, recv_ev.event_id, "pp_comm")
                    )

    # SEND_B → RECV_B: backward gradient flows from later to earlier stage.
    # SEND_B(stage=s) sends gradient; RECV_B(stage=s-1) receives it.
    for (etype, mb, stage), (pp_r, ev) in list(send_recv_lookup.items()):
        if etype == "pp_send_gradient":
            recv_stage = stage - 1
            recv_key = ("pp_recv_gradient", mb, recv_stage)
            if recv_key in send_recv_lookup:
                recv_pp_r, recv_ev = send_recv_lookup[recv_key]
                if recv_pp_r != pp_r:
                    ts.add_dep(
                        ScheduleDep(ev.event_id, recv_ev.event_id, "pp_comm")
                    )

    # ── Replicate PP-group events across TP and DP ranks ─────────────
    # PP events are generated per PP rank.  Replicate them to sibling
    # ranks in the same PP group so the swimlane shows balanced work.

    group_size = tp_degree * dp_degree
    if group_size > 1:
        original_events = [
            e
            for e in ts.events
            if e.metadata.get("strategy") == "pp"
        ]
        original_deps = list(ts.deps)

        eid_remap: dict[str, dict[int, str]] = {}
        for ev in original_events:
            base_rank = ev.rank
            pp_group_base = (base_rank // group_size) * group_size
            for r_offset in range(1, group_size):
                r = pp_group_base + r_offset
                new_eid = _next_id(ev.event_type)
                new_ev = ScheduleEvent(
                    event_id=new_eid,
                    event_type=ev.event_type,
                    rank=r,
                    pp_rank=ev.pp_rank,
                    pp_stage=ev.pp_stage,
                    microbatch_idx=ev.microbatch_idx,
                    logical_clock=ev.logical_clock,
                    metadata=dict(ev.metadata),
                )
                ts.add_event(new_ev)

                if ev.event_id not in eid_remap:
                    eid_remap[ev.event_id] = {}
                eid_remap[ev.event_id][r] = new_eid

                if r in per_rank_prev:
                    ts.add_dep(ScheduleDep(per_rank_prev[r], new_eid, "control"))
                per_rank_prev[r] = new_eid

        for dep in original_deps:
            from_eid = dep.from_event_id
            to_eid = dep.to_event_id
            remap_from = eid_remap.get(from_eid, {})
            remap_to = eid_remap.get(to_eid, {})
            for r, to_copy in remap_to.items():
                from_copy = remap_from.get(r)
                if from_copy:
                    ts.add_dep(ScheduleDep(from_copy, to_copy, dep.dep_type))

    # ── Add optimizer step per rank ───────────────────────────────────
    # After REDUCE_GRAD (or last backward if no REDUCE_GRAD), add optimizer step.
    for rank in range(total_ranks):
        pp_rank_for_rank = rank // (tp_degree * dp_degree)
        # Find last event for this rank
        rank_events = [e for e in ts.events if e.rank == rank]
        if rank_events:
            last_eid = rank_events[-1].event_id
            opt_eid = _next_id("optimizer_step")
            ts.add_event(
                ScheduleEvent(
                    event_id=opt_eid,
                    event_type="optimizer_step",
                    rank=rank,
                    pp_rank=pp_rank_for_rank,
                    logical_clock=rank_events[-1].logical_clock + 1,
                    metadata={"strategy": "optimizer", "step": 0},
                )
            )
            ts.add_dep(ScheduleDep(last_eid, opt_eid, "control"))

    return ts