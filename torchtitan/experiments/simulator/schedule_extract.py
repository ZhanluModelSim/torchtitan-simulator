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
        get_schedule_class,
        PipelineScheduleMulti,
        PipelineScheduleSingle,
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


def _stage_to_pp_rank(stage_index: int, schedule: Any, pp_degree: int) -> int:
    if hasattr(schedule, "stage_index_to_group_rank"):
        mapping = schedule.stage_index_to_group_rank
        if mapping and stage_index in mapping:
            return mapping[stage_index]
    return stage_index // pp_degree if pp_degree > 0 else 0


def replicate_events_to_ranks(
    schedule: TrainingSchedule,
    group_size: int,
    strategies: set[str],
    per_rank_prev: dict[Any, str],
    next_id_fn: Any,
    rank_clock: dict[int, int] | None = None,
) -> None:
    if group_size <= 1:
        return
    original_events = [
        e for e in schedule.events if e.metadata.get("strategy") in strategies
    ]
    original_deps = list(schedule.deps)
    eid_remap: dict[str, dict[int, str]] = {}

    for ev in original_events:
        base_rank = ev.rank
        group_base = (base_rank // group_size) * group_size
        for r_offset in range(1, group_size):
            r = group_base + r_offset
            new_eid = next_id_fn(ev.event_type)
            if rank_clock is not None:
                clock = rank_clock.get(r, 0)
                rank_clock[r] = clock + 1
            else:
                clock = ev.logical_clock
            new_ev = ScheduleEvent(
                event_id=new_eid,
                event_type=ev.event_type,
                rank=r,
                pp_rank=ev.pp_rank,
                pp_stage=ev.pp_stage,
                microbatch_idx=ev.microbatch_idx,
                logical_clock=clock,
                metadata=dict(ev.metadata),
            )
            schedule.add_event(new_ev)
            if ev.event_id not in eid_remap:
                eid_remap[ev.event_id] = {}
            eid_remap[ev.event_id][r_offset] = new_eid
            if r in per_rank_prev:
                schedule.add_dep(ScheduleDep(per_rank_prev[r], new_eid, "control"))
            per_rank_prev[r] = new_eid

    for dep in original_deps:
        remap_from = eid_remap.get(dep.from_event_id, {})
        remap_to = eid_remap.get(dep.to_event_id, {})
        for r_offset, to_copy in remap_to.items():
            from_copy = remap_from.get(r_offset)
            if from_copy:
                schedule.add_dep(ScheduleDep(from_copy, to_copy, dep.dep_type))


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

    per_rank_prev: dict[Any, str] = {}

    for pp_rank_key, action_list in pipeline_order.items():
        logical_clock = 0
        for action in action_list:
            if action is None:
                logical_clock += 1
                continue

            event_type = _action_to_event_type(action)
            stage_index = action.stage_index
            mb_index = action.microbatch_index

            actual_pp_rank = pp_rank_key

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

            pp_prev_key = ("pp", pp_rank_key)
            if pp_prev_key in per_rank_prev:
                ts.add_dep(ScheduleDep(per_rank_prev[pp_prev_key], eid, "control"))
            per_rank_prev[pp_prev_key] = eid

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
        if etype in (
            "pp_send_activation",
            "pp_recv_activation",
            "pp_send_gradient",
            "pp_recv_gradient",
        ):
            send_recv_lookup[(etype, mb, stage)] = (pp_r, ev)

    # SEND_F(stage=s) → RECV_F(stage=s+1) for same microbatch
    for (etype, mb, stage), (pp_r, ev) in list(send_recv_lookup.items()):
        if etype == "pp_send_activation":
            recv_stage = stage + 1
            recv_key = ("pp_recv_activation", mb, recv_stage)
            if recv_key in send_recv_lookup:
                recv_pp_r, recv_ev = send_recv_lookup[recv_key]
                if recv_pp_r != pp_r:
                    ts.add_dep(ScheduleDep(ev.event_id, recv_ev.event_id, "pp_comm"))

    # SEND_B → RECV_B: backward gradient flows from later to earlier stage.
    # SEND_B(stage=s) sends gradient; RECV_B(stage=s-1) receives it.
    for (etype, mb, stage), (pp_r, ev) in list(send_recv_lookup.items()):
        if etype == "pp_send_gradient":
            recv_stage = stage - 1
            recv_key = ("pp_recv_gradient", mb, recv_stage)
            if recv_key in send_recv_lookup:
                recv_pp_r, recv_ev = send_recv_lookup[recv_key]
                if recv_pp_r != pp_r:
                    ts.add_dep(ScheduleDep(ev.event_id, recv_ev.event_id, "pp_comm"))

    group_size = tp_degree * dp_degree
    if group_size > 1:
        replicate_events_to_ranks(
            ts,
            group_size,
            {"pp"},
            per_rank_prev,
            lambda prefix: _next_id(prefix),
        )

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


# ---------------------------------------------------------------------------
# PPScheduleExtractor helpers
# ---------------------------------------------------------------------------


def _make_event_id(prefix: str, counter: list[int]) -> str:
    counter[0] += 1
    return f"{prefix}_{counter[0]:07d}"


_ACTION_FWD = "F"
_ACTION_BWD = "B"
_ACTION_SEND_FWD = "SEND_F"
_ACTION_RECV_FWD = "RECV_F"
_ACTION_SEND_BWD = "SEND_B"
_ACTION_RECV_BWD = "RECV_B"
_ACTION_SEND_ACT = "SEND_ACT"
_ACTION_RECV_ACT = "RECV_ACT"


def _pp_action_str_to_event_type(action: str) -> str:
    mapping = {
        _ACTION_FWD: "fwd",
        _ACTION_BWD: "bwd",
        _ACTION_SEND_FWD: "send_fwd",
        _ACTION_RECV_FWD: "recv_fwd",
        _ACTION_SEND_BWD: "send_bwd",
        _ACTION_RECV_BWD: "recv_bwd",
        _ACTION_SEND_ACT: "send_activation",
        _ACTION_RECV_ACT: "recv_activation",
    }
    return mapping.get(action, action.lower())


# ---------------------------------------------------------------------------
# PPScheduleExtractor
# ---------------------------------------------------------------------------


class PPScheduleExtractor:
    """
    Extracts the static schedule from a ``_PipelineSchedule`` instance.

    The extracted :class:`TrainingSchedule` contains one
    :class:`ScheduleEvent` per (rank, clock-cycle, action) triple, plus
    causal dependency edges between matching send/recv pairs.

    Args:
        schedule: An instantiated pipeline schedule.
        pp_rank: The rank for which to extract the schedule.
            Pass -1 (default) to extract for *all* ranks.
        world_size: Number of PP ranks (used when extracting all ranks).
        n_microbatches: Number of microbatches.  If ``None``, read from
            ``schedule.n_microbatches``.
    """

    def __init__(
        self,
        schedule: Any,
        pp_rank: int = -1,
        world_size: int = 1,
        n_microbatches: int | None = None,
    ) -> None:
        self.schedule = schedule
        self.pp_rank = pp_rank
        self.world_size = world_size
        self.n_microbatches = n_microbatches or getattr(
            schedule, "n_microbatches", getattr(schedule, "_n_microbatches", 1)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> TrainingSchedule:
        """Return a :class:`TrainingSchedule` describing the full pipeline schedule.

        Primary path: read ``pipeline_order_with_comms`` (multi-stage) or
        ``pipeline_order`` (single-stage) from the schedule object, then
        convert to TrainingSchedule via the shared conversion function.
        Falls back to heuristic reconstruction if neither attribute exists.
        """
        pipeline_order = None
        if hasattr(self.schedule, "pipeline_order_with_comms"):
            pipeline_order = self.schedule.pipeline_order_with_comms
        elif hasattr(self.schedule, "pipeline_order"):
            pipeline_order = self.schedule.pipeline_order

        if pipeline_order is not None:
            pp_group_size = getattr(self.schedule, "pp_group_size", self.world_size)
            return _convert_pipeline_order_to_training_schedule(
                pipeline_order,
                schedule=self.schedule,
                pp_degree=pp_group_size,
                tp_degree=1,
                dp_degree=1,
                n_microbatches=self.n_microbatches,
                schedule_name=type(self.schedule).__name__,
                virtual_stages_per_rank=1,
            )

        logger.warning(
            "No pipeline_order found on schedule %s; using heuristic 1F1B",
            type(self.schedule).__name__,
        )
        metadata = {
            "schedule_type": type(self.schedule).__name__,
            "n_microbatches": self.n_microbatches,
            "pp_rank": self.pp_rank,
            "world_size": self.world_size,
            "extraction_method": "heuristic_fallback",
        }
        ts = TrainingSchedule(metadata=metadata)
        self._build_schedule_heuristic(ts)
        self._add_send_recv_deps(ts)
        return ts

    # ------------------------------------------------------------------
    # Action-table extraction strategies
    # ------------------------------------------------------------------

    def _extract_action_table(self) -> dict[int, list[tuple[str, int]]] | None:
        """
        Try to read the internal action table from the schedule.

        Returns a dict mapping ``rank -> list[(action_type, microbatch_id)]``
        covering one full pipeline step, or ``None`` if unavailable.
        """
        schedule = self.schedule

        if hasattr(schedule, "_actions"):
            actions = schedule._actions
            if actions:
                return self._parse_runtime_actions(actions)

        if hasattr(schedule, "_compute_clock_cycles"):
            try:
                clock_cycles = schedule._compute_clock_cycles()
                return self._parse_clock_cycles(clock_cycles)
            except Exception:
                pass

        return None

    def _parse_runtime_actions(self, actions: Any) -> dict[int, list[tuple[str, int]]]:
        """
        Parse the actions dict from ``_PipelineScheduleRuntime._actions``.
        """
        table: dict[int, list[tuple[str, int]]] = {}

        if isinstance(actions, list):
            for item in actions:
                if isinstance(item, dict):
                    rank = item.get("rank", 0)
                    action = item.get("type", item.get("action", "F"))
                    mb = item.get("microbatch", item.get("mb", 0))
                    table.setdefault(rank, []).append((str(action).upper(), int(mb)))
                elif hasattr(item, "computation_type") and hasattr(
                    item, "microbatch_index"
                ):
                    rank = int(getattr(item, "stage_index", 0))
                    action_type, mb = self._unpack_action(item)
                    table.setdefault(rank, []).append((action_type, mb))
                else:
                    action_type, mb = self._unpack_action(item)
                    table.setdefault(0, []).append((action_type, mb))
            return table

        if isinstance(actions, dict):
            for rank, rank_actions in actions.items():
                rank = int(rank)
                parsed: list[tuple[str, int]] = []
                for act in rank_actions:
                    if isinstance(act, dict):
                        parsed.append(
                            (
                                str(act.get("type", "F")).upper(),
                                int(act.get("microbatch", 0)),
                            )
                        )
                    elif hasattr(act, "type") and hasattr(act, "microbatch"):
                        parsed.append((str(act.type).upper(), int(act.microbatch)))
                    else:
                        action_type, mb = self._unpack_action(act)
                        parsed.append((action_type, mb))
                table[rank] = parsed
            return table

        return {}

    def _parse_clock_cycles(
        self, clock_cycles: Any
    ) -> dict[int, list[tuple[str, int]]]:
        """Parse output of ``_compute_clock_cycles()``."""
        table: dict[int, list[tuple[str, int]]] = {}

        if not clock_cycles:
            return table

        if isinstance(clock_cycles[0], list):
            for clock_idx, rank_actions in enumerate(clock_cycles):
                for rank_idx, action in enumerate(rank_actions):
                    if action is None:
                        continue
                    action_type, mb = self._unpack_action(action)
                    table.setdefault(rank_idx, []).append((action_type, mb))
        else:
            for idx, action in enumerate(clock_cycles):
                if action is None:
                    continue
                action_type, mb = self._unpack_action(action)
                table.setdefault(0, []).append((action_type, mb))

        return table

    @staticmethod
    def _unpack_action(action: Any) -> tuple[str, int]:
        """Extract (action_type_str, microbatch_id) from an action object."""
        if hasattr(action, "computation_type") and hasattr(action, "microbatch_index"):
            t = str(action.computation_type).upper()
            mb = int(action.microbatch_index)
        elif hasattr(action, "type") and hasattr(action, "microbatch"):
            t = str(action.type).upper()
            mb = int(action.microbatch)
        elif isinstance(action, tuple) and len(action) >= 2:
            t, mb = str(action[0]).upper(), int(action[1])
        elif isinstance(action, str):
            t, mb = action.upper(), 0
        else:
            t, mb = str(action).upper(), 0
        if "FORWARD" in t or t == "F":
            t = "F"
        elif "BACKWARD" in t or t == "B":
            t = "B"
        elif "SEND" in t and "FWD" in t:
            t = "SEND_F"
        elif "RECV" in t and "FWD" in t:
            t = "RECV_F"
        elif "SEND" in t and "BWD" in t:
            t = "SEND_B"
        elif "RECV" in t and "BWD" in t:
            t = "RECV_B"
        return t, mb

    # ------------------------------------------------------------------
    # Schedule construction from action table
    # ------------------------------------------------------------------

    def _build_schedule_from_table(
        self,
        ts: TrainingSchedule,
        action_table: dict[int, list[tuple[str, int]]],
    ) -> None:
        counter = [0]

        ranks = list(action_table.keys()) if self.pp_rank == -1 else [self.pp_rank]

        for rank in ranks:
            actions = action_table.get(rank, [])
            for clock, (action_type, mb) in enumerate(actions):
                event = ScheduleEvent(
                    event_id=_make_event_id("pp", counter),
                    event_type=_pp_action_str_to_event_type(action_type),
                    rank=rank,
                    pp_rank=rank,
                    microbatch_idx=mb,
                    logical_clock=clock,
                    metadata={"schedule_action": action_type},
                )
                ts.add_event(event)

    def _build_schedule_heuristic(self, ts: TrainingSchedule) -> None:
        """
        Reconstruct a 1F1B-style schedule heuristically when the schedule
        does not expose an action table.
        """
        n_mb = self.n_microbatches
        n_ranks = self.world_size
        counter = [0]

        ranks = list(range(n_ranks)) if self.pp_rank == -1 else [self.pp_rank]

        for rank in ranks:
            clock = 0
            for mb in range(min(rank + 1, n_mb)):
                ts.add_event(
                    ScheduleEvent(
                        event_id=_make_event_id("pp", counter),
                        event_type="fwd",
                        rank=rank,
                        pp_rank=rank,
                        microbatch_idx=mb,
                        logical_clock=clock,
                    )
                )
                clock += 1
            fwd_mb = rank + 1
            bwd_mb = 0
            while fwd_mb < n_mb or bwd_mb < rank + 1:
                if fwd_mb < n_mb:
                    ts.add_event(
                        ScheduleEvent(
                            event_id=_make_event_id("pp", counter),
                            event_type="fwd",
                            rank=rank,
                            pp_rank=rank,
                            microbatch_idx=fwd_mb,
                            logical_clock=clock,
                        )
                    )
                    fwd_mb += 1
                    clock += 1
                if bwd_mb < rank + 1:
                    ts.add_event(
                        ScheduleEvent(
                            event_id=_make_event_id("pp", counter),
                            event_type="bwd",
                            rank=rank,
                            pp_rank=rank,
                            microbatch_idx=bwd_mb,
                            logical_clock=clock,
                        )
                    )
                    bwd_mb += 1
                    clock += 1
            while bwd_mb < n_mb:
                ts.add_event(
                    ScheduleEvent(
                        event_id=_make_event_id("pp", counter),
                        event_type="bwd",
                        rank=rank,
                        pp_rank=rank,
                        microbatch_idx=bwd_mb,
                        logical_clock=clock,
                    )
                )
                bwd_mb += 1
                clock += 1

    # ------------------------------------------------------------------
    # Dependency construction
    # ------------------------------------------------------------------

    def _add_send_recv_deps(self, ts: TrainingSchedule) -> None:
        """
        Add causal ``pp_comm`` dependency edges between matching send/recv
        event pairs.
        """
        index: dict[tuple[str, int | None, int], ScheduleEvent] = {}
        for ev in ts.events:
            key = (ev.event_type, ev.microbatch_idx, ev.rank)
            index[key] = ev

        def _dep(src: ScheduleEvent, dst: ScheduleEvent) -> None:
            ts.add_dep(
                ScheduleDep(
                    from_event_id=src.event_id,
                    to_event_id=dst.event_id,
                    dep_type="pp_comm",
                )
            )

        for (etype, mb, rank), ev in list(index.items()):
            if etype == "send_fwd":
                recv_key = ("recv_fwd", mb, rank + 1)
                if recv_key in index:
                    _dep(ev, index[recv_key])

        for (etype, mb, rank), ev in list(index.items()):
            if etype == "send_bwd" and rank > 0:
                recv_key = ("recv_bwd", mb, rank - 1)
                if recv_key in index:
                    _dep(ev, index[recv_key])

        rank_clocks: dict[int, list[ScheduleEvent]] = {}
        for ev in ts.events:
            rank_clocks.setdefault(ev.rank, []).append(ev)

        for rank, evs in rank_clocks.items():
            evs_sorted = sorted(evs, key=lambda e: e.logical_clock)
            for prev, curr in zip(evs_sorted, evs_sorted[1:]):
                ts.add_dep(
                    ScheduleDep(
                        from_event_id=prev.event_id,
                        to_event_id=curr.event_id,
                        dep_type="control",
                    )
                )
