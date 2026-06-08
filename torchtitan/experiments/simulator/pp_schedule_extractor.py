# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Static extraction of the Pipeline Parallel schedule.

This module inspects a :class:`_PipelineSchedule` object to determine the
complete ordering of forward passes, backward passes, send operations, and
receive operations across all ranks and microbatches — without actually
executing any computation.

Primary strategy: read ``pipeline_order`` / ``pipeline_order_with_comms``
from the schedule object (populated during ``__init__``).  Falls back to
heuristic reconstruction for very old PyTorch versions that lack these
attributes.

Supported built-in schedules
-----------------------------
All schedules registered in ``torch.distributed.pipelining.schedules``:
Schedule1F1B, ScheduleGPipe, ScheduleInterleaved1F1B, ScheduleLoopedBFS,
ScheduleInterleavedZeroBubble, ScheduleZBVZeroBubble, ScheduleDualPipeV,
and ``_PipelineScheduleRuntime`` (CSV-driven).

Usage::

    extractor = PPScheduleExtractor(schedule, pp_rank=-1, world_size=4)
    training_schedule = extractor.extract()
    training_schedule.to_dict()  # -> JSON-serializable dict
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

from .nodes import ScheduleDep, ScheduleEvent, TrainingSchedule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_event_id(prefix: str, counter: list[int]) -> str:
    counter[0] += 1
    return f"{prefix}_{counter[0]:07d}"


# Action type constants produced by the schedule internals
_ACTION_FWD = "F"
_ACTION_BWD = "B"
_ACTION_SEND_FWD = "SEND_F"
_ACTION_RECV_FWD = "RECV_F"
_ACTION_SEND_BWD = "SEND_B"
_ACTION_RECV_BWD = "RECV_B"
_ACTION_SEND_ACT = "SEND_ACT"  # activation send (DP-style)
_ACTION_RECV_ACT = "RECV_ACT"


def _action_to_event_type(action: str) -> str:
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
        from .schedule_extract import _convert_pipeline_order_to_training_schedule

        # Primary: read pipeline_order from the schedule (populated at __init__)
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

        # Fallback: heuristic reconstruction (for very old PyTorch versions)
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

        # Strategy 1: _PipelineScheduleRuntime exposes _actions
        if hasattr(schedule, "_actions"):
            actions = schedule._actions
            if actions:
                return self._parse_runtime_actions(actions)

        # Strategy 2: _compute_clock_cycles exists (internal API)
        if hasattr(schedule, "_compute_clock_cycles"):
            try:
                clock_cycles = schedule._compute_clock_cycles()
                return self._parse_clock_cycles(clock_cycles)
            except Exception:
                pass

        # Strategy 3: _step_microbatches is patchable for dry-run
        # (only for single-stage schedules, which store per-rank ops
        #  in a predictable pattern)
        return None

    def _parse_runtime_actions(self, actions: Any) -> dict[int, list[tuple[str, int]]]:
        """
        Parse the actions dict from ``_PipelineScheduleRuntime._actions``.

        The attribute is typically ``list[dict]`` where each dict has keys
        like ``"type"``, ``"microbatch"``, ``"rank"`` etc., or it may be a
        nested ``{rank: [Action]}`` structure, or a list of action objects
        with ``computation_type`` / ``microbatch_index`` / ``stage_index``
        attributes.
        """
        table: dict[int, list[tuple[str, int]]] = {}

        # Handle list-of-dicts or list-of-objects format
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
                    # Action object with stage_index, computation_type, microbatch_index
                    rank = int(getattr(item, "stage_index", 0))
                    action_type, mb = self._unpack_action(item)
                    table.setdefault(rank, []).append((action_type, mb))
                else:
                    action_type, mb = self._unpack_action(item)
                    table.setdefault(0, []).append((action_type, mb))
            return table

        # Handle dict-of-lists / dict-of-dicts
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

        # clock_cycles is typically ``list[list[Action_or_None]]`` indexed by
        # [clock][rank], or it may be a list of per-rank action lists.
        if not clock_cycles:
            return table

        if isinstance(clock_cycles[0], list):
            # [clock_idx][rank_idx] -> action
            for clock_idx, rank_actions in enumerate(clock_cycles):
                for rank_idx, action in enumerate(rank_actions):
                    if action is None:
                        continue
                    action_type, mb = self._unpack_action(action)
                    table.setdefault(rank_idx, []).append((action_type, mb))
        else:
            # Flat list per rank (single-rank schedule)
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
        # Normalise common strings
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
                    event_type=_action_to_event_type(action_type),
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

        This generates a canonical 1F1B ordering:
          warm-up  (stage-index forward passes)
          steady   (alternating 1F1B)
          cool-down (remaining backward passes)
        """
        n_mb = self.n_microbatches
        n_ranks = self.world_size
        counter = [0]

        ranks = list(range(n_ranks)) if self.pp_rank == -1 else [self.pp_rank]

        for rank in ranks:
            clock = 0
            # Warm-up: rank fills its pipeline
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
            # Steady-state 1F1B
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
            # Cool-down
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

        Rules:
          - A ``send_fwd`` on rank R must precede the ``recv_fwd`` on rank R+1
            for the same microbatch.
          - A ``send_bwd`` on rank R+1 must precede the ``recv_bwd`` on rank R
            for the same microbatch.
        """
        # Index events by (event_type, microbatch_idx, rank)
        index: dict[tuple[str, int | None, int], ScheduleEvent] = {}
        for ev in ts.events:
            key = (ev.event_type, ev.microbatch_idx, ev.rank)
            index[key] = ev

        counter = [0]

        def _dep(src: ScheduleEvent, dst: ScheduleEvent) -> None:
            ts.add_dep(
                ScheduleDep(
                    from_event_id=src.event_id,
                    to_event_id=dst.event_id,
                    dep_type="pp_comm",
                )
            )

        # Forward: send_fwd(rank=r) → recv_fwd(rank=r+1)
        for (etype, mb, rank), ev in list(index.items()):
            if etype == "send_fwd":
                recv_key = ("recv_fwd", mb, rank + 1)
                if recv_key in index:
                    _dep(ev, index[recv_key])

        # Backward: send_bwd(rank=r) → recv_bwd(rank=r-1)
        for (etype, mb, rank), ev in list(index.items()):
            if etype == "send_bwd" and rank > 0:
                recv_key = ("recv_bwd", mb, rank - 1)
                if recv_key in index:
                    _dep(ev, index[recv_key])

        # Sequential per-rank: each event depends on the previous for same rank+mb
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
