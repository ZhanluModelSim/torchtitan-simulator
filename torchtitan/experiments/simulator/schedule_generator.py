# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generate semantic PP / TP / DP / FSDP2 training schedules.

When the simulator runs on a single CPU process (e.g. ``fake_backend``),
no real multi-rank parallelism is active.  This module creates a
:class:`TrainingSchedule` that mirrors what an Interleaved1F1B run with
the configured degrees would look like across all ranks, so the HTML
visualisation shows the full parallel topology.
"""

from __future__ import annotations

from typing import Any

from .nodes import (
    ScheduleDep,
    ScheduleEvent,
    TrainingSchedule,
)


def _unique_id(prefix: str, counter: list[int]) -> str:
    counter[0] += 1
    return f"{prefix}_{counter[0]:06d}"


def generate_interleaved_1f1b_schedule(
    *,
    pp_degree: int,
    tp_degree: int,
    dp_shard_degree: int,
    dp_replicate_degree: int = 1,
    num_microbatches: int = 8,
    num_steps: int = 1,
    virtual_stages_per_rank: int = 2,
) -> TrainingSchedule:
    """Generate a semantic Interleaved1F1B-style training schedule.

    The generated schedule covers all ranks in the pp × tp × dp topology
    with explicit dependencies for PP send/recv, TP all-reduce, FSDP2
    all-gather / reduce-scatter, and DP gradient synchronisation.

    Args:
        pp_degree: Number of pipeline-parallel ranks.
        tp_degree: Tensor-parallel degree.
        dp_shard_degree: FSDP / DP-shard degree.
        dp_replicate_degree: DP-replicate (HSDP) degree.
        num_microbatches: Microbatches per training step.
        num_steps: Number of training steps.
        virtual_stages_per_rank: Virtual pipeline stages per rank
            (for Interleaved1F1B, typically >= 2).

    Returns:
        A populated :class:`TrainingSchedule`.
    """
    total_stages = pp_degree * virtual_stages_per_rank
    dp_degree = dp_shard_degree * dp_replicate_degree
    total_ranks = pp_degree * tp_degree * dp_degree

    # Ranks in each TP group share the same PP rank in the pipeline view.
    def _tp_group(pp_idx: int, dp_idx: int) -> list[int]:
        base = pp_idx * tp_degree * dp_degree + dp_idx * tp_degree
        return [base + t for t in range(tp_degree)]

    # All DP ranks for a given PP rank share parameters / FSDP groups.
    def _dp_group(pp_idx: int) -> list[int]:
        ranks: list[int] = []
        for d in range(dp_degree):
            ranks.extend(_tp_group(pp_idx, d))
        return ranks

    schedule = TrainingSchedule(
        metadata={
            "pp_degree": pp_degree,
            "tp_degree": tp_degree,
            "dp_shard_degree": dp_shard_degree,
            "dp_replicate_degree": dp_replicate_degree,
            "virtual_stages_per_rank": virtual_stages_per_rank,
            "num_microbatches": num_microbatches,
            "num_steps": num_steps,
            "total_ranks": total_ranks,
            "schedule_type": "Interleaved1F1B_semantic",
        }
    )

    counter = [0]
    nid = lambda prefix: _unique_id(prefix, counter)

    # Per-rank clock for implicit same-rank ordering.
    rank_clock: dict[int, int] = {}
    prev_per_rank: dict[int, str] = {}

    def _add(
        rank: int,
        event_type: str,
        strategy: str,
        *,
        pp_rank: int | None = None,
        pp_stage: int | None = None,
        mb: int | None = None,
        step: int | None = None,
        deps: list[tuple[str, str]] | None = None,
    ) -> str:
        eid = nid(event_type)
        ev = ScheduleEvent(
            event_id=eid,
            event_type=event_type,
            rank=rank,
            pp_rank=pp_rank,
            pp_stage=pp_stage,
            microbatch_idx=mb,
            logical_clock=rank_clock.get(rank, 0),
            metadata={
                "strategy": strategy,
                "step": step,
                "mb": mb,
            },
        )
        schedule.add_event(ev)
        rank_clock[rank] = rank_clock.get(rank, 0) + 1

        if rank in prev_per_rank:
            schedule.add_dep(
                ScheduleDep(prev_per_rank[rank], eid, "control")
            )
        prev_per_rank[rank] = eid

        for dep_id, dep_type in (deps or []):
            schedule.add_dep(ScheduleDep(dep_id, eid, dep_type))
        return eid

    # ------------------------------------------------------------------
    # Helper: a single forward/backward pair for one PP-rank stage+mb
    # ------------------------------------------------------------------

    # Stage → rank mapping (which pp_rank owns which virtual stages).
    stage_to_pp_rank = {s: s // virtual_stages_per_rank for s in range(total_stages)}

    def _forward_pass(
        step: int,
        pp_rank: int,
        stage: int,
        mb: int,
        *,
        fsdp_dep: str | None = None,
        pp_recv_dep: str | None = None,
    ) -> str:
        """Emit FSDP all-gather → TP all-reduce → PP forward → PP send."""
        rank = pp_rank * tp_degree * dp_degree
        deps: list[tuple[str, str]] = []
        if fsdp_dep is not None:
            deps.append((fsdp_dep, "fsdp_comm"))
        if pp_recv_dep is not None:
            deps.append((pp_recv_dep, "pp_comm"))

        # FSDP2 all-gather (one event per DP group, pinned to rank 0 of group)
        ag_eid = _add(
            rank, "fsdp2_all_gather", "fsdp2",
            pp_rank=pp_rank, pp_stage=stage, mb=mb, step=step,
            deps=deps,
        )
        # TP all-reduce activation
        tp_eid = _add(
            rank, "tp_all_reduce", "tp",
            pp_rank=pp_rank, pp_stage=stage, mb=mb, step=step,
            deps=[(ag_eid, "control")],
        )
        # PP forward compute
        fwd_eid = _add(
            rank, "pp_forward", "pp",
            pp_rank=pp_rank, pp_stage=stage, mb=mb, step=step,
            deps=[(tp_eid, "control")],
        )
        return fwd_eid

    def _send_activation(
        step: int, pp_rank: int, stage: int, mb: int, fwd_dep: str,
    ) -> str:
        rank = pp_rank * tp_degree * dp_degree
        return _add(
            rank, "pp_send_activation", "pp",
            pp_rank=pp_rank, pp_stage=stage, mb=mb, step=step,
            deps=[(fwd_dep, "pp_comm")],
        )

    def _backward_pass(
        step: int,
        pp_rank: int,
        stage: int,
        mb: int,
        *,
        bwd_trigger_dep: str | None = None,
        pp_recv_grad_dep: str | None = None,
    ) -> str:
        rank = pp_rank * tp_degree * dp_degree
        deps: list[tuple[str, str]] = []
        if bwd_trigger_dep is not None:
            deps.append((bwd_trigger_dep, "control"))
        if pp_recv_grad_dep is not None:
            deps.append((pp_recv_grad_dep, "pp_comm"))

        bwd_eid = _add(
            rank, "pp_backward", "pp",
            pp_rank=pp_rank, pp_stage=stage, mb=mb, step=step,
            deps=deps,
        )
        rs_eid = _add(
            rank, "fsdp2_reduce_scatter", "fsdp2",
            pp_rank=pp_rank, pp_stage=stage, mb=mb, step=step,
            deps=[(bwd_eid, "fsdp_comm")],
        )
        return rs_eid

    def _send_gradient(
        step: int, pp_rank: int, stage: int, mb: int, rs_dep: str,
    ) -> str:
        rank = pp_rank * tp_degree * dp_degree
        return _add(
            rank, "pp_send_gradient", "pp",
            pp_rank=pp_rank, pp_stage=stage, mb=mb, step=step,
            deps=[(rs_dep, "pp_comm")],
        )

    # ------------------------------------------------------------------
    # Interleaved1F1B over all steps and microbatches
    # ------------------------------------------------------------------

    for step in range(num_steps):
        # Per-(stage, mb) tracking: (fwd_eid, bwd_trigger)
        fwd_done: dict[tuple[int, int], str] = {}   # (stage, mb) → fwd_event_id
        fwd_sent: dict[tuple[int, int], str] = {}    # (stage, mb) → send_event_id
        bwd_done: dict[tuple[int, int], str] = {}    # (stage, mb) → rs_event_id
        bwd_grad_sent: dict[tuple[int, int], str] = {}  # (stage, mb) → send_grad_event_id

        # --- Warmup: pipeline fill ---
        # Each pipeline "clock" step introduces a new microbatch and shifts
        # existing ones forward.  Stage 's' at clock 'clock' processes
        # microbatch 'clock - s' (if it hasn't reached the end yet).
        for clock in range(total_stages + num_microbatches - 1):
            microbatch = clock
            for s in range(min(total_stages, clock + 1)):
                stage = s
                mb = clock - s
                if mb >= num_microbatches:
                    continue
                if (stage, mb) in fwd_done:
                    continue
                pp_rank = stage_to_pp_rank[stage]

                pp_recv_dep = None
                if stage > 0:
                    pp_recv_dep = fwd_sent.get((stage - 1, mb))

                fwd_eid = _forward_pass(
                    step, pp_rank, stage, mb,
                    pp_recv_dep=pp_recv_dep,
                )
                send_eid = _send_activation(step, pp_rank, stage, mb, fwd_eid)
                fwd_done[(stage, mb)] = fwd_eid
                fwd_sent[(stage, mb)] = send_eid

                # Once a microbatch reaches the last stage, trigger backward
                if stage == total_stages - 1:
                    loss_eid = _add(
                        pp_rank * tp_degree * dp_degree, "loss_compute", "compute",
                        pp_rank=pp_rank, pp_stage=stage, mb=mb, step=step,
                    )
                    rs_eid = _backward_pass(
                        step, pp_rank, stage, mb,
                        bwd_trigger_dep=loss_eid,
                    )
                    bwd_done[(stage, mb)] = rs_eid
                    send_grad_eid = _send_gradient(step, pp_rank, stage, mb, rs_eid)
                    bwd_grad_sent[(stage, mb)] = send_grad_eid

                    # Shift backward up the pipeline for this microbatch
                    for bs in range(total_stages - 2, -1, -1):
                        bwd_pp_rank = stage_to_pp_rank[bs]
                        pp_recv_grad = bwd_grad_sent.get((bs + 1, mb))
                        rs_eid = _backward_pass(
                            step, bwd_pp_rank, bs, mb,
                            pp_recv_grad_dep=pp_recv_grad,
                        )
                        bwd_done[(bs, mb)] = rs_eid
                        send_grad = _send_gradient(step, bwd_pp_rank, bs, mb, rs_eid)
                        bwd_grad_sent[(bs, mb)] = send_grad

        # --- Per-step DP gradient sync + optimizer ---
        for rank in range(total_ranks):
            pp_rank = rank // (tp_degree * dp_degree)
            stage = (pp_rank * virtual_stages_per_rank) % total_stages

            dp_sync = _add(
                rank, "dp_gradient_sync", "dp",
                pp_rank=pp_rank, pp_stage=stage, step=step,
            )
            _add(
                rank, "optimizer_step", "optimizer",
                pp_rank=pp_rank, pp_stage=stage, step=step,
                deps=[(dp_sync, "control")],
            )

    # ── Post-hoc: replicate PP-group schedule across TP/DP ranks ─────
    # PP/FSDP/TP events are generated for the base rank of each PP group.
    # We copy them to sibling ranks so the swimlane shows balanced work.
    # dp_gradient_sync and optimizer_step are already emitted per-rank.
    group_size = tp_degree * dp_degree
    original_events = [e for e in schedule.events
                       if e.metadata.get("strategy") in ("pp", "fsdp2", "tp", "compute")]
    original_deps = list(schedule.deps)

    eid_remap: dict[str, dict[int, str]] = {}
    for ev in original_events:
        base_rank = ev.rank
        pp_group_base = (base_rank // group_size) * group_size
        for r in range(pp_group_base, pp_group_base + group_size):
            if r == base_rank:
                continue
            new_eid = nid(ev.event_type)
            new_ev = ScheduleEvent(
                event_id=new_eid,
                event_type=ev.event_type,
                rank=r,
                pp_rank=ev.pp_rank,
                pp_stage=ev.pp_stage,
                microbatch_idx=ev.microbatch_idx,
                logical_clock=rank_clock.get(r, 0),
                metadata=dict(ev.metadata),
            )
            schedule.add_event(new_ev)
            rank_clock[r] = rank_clock.get(r, 0) + 1

            if ev.event_id not in eid_remap:
                eid_remap[ev.event_id] = {}
            eid_remap[ev.event_id][r] = new_eid

            if r in prev_per_rank:
                schedule.add_dep(ScheduleDep(prev_per_rank[r], new_eid, "control"))
            prev_per_rank[r] = new_eid

    for dep in original_deps:
        from_eid = dep.from_event_id
        to_eid = dep.to_event_id
        remap_from = eid_remap.get(from_eid, {})
        remap_to = eid_remap.get(to_eid, {})
        for r, to_copy in remap_to.items():
            from_copy = remap_from.get(r)
            if from_copy:
                schedule.add_dep(ScheduleDep(from_copy, to_copy, dep.dep_type))

    return schedule
