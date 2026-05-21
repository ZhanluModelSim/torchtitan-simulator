# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Runtime capture: combines all interceptors (dispatch, comm, FSDP, PP) into a
single context manager for recording a live training step.

Usage::

    rec = RuntimeCapture(rank=0)
    with rec.activate(model_parts, pp_schedule=pp_sched):
        # Run one forward+backward step
        output = model(inputs)
        loss.backward()
    result = rec.build_result()
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn as nn

from .comm_interceptor import CommRecorder, capture_comms
from .dispatch_interceptor import OpRecorder, capture_ops
from .fsdp_tracer import FSDPEventRecorder, capture_fsdp_events
from .graph_assembler import GraphAssembler
from .memory_estimator import (
    estimate_comm_memory,
    estimate_graph_memory,
    merge_memory_summary,
)
from .nodes import (
    ScheduleDep,
    ScheduleEvent,
    SimulationResult,
    TrainingSchedule,
)


class RuntimeCapture:
    """
    Aggregate context manager that activates all interceptors simultaneously
    and provides a single ``build_result()`` call to assemble the collected
    data into a :class:`SimulationResult`.

    Attributes:
        rank: Process rank.
        op_recorder: Captures every dispatched tensor op.
        comm_recorder: Captures every distributed collective / P2P op.
        fsdp_recorder: Captures FSDP allgather / reduce-scatter lifecycle.
    """

    def __init__(self, rank: int = 0) -> None:
        self.rank = rank
        self.op_recorder = OpRecorder()
        self.comm_recorder = CommRecorder(rank=rank)
        self.fsdp_recorder = FSDPEventRecorder(rank=rank)
        self._pp_events: list[dict[str, Any]] = []
        self._pp_deps: list[dict[str, Any]] = []
        self._active = False

    # ------------------------------------------------------------------
    # Phase control helpers (call between forward / backward / optimizer)
    # ------------------------------------------------------------------

    def set_phase(self, phase: str) -> None:
        """Update the current phase label on all recorders."""
        self.op_recorder.current_phase = phase
        self.comm_recorder.current_phase = phase
        self.fsdp_recorder.current_phase = phase

    def set_pp_stage(self, stage: int | None) -> None:
        self.op_recorder.current_pp_stage = stage
        self.comm_recorder.current_pp_stage = stage

    def set_microbatch(self, mb: int | None) -> None:
        self.op_recorder.current_microbatch = mb
        self.comm_recorder.current_microbatch = mb

    # ------------------------------------------------------------------
    # PP schedule hooks (optional)
    # ------------------------------------------------------------------

    def attach_pp_hooks(self, pp_schedule: Any, pp_stages: list[Any]) -> list[Any]:
        """
        Attach lightweight hooks to *pp_schedule* and each stage in
        *pp_stages* that record PP events into ``self._pp_events``.

        Returns a list of (object, attr_name, original_value) triples that
        ``detach_pp_hooks`` uses to restore originals.
        """
        saved: list[tuple[Any, str, Any]] = []
        counter = [0]

        def _next_id() -> str:
            counter[0] += 1
            return f"pp_rt_{counter[0]:07d}"

        def _make_fwd_wrapper(stage_idx: int, orig_fwd: Any) -> Any:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                mb = self.op_recorder.current_microbatch
                ev_id = _next_id()
                prev_stage = self.op_recorder.current_pp_stage
                self.set_pp_stage(stage_idx)
                self._pp_events.append(
                    {
                        "event_id": ev_id,
                        "event_type": "fwd_start",
                        "rank": self.rank,
                        "pp_stage": stage_idx,
                        "microbatch": mb,
                        "logical_clock": len(self._pp_events),
                    }
                )
                try:
                    result = orig_fwd(*args, **kwargs)
                finally:
                    self.set_pp_stage(prev_stage)
                end_id = _next_id()
                self._pp_events.append(
                    {
                        "event_id": end_id,
                        "event_type": "fwd_end",
                        "rank": self.rank,
                        "pp_stage": stage_idx,
                        "microbatch": mb,
                        "logical_clock": len(self._pp_events),
                    }
                )
                self._pp_deps.append({"from": ev_id, "to": end_id, "type": "control"})
                return result

            return wrapped

        def _make_bwd_wrapper(stage_idx: int, orig_bwd: Any) -> Any:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                mb = self.op_recorder.current_microbatch
                ev_id = _next_id()
                prev_stage = self.op_recorder.current_pp_stage
                self.set_pp_stage(stage_idx)
                self._pp_events.append(
                    {
                        "event_id": ev_id,
                        "event_type": "bwd_start",
                        "rank": self.rank,
                        "pp_stage": stage_idx,
                        "microbatch": mb,
                        "logical_clock": len(self._pp_events),
                    }
                )
                try:
                    result = orig_bwd(*args, **kwargs)
                finally:
                    self.set_pp_stage(prev_stage)
                end_id = _next_id()
                self._pp_events.append(
                    {
                        "event_id": end_id,
                        "event_type": "bwd_end",
                        "rank": self.rank,
                        "pp_stage": stage_idx,
                        "microbatch": mb,
                        "logical_clock": len(self._pp_events),
                    }
                )
                self._pp_deps.append({"from": ev_id, "to": end_id, "type": "control"})
                return result

            return wrapped

        for idx, stage in enumerate(pp_stages):
            idx = getattr(stage, "stage_index", idx)
            orig_fwd = stage.forward
            stage.forward = _make_fwd_wrapper(idx, orig_fwd)
            saved.append((stage, "forward", orig_fwd))

            bwd_attr = "_backward_one_chunk"
            if hasattr(stage, bwd_attr):
                orig_bwd = getattr(stage, bwd_attr)
                setattr(stage, bwd_attr, _make_bwd_wrapper(idx, orig_bwd))
                saved.append((stage, bwd_attr, orig_bwd))

        if hasattr(pp_schedule, "step"):
            orig_step = pp_schedule.step

            def wrapped_step(*args: Any, **kwargs: Any) -> Any:
                start_id = _next_id()
                self._pp_events.append(
                    {
                        "event_id": start_id,
                        "event_type": "pp_step_start",
                        "rank": self.rank,
                        "pp_stage": None,
                        "microbatch": self.op_recorder.current_microbatch,
                        "logical_clock": len(self._pp_events),
                    }
                )
                result = orig_step(*args, **kwargs)
                end_id = _next_id()
                self._pp_events.append(
                    {
                        "event_id": end_id,
                        "event_type": "pp_step_end",
                        "rank": self.rank,
                        "pp_stage": None,
                        "microbatch": self.op_recorder.current_microbatch,
                        "logical_clock": len(self._pp_events),
                    }
                )
                self._pp_deps.append({"from": start_id, "to": end_id, "type": "control"})
                return result

            pp_schedule.step = wrapped_step
            saved.append((pp_schedule, "step", orig_step))

        return saved

    @staticmethod
    def detach_pp_hooks(saved: list[tuple[Any, str, Any]]) -> None:
        for obj, attr, orig in saved:
            setattr(obj, attr, orig)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def activate(
        self,
        model_parts: list[nn.Module],
        phase: str = "forward",
        pp_schedule: Any = None,
        pp_stages: list[Any] | None = None,
    ):
        """
        Activate all interceptors for the duration of the block.

        Args:
            model_parts: Model modules to attach FSDP hooks to.
            phase: Initial phase (``"forward"``).
            pp_schedule: Optional PP schedule to hook.
            pp_stages: Optional list of PipelineStage instances to hook.

        Yields:
            ``self`` for convenience.
        """
        self.set_phase(phase)
        pp_hook_saved: list[Any] = []
        if pp_schedule is not None and pp_stages:
            pp_hook_saved = self.attach_pp_hooks(pp_schedule, pp_stages or [])

        with contextlib.ExitStack() as stack:
            stack.enter_context(capture_ops(self.op_recorder, phase=phase))
            stack.enter_context(capture_comms(self.comm_recorder))
            for m in model_parts:
                stack.enter_context(capture_fsdp_events(m, self.fsdp_recorder))
            yield self

        if pp_hook_saved:
            self.detach_pp_hooks(pp_hook_saved)

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------

    def build_result(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> SimulationResult:
        """
        Assemble all captured data into a :class:`SimulationResult`.

        The :class:`ComputeGraph` is built from ``op_recorder.nodes``.
        The :class:`TrainingSchedule` is built from FSDP + PP events.
        """
        graph = GraphAssembler.from_runtime(
            self.op_recorder.nodes,
            edges=self.op_recorder.edges,
            metadata={
                "rank": self.rank,
                "capture": "runtime",
                **(metadata or {}),
            },
        )
        GraphAssembler.merge_comm_events(graph, self.comm_recorder.events)

        # Build schedule from FSDP + PP events
        schedule = TrainingSchedule(
            metadata={
                "rank": self.rank,
                "has_fsdp": bool(self.fsdp_recorder.events),
                "has_pp": bool(self._pp_events),
            }
        )

        # Add FSDP events
        for ev in self.fsdp_recorder.events:
            schedule.add_event(
                ScheduleEvent(
                    event_id=ev["event_id"],
                    event_type=ev["event_type"],
                    rank=self.rank,
                    logical_clock=ev["logical_clock"],
                    metadata=ev.get("metadata", {}),
                )
            )

        # Add PP events
        for ev in self._pp_events:
            schedule.add_event(
                ScheduleEvent(
                    event_id=ev["event_id"],
                    event_type=ev["event_type"],
                    rank=ev.get("rank", self.rank),
                    pp_stage=ev.get("pp_stage"),
                    microbatch_idx=ev.get("microbatch"),
                    logical_clock=ev.get("logical_clock", 0),
                )
            )

        for dep in self._pp_deps:
            schedule.add_dep(ScheduleDep(dep["from"], dep["to"], dep["type"]))

        graph_memory_events, graph_memory_summary = estimate_graph_memory(graph)
        comm_memory_events = estimate_comm_memory(self.comm_recorder.events)
        comm_memory_summary = {
            **merge_memory_summary(
                graph_memory_summary,
                {
                    "total_event_bytes": sum(e.bytes for e in comm_memory_events),
                    "by_category": {
                        "comm_event_buffer": sum(e.bytes for e in comm_memory_events)
                    },
                },
            ),
            "graph_peak_live_bytes": graph_memory_summary.get("peak_live_bytes", 0),
        }
        metadata = metadata or {}
        metadata["memory"] = merge_memory_summary(
            metadata.get("memory", {}),
            comm_memory_summary,
        )

        return SimulationResult(
            compute_graph=graph,
            schedule=schedule,
            comm_events=list(self.comm_recorder.events),
            fsdp_events=list(self.fsdp_recorder.events),
            pp_events=list(self._pp_events),
            memory_events=[*graph_memory_events, *comm_memory_events],
            metadata=metadata,
        )
