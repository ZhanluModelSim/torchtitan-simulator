# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Main ``Simulator`` class: high-level API for CPU-only computation graph and
schedule capture on top of TorchTitan.

Three simulation modes
-----------------------
1. ``simulate_fx(model, inputs)``
   — Static forward (or joint fwd+bwd) capture using ``make_fx`` +
     ``FakeTensorMode``.  No actual forward pass is run; tensor shapes are
     inferred symbolically.  Fast and works without FSDP initialisation.

2. ``simulate_runtime(model_parts, inputs, ...)``
   — Dynamic capture: runs **one real training step** on CPU (gloo backend)
     while intercepting every dispatched op, collective, and FSDP lifecycle
     event.  The model must already be initialised and moved to CPU.

3. ``simulate_pp_schedule(pp_schedule)``
   — Pure schedule extraction: no model execution, just reads the PP
     schedule's action table (or reconstructs 1F1B heuristically) and builds
     a :class:`TrainingSchedule`.

In all modes, results are returned as :class:`SimulationResult` which can be
serialised with :mod:`export`.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import torch
import torch.nn as nn

from .comm_interceptor import capture_comms, CommRecorder
from .cpu_env import cpu_distributed_context, patch_device_type_to_cpu
from .export import (
    export_chrome_trace,
    export_dot,
    export_html,
    export_json,
    export_text_summary,
)
from .fx_capture import capture_forward_fx, capture_joint_fx
from .nodes import SimulationResult, TrainingSchedule
from .pp_schedule_extractor import PPScheduleExtractor
from .runtime_capture import RuntimeCapture


class Simulator:
    """
    High-level simulator that orchestrates CPU-only graph and schedule capture.

    Args:
        rank: Process rank (0 for single-process simulation).
        world_size: Total number of simulated processes.
        verbose: If ``True``, print progress messages to stdout.
    """

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        verbose: bool = True,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[Simulator rank={self.rank}] {msg}")

    # ------------------------------------------------------------------
    # Mode 1: Static FX capture
    # ------------------------------------------------------------------

    def simulate_fx(
        self,
        model: nn.Module,
        example_inputs: tuple[Any, ...],
        example_kwargs: dict[str, Any] | None = None,
        capture_joint: bool = False,
        loss_fn: Any | None = None,
        example_labels: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SimulationResult:
        """
        Trace *model* statically using ``make_fx`` + ``FakeTensorMode``.

        No actual computation is performed; tensor shapes are propagated via
        FakeTensor metadata.

        Args:
            model: The model to trace (eager, FSDP2, or TP-wrapped).
            example_inputs: Positional CPU tensor inputs.
            example_kwargs: Optional keyword CPU tensor inputs.
            capture_joint: If ``True``, trace the joint fwd+bwd graph.
            loss_fn: Custom loss function for joint capture.
            example_labels: Labels for ``loss_fn`` in joint capture.
            metadata: Extra metadata to attach to the result.

        Returns:
            :class:`SimulationResult` with compute graph (no runtime events).
        """
        self._log("Starting static FX capture …")
        patch_device_type_to_cpu()

        if capture_joint:
            self._log("  mode: joint fwd+bwd")
            graph = capture_joint_fx(
                model,
                example_inputs,
                loss_fn=loss_fn,
                example_labels=example_labels,
                example_kwargs=example_kwargs,
            )
        else:
            self._log("  mode: forward only")
            graph = capture_forward_fx(
                model,
                example_inputs,
                example_kwargs=example_kwargs,
            )

        self._log(f"  captured {len(graph.nodes)} ops, {len(graph.edges)} edges")

        return SimulationResult(
            compute_graph=graph,
            metadata={
                "mode": "fx",
                "rank": self.rank,
                **(metadata or {}),
            },
        )

    # ------------------------------------------------------------------
    # Mode 2: Dynamic runtime capture
    # ------------------------------------------------------------------

    def simulate_runtime(
        self,
        model_parts: list[nn.Module],
        example_inputs: tuple[Any, ...],
        loss_fn: Any | None = None,
        example_labels: torch.Tensor | None = None,
        pp_schedule: Any | None = None,
        pp_stages: list[Any] | None = None,
        optimizer: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SimulationResult:
        """
        Run one training step on CPU while capturing all ops and events.

        Activates :class:`OpCaptureMode`, :class:`CommRecorder`,
        :class:`FSDPEventRecorder`, and optional PP hooks simultaneously, then
        runs:
            1. Forward pass (or ``pp_schedule.step()`` for PP)
            2. ``loss.backward()``
            3. ``optimizer.step()`` (if provided)

        All model parts must be on CPU.

        Args:
            model_parts: List of model module(s).  For PP, one per stage.
            example_inputs: Positional CPU input tensors.
            loss_fn: Optional loss function; defaults to ``output.sum()``.
            example_labels: Labels for ``loss_fn``.
            pp_schedule: Optional PP schedule object (from TorchTitan).
            pp_stages: Optional list of ``PipelineStage`` objects.
            optimizer: Optional optimizer; if given, ``optimizer.step()`` is
                captured too.
            metadata: Extra metadata.

        Returns:
            :class:`SimulationResult` with all captured events.
        """
        self._log("Starting runtime capture …")
        patch_device_type_to_cpu()

        capture = RuntimeCapture(rank=self.rank)

        with capture.activate(
            model_parts,
            phase="forward",
            pp_schedule=pp_schedule,
            pp_stages=pp_stages,
        ):
            if pp_schedule is not None:
                self._log("  running PP schedule step …")
                capture.set_phase("forward")
                pp_schedule.step(*example_inputs)
            else:
                model = model_parts[0]
                self._log("  running forward pass …")
                capture.set_phase("forward")
                output = model(*example_inputs)

                if loss_fn is not None and example_labels is not None:
                    loss = loss_fn(output, example_labels)
                elif isinstance(output, torch.Tensor):
                    loss = output.sum()
                else:
                    import torch.utils._pytree as pytree

                    flat, _ = pytree.tree_flatten(output)
                    loss = sum(t.sum() for t in flat if isinstance(t, torch.Tensor))

                self._log("  running backward pass …")
                capture.set_phase("backward")
                loss.backward()

                if optimizer is not None:
                    self._log("  running optimizer step …")
                    capture.set_phase("optimizer")
                    optimizer.step()
                    optimizer.zero_grad()

        result = capture.build_result(metadata={"mode": "runtime", **(metadata or {})})

        self._log(
            f"  captured {len(result.compute_graph.nodes)} ops, "
            f"{len(result.comm_events)} comm events, "
            f"{len(result.fsdp_events)} FSDP events"
        )
        return result

    # ------------------------------------------------------------------
    # Mode 3: PP schedule only
    # ------------------------------------------------------------------

    def simulate_pp_schedule(
        self,
        pp_schedule: Any,
        num_stages: int | None = None,
        num_microbatches: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SimulationResult:
        """
        Extract the PP schedule (no model execution).

        Reads the schedule's action table or reconstructs it heuristically.
        Returns a :class:`SimulationResult` with an empty compute graph and a
        populated :class:`TrainingSchedule`.

        Args:
            pp_schedule: A TorchTitan pipeline schedule object.
            num_stages: Override for number of stages (auto-detected if ``None``).
            num_microbatches: Override for number of microbatches.
            metadata: Extra metadata.

        Returns:
            :class:`SimulationResult` with populated schedule, empty graph.
        """
        self._log("Extracting PP schedule …")
        extractor = PPScheduleExtractor(
            schedule=pp_schedule,
            pp_rank=self.rank,
            world_size=self.world_size,
            n_microbatches=num_microbatches,
        )
        schedule = extractor.extract()
        self._log(
            f"  schedule has {len(schedule.events)} events, "
            f"{len(schedule.deps)} deps"
        )

        from .nodes import ComputeGraph

        return SimulationResult(
            compute_graph=ComputeGraph(metadata={}),
            schedule=schedule,
            metadata={"mode": "pp_schedule_only", **(metadata or {})},
        )

    # ------------------------------------------------------------------
    # Convenience: run all modes and export
    # ------------------------------------------------------------------

    def simulate_all(
        self,
        model_parts: list[nn.Module],
        example_inputs: tuple[Any, ...],
        pp_schedule: Any | None = None,
        pp_stages: list[Any] | None = None,
        loss_fn: Any | None = None,
        example_labels: torch.Tensor | None = None,
        output_dir: str = "./simulator_output",
        output_formats: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SimulationResult:
        """
        Run static FX capture + runtime capture (+ PP schedule if given).

        Merges the results and exports to *output_dir* in all requested
        formats.

        Args:
            model_parts: List of model modules (CPU).
            example_inputs: CPU example inputs.
            pp_schedule: Optional PP schedule.
            pp_stages: Optional PP stage list.
            loss_fn: Optional loss function.
            example_labels: Optional labels.
            output_dir: Directory for output files.
            output_formats: List of ``"json"``, ``"dot"``, ``"chrome_trace"``,
                ``"text"``.  Defaults to all four.
            metadata: Extra metadata.

        Returns:
            The runtime :class:`SimulationResult` (augmented with FX graph
            and schedule if applicable).
        """
        if output_formats is None:
            output_formats = ["json", "dot", "chrome_trace", "html", "text"]

        # 1. Static FX capture
        self._log("=== Phase 1: Static FX capture ===")
        fx_result = self.simulate_fx(
            model_parts[0],
            example_inputs,
            metadata=metadata,
        )

        # 2. Runtime capture
        self._log("=== Phase 2: Runtime capture ===")
        rt_result = self.simulate_runtime(
            model_parts,
            example_inputs,
            loss_fn=loss_fn,
            example_labels=example_labels,
            pp_schedule=pp_schedule,
            pp_stages=pp_stages,
            metadata=metadata,
        )

        # 3. PP schedule (if present)
        if pp_schedule is not None:
            self._log("=== Phase 3: PP schedule extraction ===")
            pp_result = self.simulate_pp_schedule(pp_schedule, metadata=metadata)
            rt_result.schedule = pp_result.schedule
            rt_result.pp_events = pp_result.pp_events

        # Attach FX graph as a nested metadata entry
        rt_result.metadata["fx_graph_node_count"] = len(fx_result.compute_graph.nodes)

        # 4. Export
        os.makedirs(output_dir, exist_ok=True)
        self._log(f"=== Exporting to {output_dir} ===")

        if "json" in output_formats:
            p = os.path.join(output_dir, "simulation_result.json")
            export_json(rt_result, p)
            self._log(f"  JSON → {p}")

        if "dot" in output_formats:
            p = os.path.join(output_dir, "compute_graph.dot")
            export_dot(rt_result.compute_graph, p, title="ComputeGraph")
            self._log(f"  DOT  → {p}")

        if "chrome_trace" in output_formats:
            p = os.path.join(output_dir, "trace.json")
            export_chrome_trace(rt_result, p)
            self._log(f"  Chrome trace → {p}")

        if "html" in output_formats:
            p = os.path.join(output_dir, "trace.html")
            export_html(rt_result, p)
            self._log(f"  HTML trace → {p}")

        if "text" in output_formats:
            summary = export_text_summary(rt_result)
            p = os.path.join(output_dir, "summary.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(summary)
            self._log(f"  Text summary → {p}")
            if self.verbose:
                print(summary)

        return rt_result
