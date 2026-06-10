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

from typing import Any

import torch
import torch.nn as nn

from .cpu_env import patch_device_type_to_cpu
from .export import export_result
from .fx_capture import capture_forward_fx, capture_joint_fx
from .meta_env import patch_device_type_to_meta
from .nodes import SimulationResult
from .pp_schedule_extractor import PPScheduleExtractor
from .unified_trace import compute_loss, TraceRecorder, unified_trace


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

        Uses :class:`TraceRecorder` + :func:`unified_trace` to intercept
        every dispatched op, collective, and FSDP lifecycle event, then
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

        recorder = TraceRecorder(rank=self.rank)

        with unified_trace(
            recorder,
            model_parts[0],
            example_inputs,
            use_fake_mode=False,
            phase="forward",
            capture_comm=True,
            capture_fsdp=True,
            model_parts=model_parts,
        ):
            if pp_schedule is not None:
                self._log("  running PP schedule step …")
                pp_schedule.step(*example_inputs)
            else:
                model = model_parts[0]
                self._log("  running forward pass …")
                output = model(*example_inputs)

                loss = compute_loss(output, loss_fn=loss_fn, labels=example_labels)

                self._log("  running backward pass …")
                recorder.current_phase = "backward"
                loss.backward()

                if optimizer is not None:
                    self._log("  running optimizer step …")
                    recorder.current_phase = "optimizer"
                    optimizer.step()
                    optimizer.zero_grad()

        result = recorder.build_result(metadata={"mode": "runtime", **(metadata or {})})

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
    # Mode 4: Unified dispatch trace (FakeTensorMode + TorchDispatchMode)
    # ------------------------------------------------------------------

    def simulate_unified(
        self,
        model: nn.Module,
        example_inputs: tuple[Any, ...],
        loss_fn: Any | None = None,
        example_labels: torch.Tensor | None = None,
        device_mode: str = "meta",
        metadata: dict[str, Any] | None = None,
    ) -> SimulationResult:
        """
        Trace *model* using the unified dispatch capture mode.

        Combines ``FakeTensorMode`` with ``TorchDispatchMode`` to capture
        every dispatched operation **without allocating any real memory**.
        This is the preferred mode for ``fake_backend`` simulation because
        it works on meta/FakeTensor inputs and produces a single coherent
        compute graph in one pass.

        When ``device_mode=\"meta\"``, the device environment is patched to
        meta so that model parameters are shape-only (0 bytes).  When
        ``device_mode=\"cpu\"``, the existing CPU patching is used and
        ``FakeTensorMode`` is still active inside the unified trace context
        (but real tensors are allocated for the model).

        Args:
            model: The model to trace.  Must be on meta (if
                ``device_mode=\"meta\"``) or CPU (if ``device_mode=\"cpu\"``).
            example_inputs: Positional input tensors.  Must match the model's
                device (meta or CPU).
            loss_fn: Optional loss function; defaults to ``output.sum()``.
            example_labels: Labels for ``loss_fn``.
            device_mode: ``\"meta\"`` or ``\"cpu\"``.  Controls which device
                patching to apply.
            metadata: Extra metadata.

        Returns:
            :class:`SimulationResult` with the unified compute graph.
        """
        self._log(f"Starting unified trace (device_mode={device_mode}) …")

        if device_mode == "meta":
            patch_device_type_to_meta()
        else:
            patch_device_type_to_cpu()

        recorder = TraceRecorder(rank=self.rank)

        use_fake = device_mode == "meta"

        with unified_trace(
            recorder,
            model,
            example_inputs,
            use_fake_mode=use_fake,
            phase="forward",
        ):
            self._log("  running forward pass …")
            output = model(*example_inputs)
            loss = compute_loss(output, loss_fn=loss_fn, labels=example_labels)

            self._log("  running backward pass …")
            recorder.current_phase = "backward"
            loss.backward()

        result = recorder.build_result(
            metadata={
                "mode": "unified_trace",
                "device_mode": device_mode,
                "rank": self.rank,
                **(metadata or {}),
            },
        )

        self._log(
            f"  captured {len(result.compute_graph.nodes)} ops, "
            f"{len(result.compute_graph.edges)} edges"
        )
        return result

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
        export_result(
            rt_result,
            output_dir,
            output_formats,
            log_fn=self._log if self.verbose else None,
            print_summary=self.verbose,
        )

        return rt_result
