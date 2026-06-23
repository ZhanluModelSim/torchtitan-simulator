# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Top-level orchestrator that projects a captured :class:`SimulationResult`
into the spec's L1/L2/L3 layered IR.

Everything here is a projection of *captured* data plus *declared* config.
No torchtitan training / parallelism logic is re-implemented.
"""

from __future__ import annotations

import os
from typing import Any

from ..nodes import SimulationResult
from .schedule_graph import ScheduleBuilder, ScheduleGraph
from .step_graph import StepBuilder, StepGraph
from .workload_graph import WorkloadBuilder, WorkloadGraph


def build_step_graphs(result: SimulationResult) -> dict[str, StepGraph]:
    """L1: partition the captured compute graph into per-phase templates."""
    return StepBuilder.from_compute_graph(result.compute_graph)


def build_schedule_graph(
    result: SimulationResult,
    config: Any,
    step_templates: dict[str, StepGraph] | None = None,
) -> ScheduleGraph:
    """L2: orchestrate step templates using captured schedule + config."""
    if step_templates is None:
        step_templates = build_step_graphs(result)
    parallelism = getattr(config, "parallelism", None)
    gradient_accumulation = _gradient_accumulation(result, config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return ScheduleBuilder.from_capture(
        step_templates,
        result.schedule,
        parallelism,
        gradient_accumulation=gradient_accumulation,
        world_size=world_size,
    )


def build_workload_graph(result: SimulationResult, config: Any) -> WorkloadGraph:
    """L3: wrap the schedule with iteration semantics + data flow."""
    step_templates = build_step_graphs(result)
    schedule_graph = build_schedule_graph(result, config, step_templates)
    training = getattr(config, "training", None)
    return WorkloadBuilder.from_capture(
        schedule_graph, step_templates, training
    )


def _gradient_accumulation(result: SimulationResult, config: Any) -> int:
    # Prefer the value captured from the live Trainer (capture-faithful).
    meta_val = (result.metadata or {}).get("gradient_accumulation_steps")
    if meta_val:
        return int(meta_val)
    training = getattr(config, "training", None)
    for attr in ("gradient_accumulation_steps", "gradient_accumulation"):
        val = getattr(training, attr, None)
        if val:
            return int(val)
    return 1
