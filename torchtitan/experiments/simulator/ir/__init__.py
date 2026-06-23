# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Layered IR (L0-L3) projection over captured simulation results.

The layers mirror ``ZhanluModelSim/workload-model-platform`` ``spec/``:

* L0 :class:`SpecOpNode`     — single op
* L1 :class:`StepGraph`      — fwd/bwd/opt DAG template
* L2 :class:`ScheduleGraph`  — StepInstance / DataPass orchestration
* L3 :class:`WorkloadGraph`  — iteration semantics + data flow

All builders project from *captured* data; they do not re-implement
torchtitan's training or parallelism logic.
"""

from __future__ import annotations

from .builder import build_workload_graph
from .op_node import project_op_nodes, SpecOpNode
from .schedule_graph import (
    DataPass,
    ScheduleBuilder,
    ScheduleGraph,
    StepInstance,
    TensorSlot,
)
from .step_graph import StepBuilder, StepGraph
from .workload_graph import DataFlow, IterationSpec, WorkloadBuilder, WorkloadGraph

__all__ = [
    "SpecOpNode",
    "project_op_nodes",
    "StepGraph",
    "StepBuilder",
    "StepInstance",
    "TensorSlot",
    "DataPass",
    "ScheduleGraph",
    "ScheduleBuilder",
    "DataFlow",
    "IterationSpec",
    "WorkloadGraph",
    "WorkloadBuilder",
    "build_workload_graph",
]
