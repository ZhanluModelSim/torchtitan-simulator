# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TorchTitan CPU Simulator — Public API
======================================

Capture forward/backward computation graphs and training schedules (PP, FSDP)
on a pure CPU environment without any GPU hardware.

The primary entry point is :class:`SimulationTrainer` (via ``run_train.sh``).
"""

import importlib
import sys

from .capture.unified_trace import TraceRecorder, unified_trace
from .cost_model import apply_cost_model, CostModel, MockCostModel
from .des_engine import DESEngine, simulate_multi_rank_des, simulate_single_rank_des
from .export import (
    export_chrome_trace,
    export_dot,
    export_html,
    export_json,
    export_kernel_summary_csv,
    export_text_summary,
)
from .ir import build_workload_graph
from .nodes import (
    ComputeGraph,
    DataEdge,
    OpNode,
    PerfResult,
    ScheduleDep,
    ScheduleEvent,
    SimulationResult,
    TensorMeta,
    TrainingSchedule,
)
from .trainer import SimulationConfig, SimulationTrainer

# Backward-compat module aliases so that old import paths still resolve.
_MODULE_ALIASES = {
    "torchtitan.experiments.simulator.unified_trace": (
        "torchtitan.experiments.simulator.capture.unified_trace"
    ),
    "torchtitan.experiments.simulator.pp_schedule_extractor": (
        "torchtitan.experiments.simulator.schedule.pp_schedule_extractor"
    ),
    "torchtitan.experiments.simulator.schedule_extract": (
        "torchtitan.experiments.simulator.schedule.schedule_extract"
    ),
    "torchtitan.experiments.simulator.schedule_generator": (
        "torchtitan.experiments.simulator.schedule.schedule_generator"
    ),
}

for _old_mod, _new_mod in _MODULE_ALIASES.items():
    sys.modules.setdefault(_old_mod, importlib.import_module(_new_mod))

__all__ = [
    # Entry point
    "SimulationTrainer",
    "SimulationConfig",
    # Capture
    "TraceRecorder",
    "unified_trace",
    # Data model
    "SimulationResult",
    "ComputeGraph",
    "TrainingSchedule",
    "OpNode",
    "DataEdge",
    "TensorMeta",
    "ScheduleEvent",
    "ScheduleDep",
    # DES engine
    "DESEngine",
    "simulate_single_rank_des",
    "simulate_multi_rank_des",
    # Cost model
    "CostModel",
    "MockCostModel",
    "PerfResult",
    "apply_cost_model",
    # Export helpers
    "export_json",
    "export_dot",
    "export_chrome_trace",
    "export_html",
    "export_text_summary",
    "export_kernel_summary_csv",
    # Layered IR (L0-L3) projection
    "build_workload_graph",
]
