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

Quick start::

    from torchtitan.experiments.simulator import Simulator, export_json

    sim = Simulator()

    # Static FX trace (no execution)
    result = sim.simulate_fx(model, example_inputs=(tokens,))

    # Dynamic runtime capture (1 real training step on CPU)
    result = sim.simulate_runtime([model], example_inputs=(tokens,))

    # PP schedule extraction only
    result = sim.simulate_pp_schedule(pp_sched)

    # Export to file
    export_json(result, "output/result.json")
"""

from .cost_model import CostModel, MockCostModel, apply_cost_model
from .export import (
    export_chrome_trace,
    export_dot,
    export_html,
    export_json,
    export_text_summary,
)
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
from .simulator import Simulator

__all__ = [
    # Main class
    "Simulator",
    # Data model
    "SimulationResult",
    "ComputeGraph",
    "TrainingSchedule",
    "OpNode",
    "DataEdge",
    "TensorMeta",
    "ScheduleEvent",
    "ScheduleDep",
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
]
