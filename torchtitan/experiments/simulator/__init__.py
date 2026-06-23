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

import importlib
import sys

from .cost_model import apply_cost_model, CostModel, MockCostModel
from .des_engine import DESEngine, simulate_multi_rank_des, simulate_single_rank_des
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

_MODULE_ALIASES = {
    "torchtitan.experiments.simulator.comm_interceptor": (
        "torchtitan.experiments.simulator.capture.comm_interceptor"
    ),
    "torchtitan.experiments.simulator.dispatch_interceptor": (
        "torchtitan.experiments.simulator.capture.dispatch_interceptor"
    ),
    "torchtitan.experiments.simulator.fsdp_tracer": (
        "torchtitan.experiments.simulator.capture.fsdp_tracer"
    ),
    "torchtitan.experiments.simulator.fx_capture": (
        "torchtitan.experiments.simulator.capture.fx_capture"
    ),
    "torchtitan.experiments.simulator.graph_assembler": (
        "torchtitan.experiments.simulator.capture.graph_assembler"
    ),
    "torchtitan.experiments.simulator.runtime_capture": (
        "torchtitan.experiments.simulator.capture.runtime_capture"
    ),
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
]
