# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L3 — WorkloadGraph.

Outermost container: holds the :class:`ScheduleGraph`, iteration semantics,
and data-flow rhythm.  Built by projecting the captured schedule plus the
declared training/dataloader config — no training loop is re-implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schedule_graph import DataPass, ScheduleGraph, TensorSlot
from .step_graph import StepGraph


@dataclass
class DataFlow:
    source: str
    tensor_shape: tuple[Any, ...]
    dtype: str
    volume_per_iter: int = 0
    is_streaming: bool = False
    interleave_strategy: str = "synced"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "tensor_shape": list(self.tensor_shape),
            "dtype": self.dtype,
            "volume_per_iter": self.volume_per_iter,
            "is_streaming": self.is_streaming,
            "interleave_strategy": self.interleave_strategy,
        }


@dataclass
class IterationSpec:
    schedule: ScheduleGraph
    microbatch_count: int = 1
    iteration_time_est: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule": self.schedule.to_dict(),
            "microbatch_count": self.microbatch_count,
            "iteration_time_est": self.iteration_time_est,
        }


@dataclass
class WorkloadGraph:
    workload_id: str
    workload_type: str
    step_templates: dict[str, StepGraph]
    iteration: IterationSpec
    num_iterations: int = 1
    warmup_iterations: int = 0
    data_inputs: list[DataFlow] = field(default_factory=list)
    data_outputs: list[DataFlow] = field(default_factory=list)
    cross_iter_passes: list[DataPass] = field(default_factory=list)
    total_runtime_est: float = 0.0
    total_cost_est: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "workload_type": self.workload_type,
            "step_templates": {k: v.to_dict() for k, v in self.step_templates.items()},
            "iteration": self.iteration.to_dict(),
            "num_iterations": self.num_iterations,
            "warmup_iterations": self.warmup_iterations,
            "data_inputs": [d.to_dict() for d in self.data_inputs],
            "data_outputs": [d.to_dict() for d in self.data_outputs],
            "cross_iter_passes": [p.to_dict() for p in self.cross_iter_passes],
            "total_runtime_est": self.total_runtime_est,
            "total_cost_est": self.total_cost_est,
        }
from .op_node import _DTYPE_BYTES


class WorkloadBuilder:
    """Build a :class:`WorkloadGraph` from schedule + training/data config."""

    @staticmethod
    def from_capture(
        schedule_graph: ScheduleGraph,
        step_templates: dict[str, StepGraph],
        training: Any,
        *,
        token_dtype: str = "int64",
    ) -> WorkloadGraph:
        steps = int(getattr(training, "steps", 1) or 1)
        warmup = int(getattr(training, "warmup_steps", 0) or 0)
        seq_len = int(getattr(training, "seq_len", 0) or 0)
        batch = int(getattr(training, "local_batch_size", 0) or 0)
        ga = schedule_graph.gradient_accumulation

        iteration = IterationSpec(
            schedule=schedule_graph,
            microbatch_count=schedule_graph.num_micro_batches,
        )

        data_inputs: list[DataFlow] = []
        if seq_len and batch:
            volume = batch * seq_len * _DTYPE_BYTES.get(token_dtype, 4) * max(1, ga)
            data_inputs.append(
                DataFlow(
                    source="dataloader",
                    tensor_shape=(batch, seq_len),
                    dtype=token_dtype,
                    volume_per_iter=volume,
                    is_streaming=True,
                    interleave_strategy="synced",
                )
            )

        cross_iter_passes: list[DataPass] = []
        if "optimizer" in step_templates and "forward" in step_templates:
            cross_iter_passes.append(
                DataPass(
                    src_instance="inst_optimizer",
                    dst_instance="inst_fwd_mb0",
                    slots=[TensorSlot("parameters", "", "")],
                )
            )

        return WorkloadGraph(
            workload_id="workload_train",
            workload_type="train",
            step_templates=step_templates,
            iteration=iteration,
            num_iterations=steps,
            warmup_iterations=warmup,
            data_inputs=data_inputs,
            cross_iter_passes=cross_iter_passes,
        )
