# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L2 — ScheduleGraph.

Describes how :class:`StepGraph` templates are orchestrated: micro-batches,
pipeline stages, devices, and the tensor passes between step instances.

``ScheduleBuilder`` derives instances and data passes from the *captured*
PP schedule (``TrainingSchedule`` extracted from torchtitan's real schedule
object) and the declared parallelism degrees in the config.  No pipeline
logic is re-implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..nodes import TrainingSchedule
from .step_graph import StepGraph


@dataclass
class StepInstance:
    instance_id: str
    step_ref: str
    step_type: str
    micro_batch_idx: int = 0
    pipeline_stage: int = 0
    device_ids: list[int] = field(default_factory=list)
    dp_group: int = 0
    estimated_runtime: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "step_ref": self.step_ref,
            "step_type": self.step_type,
            "micro_batch_idx": self.micro_batch_idx,
            "pipeline_stage": self.pipeline_stage,
            "device_ids": self.device_ids,
            "dp_group": self.dp_group,
            "estimated_runtime": self.estimated_runtime,
        }


@dataclass
class TensorSlot:
    name: str
    src_exit_op: str
    dst_entry_op: str
    volume_bytes: int = 0
    is_incremental: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "src_exit_op": self.src_exit_op,
            "dst_entry_op": self.dst_entry_op,
            "volume_bytes": self.volume_bytes,
            "is_incremental": self.is_incremental,
        }


@dataclass
class DataPass:
    src_instance: str
    dst_instance: str
    slots: list[TensorSlot] = field(default_factory=list)
    src_device: int | None = None
    dst_device: int | None = None
    requires_communication: bool = False
    comm_primitive: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "src_instance": self.src_instance,
            "dst_instance": self.dst_instance,
            "slots": [s.to_dict() for s in self.slots],
            "src_device": self.src_device,
            "dst_device": self.dst_device,
            "requires_communication": self.requires_communication,
            "comm_primitive": self.comm_primitive,
        }


@dataclass
class ScheduleGraph:
    schedule_id: str
    workload_type: str
    step_templates: dict[str, StepGraph] = field(default_factory=dict)
    instances: list[StepInstance] = field(default_factory=list)
    data_passes: list[DataPass] = field(default_factory=list)
    ctrl_edges: list[tuple[str, str]] = field(default_factory=list)
    dp_degree: int = 1
    tp_degree: int = 1
    pp_degree: int = 1
    ep_degree: int = 1
    cp_degree: int = 1
    num_micro_batches: int = 1
    pipeline_schedule: str = "none"
    gradient_accumulation: int = 1
    zero_stage: int = 0

    @property
    def instance_map(self) -> dict[str, StepInstance]:
        return {i.instance_id: i for i in self.instances}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "workload_type": self.workload_type,
            "step_templates": {k: v.to_dict() for k, v in self.step_templates.items()},
            "instances": [i.to_dict() for i in self.instances],
            "data_passes": [d.to_dict() for d in self.data_passes],
            "ctrl_edges": [list(e) for e in self.ctrl_edges],
            "dp_degree": self.dp_degree,
            "tp_degree": self.tp_degree,
            "pp_degree": self.pp_degree,
            "ep_degree": self.ep_degree,
            "cp_degree": self.cp_degree,
            "num_micro_batches": self.num_micro_batches,
            "pipeline_schedule": self.pipeline_schedule,
            "gradient_accumulation": self.gradient_accumulation,
            "zero_stage": self.zero_stage,
        }


def _degree(parallelism: Any, name: str, default: int = 1) -> int:
    return int(getattr(parallelism, name, default) or default)


class ScheduleBuilder:
    """Derive a :class:`ScheduleGraph` from captured schedule + config."""

    @staticmethod
    def from_capture(
        step_templates: dict[str, StepGraph],
        schedule: TrainingSchedule | None,
        parallelism: Any,
        *,
        gradient_accumulation: int = 1,
        world_size: int = 1,
    ) -> ScheduleGraph:
        pp = _degree(parallelism, "pipeline_parallel_degree")
        tp = _degree(parallelism, "tensor_parallel_degree")
        ep = _degree(parallelism, "expert_parallel_degree")
        cp = _degree(parallelism, "context_parallel_degree")
        dp_shard = int(getattr(parallelism, "data_parallel_shard_degree", 1) or 1)
        dp_repl = _degree(parallelism, "data_parallel_replicate_degree")
        if dp_shard < 0:
            denom = max(1, pp * tp * cp)
            dp_shard = max(1, world_size // denom)
        dp = max(1, dp_shard * dp_repl)
        pipeline_schedule = str(
            getattr(parallelism, "pipeline_parallel_schedule", "none") or "none"
        )

        sg = ScheduleGraph(
            schedule_id="schedule_train",
            workload_type="train",
            step_templates=step_templates,
            dp_degree=dp,
            tp_degree=tp,
            pp_degree=pp,
            ep_degree=ep,
            cp_degree=cp,
            gradient_accumulation=gradient_accumulation,
            pipeline_schedule=pipeline_schedule if pp > 1 else "none",
        )

        instances, data_passes, num_mb = ScheduleBuilder._build_instances(
            schedule, step_templates, pp, gradient_accumulation
        )
        sg.instances = instances
        sg.data_passes = data_passes
        sg.num_micro_batches = num_mb
        return sg

    @staticmethod
    def _build_instances(
        schedule: TrainingSchedule | None,
        step_templates: dict[str, StepGraph],
        pp: int,
        gradient_accumulation: int,
    ) -> tuple[list[StepInstance], list[DataPass], int]:
        instances: list[StepInstance] = []
        data_passes: list[DataPass] = []

        fwd_events = []
        bwd_events = []
        if schedule is not None:
            for ev in schedule.events:
                et = (ev.event_type or "").lower()
                if et.startswith("fwd") or et == "f":
                    fwd_events.append(ev)
                elif et.startswith("bwd") or et == "b":
                    bwd_events.append(ev)

        if fwd_events or bwd_events:
            return ScheduleBuilder._instances_from_events(
                fwd_events, bwd_events, step_templates
            )

        # Fallback: no captured PP schedule → single fwd→bwd→opt chain per
        # gradient-accumulation micro-batch.
        num_mb = max(1, gradient_accumulation)
        prev_bwd: str | None = None
        for mb in range(num_mb):
            f_id = f"inst_fwd_mb{mb}"
            b_id = f"inst_bwd_mb{mb}"
            if "forward" in step_templates:
                instances.append(
                    StepInstance(f_id, "step_forward", "forward", micro_batch_idx=mb)
                )
            if "backward" in step_templates:
                instances.append(
                    StepInstance(b_id, "step_backward", "backward", micro_batch_idx=mb)
                )
                if "forward" in step_templates:
                    data_passes.append(
                        DataPass(f_id, b_id, [TensorSlot("activations", "", "")])
                    )
                prev_bwd = b_id
        if "optimizer" in step_templates:
            o_id = "inst_optimizer"
            instances.append(StepInstance(o_id, "step_optimizer", "optimizer"))
            if prev_bwd is not None:
                data_passes.append(
                    DataPass(prev_bwd, o_id, [TensorSlot("gradients", "", "")])
                )
        return instances, data_passes, num_mb

    @staticmethod
    def _instances_from_events(
        fwd_events: list[Any],
        bwd_events: list[Any],
        step_templates: dict[str, StepGraph],
    ) -> tuple[list[StepInstance], list[DataPass], int]:
        instances: list[StepInstance] = []
        data_passes: list[DataPass] = []
        micro_batches: set[int] = set()

        def _mk(ev: Any, step_type: str, step_ref: str) -> StepInstance:
            mb = ev.microbatch_idx if ev.microbatch_idx is not None else 0
            stage = ev.pp_stage if ev.pp_stage is not None else 0
            micro_batches.add(mb)
            return StepInstance(
                instance_id=f"inst_{step_type}_s{stage}_mb{mb}",
                step_ref=step_ref,
                step_type=step_type,
                micro_batch_idx=mb,
                pipeline_stage=stage,
                device_ids=[stage],
            )

        fwd_by_key: dict[tuple[int, int], StepInstance] = {}
        for ev in fwd_events:
            inst = _mk(ev, "forward", "step_forward")
            instances.append(inst)
            fwd_by_key[(inst.pipeline_stage, inst.micro_batch_idx)] = inst

        bwd_by_key: dict[tuple[int, int], StepInstance] = {}
        for ev in bwd_events:
            inst = _mk(ev, "backward", "step_backward")
            instances.append(inst)
            bwd_by_key[(inst.pipeline_stage, inst.micro_batch_idx)] = inst

        # PP forward pass: stage_i fwd -> stage_{i+1} fwd (same micro-batch).
        for (stage, mb), inst in fwd_by_key.items():
            nxt = fwd_by_key.get((stage + 1, mb))
            if nxt is not None:
                data_passes.append(
                    DataPass(
                        inst.instance_id,
                        nxt.instance_id,
                        [TensorSlot("activations", "", "")],
                        src_device=stage,
                        dst_device=stage + 1,
                        requires_communication=True,
                        comm_primitive="p2p_send_recv",
                    )
                )
        # PP backward pass: stage_i bwd -> stage_{i-1} bwd (same micro-batch).
        for (stage, mb), inst in bwd_by_key.items():
            nxt = bwd_by_key.get((stage - 1, mb))
            if nxt is not None:
                data_passes.append(
                    DataPass(
                        inst.instance_id,
                        nxt.instance_id,
                        [TensorSlot("gradients", "", "")],
                        src_device=stage,
                        dst_device=stage - 1,
                        requires_communication=True,
                        comm_primitive="p2p_send_recv",
                    )
                )

        num_mb = len(micro_batches) if micro_batches else 1

        if "optimizer" in step_templates:
            instances.append(
                StepInstance("inst_optimizer", "step_optimizer", "optimizer")
            )
        return instances, data_passes, num_mb
