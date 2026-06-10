# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ParallelismDegrees:
    pp: int
    tp: int
    dp_shard: int
    dp_replicate: int
    dp: int


def read_parallelism_degrees(config: Any) -> ParallelismDegrees:
    parallelism = getattr(config, "parallelism", None)
    if parallelism is None:
        return ParallelismDegrees(pp=1, tp=1, dp_shard=1, dp_replicate=1, dp=1)
    pp = int(getattr(parallelism, "pipeline_parallel_degree", 1) or 1)
    tp = int(getattr(parallelism, "tensor_parallel_degree", 1) or 1)
    ds = int(getattr(parallelism, "data_parallel_shard_degree", 1) or 1)
    if ds < 0:
        ds = 1
    dr = int(getattr(parallelism, "data_parallel_replicate_degree", 1) or 1)
    return ParallelismDegrees(pp=pp, tp=tp, dp_shard=ds, dp_replicate=dr, dp=ds * dr)


def inject_semantic_schedule(result: Any, config: Any) -> None:
    from .nodes import TrainingSchedule
    from .schedule_extract import extract_schedule_from_pytorch

    parallelism = getattr(config, "parallelism", None)
    if parallelism is None:
        return

    par = read_parallelism_degrees(config)
    schedule_name = str(
        getattr(parallelism, "pipeline_parallel_schedule", "1F1B") or "1F1B"
    )
    num_mb = int(getattr(parallelism, "pipeline_parallel_microbatch_size", 8) or 8)
    virtual = 2 if "Interleaved" in schedule_name else 1
    num_stages = par.pp * virtual

    semantic = extract_schedule_from_pytorch(
        pp_degree=par.pp,
        tp_degree=par.tp,
        dp_degree=par.dp,
        num_stages=num_stages,
        n_microbatches=num_mb,
        schedule_name=schedule_name,
        virtual_stages_per_rank=virtual,
    )

    existing = result.schedule
    if existing is None:
        result.schedule = semantic
    elif isinstance(existing, TrainingSchedule):
        for ev in semantic.events:
            existing.add_event(ev)
        for dep in semantic.deps:
            existing.add_dep(dep)
