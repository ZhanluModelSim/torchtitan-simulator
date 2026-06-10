# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
from pathlib import Path

from ..nodes import SimulationResult
from .schedule_timing import _inject_schedule_timing, _populate_des_metadata


def export_json(result: SimulationResult, path: str | os.PathLike) -> None:
    """
    Serialize a :class:`SimulationResult` to a JSON file.

    The output is pretty-printed with ``indent=2`` for readability.

    Args:
        result: The simulation result to serialize.
        path: Output file path (will be created / overwritten).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _populate_des_metadata(result)
    data = result.to_dict()
    _inject_schedule_timing(data, result)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
