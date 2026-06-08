# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Records FSDP parameter lifecycle events (allgather / reshard / reduce-scatter)
by attaching module hooks to all :class:`FSDPModule` instances in a model.

The FSDP parameter lifecycle per-module during one forward+backward pass:

  1. **forward_pre_hook**  → parameter allgather (gather shards into full param)
  2. **forward_hook**      → parameter reshard  (drop gathered params if configured)
  3. **backward_pre_hook** → parameter allgather (for gradient computation)
  4. **backward_hook**     → gradient reduce-scatter (scatter and reduce gradients)

Usage::

    recorder = FSDPEventRecorder(rank=0)
    with capture_fsdp_events(model, recorder):
        loss = model(inputs)
        loss.backward()
    print(recorder.events)
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn


class FSDPEventRecorder:
    """Thread-safe recorder for FSDP lifecycle events."""

    def __init__(self, rank: int = 0) -> None:
        self._lock = threading.Lock()
        self._counter: int = 0
        self.events: list[dict[str, Any]] = []
        self.rank: int = rank
        self.logical_clock: int = 0
        self.current_phase: str = "forward"

    def _next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"fsdp_{self._counter:07d}"

    def record(
        self,
        event_type: str,
        module_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "event_id": self._next_id(),
            "event_type": event_type,
            "module_name": module_name,
            "rank": self.rank,
            "phase": self.current_phase,
            "logical_clock": self.logical_clock,
            "metadata": metadata or {},
        }
        with self._lock:
            self.events.append(event)
            self.logical_clock += 1
        return event


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------


def _fwd_pre_hook(recorder: FSDPEventRecorder, module_name: str):
    """Fires before module.forward() — corresponds to FSDP allgather."""

    def hook(module: nn.Module, input: Any) -> None:
        recorder.record(
            "fsdp_allgather_pre_fwd",
            module_name,
            {"action": "allgather_params"},
        )

    return hook


def _fwd_post_hook(recorder: FSDPEventRecorder, module_name: str):
    """Fires after module.forward() — may correspond to FSDP reshard."""

    def hook(module: nn.Module, input: Any, output: Any) -> None:
        recorder.record(
            "fsdp_reshard_post_fwd",
            module_name,
            {"action": "reshard_params"},
        )

    return hook


def _bwd_pre_hook(recorder: FSDPEventRecorder, module_name: str):
    """Fires before module backward — corresponds to FSDP allgather for bwd."""

    def hook(module: nn.Module, grad_output: Any) -> None:
        recorder.record(
            "fsdp_allgather_pre_bwd",
            module_name,
            {"action": "allgather_params_for_bwd"},
        )

    return hook


def _bwd_post_hook(recorder: FSDPEventRecorder, module_name: str):
    """Fires after module backward — corresponds to FSDP reduce-scatter."""

    def hook(
        module: nn.Module,
        grad_input: Any,
        grad_output: Any,
    ) -> None:
        recorder.record(
            "fsdp_reduce_scatter_post_bwd",
            module_name,
            {"action": "reduce_scatter_grads"},
        )

    return hook


# ---------------------------------------------------------------------------
# Public context manager
# ---------------------------------------------------------------------------


@contextmanager
def capture_fsdp_events(
    model: nn.Module,
    recorder: FSDPEventRecorder,
) -> Generator[FSDPEventRecorder, None, None]:
    """
    Register FSDP lifecycle hooks on every :class:`FSDPModule` found in
    *model* (via ``model.named_modules()``).

    Falls back gracefully if ``torch.distributed._composable.fsdp`` is not
    available (e.g. older PyTorch).

    Args:
        model: The (FSDP-wrapped) model to instrument.
        recorder: Target recorder for events.

    Yields:
        The same *recorder* for convenience.
    """
    try:
        from torch.distributed._composable.fsdp import FSDPModule
    except ImportError:
        # FSDP not available — nothing to hook
        yield recorder
        return

    handles: list[Any] = []

    for name, module in model.named_modules():
        if isinstance(module, FSDPModule):
            handles.append(
                module.register_forward_pre_hook(_fwd_pre_hook(recorder, name))
            )
            handles.append(module.register_forward_hook(_fwd_post_hook(recorder, name)))
            handles.append(
                module.register_full_backward_pre_hook(_bwd_pre_hook(recorder, name))
            )
            handles.append(
                module.register_full_backward_hook(_bwd_post_hook(recorder, name))
            )

    try:
        yield recorder
    finally:
        for h in handles:
            h.remove()
