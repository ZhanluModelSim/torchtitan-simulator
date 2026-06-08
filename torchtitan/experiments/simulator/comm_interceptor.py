# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Monkey-patches ``torch.distributed`` and ``torch.distributed._functional_collectives``
to intercept and record every collective and point-to-point communication operation.

Usage::

    recorder = CommRecorder(rank=0)
    with capture_comms(recorder):
        dist.all_reduce(tensor, group=my_group)
        dist.send(tensor, dst=1)
    print(recorder.events)
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist

from .nodes import TensorMeta

from .unified_trace import get_current_recorder

# ---------------------------------------------------------------------------
# CommRecorder
# ---------------------------------------------------------------------------


class CommRecorder:
    """Thread-safe recorder for distributed communication events."""

    def __init__(self, rank: int = 0) -> None:
        self._lock = threading.Lock()
        self._counter: int = 0
        self.events: list[dict[str, Any]] = []
        self.rank: int = rank
        self.logical_clock: int = 0
        # Mutable context fields updated by callers
        self.current_pp_stage: int | None = None
        self.current_microbatch: int | None = None
        self.current_phase: str = "forward"

    def _next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"comm_{self._counter:07d}"

    def _meta_or_none(self, t: Any) -> dict[str, Any] | None:
        if t is None or not isinstance(t, torch.Tensor):
            return None
        try:
            return TensorMeta.from_tensor(t).to_dict()
        except Exception:
            return None

    def _tensor_ids(self, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            return [id(value)]
        if isinstance(value, (list, tuple)):
            out: list[int] = []
            for v in value:
                if isinstance(v, torch.Tensor):
                    out.append(id(v))
            return out
        return []

    def _group_size(self, group: Any) -> int:
        try:
            if group is None:
                return dist.get_world_size()
            return dist.get_world_size(group)
        except Exception:
            return -1

    def record_collective(
        self,
        op: str,
        tensor: Any,
        group: Any,
        *,
        output_tensor: Any = None,
        **extra: Any,
    ) -> dict[str, Any]:
        input_tensor_ids = self._tensor_ids(tensor)
        output_tensor_ids = self._tensor_ids(output_tensor)
        source_node_ids: list[str] = []
        recorder = get_current_recorder()
        if recorder is not None:
            seen = set()
            for t in [tensor] if isinstance(tensor, torch.Tensor) else []:
                producer = recorder.get_producer(t)
                if producer is not None and producer not in seen:
                    source_node_ids.append(producer)
                    seen.add(producer)
            if isinstance(tensor, (list, tuple)):
                for t in tensor:
                    if isinstance(t, torch.Tensor):
                        producer = recorder.get_producer(t)
                        if producer is not None and producer not in seen:
                            source_node_ids.append(producer)
                            seen.add(producer)
        event: dict[str, Any] = {
            "event_id": self._next_id(),
            "op": op,
            "tensor_meta": self._meta_or_none(tensor),
            "group_size": self._group_size(group),
            "rank": self.rank,
            "pp_stage": self.current_pp_stage,
            "microbatch": self.current_microbatch,
            "phase": self.current_phase,
            "logical_clock": self.logical_clock,
            "input_tensor_ids": input_tensor_ids,
            "output_tensor_ids": output_tensor_ids,
            "source_node_ids": source_node_ids,
            **extra,
        }
        with self._lock:
            self.events.append(event)
            self.logical_clock += 1
        return event

    def record_p2p(
        self,
        op: str,
        tensor: Any,
        peer: int,
        group: Any,
        tag: int = 0,
    ) -> dict[str, Any]:
        input_tensor_ids = self._tensor_ids(tensor)
        source_node_ids: list[str] = []
        recorder = get_current_recorder()
        if recorder is not None and isinstance(tensor, torch.Tensor):
            producer = recorder.get_producer(tensor)
            if producer is not None:
                source_node_ids.append(producer)
        event: dict[str, Any] = {
            "event_id": self._next_id(),
            "op": op,
            "tensor_meta": self._meta_or_none(tensor),
            "peer": peer,
            "group_size": self._group_size(group),
            "rank": self.rank,
            "pp_stage": self.current_pp_stage,
            "microbatch": self.current_microbatch,
            "phase": self.current_phase,
            "tag": tag,
            "logical_clock": self.logical_clock,
            "input_tensor_ids": input_tensor_ids,
            "output_tensor_ids": [],
            "source_node_ids": source_node_ids,
        }
        with self._lock:
            self.events.append(event)
            self.logical_clock += 1
        return event


# ---------------------------------------------------------------------------
# Functional-collectives intercept (torch.distributed._functional_collectives)
# ---------------------------------------------------------------------------


def _try_patch_functional_collectives(
    recorder: CommRecorder,
) -> list[tuple[Any, str, Any]]:
    """
    Patch the functional-collectives module used by FSDP and DTensor.
    Returns a list of (module, attr_name, original_fn) for later restoration.
    """
    try:
        import torch.distributed._functional_collectives as funcol
    except ImportError:
        return []

    saved: list[tuple[Any, str, Any]] = []

    def _wrap(orig_fn: Any, op_name: str, is_p2p: bool = False) -> Any:
        def wrapper(tensor, *args, **kwargs):
            group = kwargs.get("group") or (args[0] if args else None)
            try:
                if is_p2p:
                    peer = args[0] if args else kwargs.get("dst", kwargs.get("src", -1))
                    recorder.record_p2p(op_name, tensor, peer, group)
                else:
                    recorder.record_collective(op_name, tensor, group)
            except Exception:
                pass
            return orig_fn(tensor, *args, **kwargs)

        return wrapper

    # Map of (attribute_name, canonical_op_name, is_p2p)
    patches = [
        ("all_reduce", "all_reduce", False),
        ("all_gather_tensor", "all_gather", False),
        ("reduce_scatter_tensor", "reduce_scatter", False),
        ("all_to_all_single", "all_to_all", False),
        ("broadcast", "broadcast", False),
        # wait_tensor is a no-op comm marker
        ("wait_tensor", "wait_tensor", False),
    ]

    for attr, op_name, is_p2p in patches:
        orig = getattr(funcol, attr, None)
        if orig is not None:
            saved.append((funcol, attr, orig))
            setattr(funcol, attr, _wrap(orig, op_name, is_p2p))

    return saved


# ---------------------------------------------------------------------------
# Main context manager
# ---------------------------------------------------------------------------


@contextmanager
def capture_comms(
    recorder: CommRecorder,
) -> Generator[CommRecorder, None, None]:
    """
    Context manager that patches ``torch.distributed`` and
    ``torch.distributed._functional_collectives`` to record every
    communication operation into *recorder*.

    Args:
        recorder: Target :class:`CommRecorder`.

    Yields:
        The same *recorder* for convenience.
    """
    # --- Save originals ---
    orig_all_reduce = dist.all_reduce
    orig_all_gather = dist.all_gather
    orig_all_gather_into_tensor = dist.all_gather_into_tensor
    orig_reduce_scatter = dist.reduce_scatter
    orig_reduce_scatter_tensor = dist.reduce_scatter_tensor
    orig_all_to_all = dist.all_to_all
    orig_all_to_all_single = dist.all_to_all_single
    orig_send = dist.send
    orig_recv = dist.recv
    orig_isend = dist.isend
    orig_irecv = dist.irecv
    orig_broadcast = dist.broadcast
    orig_barrier = dist.barrier

    # --- Define patched versions ---

    def _all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        recorder.record_collective(
            "all_reduce", tensor, group, reduce_op=str(op), async_op=async_op
        )
        return orig_all_reduce(tensor, op=op, group=group, async_op=async_op)

    def _all_gather(tensor_list, tensor, group=None, async_op=False):
        recorder.record_collective("all_gather", tensor, group, async_op=async_op)
        return orig_all_gather(tensor_list, tensor, group=group, async_op=async_op)

    def _all_gather_into_tensor(
        output_tensor, input_tensor, group=None, async_op=False
    ):
        ev = recorder.record_collective(
            "all_gather_into_tensor",
            input_tensor,
            group,
            output_tensor=output_tensor,
            output_shape=list(output_tensor.shape),
            async_op=async_op,
        )
        out = orig_all_gather_into_tensor(
            output_tensor, input_tensor, group=group, async_op=async_op
        )
        active = get_current_recorder()
        if active is not None:
            active.set_producer(output_tensor, ev["event_id"])
        return out

    def _reduce_scatter(
        output, input_list, op=dist.ReduceOp.SUM, group=None, async_op=False
    ):
        tensor = input_list[0] if input_list else None
        ev = recorder.record_collective(
            "reduce_scatter",
            tensor,
            group,
            output_tensor=output,
            reduce_op=str(op),
            async_op=async_op,
        )
        out = orig_reduce_scatter(
            output, input_list, op=op, group=group, async_op=async_op
        )
        active = get_current_recorder()
        if active is not None:
            active.set_producer(output, ev["event_id"])
        return out

    def _reduce_scatter_tensor(
        output, input_tensor, op=dist.ReduceOp.SUM, group=None, async_op=False
    ):
        ev = recorder.record_collective(
            "reduce_scatter_tensor",
            input_tensor,
            group,
            output_tensor=output,
            output_shape=list(output.shape),
            reduce_op=str(op),
            async_op=async_op,
        )
        out = orig_reduce_scatter_tensor(
            output, input_tensor, op=op, group=group, async_op=async_op
        )
        active = get_current_recorder()
        if active is not None:
            active.set_producer(output, ev["event_id"])
        return out

    def _all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
        tensor = input_tensor_list[0] if input_tensor_list else None
        output_tensor = output_tensor_list[0] if output_tensor_list else None
        ev = recorder.record_collective(
            "all_to_all",
            tensor,
            group,
            output_tensor=output_tensor,
            async_op=async_op,
        )
        out = orig_all_to_all(
            output_tensor_list, input_tensor_list, group=group, async_op=async_op
        )
        active = get_current_recorder()
        if active is not None:
            for t in output_tensor_list:
                active.set_producer(t, ev["event_id"])
        return out

    def _all_to_all_single(output, input, *args, group=None, async_op=False, **kwargs):
        ev = recorder.record_collective(
            "all_to_all_single",
            input,
            group,
            output_tensor=output,
            async_op=async_op,
        )
        out = orig_all_to_all_single(
            output, input, *args, group=group, async_op=async_op, **kwargs
        )
        active = get_current_recorder()
        if active is not None:
            active.set_producer(output, ev["event_id"])
        return out

    def _send(tensor, dst, group=None, tag=0):
        recorder.record_p2p("send", tensor, dst, group, tag=tag)
        return orig_send(tensor, dst, group=group, tag=tag)

    def _recv(tensor, src=None, group=None, tag=0):
        ev = recorder.record_p2p(
            "recv", tensor, src if src is not None else -1, group, tag=tag
        )
        out = orig_recv(tensor, src=src, group=group, tag=tag)
        active = get_current_recorder()
        if active is not None:
            active.set_producer(tensor, ev["event_id"])
        return out

    def _isend(tensor, dst, group=None, tag=0):
        recorder.record_p2p("isend", tensor, dst, group, tag=tag)
        return orig_isend(tensor, dst, group=group, tag=tag)

    def _irecv(tensor, src=None, group=None, tag=0):
        recorder.record_p2p(
            "irecv", tensor, src if src is not None else -1, group, tag=tag
        )
        return orig_irecv(tensor, src=src, group=group, tag=tag)

    def _broadcast(tensor, src=0, group=None, async_op=False):
        recorder.record_collective(
            "broadcast", tensor, group, src=src, async_op=async_op
        )
        return orig_broadcast(tensor, src=src, group=group, async_op=async_op)

    def _barrier(group=None, async_op=False, device_ids=None):
        recorder.record_collective("barrier", None, group, async_op=async_op)
        return orig_barrier(group=group, async_op=async_op, device_ids=device_ids)

    # --- Apply patches ---
    dist.all_reduce = _all_reduce
    dist.all_gather = _all_gather
    dist.all_gather_into_tensor = _all_gather_into_tensor
    dist.reduce_scatter = _reduce_scatter
    dist.reduce_scatter_tensor = _reduce_scatter_tensor
    dist.all_to_all = _all_to_all
    dist.all_to_all_single = _all_to_all_single
    dist.send = _send
    dist.recv = _recv
    dist.isend = _isend
    dist.irecv = _irecv
    dist.broadcast = _broadcast
    dist.barrier = _barrier

    # Also patch functional_collectives
    funcol_saved = _try_patch_functional_collectives(recorder)

    try:
        yield recorder
    finally:
        # --- Restore originals ---
        dist.all_reduce = orig_all_reduce
        dist.all_gather = orig_all_gather
        dist.all_gather_into_tensor = orig_all_gather_into_tensor
        dist.reduce_scatter = orig_reduce_scatter
        dist.reduce_scatter_tensor = orig_reduce_scatter_tensor
        dist.all_to_all = orig_all_to_all
        dist.all_to_all_single = orig_all_to_all_single
        dist.send = orig_send
        dist.recv = orig_recv
        dist.isend = orig_isend
        dist.irecv = orig_irecv
        dist.broadcast = orig_broadcast
        dist.barrier = orig_barrier

        for mod, attr, orig_fn in funcol_saved:
            setattr(mod, attr, orig_fn)
