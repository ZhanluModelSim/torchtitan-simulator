# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified op classification logic shared between FX tracing and dispatch
interception.

Both ``fx_capture.py`` and ``dispatch_interceptor.py`` previously maintained
their own copies of marker lists and classification functions with subtle
differences (e.g. ``"broadcast_"`` vs ``"broadcast"``).  This module
consolidates them into a single source of truth so that every compute-graph
node receives the same ``op_type`` and ``comm_op`` labels regardless of which
capture path produced it.

Usage::

    from .op_classification import classify_op, is_trivial, TRIVIAL_TARGETS

    op_type, comm_op = classify_op("aten.addmm.default")
    # ("compute", None)

    op_type, comm_op = classify_op("c10d_functional.all_reduce")
    # ("comm_collective", "all_reduce")
"""

from __future__ import annotations

_COMM_MARKERS = (
    "_c10d_functional",
    "c10d_functional",
    "c10d.",
    "all_reduce",
    "all_gather",
    "allgather",
    "_allgather_base",
    "reduce_scatter",
    "_reduce_scatter_base",
    "all_to_all",
    "broadcast",
    "wait_tensor",
    "barrier",
)

_P2P_MARKERS = ("_send", "_recv", ".send", ".recv")

_DATA_MOVE_MARKERS = ("_to_copy", "copy_", ".to.")

_MEMORY_MARKERS = (
    "aten.empty",
    "aten.zeros",
    "aten.ones",
    "aten.full",
    "aten.arange",
    "aten.rand",
)

TRIVIAL_TARGETS = frozenset(
    [
        "aten.detach.default",
        "aten.detach_.default",
        "aten.alias.default",
        "aten.t.default",
        "aten.as_strided.default",
        "aten._unsafe_view.default",
        "aten.view.default",
        "aten.lift_fresh_copy.default",
        "aten.lift.default",
    ]
)

COMM_OP_MAP: list[tuple[str, str]] = [
    ("reduce_scatter", "reduce_scatter"),
    ("reduce_scatter_base", "reduce_scatter"),
    ("all_gather", "all_gather"),
    ("allgather", "all_gather"),
    ("all_gather_base", "all_gather"),
    ("all_reduce", "all_reduce"),
    ("all_to_all", "all_to_all"),
    ("broadcast", "broadcast"),
    ("wait_tensor", "wait"),
    ("barrier", "barrier"),
    ("_send", "send"),
    ("_recv", "recv"),
]


def classify_op(target: str) -> tuple[str, str | None]:
    """Return ``(op_type, comm_op_or_None)`` for any op target string.

    The classification rules are applied in priority order:

    1.  P2P markers → ``"comm_p2p"``
    2.  Collective markers → ``"comm_collective"``
    3.  Data-move markers → ``"data_move"``
    4.  Memory/allocation markers → ``"memory"``
    5.  Everything else → ``"compute"``

    When a comm marker matches, ``comm_op`` is the canonical name from
    ``COMM_OP_MAP``; if no specific canonical name matches, a generic
    ``"p2p_unknown"`` or ``"collective_unknown"`` is returned.
    """
    if any(m in target for m in _P2P_MARKERS):
        for substr, canonical in COMM_OP_MAP:
            if substr in target:
                return "comm_p2p", canonical
        return "comm_p2p", "p2p_unknown"

    if any(m in target for m in _COMM_MARKERS):
        for substr, canonical in COMM_OP_MAP:
            if substr in target:
                return "comm_collective", canonical
        return "comm_collective", "collective_unknown"

    if any(m in target for m in _DATA_MOVE_MARKERS):
        return "data_move", None

    if any(target.startswith(m) or m in target for m in _MEMORY_MARKERS):
        return "memory", None

    return "compute", None


def is_trivial(target: str) -> bool:
    """Return ``True`` if *target* is a trivial op that should be skipped."""
    return target in TRIVIAL_TARGETS
