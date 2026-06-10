# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from .memory_estimator import dtype_size as _dtype_size
from .nodes import OpNode


class OverlapStrategy:
    """Base class for compute/comm overlap estimation strategies."""

    def overlap_factor(self, compute_us: float, comm_us: float) -> float:
        """Return effective total time given compute and comm durations."""
        raise NotImplementedError


class NoOverlap(OverlapStrategy):
    """No overlap: total = compute + comm."""

    def overlap_factor(self, compute_us: float, comm_us: float) -> float:
        return compute_us + comm_us


class FixedOverlap(OverlapStrategy):
    """Fixed-ratio overlap: total = compute + max(0, comm - compute * factor)."""

    def __init__(self, factor: float = 0.5) -> None:
        self.factor = factor

    def overlap_factor(self, compute_us: float, comm_us: float) -> float:
        return compute_us + max(0.0, comm_us - compute_us * self.factor)


_DEFAULT_MOCK_TFLOPS = 10.0
_DEFAULT_MOCK_GB_PER_S = 100.0
_DEFAULT_MOCK_COMM_GB_PER_S = 50.0
_DEFAULT_MOCK_COMM_LATENCY_US = 5.0


def _estimate_flops(node: OpNode, default_seq_len: int = 4096) -> int:
    """Heuristic FLOPs estimate from op name and input/output shapes.

    Uses lightweight rules for the most common ATen ops.  Returns 0 for ops
    that cannot be estimated (comm, memory, data-move, etc.).
    """
    op = node.op_name
    if node.op_type not in ("compute",):
        return 0

    in_shapes = [t.shape for t in node.inputs]
    out_shapes = [t.shape for t in node.outputs]

    if any(kw in op for kw in ("mm", "matmul", "bmm", "baddbmm", "addmm", "linear")):
        left_idx = 1 if "addmm" in op else 0
        right_idx = 2 if "addmm" in op else 1
        if (
            len(in_shapes) > max(left_idx, right_idx)
            and len(in_shapes[left_idx]) >= 2
            and len(in_shapes[right_idx]) >= 2
        ):
            batch_dims = (
                _numel(in_shapes[left_idx][:-2]) if len(in_shapes[left_idx]) > 2 else 1
            )
            M = in_shapes[left_idx][-2]
            K = in_shapes[left_idx][-1]
            N = in_shapes[right_idx][-1]
            return 2 * batch_dims * M * K * N
        total = 0
        for out in out_shapes:
            if len(out) >= 2:
                total += 2 * _numel(out)
        return total

    if "scaled_dot_product_attention" in op or "flash_attention" in op:
        if len(in_shapes) >= 3:
            q, k, v = in_shapes[0], in_shapes[1], in_shapes[2]
            if len(q) >= 3 and len(k) >= 3:
                flops_qk = 2 * _numel(q[:-1]) * k[-2]
                flops_av = 2 * _numel(q[:-1]) * v[-1]
                return flops_qk + flops_av
        return 0

    if "conv" in op:
        total = 0
        for out in out_shapes:
            total += 2 * _numel(out)
        for inp in in_shapes:
            if len(inp) >= 2:
                total *= max(1, inp[1])
        return total

    flops_per_elem = 0
    if "gelu" in op or "silu" in op or "swish" in op:
        flops_per_elem = 5
    elif "tanh" in op:
        flops_per_elem = 5
    elif "sigmoid" in op:
        flops_per_elem = 5
    elif "exp" in op:
        flops_per_elem = 4
    elif "sqrt" in op or "rsqrt" in op:
        flops_per_elem = 3
    elif op.startswith("aten.add") or op == "add":
        flops_per_elem = 1
    elif op.startswith("aten.mul") or op == "mul":
        flops_per_elem = 1
    elif op.startswith("aten.div") or op == "div":
        flops_per_elem = 1
    elif op.startswith("aten.sub") or op == "sub":
        flops_per_elem = 1
    elif "norm" in op or "rms_norm" in op or "layer_norm" in op:
        flops_per_elem = 5
    elif "softmax" in op:
        flops_per_elem = 5
    else:
        flops_per_elem = 2

    total = 0
    for out in out_shapes:
        total += flops_per_elem * _numel(out)
    return total


def _estimate_bytes(node: OpNode, default_seq_len: int = 4096) -> tuple[int, int]:
    """Estimate bytes read / written from tensor shapes.

    Returns:
        (bytes_read, bytes_written)
    """
    bytes_read = 0
    for inp in node.inputs:
        bytes_read += _tensor_bytes(inp.shape, inp.dtype, default_seq_len)
    bytes_written = 0
    for out in node.outputs:
        bytes_written += _tensor_bytes(out.shape, out.dtype, default_seq_len)
    return bytes_read, bytes_written


def _estimate_comm_bytes(node: OpNode, default_seq_len: int = 4096) -> int:
    """Estimate bytes communicated by a collective or P2P op."""
    if node.comm_op == "reduce_scatter":
        total = 0
        for inp in node.inputs:
            total += _tensor_bytes(inp.shape, inp.dtype, default_seq_len)
        return total
    if node.comm_op == "all_gather":
        total = 0
        for out in node.outputs:
            total += _tensor_bytes(out.shape, out.dtype, default_seq_len)
        return total
    total = 0
    for out in node.outputs:
        total += _tensor_bytes(out.shape, out.dtype, default_seq_len)
    if total == 0:
        for inp in node.inputs:
            total += _tensor_bytes(inp.shape, inp.dtype, default_seq_len)
    return total


def _numel(shape: tuple[int, ...], default_seq_len: int = 4096) -> int:
    """Product of shape dimensions, handling dynamic dims (None or -1).

    Dynamic dimensions (commonly sequence length in LLM training) are
    replaced by *default_seq_len* rather than a hardcoded constant.
    """
    prod = 1
    for d in shape:
        if d is None or d < 0:
            prod *= default_seq_len
        else:
            prod *= d
    return prod


def _tensor_bytes(
    shape: tuple[int, ...], dtype: str, default_seq_len: int = 4096
) -> int:
    size = _dtype_size(dtype)
    return _numel(shape, default_seq_len) * (size if size > 0 else 2)
