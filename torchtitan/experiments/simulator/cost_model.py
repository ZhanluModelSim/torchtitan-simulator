# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Cost model for annotating :class:`OpNode` instances with performance estimates.

The module provides:

* :class:`CostModel` — abstract base class defining the estimation interface.
* :class:`MockCostModel` — concrete implementation that produces synthetic
  performance numbers from tensor shapes and mock hardware parameters.
* :func:`apply_cost_model` — convenience function that runs a cost model over
  a :class:`SimulationResult` and returns per-phase timing aggregates.
"""

from __future__ import annotations

import math
from typing import Any

from .nodes import ComputeGraph, OpNode, PerfResult, SimulationResult

# ---------------------------------------------------------------------------
# Mock hardware parameters
# ---------------------------------------------------------------------------


# Default mock: a mid-range GPU-class accelerator.
_DEFAULT_MOCK_TFLOPS = 10.0  # FP16/BF16 TFLOPS
_DEFAULT_MOCK_GB_PER_S = 100.0  # HBM bandwidth (GB/s) for compute mem-bound ops
_DEFAULT_MOCK_COMM_GB_PER_S = 50.0  # inter-node / NVLink bandwidth (GB/s)
_DEFAULT_MOCK_COMM_LATENCY_US = 5.0  # fixed per-collective latency (µs)


def _estimate_flops(node: OpNode) -> int:
    """Heuristic FLOPs estimate from op name and input/output shapes.

    Uses lightweight rules for the most common ATen ops.  Returns 0 for ops
    that cannot be estimated (comm, memory, data-move, etc.).
    """
    op = node.op_name
    # Only estimate compute ops
    if node.op_type not in ("compute",):
        return 0

    # Gather shapes from inputs/outputs
    in_shapes = [t.shape for t in node.inputs]
    out_shapes = [t.shape for t in node.outputs]

    # --- matmul-like ---
    if any(kw in op for kw in ("mm", "matmul", "bmm", "baddbmm", "addmm", "linear")):
        # Roughly 2 * M * K * N for each matmul
        total = 0
        for inp in node.inputs:
            s = inp.shape
            if len(s) >= 2:
                total += 2 * _numel(s[:-2]) * s[-2] * s[-1]
        for out in node.outputs:
            s = out.shape
            if len(s) >= 2:
                total += _numel(s)
        return total

    # --- scaled_dot_product_attention ---
    if "scaled_dot_product_attention" in op or "flash_attention" in op:
        if len(in_shapes) >= 3:
            q, k, v = in_shapes[0], in_shapes[1], in_shapes[2]
            # QK^T: 2 * B * H * S * S * D_head   (assume last dim is head_dim)
            if len(q) >= 3 and len(k) >= 3:
                flops_qk = 2 * _numel(q[:-1]) * k[-2]
                flops_av = 2 * _numel(q[:-1]) * v[-1]
                return flops_qk + flops_av
        return 0

    # --- convolution ---
    if "conv" in op:
        total = 0
        for out in out_shapes:
            total += 2 * _numel(out)
        for inp in in_shapes:
            if len(inp) >= 2:
                total *= max(1, inp[1])  # rough kernel scaling
        return total

    # --- element-wise / activation / norm ---
    # Roughly 1-5 FLOPs per output element
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
    elif "div" in op or "mul" in op or "add" in op or "sub" in op:
        flops_per_elem = 1
    elif "norm" in op or "rms_norm" in op or "layer_norm" in op:
        flops_per_elem = 5
    elif "softmax" in op:
        flops_per_elem = 5
    elif "silu" in op or "gelu" in op:
        flops_per_elem = 5
    else:
        # Generic compute op: assume ~2 FLOPs per output element
        flops_per_elem = 2

    total = 0
    for out in out_shapes:
        total += flops_per_elem * _numel(out)
    return total


def _estimate_bytes(node: OpNode) -> tuple[int, int]:
    """Estimate bytes read / written from tensor shapes.

    Returns:
        (bytes_read, bytes_written)
    """
    bytes_read = 0
    for inp in node.inputs:
        bytes_read += _tensor_bytes(inp.shape, inp.dtype)
    bytes_written = 0
    for out in node.outputs:
        bytes_written += _tensor_bytes(out.shape, out.dtype)
    return bytes_read, bytes_written


def _estimate_comm_bytes(node: OpNode) -> int:
    """Estimate bytes communicated by a collective or P2P op."""
    total = 0
    for out in node.outputs:
        total += _tensor_bytes(out.shape, out.dtype)
    if total == 0:
        for inp in node.inputs:
            total += _tensor_bytes(inp.shape, inp.dtype)
    return total


def _numel(shape: tuple[int, ...]) -> int:
    """Product of shape dimensions, handling dynamic dims (None or -1)."""
    prod = 1
    for d in shape:
        if d is None or d < 0:
            # Dynamic dimension: use a reasonable default
            prod *= 1024
        else:
            prod *= d
    return prod


def _tensor_bytes(shape: tuple[int, ...], dtype: str) -> int:
    """Bytes for a tensor of given shape and dtype string."""
    dtype_bytes = {
        "torch.float32": 4,
        "torch.float": 4,
        "torch.float16": 2,
        "torch.half": 2,
        "torch.bfloat16": 2,
        "torch.float8_e4m3fn": 1,
        "torch.float8_e5m2": 1,
        "torch.int64": 8,
        "torch.long": 8,
        "torch.int32": 4,
        "torch.int": 4,
        "torch.int16": 2,
        "torch.short": 2,
        "torch.int8": 1,
        "torch.uint8": 1,
        "torch.bool": 1,
    }
    return _numel(shape) * dtype_bytes.get(dtype, 2)


# ---------------------------------------------------------------------------
# CostModel — abstract base
# ---------------------------------------------------------------------------


class CostModel:
    """Abstract interface for performance cost estimation.

    Subclasses implement hardware-specific models that take an :class:`OpNode`
    (with its tensor shapes, op type, and attributes) and return a
    :class:`PerfResult` with estimated compute/communication time, FLOPs, and
    memory traffic.

    Usage::

        model = MyCostModel(hardware_params)
        model.estimate_graph(result.compute_graph)
        step_time_us = model.predict_step_time_us(result.compute_graph)
    """

    def estimate_node(self, node: OpNode) -> PerfResult:
        """Estimate performance for a single node.

        Args:
            node: The operator node to estimate.

        Returns:
            A :class:`PerfResult` with populated fields.
        """
        raise NotImplementedError("Subclasses must implement estimate_node()")

    def estimate_graph(self, graph: ComputeGraph) -> None:
        """Estimate performance for every node in *graph*.

        Calls :meth:`estimate_node` for each node and stores the result in
        ``node.perf_result``.  Existing ``perf_result`` values are overwritten.

        Args:
            graph: The compute graph to annotate (mutated in-place).
        """
        for node in graph.nodes.values():
            node.perf_result = self.estimate_node(node)

    def estimate_result(self, result: SimulationResult) -> None:
        """Convenience wrapper that calls :meth:`estimate_graph` on
        ``result.compute_graph``.
        """
        self.estimate_graph(result.compute_graph)

    def predict_step_time_us(self, graph: ComputeGraph) -> float:
        """Predict total step time from annotated performance data.

        The default implementation computes the critical-path time through
        the graph using a topological longest-path algorithm.  Subclasses may
        override this with more sophisticated models (e.g. accounting for
        operator fusion, wave-level parallelism, or communication overlap).

        Args:
            graph: Annotated graph (must have ``perf_result`` on every node).

        Returns:
            Predicted step time in microseconds.
        """
        return _critical_path_time_us(graph)


# ---------------------------------------------------------------------------
# MockCostModel
# ---------------------------------------------------------------------------


class MockCostModel(CostModel):
    """Synthetic cost model using mock hardware parameters.

    Estimates:
    * **Compute time**: ``flops / (tflops * 1e6)``  (µs)
    * **Comm time**: ``bytes / (comm_gb_per_s * 1e3) + latency_us`` (µs)
    * **Memory-bound time**: ``bytes / (gb_per_s * 1e3)`` applied when
      arithmetic intensity is below a configurable threshold.

    Args:
        tflops: Mock compute throughput in FP16/BF16 TFLOPS (default 10).
        gb_per_s: Mock HBM bandwidth in GB/s (default 100).
        comm_gb_per_s: Mock inter-device communication bandwidth in GB/s
            (default 50).
        comm_latency_us: Fixed per-collective latency in µs (default 5).
        arithmetic_intensity_threshold: Minimum FLOPs/byte to be considered
            compute-bound (default 10.0).
        noise_std: Standard deviation of Gaussian noise added to each
            estimate as a fraction (default 0.05 = 5%).
        seed: Random seed for reproducible noise (default 42).
    """

    def __init__(
        self,
        tflops: float = _DEFAULT_MOCK_TFLOPS,
        gb_per_s: float = _DEFAULT_MOCK_GB_PER_S,
        comm_gb_per_s: float = _DEFAULT_MOCK_COMM_GB_PER_S,
        comm_latency_us: float = _DEFAULT_MOCK_COMM_LATENCY_US,
        arithmetic_intensity_threshold: float = 10.0,
        noise_std: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.tflops = tflops
        self.gb_per_s = gb_per_s
        self.comm_gb_per_s = comm_gb_per_s
        self.comm_latency_us = comm_latency_us
        self.arithmetic_intensity_threshold = arithmetic_intensity_threshold
        self.noise_std = noise_std
        self._rng = __import__("random").Random(seed)

    def estimate_node(self, node: OpNode) -> PerfResult:
        """Estimate performance for a single node using mock parameters."""
        flops = _estimate_flops(node)
        bytes_read, bytes_written = _estimate_bytes(node)
        comm_bytes = _estimate_comm_bytes(node) if node.op_type.startswith("comm_") else 0

        compute_time_us = 0.0
        comm_time_us = 0.0

        if node.op_type == "compute" and flops > 0:
            # Compute time from FLOPs
            compute_time_us = flops / (self.tflops * 1e6)
            # Also check memory-bound ceiling
            total_bytes = bytes_read + bytes_written
            if total_bytes > 0:
                ai = flops / total_bytes if total_bytes > 0 else float("inf")
                mem_bound_time_us = total_bytes / (self.gb_per_s * 1e3)
                if ai < self.arithmetic_intensity_threshold:
                    # Memory-bound: use the larger of compute and memory time
                    compute_time_us = max(compute_time_us, mem_bound_time_us)

        elif node.op_type in ("comm_collective", "comm_p2p") and comm_bytes > 0:
            comm_time_us = self.comm_latency_us + comm_bytes / (self.comm_gb_per_s * 1e3)
            # Scale by group size heuristic (all_reduce has log(P) factor)
            if node.comm_op == "all_reduce" and node.comm_group_size:
                gs = max(node.comm_group_size, 1)
                comm_time_us *= (1 + math.log2(gs) * 0.5)

        elif node.op_type in ("data_move", "memory"):
            # Data movement: bandwidth-bound
            total_bytes = bytes_read + bytes_written
            if total_bytes > 0:
                compute_time_us = total_bytes / (self.gb_per_s * 1e3)

        # Apply Gaussian noise
        noise = self._rng.gauss(0, self.noise_std)
        compute_time_us *= max(0.01, 1.0 + noise)
        if comm_time_us > 0:
            noise2 = self._rng.gauss(0, self.noise_std)
            comm_time_us *= max(0.01, 1.0 + noise2)

        total_time_us = compute_time_us + comm_time_us

        return PerfResult(
            compute_time_us=round(compute_time_us, 3),
            comm_time_us=round(comm_time_us, 3),
            total_time_us=round(total_time_us, 3),
            flops=flops,
            bytes_read=bytes_read,
            bytes_written=bytes_written,
            metadata={
                "model": "mock",
                "tflops": self.tflops,
                "gb_per_s": self.gb_per_s,
                "comm_gb_per_s": self.comm_gb_per_s,
            },
        )

    def predict_step_time_us(self, graph: ComputeGraph) -> float:
        """Predict step time via critical-path analysis of the annotated graph."""
        return _critical_path_time_us(graph)


# ---------------------------------------------------------------------------
# Critical-path analysis
# ---------------------------------------------------------------------------


def _critical_path_time_us(graph: ComputeGraph) -> float:
    """Topological longest-path algorithm on the compute graph.

    Uses ``perf_result.total_time_us`` as the node weight.  Comm and compute
    are assumed to overlap partially — comm edges only add the comm portion
    of the destination node when both nodes have compute time.

    Returns:
        Longest path duration in microseconds, or 0.0 if graph is unannotated.
    """
    # Build adjacency list
    adj: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
    in_degree: dict[str, int] = {nid: 0 for nid in graph.nodes}
    for edge in graph.edges:
        if edge.src_node_id in adj and edge.dst_node_id in adj:
            adj[edge.src_node_id].append(edge.dst_node_id)
            in_degree[edge.dst_node_id] = in_degree.get(edge.dst_node_id, 0) + 1

    # Topological sort
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    topo: list[str] = []
    while queue:
        u = queue.pop(0)
        topo.append(u)
        for v in adj.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # DP longest path
    dist: dict[str, float] = {}
    max_time = 0.0
    for u in topo:
        node = graph.nodes.get(u)
        if node is None or node.perf_result is None:
            dur = 0.0
        else:
            dur = node.perf_result.total_time_us
        prev = dist.get(u, 0.0)
        dist[u] = prev + dur
        if dist[u] > max_time:
            max_time = dist[u]
        for v in adj.get(u, []):
            # Partial overlap: only add comm portion when crossing a comm edge
            v_node = graph.nodes.get(v)
            edge_cost = dur
            if v_node and v_node.perf_result and v_node.perf_result.comm_time_us > 0:
                # Communication can overlap with upstream compute;
                # only add the portion that isn't overlapped.
                edge_cost = max(0.0, dur - v_node.perf_result.comm_time_us * 0.5)
            dist[v] = max(dist.get(v, 0.0), dist[u] + edge_cost)

    return round(max_time, 3)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def apply_cost_model(
    result: SimulationResult,
    cost_model: CostModel | None = None,
) -> dict[str, Any]:
    """Run a cost model over *result* and return aggregate timing statistics.

    If *cost_model* is ``None``, a default :class:`MockCostModel` is used.

    Args:
        result: The simulation result to annotate and analyse.
        cost_model: Optional cost model instance.

    Returns:
        Dict with keys ``step_time_us``, ``compute_time_us``,
        ``comm_time_us``, and ``per_phase`` breakdown.
    """
    if cost_model is None:
        cost_model = MockCostModel()

    cost_model.estimate_graph(result.compute_graph)
    step_time_us = cost_model.predict_step_time_us(result.compute_graph)

    # Per-phase breakdown
    per_phase: dict[str, dict[str, float]] = {}
    for node in result.compute_graph.nodes.values():
        phase = node.phase or "unknown"
        if phase not in per_phase:
            per_phase[phase] = {"compute_time_us": 0.0, "comm_time_us": 0.0, "total_time_us": 0.0}
        if node.perf_result:
            per_phase[phase]["compute_time_us"] += node.perf_result.compute_time_us
            per_phase[phase]["comm_time_us"] += node.perf_result.comm_time_us
            per_phase[phase]["total_time_us"] += node.perf_result.total_time_us

    # Round values
    total_compute = round(sum(p["compute_time_us"] for p in per_phase.values()), 3)
    total_comm = round(sum(p["comm_time_us"] for p in per_phase.values()), 3)

    return {
        "step_time_us": step_time_us,
        "total_compute_time_us": total_compute,
        "total_comm_time_us": total_comm,
        "per_phase": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in per_phase.items()},
    }
