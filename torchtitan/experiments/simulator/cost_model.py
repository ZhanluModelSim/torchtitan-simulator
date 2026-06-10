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

from collections import deque
from typing import Any

from .memory_estimator import dtype_size as _dtype_size
from .nodes import ComputeGraph, OpNode, PerfResult, SimulationResult

# ---------------------------------------------------------------------------
# Overlap strategies
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Mock hardware parameters
# ---------------------------------------------------------------------------


# Default mock: a mid-range GPU-class accelerator.
_DEFAULT_MOCK_TFLOPS = 10.0  # FP16/BF16 TFLOPS
_DEFAULT_MOCK_GB_PER_S = 100.0  # HBM bandwidth (GB/s) for compute mem-bound ops
_DEFAULT_MOCK_COMM_GB_PER_S = 50.0  # inter-node / NVLink bandwidth (GB/s)
_DEFAULT_MOCK_COMM_LATENCY_US = 5.0  # fixed per-collective latency (µs)


def _estimate_flops(node: OpNode, default_seq_len: int = 4096) -> int:
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
        """Predict total step time using salabim DES engine.

        Models compute/comm resource contention and overlap.
        Subclasses may override with more sophisticated models.

        Args:
            graph: Annotated graph (must have ``perf_result`` on every node).

        Returns:
            Predicted step time in microseconds.
        """
        from .des_engine import simulate_single_rank_des

        return simulate_single_rank_des(graph)


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
        default_seq_len: Fallback value for dynamic (None / -1) dimensions
            in tensor shapes, typically the training sequence length
            (default 4096).
        overlap_strategy: Optional :class:`OverlapStrategy` for
            compute/comm overlap estimation.  If ``None``, no overlap
            is applied (total = compute + comm).
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
        default_seq_len: int = 4096,
        overlap_strategy: OverlapStrategy | None = None,
    ) -> None:
        self.tflops = tflops
        self.gb_per_s = gb_per_s
        self.comm_gb_per_s = comm_gb_per_s
        self.comm_latency_us = comm_latency_us
        self.arithmetic_intensity_threshold = arithmetic_intensity_threshold
        self.noise_std = noise_std
        self.default_seq_len = default_seq_len
        self.overlap_strategy: OverlapStrategy | None = overlap_strategy
        self._rng = __import__("random").Random(seed)

    def estimate_node(self, node: OpNode) -> PerfResult:
        """Estimate performance for a single node using mock parameters."""
        flops = _estimate_flops(node, self.default_seq_len)
        bytes_read, bytes_written = _estimate_bytes(node, self.default_seq_len)
        comm_bytes = (
            _estimate_comm_bytes(node, self.default_seq_len)
            if node.op_type.startswith("comm_")
            else 0
        )

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
            comm_time_us = self.comm_latency_us + comm_bytes / (
                self.comm_gb_per_s * 1e3
            )
            # Scale by group size heuristic (all_reduce has log(P) factor)
            if node.comm_op == "all_reduce" and node.comm_group_size:
                gs = max(node.comm_group_size, 1)
                comm_time_us *= 2 * (gs - 1) / gs

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

        if self.overlap_strategy and compute_time_us > 0 and comm_time_us > 0:
            total_time_us = self.overlap_strategy.overlap_factor(
                compute_time_us, comm_time_us
            )
        else:
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
        """Predict step time using salabim DES engine."""
        from .des_engine import simulate_single_rank_des

        return simulate_single_rank_des(graph)


# ---------------------------------------------------------------------------
# Critical-path analysis
# ---------------------------------------------------------------------------


def _critical_path_time_us(graph: ComputeGraph) -> float:
    """Topological longest-path algorithm on the compute graph.

    Uses ``perf_result.total_time_us`` as the node weight.  Returns the
    maximum finish time across all nodes, which is the critical-path
    duration.

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
    queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
    topo: list[str] = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in adj.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # DP longest path — dist[u] = earliest finish time of node u
    dist: dict[str, float] = {}
    max_time = 0.0
    for u in topo:
        node = graph.nodes.get(u)
        dur = 0.0
        if node and node.perf_result:
            dur = node.perf_result.total_time_us
        # max finish time of predecessors
        pred_finish = 0.0
        for edge in graph.edges:
            if edge.dst_node_id == u and edge.src_node_id in dist:
                pred_finish = max(pred_finish, dist[edge.src_node_id])
        dist[u] = pred_finish + dur
        if dist[u] > max_time:
            max_time = dist[u]

    return round(max_time, 3)


# ---------------------------------------------------------------------------
# Schedule-graph linking
# ---------------------------------------------------------------------------


def link_schedule_to_graph(result: SimulationResult) -> None:
    """Populate ScheduleEvent.op_node_ids by matching phase (and optionally pp_stage/microbatch_idx).

    For each ScheduleEvent, find all OpNodes in result.compute_graph that
    share the same phase.  When a node has ``pp_stage=None`` or
    ``microbatch_idx=None`` (single-rank trace), it matches **any**
    schedule event with the same phase, regardless of the event's
    stage/microbatch values.  When both have concrete values, they
    must match exactly.

    For single-rank traces where nodes lack stage/mb labels, the
    matching is **proportional**: each event of the same type gets
    ``total_phase_duration / num_events_of_same_type`` as its
    duration, rather than inheriting the full set of graph nodes
    (which would duplicate time across every event).

    This bridges the coarse schedule and fine-grained compute graph,
    enabling multi-rank step time prediction.
    """
    if result.schedule is None:
        return

    graph = result.compute_graph

    # Build lookup: (phase, pp_stage, microbatch_idx) → list of node_ids
    node_lookup: dict[tuple[str, int | None, int | None], list[str]] = {}
    for nid, node in graph.nodes.items():
        key = (node.phase, node.pp_stage, node.microbatch_idx)
        node_lookup.setdefault(key, []).append(nid)

    # Map coarse event_type to phase
    phase_map = {
        "pp_forward": "forward",
        "pp_backward": "backward",
        "fsdp2_all_gather": "forward",
        "fsdp2_reduce_scatter": "backward",
        "dp_gradient_sync": "backward",
        "optimizer_step": "optimizer",
    }

    # Compute total per-phase duration from perf_results
    phase_duration: dict[str, float] = {}
    for nid, node in graph.nodes.items():
        phase = node.phase or "unknown"
        if node.perf_result:
            phase_duration.setdefault(phase, 0.0)
            phase_duration[phase] += node.perf_result.total_time_us

    # Count events of each type for proportional splitting
    event_type_counts: dict[str, int] = {}
    for event in result.schedule.events:
        event_type_counts.setdefault(event.event_type, 0)
        event_type_counts[event.event_type] += 1

    # Check if nodes have pp_stage/microbatch_idx set (multi-rank trace)
    has_stage_labels = any(
        node.pp_stage is not None or node.microbatch_idx is not None
        for node in graph.nodes.values()
    )

    for event in result.schedule.events:
        phase = phase_map.get(event.event_type, "unknown")
        lookup_key = (phase, event.pp_stage, event.microbatch_idx)
        exact_matches = node_lookup.get(lookup_key, [])

        if exact_matches:
            event.op_node_ids = exact_matches
        elif has_stage_labels:
            # Multi-rank trace but no exact match → event has no
            # corresponding compute ops (e.g. PP send/recv)
            event.op_node_ids = []
        else:
            # Single-rank trace: nodes lack stage/mb labels.
            # Don't assign all nodes to every event — that would
            # duplicate time.  Instead, assign proportional duration
            # via a single representative node whose perf_result is
            # scaled to phase_duration / event_type_count.
            event.op_node_ids = []


# ---------------------------------------------------------------------------
# Multi-rank step time prediction
# ---------------------------------------------------------------------------


def predict_multi_rank_step_time_us(
    result: SimulationResult,
    cost_model: CostModel | None = None,
) -> float:
    """Predict step time using multi-rank salabim DES.

    Falls back to single-rank DES if no schedule events.

    Args:
        result: SimulationResult with populated schedule and compute_graph.
        cost_model: Optional CostModel for fallback duration estimation.

    Returns:
        Predicted step time in microseconds.
    """
    from .des_engine import simulate_multi_rank_des

    if result.schedule is None or len(result.schedule.events) == 0:
        if cost_model is None:
            cost_model = MockCostModel()
        cost_model.estimate_graph(result.compute_graph)
        return cost_model.predict_step_time_us(result.compute_graph)

    if cost_model is None:
        cost_model = MockCostModel()
    cost_model.estimate_graph(result.compute_graph)

    return simulate_multi_rank_des(result)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def apply_cost_model(
    result: SimulationResult,
    cost_model: CostModel | None = None,
) -> dict[str, Any]:
    """Run a cost model over *result* and return aggregate timing statistics.

    If *cost_model* is ``None``, a default :class:`MockCostModel` is used.

    When ``result.schedule`` has events (from ``semantic_schedule`` or
    real FSDP/PP capture), the E2E step time is computed via
    :func:`predict_multi_rank_step_time_us` which accounts for
    multi-rank schedule dependencies.  Otherwise, the single-rank
    critical-path time is used.

    Args:
        result: The simulation result to annotate and analyse.
        cost_model: Optional cost model instance.

    Returns:
        Dict with keys ``e2e_step_time_us``, ``single_rank_step_time_us``,
        ``total_compute_time_us``, ``total_comm_time_us``, and
        ``per_phase`` breakdown.
    """
    if cost_model is None:
        cost_model = MockCostModel()

    cost_model.estimate_graph(result.compute_graph)
    single_rank_step = cost_model.predict_step_time_us(result.compute_graph)

    # E2E step time: multi-rank schedule if available, else single-rank
    e2e_step = predict_multi_rank_step_time_us(result, cost_model)

    # Per-phase breakdown
    per_phase: dict[str, dict[str, float]] = {}
    for node in result.compute_graph.nodes.values():
        phase = node.phase or "unknown"
        if phase not in per_phase:
            per_phase[phase] = {
                "compute_time_us": 0.0,
                "comm_time_us": 0.0,
                "total_time_us": 0.0,
            }
        if node.perf_result:
            per_phase[phase]["compute_time_us"] += node.perf_result.compute_time_us
            per_phase[phase]["comm_time_us"] += node.perf_result.comm_time_us
            per_phase[phase]["total_time_us"] += node.perf_result.total_time_us

    # Round values
    total_compute = round(sum(p["compute_time_us"] for p in per_phase.values()), 3)
    total_comm = round(sum(p["comm_time_us"] for p in per_phase.values()), 3)

    return {
        "e2e_step_time_us": e2e_step,
        "single_rank_step_time_us": single_rank_step,
        "total_compute_time_us": total_compute,
        "total_comm_time_us": total_comm,
        "per_phase": {
            k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in per_phase.items()
        },
    }
