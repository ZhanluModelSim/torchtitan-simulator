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

from typing import Any

from .cost_estimators import (
    _DEFAULT_MOCK_COMM_GB_PER_S,
    _DEFAULT_MOCK_COMM_LATENCY_US,
    _DEFAULT_MOCK_GB_PER_S,
    _DEFAULT_MOCK_TFLOPS,
    _estimate_bytes,
    _estimate_comm_bytes,
    _estimate_flops,
    _numel,
    _tensor_bytes,
    FixedOverlap,
    NoOverlap,
    OverlapStrategy,
)
from .nodes import ComputeGraph, OpNode, PerfResult, SimulationResult
from .schedule_analysis import (
    _critical_path_time_us,
    link_schedule_to_graph,
    predict_multi_rank_step_time_us,
)

__all__ = [
    "CostModel",
    "MockCostModel",
    "apply_cost_model",
    "OverlapStrategy",
    "NoOverlap",
    "FixedOverlap",
    "_estimate_flops",
    "_estimate_bytes",
    "_estimate_comm_bytes",
    "_numel",
    "_tensor_bytes",
    "_DEFAULT_MOCK_TFLOPS",
    "_DEFAULT_MOCK_GB_PER_S",
    "_DEFAULT_MOCK_COMM_GB_PER_S",
    "_DEFAULT_MOCK_COMM_LATENCY_US",
    "link_schedule_to_graph",
    "predict_multi_rank_step_time_us",
    "_critical_path_time_us",
]


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
            compute_time_us = flops / (self.tflops * 1e6)
            total_bytes = bytes_read + bytes_written
            if total_bytes > 0:
                ai = flops / total_bytes if total_bytes > 0 else float("inf")
                mem_bound_time_us = total_bytes / (self.gb_per_s * 1e3)
                if ai < self.arithmetic_intensity_threshold:
                    compute_time_us = max(compute_time_us, mem_bound_time_us)

        elif node.op_type in ("comm_collective", "comm_p2p") and comm_bytes > 0:
            comm_time_us = self.comm_latency_us + comm_bytes / (
                self.comm_gb_per_s * 1e3
            )
            if node.comm_op == "all_reduce" and node.comm_group_size:
                gs = max(node.comm_group_size, 1)
                comm_time_us *= 2 * (gs - 1) / gs

        elif node.op_type in ("data_move", "memory"):
            total_bytes = bytes_read + bytes_written
            if total_bytes > 0:
                compute_time_us = total_bytes / (self.gb_per_s * 1e3)

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

    e2e_step = predict_multi_rank_step_time_us(result, cost_model)

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
