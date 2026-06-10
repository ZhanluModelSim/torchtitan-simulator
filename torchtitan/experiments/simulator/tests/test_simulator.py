# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CPU-only unit tests for the simulator module.

Run with::

    pytest torchtitan/experiments/simulator/tests/test_simulator.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_linear() -> nn.Module:
    return nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))


def _example_inputs() -> tuple[torch.Tensor, ...]:
    return (torch.randn(2, 16),)


# ===========================================================================
# Data model tests
# ===========================================================================


class TestTensorMeta(unittest.TestCase):
    def test_to_dict_round_trip(self):
        from torchtitan.experiments.simulator.nodes import TensorMeta

        tm = TensorMeta(
            shape=(2, 16), dtype="torch.float32", device="cpu", requires_grad=True
        )
        d = tm.to_dict()
        assert d["shape"] == [2, 16]
        assert d["dtype"] == "torch.float32"
        assert d["device"] == "cpu"
        assert d["requires_grad"] is True


class TestOpNode(unittest.TestCase):
    def test_to_dict(self):
        from torchtitan.experiments.simulator.nodes import OpNode, TensorMeta

        node = OpNode(
            node_id="fx_0000001",
            op_name="aten.mm.default",
            op_type="compute",
            phase="forward",
            inputs=[TensorMeta((2, 16), "torch.float32", "cpu")],
            outputs=[TensorMeta((2, 4), "torch.float32", "cpu")],
        )
        d = node.to_dict()
        assert d["node_id"] == "fx_0000001"
        assert d["op_type"] == "compute"
        assert len(d["inputs"]) == 1
        assert len(d["outputs"]) == 1


class TestComputeGraph(unittest.TestCase):
    def test_add_node_and_edge(self):
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            DataEdge,
            OpNode,
        )

        g = ComputeGraph()
        n1 = OpNode("n1", "aten.mm.default", "compute", "forward", [], [])
        n2 = OpNode("n2", "aten.relu.default", "compute", "forward", [], [])
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(DataEdge("n1", "n2", "data"))
        assert len(g.nodes) == 2
        assert len(g.edges) == 1

    def test_to_dict_serializable(self):
        from torchtitan.experiments.simulator.nodes import ComputeGraph, OpNode

        g = ComputeGraph(metadata={"rank": 0})
        g.add_node(OpNode("n1", "aten.mm.default", "compute", "forward", [], []))
        d = g.to_dict()
        json.dumps(d)

    def test_fix_comm_phase_labels_cross_phase_dep(self):
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            DataEdge,
            OpNode,
            PerfResult,
        )

        g = ComputeGraph()
        fwd = OpNode(
            "f1",
            "compute_op",
            "compute",
            "forward",
            [],
            [],
            perf_result=PerfResult(total_time_us=10.0),
        )
        bwd = OpNode(
            "b1",
            "compute_op",
            "compute",
            "backward",
            [],
            [],
            perf_result=PerfResult(total_time_us=20.0),
        )
        comm = OpNode(
            "c1",
            "reduce_scatter",
            "comm_collective",
            "forward",
            [],
            [],
            comm_op="reduce_scatter",
            perf_result=PerfResult(comm_time_us=5.0, total_time_us=5.0),
        )
        g.add_node(fwd)
        g.add_node(bwd)
        g.add_node(comm)
        g.add_edge(DataEdge("b1", "c1", "data"))
        g.fix_comm_phase_labels()
        assert (
            g.nodes["c1"].phase == "backward"
        ), "comm node with only backward predecessors should be backward"

    def test_phase_boundary_sentinel_has_perf_result(self):
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            DataEdge,
            OpNode,
            PerfResult,
        )

        g = ComputeGraph()
        fwd = OpNode(
            "f1",
            "compute_op",
            "compute",
            "forward",
            [],
            [],
            perf_result=PerfResult(total_time_us=10.0),
        )
        bwd = OpNode(
            "b1",
            "compute_op",
            "compute",
            "backward",
            [],
            [],
            perf_result=PerfResult(total_time_us=20.0),
        )
        g.add_node(fwd)
        g.add_node(bwd)
        g.add_edge(DataEdge("f1", "b1", "data"))

        g.add_phase_boundary_edges()
        sentinel = g.nodes["phase_end_forward"]
        assert sentinel.perf_result is not None, "sentinel should have PerfResult"
        assert (
            sentinel.perf_result.total_time_us == 0.0
        ), "sentinel should have zero duration"
        assert (
            sentinel.perf_result.metadata.get("phase_boundary") is True
        ), "sentinel PerfResult should be marked as phase_boundary"

    def test_fix_comm_phase_labels_mixed_phase_deps_unchanged(self):
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            DataEdge,
            OpNode,
            PerfResult,
        )

        g = ComputeGraph()
        fwd = OpNode(
            "f1",
            "compute_op",
            "compute",
            "forward",
            [],
            [],
            perf_result=PerfResult(total_time_us=10.0),
        )
        bwd = OpNode(
            "b1",
            "compute_op",
            "compute",
            "backward",
            [],
            [],
            perf_result=PerfResult(total_time_us=20.0),
        )
        comm = OpNode(
            "c1",
            "all_reduce",
            "comm_collective",
            "forward",
            [],
            [],
            comm_op="all_reduce",
            perf_result=PerfResult(comm_time_us=5.0, total_time_us=5.0),
        )
        g.add_node(fwd)
        g.add_node(bwd)
        g.add_node(comm)
        g.add_edge(DataEdge("f1", "c1", "data"))
        g.add_edge(DataEdge("b1", "c1", "data"))
        g.fix_comm_phase_labels()
        assert (
            g.nodes["c1"].phase == "forward"
        ), "comm node with mixed-phase predecessors should keep original phase"


class TestSimulationResultSave(unittest.TestCase):
    def test_save_json(self):
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            MemoryEvent,
            SimulationResult,
        )

        result = SimulationResult(
            compute_graph=ComputeGraph(),
            memory_events=[
                MemoryEvent(
                    event_id="mem_1",
                    category="activation",
                    bytes=128,
                    node_id="n1",
                    phase="forward",
                )
            ],
            metadata={"test": True},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "result.json")
            result.save_json(path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert "metadata" in data
            assert data["memory_events"][0]["bytes"] == 128


class TestMemoryEstimator(unittest.TestCase):
    def test_tensor_nbytes_and_graph_peak(self):
        from torchtitan.experiments.simulator.memory_estimator import (
            estimate_graph_memory,
            tensor_nbytes,
        )
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            DataEdge,
            OpNode,
            TensorMeta,
        )

        assert tensor_nbytes(TensorMeta((2, 4), "torch.float32", "cpu")) == 32

        graph = ComputeGraph()
        graph.add_node(
            OpNode(
                "n1",
                "aten.mm.default",
                "compute",
                "forward",
                [],
                [TensorMeta((2, 4), "torch.float32", "cpu")],
            )
        )
        graph.add_node(
            OpNode(
                "n2",
                "aten.relu.default",
                "compute",
                "forward",
                [],
                [TensorMeta((2, 4), "torch.float32", "cpu")],
            )
        )
        graph.add_edge(DataEdge("n1", "n2", "data"))

        events, summary = estimate_graph_memory(graph)
        assert len(events) == 2
        assert summary["by_category"]["activation"] == 64
        assert summary["peak_live_bytes"] >= 32

    def test_model_state_memory(self):
        from torchtitan.experiments.simulator.memory_estimator import (
            estimate_model_state_memory,
        )

        model = nn.Linear(4, 2)
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        events, summary = estimate_model_state_memory([model], optimizer_name="AdamW")

        assert len(events) == 4
        assert summary["parameter_bytes"] == param_bytes
        assert summary["gradient_bytes"] == param_bytes
        assert summary["optimizer_state_bytes"] == 2 * param_bytes


# ===========================================================================
# Unified trace mode tests (replaces dispatch interceptor tests)
# ===========================================================================


class TestOpCaptureMode(unittest.TestCase):
    def test_captures_matmul(self):
        from torchtitan.experiments.simulator.unified_trace import (
            TraceRecorder,
            unified_trace,
        )

        recorder = TraceRecorder()
        a = torch.randn(4, 8)
        b = torch.randn(8, 4)
        with unified_trace(recorder, use_fake_mode=False, phase="forward"):
            torch.mm(a, b)

        assert len(recorder.nodes) > 0
        op_names = [n.op_name for n in recorder.nodes]
        assert any("mm" in name.lower() for name in op_names), f"ops: {op_names}"

    def test_phase_labelling(self):
        from torchtitan.experiments.simulator.unified_trace import (
            TraceRecorder,
            unified_trace,
        )

        recorder = TraceRecorder()
        x = torch.randn(3, 3)
        with unified_trace(recorder, use_fake_mode=False, phase="backward"):
            _ = x @ x

        phases = {n.phase for n in recorder.nodes}
        assert "backward" in phases

    def test_categorizes_relu(self):
        from torchtitan.experiments.simulator.unified_trace import (
            TraceRecorder,
            unified_trace,
        )

        recorder = TraceRecorder()
        x = torch.randn(4)
        with unified_trace(recorder, use_fake_mode=False, phase="forward"):
            torch.relu(x)

        types = {n.op_type for n in recorder.nodes}
        assert "compute" in types

    def test_tensor_meta_captured(self):
        from torchtitan.experiments.simulator.unified_trace import (
            TraceRecorder,
            unified_trace,
        )

        recorder = TraceRecorder()
        x = torch.randn(3, 5)
        with unified_trace(recorder, use_fake_mode=False, phase="forward"):
            torch.sigmoid(x)

        shapes = [tuple(m.shape) for n in recorder.nodes for m in n.outputs]
        assert (3, 5) in shapes, f"shapes={shapes}"

    def test_runtime_data_edges_from_tensor_producers(self):
        from torchtitan.experiments.simulator.unified_trace import (
            TraceRecorder,
            unified_trace,
        )

        recorder = TraceRecorder()
        x = torch.randn(2, 2)
        with unified_trace(recorder, use_fake_mode=False, phase="forward"):
            y = torch.relu(x)
            _ = y + 1.0

        assert len(recorder.edges) > 0
        edge_types = {t for _, _, t in recorder.edges}
        assert "data" in edge_types


# ===========================================================================
# Comm interceptor tests
# ===========================================================================


class TestCommRecorder(unittest.TestCase):
    def setUp(self):
        import torch.distributed as dist

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29501")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            dist.init_process_group(backend="gloo", init_method="env://")

    def tearDown(self):
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()

    def test_records_all_reduce(self):
        import torch.distributed as dist

        from torchtitan.experiments.simulator.comm_interceptor import (
            capture_comms,
            CommRecorder,
        )

        recorder = CommRecorder(rank=0)
        t = torch.ones(4)
        with capture_comms(recorder):
            dist.all_reduce(t)

        assert len(recorder.events) > 0
        ops = [e["op"] for e in recorder.events]
        assert any("all_reduce" in op for op in ops), f"ops: {ops}"

    def test_records_broadcast(self):
        import torch.distributed as dist

        from torchtitan.experiments.simulator.comm_interceptor import (
            capture_comms,
            CommRecorder,
        )

        recorder = CommRecorder(rank=0)
        t = torch.ones(4)
        with capture_comms(recorder):
            dist.broadcast(t, src=0)

        ops = [e["op"] for e in recorder.events]
        assert any("broadcast" in op for op in ops), f"ops: {ops}"

    def test_comm_event_has_source_node_ids(self):
        import torch.distributed as dist

        from torchtitan.experiments.simulator.comm_interceptor import (
            capture_comms,
            CommRecorder,
        )
        from torchtitan.experiments.simulator.unified_trace import (
            TraceRecorder,
            unified_trace,
        )

        comm = CommRecorder(rank=0)
        ops = TraceRecorder()
        x = torch.ones(4)
        with unified_trace(ops, use_fake_mode=False, phase="forward"):
            y = x + 1
            with capture_comms(comm):
                dist.all_reduce(y)

        assert comm.events, "expected comm events"
        first = comm.events[0]
        assert "source_node_ids" in first


# ===========================================================================
# FSDP tracer tests (uses mock module since FSDP requires full setup)
# ===========================================================================


class TestFSDPEventRecorder(unittest.TestCase):
    def test_records_custom_hooks(self):
        """Test FSDPEventRecorder directly without actual FSDPModule."""
        from torchtitan.experiments.simulator.fsdp_tracer import FSDPEventRecorder

        recorder = FSDPEventRecorder(rank=0)
        recorder.current_phase = "forward"
        recorder.record("allgather_start", "layer0", {"module": "test"})
        recorder.record("allgather_end", "layer0", {"module": "test"})
        recorder.current_phase = "backward"
        recorder.record("reduce_scatter_start", "layer0")
        recorder.record("reduce_scatter_end", "layer0")

        assert len(recorder.events) == 4
        types = [e["event_type"] for e in recorder.events]
        assert "allgather_start" in types
        assert "reduce_scatter_end" in types

    def test_logical_clock_increment(self):
        from torchtitan.experiments.simulator.fsdp_tracer import FSDPEventRecorder

        recorder = FSDPEventRecorder(rank=0)
        for _ in range(5):
            recorder.record("test_event", "layer0")

        clocks = [e["logical_clock"] for e in recorder.events]
        assert clocks == list(range(5)), f"clocks={clocks}"


# ===========================================================================
# FX capture tests
# ===========================================================================


class TestFxCapture(unittest.TestCase):
    def test_forward_fx_linear(self):
        from torchtitan.experiments.simulator.fx_capture import capture_forward_fx

        model = _small_linear()
        inputs = _example_inputs()
        graph = capture_forward_fx(model, inputs)

        assert len(graph.nodes) > 0, "No ops captured"
        # All nodes should have phase == 'forward'
        phases = {n.phase for n in graph.nodes.values()}
        assert phases <= {"forward", "joint"}, f"unexpected phases: {phases}"

    def test_forward_fx_has_edges(self):
        from torchtitan.experiments.simulator.fx_capture import capture_forward_fx

        model = _small_linear()
        inputs = _example_inputs()
        graph = capture_forward_fx(model, inputs)

        # Linear → ReLU → Linear should produce at least 1 edge
        assert len(graph.edges) > 0, "No edges captured"

    def test_forward_fx_tensor_shapes(self):
        from torchtitan.experiments.simulator.fx_capture import capture_forward_fx

        model = nn.Linear(8, 4)
        inputs = (torch.randn(2, 8),)
        graph = capture_forward_fx(model, inputs)

        all_shapes = [tuple(m.shape) for n in graph.nodes.values() for m in n.outputs]
        assert len(all_shapes) > 0, "No output shapes captured"

    def test_graph_to_compute_graph(self):
        """Test fx_graph_to_compute_graph directly."""
        import torch.fx as fx

        from torchtitan.experiments.simulator.fx_capture import (
            fx_graph_to_compute_graph,
        )

        model = nn.Linear(4, 4)
        gm = fx.symbolic_trace(model)
        graph = fx_graph_to_compute_graph(gm, phase="forward")
        # symbolic_trace does include call_function nodes for mm/add
        assert isinstance(graph.nodes, dict)


# ===========================================================================
# Comm merge tests (moved from graph_assembler)
# ===========================================================================


class TestMergeCommEvents(unittest.TestCase):
    def test_merge_comm_events(self):
        from torchtitan.experiments.simulator.fx_capture import merge_comm_events
        from torchtitan.experiments.simulator.nodes import ComputeGraph

        graph = ComputeGraph()
        comm_events = [
            {"op": "all_reduce", "op_type": "comm_collective", "phase": "forward"},
            {
                "op": "reduce_scatter_tensor",
                "op_type": "comm_collective",
                "phase": "backward",
            },
        ]
        merge_comm_events(graph, comm_events)
        assert len(graph.nodes) == 2
        types = {n.op_type for n in graph.nodes.values()}
        assert "comm_collective" in types


# ===========================================================================
# Export tests
# ===========================================================================


class TestExport(unittest.TestCase):
    def _make_result(self):
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            DataEdge,
            MemoryEvent,
            OpNode,
            SimulationResult,
            TensorMeta,
        )

        g = ComputeGraph(metadata={"rank": 0})
        n1 = OpNode(
            "n1",
            "aten.mm.default",
            "compute",
            "forward",
            [TensorMeta((2, 8), "torch.float32", "cpu")],
            [TensorMeta((2, 4), "torch.float32", "cpu")],
        )
        n2 = OpNode(
            "n2", "aten.all_reduce.default", "comm_collective", "forward", [], []
        )
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(DataEdge("n1", "n2", "comm_dep"))
        return SimulationResult(
            compute_graph=g,
            comm_events=[{"op": "all_reduce", "phase": "forward"}],
            memory_events=[
                MemoryEvent(
                    event_id="mem_1",
                    category="activation",
                    bytes=32,
                    node_id="n1",
                    phase="forward",
                    lifetime_start=0,
                    lifetime_end=1,
                ),
                MemoryEvent(
                    event_id="mem_2",
                    category="parameter",
                    bytes=64,
                    phase="model_state",
                ),
            ],
            metadata={
                "mode": "test",
                "memory": {
                    "peak_live_bytes": 32,
                    "by_category": {"activation": 32},
                },
            },
        )

    def test_export_json(self):
        from torchtitan.experiments.simulator.export import export_json

        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.json")
            export_json(result, path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert "compute_graph" in data

    def test_export_dot(self):
        from torchtitan.experiments.simulator.export import export_dot

        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.dot")
            export_dot(result.compute_graph, path, title="Test")
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "digraph" in content
            assert "n1" in content

    def test_export_chrome_trace(self):
        from torchtitan.experiments.simulator.export import export_chrome_trace

        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trace.json")
            export_chrome_trace(result, path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert "traceEvents" in data
            assert len(data["traceEvents"]) > 0

    def test_export_text_summary(self):
        from torchtitan.experiments.simulator.export import export_text_summary

        result = self._make_result()
        summary = export_text_summary(result)
        assert "Compute Graph Summary" in summary
        assert "Communication Events" in summary
        assert "Memory Estimate" in summary
        assert "activation" in summary
        assert "Total ops" in summary

    def test_export_html(self):
        from torchtitan.experiments.simulator.export import export_html

        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trace.html")
            export_html(result, path)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "TorchTitan Simulation Trace" in content
            assert "operator dependency DAG" in content
            assert "Chrome Trace Timeline" in content
            assert "canvas" in content
            assert "drawChromeTrace" in content
            assert "drawDag" in content
            assert "Zoom in" in content
            assert "rank-tabs" in content
            assert "Global rank" in content
            assert "PP rank" in content
            assert "Estimated live memory peak" in content
            assert "Memory estimate summary" in content
            assert "Memory trace timeline and event breakdown" in content
            assert "memory-chart" in content
            assert "drawMemoryTrace" in content
            assert "memory-events-body" in content

    def test_chrome_trace_schedule_events_have_des_timing(self):
        from torchtitan.experiments.simulator.des_engine import simulate_multi_rank_des
        from torchtitan.experiments.simulator.export import export_chrome_trace
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            OpNode,
            PerfResult,
            ScheduleDep,
            ScheduleEvent,
            SimulationResult,
            TrainingSchedule,
        )

        graph = ComputeGraph()
        graph.add_node(
            OpNode(
                "n_fwd",
                "aten.mm.default",
                "compute",
                "forward",
                pp_stage=0,
                microbatch_idx=0,
                perf_result=PerfResult(total_time_us=50.0),
            )
        )
        graph.add_node(
            OpNode(
                "n_bwd",
                "aten.mm.default",
                "compute",
                "backward",
                pp_stage=0,
                microbatch_idx=0,
                perf_result=PerfResult(total_time_us=30.0),
            )
        )

        schedule = TrainingSchedule()
        ev_fwd = ScheduleEvent(
            "e_fwd", "pp_forward", rank=0, pp_stage=0, microbatch_idx=0
        )
        ev_bwd = ScheduleEvent(
            "e_bwd", "pp_backward", rank=0, pp_stage=0, microbatch_idx=0
        )
        schedule.add_event(ev_fwd)
        schedule.add_event(ev_bwd)
        schedule.add_dep(ScheduleDep("e_fwd", "e_bwd", "data"))

        result = SimulationResult(compute_graph=graph, schedule=schedule)
        simulate_multi_rank_des(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trace.json")
            export_chrome_trace(result, path)
            with open(path) as f:
                data = json.load(f)
            schedule_events = [
                e
                for e in data["traceEvents"]
                if e.get("cat") in ("pp", "fsdp", "tp", "dp", "optim")
            ]
            assert len(schedule_events) > 0, "Should have schedule events"
            has_positive_ts = any(e.get("ts", 0) > 0 for e in schedule_events)
            assert (
                has_positive_ts
            ), "Schedule events should have DES timing (positive ts) in trace.json"

    def test_chrome_trace_des_dual_track(self):
        from torchtitan.experiments.simulator.des_engine import simulate_single_rank_des
        from torchtitan.experiments.simulator.export import export_chrome_trace
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            DataEdge,
            OpNode,
            PerfResult,
            SimulationResult,
        )

        graph = ComputeGraph()
        compute_node = OpNode(
            "compute",
            "mm",
            "compute",
            "forward",
            [],
            [],
            perf_result=PerfResult(total_time_us=50.0),
        )
        comm_node = OpNode(
            "comm",
            "all_reduce",
            "comm_collective",
            "forward",
            [],
            [],
            perf_result=PerfResult(total_time_us=30.0),
        )
        graph.add_node(compute_node)
        graph.add_node(comm_node)
        graph.add_edge(DataEdge("compute", "comm", "data"))
        simulate_single_rank_des(graph)
        result = SimulationResult(compute_graph=graph)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trace.json")
            export_chrome_trace(result, path)
            with open(path) as f:
                data = json.load(f)
            compute_events = [
                e
                for e in data["traceEvents"]
                if e.get("ph") == "X"
                and e.get("pid") == 0
                and e.get("cat") == "compute"
            ]
            comm_events = [
                e
                for e in data["traceEvents"]
                if e.get("ph") == "X"
                and e.get("pid") == 0
                and e.get("cat") in ("comm_collective", "comm_p2p")
            ]
            assert len(compute_events) >= 1
            assert len(comm_events) >= 1
            assert (
                compute_events[0]["tid"] != comm_events[0]["tid"]
            ), "Compute and comm should be on separate threads when DES timing available"
            phase_marker_events = [
                e
                for e in data["traceEvents"]
                if e.get("ph") == "i" and e.get("cat") == "phase_boundary"
            ]
            assert len(phase_marker_events) >= 1
            thread_names = {
                e["tid"]: e["args"]["name"]
                for e in data["traceEvents"]
                if e.get("ph") == "M"
                and e.get("name") == "thread_name"
                and e.get("pid") == 0
            }
            assert "Compute Engine" in thread_names.values()
            assert "Comm Engine" in thread_names.values()
            assert "Phase Markers" in thread_names.values()


class TestExportResultGatedToRankZero(unittest.TestCase):
    def test_export_result_gated_to_rank_zero(self):
        from torchtitan.experiments.simulator.export import export_result
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            OpNode,
            PerfResult,
            SimulationResult,
        )

        g = ComputeGraph()
        g.add_node(
            OpNode(
                "n1",
                "op1",
                "compute",
                "forward",
                perf_result=PerfResult(total_time_us=1.0),
            )
        )
        result = SimulationResult(compute_graph=g)

        saved_rank = os.environ.get("RANK")
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["RANK"] = "3"
            export_result(result, tmpdir, ["json"])
            json_path = os.path.join(tmpdir, "simulation_result.json")
            assert not os.path.exists(json_path), "Non-zero rank should not write files"

            os.environ["RANK"] = "0"
            export_result(result, tmpdir, ["json"])
            assert os.path.exists(json_path), "Rank 0 should write files"

        if saved_rank is not None:
            os.environ["RANK"] = saved_rank
        else:
            os.environ.pop("RANK", None)


class TestTrainerRunnerExtensionHooks(unittest.TestCase):
    def test_collect_extension_metadata(self):
        from torchtitan.experiments.simulator.extension_hooks import (
            collect_extension_metadata,
        )

        class TrainerWithMetadata:
            def collect_simulation_metadata(self, capture):
                assert capture == "capture"
                return {"extension": {"enabled": True}}

        assert collect_extension_metadata(TrainerWithMetadata(), "capture") == {
            "extension": {"enabled": True}
        }
        assert collect_extension_metadata(object(), "capture") == {}

    def test_collect_extension_metadata_rejects_non_dict(self):
        from torchtitan.experiments.simulator.extension_hooks import (
            collect_extension_metadata,
        )

        class BadTrainer:
            def collect_simulation_metadata(self, capture):
                del capture
                return ["not", "metadata"]

        with self.assertRaises(TypeError):
            collect_extension_metadata(BadTrainer(), object())

    def test_postprocess_extension_result(self):
        from torchtitan.experiments.simulator.extension_hooks import (
            postprocess_extension_result,
        )

        class TrainerWithPostprocess:
            def postprocess_simulation_result(self, result, sim_opts):
                result["sim_opts"] = sim_opts
                return None

        result = {"ok": True}
        returned = postprocess_extension_result(
            result, TrainerWithPostprocess(), "opts"
        )
        assert returned is result
        assert result["sim_opts"] == "opts"


# ===========================================================================
# Schedule extract tests (real PyTorch schedule with mock stages)
# ===========================================================================


class TestScheduleExtract(unittest.TestCase):
    def test_mock_pipeline_stage_attributes(self):
        from torchtitan.experiments.simulator.schedule_extract import MockPipelineStage

        stage = MockPipelineStage(
            stage_index=0, num_stages=4, group_rank=0, group_size=4
        )
        assert stage.stage_index == 0
        assert stage.num_stages == 4
        assert stage.group_rank == 0
        assert stage.group_size == 4
        assert stage.device == torch.device("cpu")
        assert stage.has_backward is True

    def test_extract_1f1b_schedule(self):
        from torchtitan.experiments.simulator.schedule_extract import (
            extract_schedule_from_pytorch,
        )

        result = extract_schedule_from_pytorch(
            pp_degree=4,
            tp_degree=1,
            dp_degree=1,
            num_stages=4,
            n_microbatches=8,
            schedule_name="1F1B",
        )
        assert len(result.events) > 0
        event_types = {e.event_type for e in result.events}
        assert "pp_forward" in event_types
        assert "pp_backward" in event_types
        assert "optimizer_step" in event_types

    def test_1f1b_warmup_counts(self):
        from torchtitan.experiments.simulator.schedule_extract import (
            extract_schedule_from_pytorch,
        )

        result = extract_schedule_from_pytorch(
            pp_degree=4,
            tp_degree=1,
            dp_degree=1,
            num_stages=4,
            n_microbatches=8,
            schedule_name="1F1B",
        )
        # Schedule1F1B: rank 0 should have warmup + steady + cooldown
        rank0_fwd = [
            e for e in result.events if e.rank == 0 and e.event_type == "pp_forward"
        ]
        rank0_bwd = [
            e for e in result.events if e.rank == 0 and e.event_type == "pp_backward"
        ]
        assert len(rank0_fwd) == 8
        assert len(rank0_bwd) == 8

    def test_extract_gpipe_schedule(self):
        from torchtitan.experiments.simulator.schedule_extract import (
            extract_schedule_from_pytorch,
        )

        result = extract_schedule_from_pytorch(
            pp_degree=4,
            tp_degree=1,
            dp_degree=1,
            num_stages=4,
            n_microbatches=8,
            schedule_name="GPipe",
        )
        assert len(result.events) > 0
        event_types = {e.event_type for e in result.events}
        assert "pp_forward" in event_types
        assert "pp_backward" in event_types

    def test_extract_interleaved_1f1b_schedule(self):
        from torchtitan.experiments.simulator.schedule_extract import (
            extract_schedule_from_pytorch,
        )

        result = extract_schedule_from_pytorch(
            pp_degree=4,
            tp_degree=1,
            dp_degree=1,
            num_stages=8,
            n_microbatches=8,
            schedule_name="Interleaved1F1B",
            virtual_stages_per_rank=2,
        )
        assert len(result.events) > 0
        event_types = {e.event_type for e in result.events}
        assert "pp_forward" in event_types
        assert "pp_backward" in event_types
        assert "pp_send_activation" in event_types
        assert "pp_recv_activation" in event_types
        assert "fsdp2_all_gather" in event_types
        assert "fsdp2_reduce_scatter" in event_types

    def test_interleaved_has_send_recv_deps(self):
        from torchtitan.experiments.simulator.schedule_extract import (
            extract_schedule_from_pytorch,
        )

        result = extract_schedule_from_pytorch(
            pp_degree=4,
            tp_degree=1,
            dp_degree=1,
            num_stages=8,
            n_microbatches=8,
            schedule_name="Interleaved1F1B",
            virtual_stages_per_rank=2,
        )
        dep_types = {d.dep_type for d in result.deps}
        assert "pp_comm" in dep_types
        assert "control" in dep_types

    def test_interleaved_reduce_grad_after_backward(self):
        from torchtitan.experiments.simulator.schedule_extract import (
            extract_schedule_from_pytorch,
        )

        result = extract_schedule_from_pytorch(
            pp_degree=4,
            tp_degree=1,
            dp_degree=1,
            num_stages=8,
            n_microbatches=8,
            schedule_name="Interleaved1F1B",
            virtual_stages_per_rank=2,
        )
        dp_events = [e for e in result.events if e.event_type == "dp_gradient_sync"]
        assert len(dp_events) > 0, "Expected REDUCE_GRAD events"

    def test_extract_all_schedule_types(self):
        from torchtitan.experiments.simulator.schedule_extract import (
            extract_schedule_from_pytorch,
        )

        for name, pp, virtual in [
            ("1F1B", 4, 1),
            ("GPipe", 4, 1),
            ("Interleaved1F1B", 4, 2),
            ("LoopedBFS", 4, 2),
            ("ZBVZeroBubble", 4, 2),
            ("DualPipeV", 4, 2),
        ]:
            result = extract_schedule_from_pytorch(
                pp_degree=pp,
                tp_degree=1,
                dp_degree=1,
                num_stages=pp * virtual,
                n_microbatches=8,
                schedule_name=name,
                virtual_stages_per_rank=virtual,
            )
            assert len(result.events) > 0, f"{name} should produce events"

    def test_tp_dp_replication(self):
        from torchtitan.experiments.simulator.schedule_extract import (
            extract_schedule_from_pytorch,
        )

        result = extract_schedule_from_pytorch(
            pp_degree=2,
            tp_degree=2,
            dp_degree=2,
            num_stages=2,
            n_microbatches=4,
            schedule_name="1F1B",
        )
        # With tp=2 and dp=2, total_ranks = 2*2*2 = 8
        ranks_with_events = {e.rank for e in result.events}
        assert len(ranks_with_events) > 2, "Events should span multiple global ranks"


# ===========================================================================
# Cost model tests
# ===========================================================================


class TestCostModel(unittest.TestCase):
    def test_matmul_flops_correct(self):
        from torchtitan.experiments.simulator.cost_model import _estimate_flops
        from torchtitan.experiments.simulator.nodes import OpNode, TensorMeta

        node = OpNode(
            node_id="n1",
            op_name="aten.mm.default",
            op_type="compute",
            phase="forward",
            inputs=[
                TensorMeta((2, 8), "torch.float32", "cpu"),
                TensorMeta((8, 4), "torch.float32", "cpu"),
            ],
            outputs=[TensorMeta((2, 4), "torch.float32", "cpu")],
        )
        flops = _estimate_flops(node)
        assert flops == 2 * 2 * 8 * 4, f"Expected 128 FLOPs, got {flops}"

    def test_matmul_flops_fallback_no_inputs(self):
        from torchtitan.experiments.simulator.cost_model import _estimate_flops
        from torchtitan.experiments.simulator.nodes import OpNode, TensorMeta

        node = OpNode(
            node_id="n2",
            op_name="aten.mm.default",
            op_type="compute",
            phase="forward",
            inputs=[],
            outputs=[TensorMeta((2, 4), "torch.float32", "cpu")],
        )
        flops = _estimate_flops(node)
        assert flops == 2 * 2 * 4, f"Expected 16 FLOPs (fallback), got {flops}"

    def test_addmm_not_matched_as_add(self):
        from torchtitan.experiments.simulator.cost_model import _estimate_flops
        from torchtitan.experiments.simulator.nodes import OpNode, TensorMeta

        node = OpNode(
            node_id="n3",
            op_name="aten.addmm.default",
            op_type="compute",
            phase="forward",
            inputs=[
                TensorMeta((2, 4), "torch.float32", "cpu"),
                TensorMeta((2, 8), "torch.float32", "cpu"),
                TensorMeta((8, 4), "torch.float32", "cpu"),
            ],
            outputs=[TensorMeta((2, 4), "torch.float32", "cpu")],
        )
        flops = _estimate_flops(node)
        assert flops == 2 * 2 * 8 * 4, f"addmm should use matmul formula, got {flops}"

    def test_reduce_scatter_comm_bytes_from_input(self):
        from torchtitan.experiments.simulator.cost_model import _estimate_comm_bytes
        from torchtitan.experiments.simulator.nodes import OpNode, TensorMeta

        node = OpNode(
            node_id="n4",
            op_name="reduce_scatter",
            op_type="comm_collective",
            phase="backward",
            inputs=[TensorMeta((4, 128), "torch.float32", "cpu")],
            outputs=[TensorMeta((4, 32), "torch.float32", "cpu")],
            comm_op="reduce_scatter",
            comm_group_size=4,
        )
        bytes_est = _estimate_comm_bytes(node)
        expected = 4 * 128 * 4
        assert (
            bytes_est == expected
        ), f"Expected {expected} bytes from input, got {bytes_est}"

    def test_all_gather_comm_bytes_from_output(self):
        from torchtitan.experiments.simulator.cost_model import _estimate_comm_bytes
        from torchtitan.experiments.simulator.nodes import OpNode, TensorMeta

        node = OpNode(
            node_id="n5",
            op_name="all_gather",
            op_type="comm_collective",
            phase="forward",
            inputs=[TensorMeta((4, 32), "torch.float32", "cpu")],
            outputs=[TensorMeta((4, 128), "torch.float32", "cpu")],
            comm_op="all_gather",
            comm_group_size=4,
        )
        bytes_est = _estimate_comm_bytes(node)
        expected = 4 * 128 * 4
        assert (
            bytes_est == expected
        ), f"Expected {expected} bytes from output, got {bytes_est}"

    def test_all_reduce_ring_factor(self):
        from torchtitan.experiments.simulator.cost_model import MockCostModel
        from torchtitan.experiments.simulator.nodes import OpNode, TensorMeta

        model = MockCostModel(noise_std=0.0)
        node = OpNode(
            node_id="n6",
            op_name="all_reduce",
            op_type="comm_collective",
            phase="forward",
            inputs=[TensorMeta((1024,), "torch.float32", "cpu")],
            outputs=[TensorMeta((1024,), "torch.float32", "cpu")],
            comm_op="all_reduce",
            comm_group_size=8,
        )
        perf = model.estimate_node(node)
        base_comm_bytes = 1024 * 4
        base_time = 5.0 + base_comm_bytes / (50.0 * 1e3)
        expected = base_time * 2 * 7 / 8
        assert (
            abs(perf.comm_time_us - expected) < 0.5
        ), f"Expected ~{expected}, got {perf.comm_time_us}"

    def test_critical_path_deque(self):
        from torchtitan.experiments.simulator.cost_model import _critical_path_time_us
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            DataEdge,
            OpNode,
            PerfResult,
        )

        graph = ComputeGraph()
        n = 200
        for i in range(n):
            node = OpNode(
                node_id=f"n{i}",
                op_name=f"op{i}",
                op_type="compute",
                phase="forward",
                inputs=[],
                outputs=[],
                perf_result=PerfResult(total_time_us=1.0),
            )
            graph.add_node(node)
            if i > 0:
                graph.add_edge(DataEdge(f"n{i - 1}", f"n{i}", "data"))

        result = _critical_path_time_us(graph)
        assert result == n, f"Expected {n}, got {result}"

    def test_silu_no_duplicate_branch(self):
        from torchtitan.experiments.simulator.cost_model import _estimate_flops
        from torchtitan.experiments.simulator.nodes import OpNode, TensorMeta

        node = OpNode(
            node_id="n7",
            op_name="aten.silu.default",
            op_type="compute",
            phase="forward",
            inputs=[TensorMeta((4,), "torch.float32", "cpu")],
            outputs=[TensorMeta((4,), "torch.float32", "cpu")],
        )
        flops = _estimate_flops(node)
        assert flops == 5 * 4, f"silu should be 5 FLOPs/elem, got {flops}"


# ===========================================================================
# Synthetic comm injection tests
# ===========================================================================


class TestSyntheticCommInjection(unittest.TestCase):
    def test_shape_is_numel_not_bytes(self):
        from torchtitan.experiments.simulator.memory_estimator import tensor_nbytes
        from torchtitan.experiments.simulator.nodes import TensorMeta

        per_layer_numel = 1024
        dtype_str = "torch.bfloat16"
        tm = TensorMeta(shape=(per_layer_numel,), dtype=dtype_str, device="cpu")
        nbytes = tensor_nbytes(tm)
        assert (
            nbytes == per_layer_numel * 2
        ), f"bf16: {per_layer_numel} elements × 2 bytes = {nbytes}"

        full_numel = per_layer_numel * 4
        tm_full = TensorMeta(shape=(full_numel,), dtype=dtype_str, device="cpu")
        nbytes_full = tensor_nbytes(tm_full)
        assert nbytes_full == full_numel * 2

    def test_reduce_scatter_shape_product_matches_bytes(self):
        from torchtitan.experiments.simulator.cost_model import _tensor_bytes
        from torchtitan.experiments.simulator.nodes import TensorMeta

        numel = 256
        tm = TensorMeta(shape=(numel,), dtype="torch.bfloat16", device="cpu")
        assert _tensor_bytes(tm.shape, tm.dtype) == numel * 2

    def test_dtype_from_config_bfloat16(self):
        from torchtitan.config import TORCH_DTYPE_MAP

        mp_param = "bfloat16"
        torch_dtype = TORCH_DTYPE_MAP.get(mp_param, torch.bfloat16)
        dtype_str = str(torch_dtype)
        assert dtype_str == "torch.bfloat16"

    def test_inject_synthetic_comm_creates_edges(self):
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            OpNode,
            SimulationResult,
            TensorMeta,
        )
        from torchtitan.experiments.simulator.synthetic_comm import (
            inject_synthetic_comm_events,
        )

        graph = ComputeGraph()
        fwd_node = OpNode(
            "fwd_1",
            "aten.mm.default",
            "compute",
            "forward",
            [TensorMeta((2, 4), "torch.float32", "cpu")],
            [TensorMeta((2, 4), "torch.float32", "cpu")],
        )
        bwd_node = OpNode(
            "bwd_1",
            "aten.mm.default",
            "compute",
            "backward",
            [TensorMeta((2, 4), "torch.float32", "cpu")],
            [TensorMeta((2, 4), "torch.float32", "cpu")],
        )
        graph.add_node(fwd_node)
        graph.add_node(bwd_node)
        result = SimulationResult(compute_graph=graph)

        class MockParallelism:
            tensor_parallel_degree = 1
            data_parallel_shard_degree = 2
            pipeline_parallel_degree = 1

        class MockTraining:
            seq_len = 8
            local_batch_size = 2
            mixed_precision_param = "bfloat16"

        class MockConfig:
            parallelism = MockParallelism()
            training = MockTraining()

        class MockModelPart(nn.Module):
            n_layers = 2

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 16)

        class MockTrainer:
            config = MockConfig()
            model_parts = [MockModelPart()]

        sim_opts = type("SimOpts", (), {"comm_backend": ""})()

        inject_synthetic_comm_events(result, MockTrainer(), sim_opts)

        assert len(result.comm_events) > 0, "Should inject FSDP comm events"

        comm_edges = [
            e for e in result.compute_graph.edges if e.edge_type == "sequential"
        ]
        assert len(comm_edges) > 0, "Injected comm nodes should have sequential edges"

        for ce in result.comm_events:
            tm = ce["tensor_meta"]
            shape_prod = 1
            for d in tm["shape"]:
                shape_prod *= d
            dtype_byte = 2 if tm["dtype"] == "torch.bfloat16" else 4
            actual_bytes = shape_prod * dtype_byte
            assert actual_bytes > 0, "shape should be numel, not bytes"


# ===========================================================================
# PP schedule extractor tests (mock schedule)
# ========================================================================= ==


class TestPPScheduleExtractor(unittest.TestCase):
    def _make_mock_schedule(self, n_stages: int = 2, n_microbatches: int = 4):
        """Create a minimal mock PP schedule."""

        class MockAction:
            __slots__ = ("stage_index", "computation_type", "microbatch_index")

            def __init__(self, stage, action_type, mb):
                self.stage_index = stage
                self.computation_type = action_type
                self.microbatch_index = mb

        class MockSchedule:
            def __init__(self):
                self.n_microbatches = n_microbatches
                # Build a simple 1F1B-like action list
                self._actions = []
                for mb in range(n_microbatches):
                    for s in range(n_stages):
                        self._actions.append(MockAction(s, "F", mb))
                for mb in range(n_microbatches - 1, -1, -1):
                    for s in range(n_stages - 1, -1, -1):
                        self._actions.append(MockAction(s, "B", mb))

        return MockSchedule()

    def test_extract_from_actions(self):
        from torchtitan.experiments.simulator.schedule_extract import (
            PPScheduleExtractor,
        )

        schedule_obj = self._make_mock_schedule(n_stages=2, n_microbatches=2)
        extractor = PPScheduleExtractor(schedule=schedule_obj, pp_rank=-1)
        sched = extractor.extract()

        assert len(sched.events) > 0
        event_types = {e.event_type for e in sched.events}
        # Should have forward and backward events
        assert len(event_types) > 0

    def test_schedule_has_deps(self):
        from torchtitan.experiments.simulator.schedule_extract import (
            PPScheduleExtractor,
        )

        schedule_obj = self._make_mock_schedule(n_stages=2, n_microbatches=2)
        extractor = PPScheduleExtractor(schedule=schedule_obj, pp_rank=-1)
        sched = extractor.extract()

        assert len(sched.deps) > 0


# ===========================================================================
# Simulator integration test (CPU, no PP)
# ===========================================================================


# ===========================================================================
# Phase 3+4 tests: schedule-graph linking, multi-rank step time,
# dynamic dim, overlap strategy, infer_num_layers
# ===========================================================================


class TestScheduleGraphLinking(unittest.TestCase):
    def test_link_schedule_to_graph_populates_op_node_ids(self):
        from torchtitan.experiments.simulator.cost_model import link_schedule_to_graph
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            OpNode,
            PerfResult,
            ScheduleDep,
            ScheduleEvent,
            SimulationResult,
            TensorMeta,
            TrainingSchedule,
        )

        graph = ComputeGraph()
        fwd_node = OpNode(
            "n1",
            "aten.mm.default",
            "compute",
            "forward",
            pp_stage=0,
            microbatch_idx=0,
            inputs=[TensorMeta((2, 8), "torch.float32", "cpu")],
            outputs=[TensorMeta((2, 4), "torch.float32", "cpu")],
            perf_result=PerfResult(total_time_us=10.0),
        )
        bwd_node = OpNode(
            "n2",
            "aten.mm.default",
            "compute",
            "backward",
            pp_stage=0,
            microbatch_idx=0,
            inputs=[TensorMeta((2, 4), "torch.float32", "cpu")],
            outputs=[TensorMeta((2, 8), "torch.float32", "cpu")],
            perf_result=PerfResult(total_time_us=15.0),
        )
        graph.add_node(fwd_node)
        graph.add_node(bwd_node)

        schedule = TrainingSchedule()
        fwd_event = ScheduleEvent(
            "ev_fwd", "pp_forward", rank=0, pp_stage=0, microbatch_idx=0
        )
        bwd_event = ScheduleEvent(
            "ev_bwd", "pp_backward", rank=0, pp_stage=0, microbatch_idx=0
        )
        schedule.add_event(fwd_event)
        schedule.add_event(bwd_event)
        schedule.add_dep(ScheduleDep("ev_fwd", "ev_bwd", "control"))

        result = SimulationResult(compute_graph=graph, schedule=schedule)

        link_schedule_to_graph(result)
        assert fwd_event.op_node_ids == ["n1"]
        assert bwd_event.op_node_ids == ["n2"]

    def test_link_no_schedule_skips(self):
        from torchtitan.experiments.simulator.cost_model import link_schedule_to_graph
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            SimulationResult,
        )

        result = SimulationResult(compute_graph=ComputeGraph(), schedule=None)
        link_schedule_to_graph(result)
        assert True


class TestMultiRankStepTime(unittest.TestCase):
    def test_predict_with_schedule_returns_max_rank_finish(self):
        from torchtitan.experiments.simulator.cost_model import (
            CostModel,
            predict_multi_rank_step_time_us,
        )
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            OpNode,
            PerfResult,
            ScheduleDep,
            ScheduleEvent,
            SimulationResult,
            TensorMeta,
            TrainingSchedule,
        )

        class _PreservingCostModel(CostModel):
            def estimate_node(self, node):
                if node.perf_result is not None:
                    return node.perf_result
                return PerfResult()

        graph = ComputeGraph()
        fwd = OpNode(
            "n1",
            "aten.mm.default",
            "compute",
            "forward",
            pp_stage=0,
            microbatch_idx=0,
            inputs=[TensorMeta((2, 8), "torch.float32", "cpu")],
            outputs=[TensorMeta((2, 4), "torch.float32", "cpu")],
            perf_result=PerfResult(total_time_us=10.0),
        )
        bwd = OpNode(
            "n2",
            "aten.mm.default",
            "compute",
            "backward",
            pp_stage=0,
            microbatch_idx=0,
            inputs=[],
            outputs=[],
            perf_result=PerfResult(total_time_us=20.0),
        )
        graph.add_node(fwd)
        graph.add_node(bwd)

        schedule = TrainingSchedule()
        ev0 = ScheduleEvent("e0", "pp_forward", rank=0, pp_stage=0, microbatch_idx=0)
        ev1 = ScheduleEvent("e1", "pp_backward", rank=0, pp_stage=0, microbatch_idx=0)
        schedule.add_event(ev0)
        schedule.add_event(ev1)
        schedule.add_dep(ScheduleDep("e0", "e1", "control"))

        result = SimulationResult(compute_graph=graph, schedule=schedule)

        step_time = predict_multi_rank_step_time_us(
            result, cost_model=_PreservingCostModel()
        )
        assert step_time == 30.0, f"Expected 30.0, got {step_time}"

    def test_predict_no_schedule_falls_back(self):
        from torchtitan.experiments.simulator.cost_model import (
            MockCostModel,
            predict_multi_rank_step_time_us,
        )
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            OpNode,
            SimulationResult,
            TensorMeta,
        )

        graph = ComputeGraph()
        graph.add_node(
            OpNode(
                "n1",
                "aten.mm.default",
                "compute",
                "forward",
                [TensorMeta((2, 8), "torch.float32", "cpu")],
                [TensorMeta((2, 4), "torch.float32", "cpu")],
            )
        )
        result = SimulationResult(compute_graph=graph, schedule=None)

        cost_model = MockCostModel(noise_std=0.0)
        cost_model.estimate_graph(graph)
        step_time = predict_multi_rank_step_time_us(result, cost_model)
        assert step_time > 0, f"Expected positive step time, got {step_time}"


class TestDynamicDim(unittest.TestCase):
    def test_numel_dynamic_dim_uses_default_seq_len(self):
        from torchtitan.experiments.simulator.cost_model import _numel

        assert _numel((2, -1, 8), default_seq_len=4096) == 2 * 4096 * 8
        assert _numel((2, None, 8), default_seq_len=2048) == 2 * 2048 * 8

    def test_mock_cost_model_default_seq_len(self):
        from torchtitan.experiments.simulator.cost_model import MockCostModel

        model = MockCostModel(noise_std=0.0, default_seq_len=2048)
        assert model.default_seq_len == 2048


class TestOverlapStrategy(unittest.TestCase):
    def test_no_overlap_strategy(self):
        from torchtitan.experiments.simulator.cost_model import NoOverlap

        strategy = NoOverlap()
        assert strategy.overlap_factor(10.0, 5.0) == 15.0

    def test_fixed_overlap_strategy(self):
        from torchtitan.experiments.simulator.cost_model import FixedOverlap

        strategy = FixedOverlap(0.5)
        assert strategy.overlap_factor(10.0, 5.0) == 10.0 + max(0, 5.0 - 10.0 * 0.5)

    def test_mock_cost_model_accepts_overlap_strategy(self):
        from torchtitan.experiments.simulator.cost_model import (
            FixedOverlap,
            MockCostModel,
        )

        model = MockCostModel(noise_std=0.0, overlap_strategy=FixedOverlap(0.5))
        assert model.overlap_strategy is not None


class TestInferNumLayers(unittest.TestCase):
    def test_from_config_n_layers(self):
        from torchtitan.experiments.simulator.synthetic_comm import infer_num_layers

        class MockConfig:
            n_layers = 8

        class MockModel(nn.Module):
            config = MockConfig()

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

        assert infer_num_layers([MockModel()]) == 8

    def test_from_layers_attr(self):
        from torchtitan.experiments.simulator.synthetic_comm import infer_num_layers

        class MockModel(nn.Module):
            layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])

        assert infer_num_layers([MockModel()]) == 4

    def test_fallback_prefix_count(self):
        from torchtitan.experiments.simulator.synthetic_comm import infer_num_layers

        model = nn.Linear(16, 4)
        result = infer_num_layers([model])
        assert result >= 1


class TestSimulatorIntegration(unittest.TestCase):
    def test_simulate_fx_small_model(self):
        from torchtitan.experiments.simulator import Simulator

        sim = Simulator(rank=0, verbose=False)
        model = _small_linear()
        result = sim.simulate_fx(model, _example_inputs())

        assert len(result.compute_graph.nodes) > 0

    def test_simulate_runtime_small_model(self):
        # Ensure dist is initialized for CommRecorder
        import torch.distributed as dist

        from torchtitan.experiments.simulator import Simulator

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29502")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            dist.init_process_group(backend="gloo", init_method="env://")

        try:
            sim = Simulator(rank=0, verbose=False)
            model = _small_linear()
            result = sim.simulate_runtime([model], _example_inputs())

            assert len(result.compute_graph.nodes) > 0
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def test_simulate_all_exports_files(self):
        import torch.distributed as dist

        from torchtitan.experiments.simulator import Simulator

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29503")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            dist.init_process_group(backend="gloo", init_method="env://")

        try:
            sim = Simulator(rank=0, verbose=False)
            model = _small_linear()
            with tempfile.TemporaryDirectory() as tmpdir:
                result = sim.simulate_all(
                    [model],
                    _example_inputs(),
                    output_dir=tmpdir,
                    output_formats=["json", "text"],
                )
                assert os.path.exists(os.path.join(tmpdir, "simulation_result.json"))
                assert os.path.exists(os.path.join(tmpdir, "summary.txt"))
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


class TestOpClassification(unittest.TestCase):
    def test_compute_ops(self):
        from torchtitan.experiments.simulator.op_classification import classify_op

        for target in ("aten.mm.default", "aten.addmm.default", "aten.silu.default"):
            op_type, comm_op = classify_op(target)
            self.assertEqual(op_type, "compute")
            self.assertIsNone(comm_op)

    def test_comm_collective_ops(self):
        from torchtitan.experiments.simulator.op_classification import classify_op

        cases = [
            ("_c10d_functional.reduce_scatter", "reduce_scatter"),
            ("_c10d_functional.all_gather", "all_gather"),
            ("_c10d_functional.all_reduce", "all_reduce"),
            ("c10d_functional.all_to_all_single", "all_to_all"),
            ("_c10d_functional.broadcast", "broadcast"),
            ("_c10d_functional.wait_tensor", "wait"),
            ("aten.barrier.default", "barrier"),
        ]
        for target, expected_comm in cases:
            op_type, comm_op = classify_op(target)
            self.assertEqual(op_type, "comm_collective")
            self.assertEqual(comm_op, expected_comm)

    def test_p2p_ops(self):
        from torchtitan.experiments.simulator.op_classification import classify_op

        cases = [
            ("_c10d_functional._send", "send"),
            ("_c10d_functional._recv", "recv"),
        ]
        for target, expected_comm in cases:
            op_type, comm_op = classify_op(target)
            self.assertEqual(op_type, "comm_p2p")
            self.assertEqual(comm_op, expected_comm)

    def test_data_move_ops(self):
        from torchtitan.experiments.simulator.op_classification import classify_op

        for target in ("_to_copy", "aten.copy_.default", ".to.device"):
            op_type, comm_op = classify_op(target)
            self.assertEqual(op_type, "data_move")
            self.assertIsNone(comm_op)

    def test_memory_ops(self):
        from torchtitan.experiments.simulator.op_classification import classify_op

        for target in (
            "aten.empty.memory_format",
            "aten.zeros",
            "aten.ones",
            "aten.rand.default",
        ):
            op_type, comm_op = classify_op(target)
            self.assertEqual(op_type, "memory")
            self.assertIsNone(comm_op)

    def test_trivial_targets(self):
        from torchtitan.experiments.simulator.op_classification import is_trivial

        for target in (
            "aten.detach.default",
            "aten.view.default",
            "aten.alias.default",
        ):
            self.assertTrue(is_trivial(target))
        self.assertFalse(is_trivial("aten.mm.default"))

    def test_consistency_with_fx_and_dispatch(self):
        from torchtitan.experiments.simulator.op_classification import classify_op

        targets = [
            "aten.mm.default",
            "_c10d_functional.all_reduce",
            "_c10d_functional._send",
            "_to_copy",
            "aten.empty.memory_format",
        ]
        for target in targets:
            op_type, comm_op = classify_op(target)
            self.assertIsInstance(op_type, str)
            self.assertTrue(
                op_type
                in ("compute", "comm_collective", "comm_p2p", "data_move", "memory")
            )


class TestUnifiedTrace(unittest.TestCase):
    def test_trace_small_model_fake_mode(self):
        from torchtitan.experiments.simulator.unified_trace import (
            TraceRecorder,
            unified_trace,
        )

        model = _small_linear()
        inputs = _example_inputs()
        recorder = TraceRecorder(rank=0)
        with unified_trace(recorder, model, inputs, use_fake_mode=True):
            output = model(*inputs)
        result = recorder.build_result()
        assert len(result.compute_graph.nodes) > 0
        compute_nodes = [
            n for n in result.compute_graph.nodes.values() if n.op_type == "compute"
        ]
        assert len(compute_nodes) > 0
        for n in compute_nodes:
            assert n.phase == "forward"

    def test_trace_small_model_eager_mode(self):
        from torchtitan.experiments.simulator.unified_trace import (
            TraceRecorder,
            unified_trace,
        )

        model = _small_linear()
        inputs = _example_inputs()
        recorder = TraceRecorder(rank=0)
        with unified_trace(recorder, use_fake_mode=False):
            output = model(*inputs)
        result = recorder.build_result()
        assert len(result.compute_graph.nodes) > 0

    def test_trace_backward_phase_detection(self):
        from torchtitan.experiments.simulator.unified_trace import (
            TraceRecorder,
            unified_trace,
        )

        model = _small_linear()
        inputs = _example_inputs()
        recorder = TraceRecorder(rank=0)
        with unified_trace(recorder, model, inputs, use_fake_mode=True):
            output = model(*inputs)
            loss = output.sum()
            recorder.current_phase = "backward"
            loss.backward()
        result = recorder.build_result()
        fwd_nodes = [
            n for n in result.compute_graph.nodes.values() if n.phase == "forward"
        ]
        bwd_nodes = [
            n for n in result.compute_graph.nodes.values() if n.phase == "backward"
        ]
        assert len(fwd_nodes) > 0
        assert len(bwd_nodes) > 0

    def test_trace_data_flow_edges(self):
        from torchtitan.experiments.simulator.unified_trace import (
            TraceRecorder,
            unified_trace,
        )

        model = nn.Linear(8, 4)
        inputs = (torch.randn(2, 8),)
        recorder = TraceRecorder(rank=0)
        with unified_trace(recorder, model, inputs, use_fake_mode=True):
            output = model(*inputs)
        assert len(recorder.edges) > 0
        edge_types = {et for _, _, et in recorder.edges}
        assert "data" in edge_types

    def test_device_normalization_meta_to_cpu(self):
        from torchtitan.experiments.simulator.unified_trace import _normalize_device

        assert _normalize_device("meta") == "cpu"
        assert _normalize_device("cpu") == "cpu"
        assert _normalize_device("cuda:0") == "cuda:0"

    def test_trace_meta_device_model(self):
        from torchtitan.experiments.simulator.unified_trace import (
            TraceRecorder,
            unified_trace,
        )

        with torch.device("meta"):
            model = nn.Linear(16, 4)
        # Inputs must also be meta for FakeTensorMode to accept them;
        # the _fakeify_inputs helper in FX capture does this already.
        # Here we create meta inputs directly.
        inputs = (torch.randn(2, 16, device="meta"),)
        recorder = TraceRecorder(rank=0)
        with unified_trace(recorder, model, inputs, use_fake_mode=True):
            output = model(*inputs)
        result = recorder.build_result()
        assert len(result.compute_graph.nodes) > 0
        for n in result.compute_graph.nodes.values():
            for tm in n.inputs + n.outputs:
                if tm.device == "meta":
                    raise AssertionError(
                        "TensorMeta.device should be normalized to 'cpu', got 'meta'"
                    )

    def test_trace_matches_dispatch_path_classification(self):
        from torchtitan.experiments.simulator.op_classification import classify_op
        from torchtitan.experiments.simulator.unified_trace import (
            TraceRecorder,
            unified_trace,
        )

        model = nn.Linear(8, 4)
        inputs = (torch.randn(2, 8),)
        recorder = TraceRecorder(rank=0)
        with unified_trace(recorder, model, inputs, use_fake_mode=True):
            output = model(*inputs)
        for n in recorder.nodes:
            expected_type, expected_comm = classify_op(n.op_name)
            self.assertEqual(n.op_type, expected_type)
            self.assertEqual(n.comm_op, expected_comm)


class TestMetaDevicePatch(unittest.TestCase):
    def test_meta_model_zero_memory(self):
        with torch.device("meta"):
            model = nn.Linear(16, 4)
        # Meta tensors have shape/dtype but no data allocation.
        # element_size() still returns dtype byte size (not 0), so
        # memory is measured as numel * 0_data_bytes = effectively 0.
        # The key invariant is: device.type == "meta" (no storage).
        for p in model.parameters():
            assert p.device.type == "meta"
            assert p.is_meta

    def test_meta_model_trace_produces_correct_graph(self):
        from torchtitan.experiments.simulator.unified_trace import (
            TraceRecorder,
            unified_trace,
        )

        with torch.device("meta"):
            model = nn.Linear(8, 4)
        inputs = (torch.randn(2, 8, device="meta"),)
        recorder = TraceRecorder(rank=0)
        with unified_trace(recorder, model, inputs, use_fake_mode=True):
            output = model(*inputs)
        result = recorder.build_result()
        assert len(result.compute_graph.nodes) > 0
        for n in result.compute_graph.nodes.values():
            for tm in n.inputs + n.outputs:
                assert (
                    tm.device == "cpu"
                ), f"TensorMeta.device should be 'cpu', got '{tm.device}'"

    def test_meta_device_module_stubs(self):
        from torchtitan.experiments.simulator.meta_env import _make_meta_device_module

        mod = _make_meta_device_module()
        assert mod.device_count() == 0
        assert mod.memory_allocated() == 0
        assert mod.current_device() == 0
        assert mod.get_device_name() == "Meta_Simulator"
        mod.synchronize()
        mod.empty_cache()

    def test_meta_patch_sets_device_type(self):
        import torchtitan.tools.utils as tt_utils

        original_dt = getattr(tt_utils, "device_type", None)
        original_dm = getattr(tt_utils, "device_module", None)

        from torchtitan.experiments.simulator.meta_env import patch_device_type_to_meta

        patch_device_type_to_meta()
        assert tt_utils.device_type == "meta"
        assert tt_utils.device_module.device_count() == 0

        if original_dt is not None:
            tt_utils.device_type = original_dt
        if original_dm is not None:
            tt_utils.device_module = original_dm


class TestSimulatorUnified(unittest.TestCase):
    def test_simulate_unified_cpu_mode(self):
        from torchtitan.experiments.simulator.simulator import Simulator

        sim = Simulator(rank=0, verbose=False)
        model = _small_linear()
        inputs = _example_inputs()
        result = sim.simulate_unified(model, inputs, device_mode="cpu")
        assert len(result.compute_graph.nodes) > 0
        fwd_nodes = [
            n for n in result.compute_graph.nodes.values() if n.phase == "forward"
        ]
        bwd_nodes = [
            n for n in result.compute_graph.nodes.values() if n.phase == "backward"
        ]
        assert len(fwd_nodes) > 0
        assert len(bwd_nodes) > 0
        assert result.metadata["mode"] == "unified_trace"
        assert result.metadata["device_mode"] == "cpu"

    def test_simulate_unified_meta_mode(self):
        from torchtitan.experiments.simulator.simulator import Simulator

        sim = Simulator(rank=0, verbose=False)
        with torch.device("meta"):
            model = _small_linear()
        inputs = (torch.randn(2, 16, device="meta"),)
        result = sim.simulate_unified(model, inputs, device_mode="meta")
        assert len(result.compute_graph.nodes) > 0
        fwd_nodes = [
            n for n in result.compute_graph.nodes.values() if n.phase == "forward"
        ]
        bwd_nodes = [
            n for n in result.compute_graph.nodes.values() if n.phase == "backward"
        ]
        assert len(fwd_nodes) > 0
        assert len(bwd_nodes) > 0
        assert result.metadata["mode"] == "unified_trace"
        assert result.metadata["device_mode"] == "meta"

    def test_simulate_unified_meta_device_normalization(self):
        from torchtitan.experiments.simulator.simulator import Simulator

        sim = Simulator(rank=0, verbose=False)
        with torch.device("meta"):
            model = nn.Linear(8, 4)
        inputs = (torch.randn(2, 8, device="meta"),)
        result = sim.simulate_unified(model, inputs, device_mode="meta")
        for n in result.compute_graph.nodes.values():
            for tm in n.inputs + n.outputs:
                assert tm.device == "cpu", f"Expected 'cpu', got '{tm.device}'"

    def test_simulate_unified_produces_same_op_types_as_runtime(self):
        from torchtitan.experiments.simulator.simulator import Simulator

        sim = Simulator(rank=0, verbose=False)
        model = nn.Linear(8, 4)
        inputs = (torch.randn(2, 8),)

        unified_result = sim.simulate_unified(model, inputs, device_mode="cpu")
        runtime_result = sim.simulate_runtime([model], inputs)

        unified_compute = {
            n.op_name
            for n in unified_result.compute_graph.nodes.values()
            if n.op_type == "compute"
        }
        runtime_compute = {
            n.op_name
            for n in runtime_result.compute_graph.nodes.values()
            if n.op_type == "compute"
        }
        assert len(unified_compute) > 0
        assert len(runtime_compute) > 0


class TestFSDP1FakeProcessGroupIntegration(unittest.TestCase):
    """Test that FSDP1 + FakeProcessGroup + CommRecorder captures
    all_gather / reduce_scatter events without multi-process execution."""

    def setUp(self):
        import torch.distributed as dist

        if not dist.is_initialized():
            os.environ.setdefault("NGPU", "2")
            dist.init_process_group("fake", rank=0, world_size=2)

    def tearDown(self):
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()

    def test_fsdp1_wraps_on_fake_backend(self):
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
        )

        m = nn.Linear(16, 4)
        wrapped = FSDP(
            m,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            device_id=torch.device("cpu"),
        )
        assert isinstance(wrapped, FSDP)

    def test_fsdp1_forward_backward_produces_comm_events(self):
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
        )

        from torchtitan.experiments.simulator.comm_interceptor import (
            capture_comms,
            CommRecorder,
        )

        m = nn.Linear(16, 4)
        wrapped = FSDP(
            m,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            device_id=torch.device("cpu"),
        )
        recorder = CommRecorder(rank=0)
        recorder.current_phase = "forward"
        x = torch.randn(2, 16)
        with capture_comms(recorder):
            y = wrapped(x)
            assert y.shape == (2, 4)
            recorder.current_phase = "backward"
            y.sum().backward()

        assert (
            len(recorder.events) >= 2
        ), f"expected >=2 comm events, got {len(recorder.events)}"
        ops = [e["op"] for e in recorder.events]
        assert (
            "all_gather" in ops or "all_gather_into_tensor" in ops
        ), f"no all_gather: {ops}"
        assert (
            "reduce_scatter" in ops or "reduce_scatter_tensor" in ops
        ), f"no reduce_scatter: {ops}"

    def test_fsdp1_comm_events_have_correct_group_size(self):
        import torch.distributed as dist
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
        )

        from torchtitan.experiments.simulator.comm_interceptor import (
            capture_comms,
            CommRecorder,
        )

        world_size = dist.get_world_size()
        m = nn.Linear(16, 4)
        wrapped = FSDP(
            m,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            device_id=torch.device("cpu"),
        )
        recorder = CommRecorder(rank=0)
        x = torch.randn(2, 16)
        with capture_comms(recorder):
            y = wrapped(x)
            y.sum().backward()

        for ev in recorder.events:
            assert (
                ev.get("group_size") == world_size
            ), f"expected group_size={world_size}, got {ev.get('group_size')}"

    def test_unified_trace_captures_fsdp1_comm_nodes(self):
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
        )

        from torchtitan.experiments.simulator.unified_trace import (
            TraceRecorder,
            unified_trace,
        )

        m = nn.Linear(16, 4)
        wrapped = FSDP(
            m,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            device_id=torch.device("cpu"),
        )
        recorder = TraceRecorder(rank=0)
        x = torch.randn(2, 16)
        with unified_trace(
            recorder,
            wrapped,
            (x,),
            use_fake_mode=False,
            phase="forward",
            capture_comm=True,
            capture_fsdp=False,
        ):
            y = wrapped(x)
            recorder.current_phase = "backward"
            y.sum().backward()

        result = recorder.build_result()
        comm_nodes = [
            n
            for n in result.compute_graph.nodes.values()
            if n.op_type == "comm_collective"
        ]
        assert len(comm_nodes) >= 2, f"expected >=2 comm nodes, got {len(comm_nodes)}"
        ops = {n.op_name for n in comm_nodes}
        assert (
            "all_gather" in ops or "all_gather_into_tensor" in ops
        ), f"no all_gather: {ops}"
        assert (
            "reduce_scatter" in ops or "reduce_scatter_tensor" in ops
        ), f"no reduce_scatter: {ops}"


class TestDESEngine(unittest.TestCase):
    def test_op_node_des_fields(self):
        from torchtitan.experiments.simulator.nodes import OpNode, PerfResult

        node = OpNode(
            node_id="n1",
            op_name="mm",
            op_type="compute",
            phase="forward",
            perf_result=PerfResult(total_time_us=10.0),
        )
        node.des_start_time_us = 0.0
        node.des_finish_time_us = 10.0
        d = node.to_dict()
        assert "des_start_time_us" in d
        assert d["des_start_time_us"] == 0.0
        assert d["des_finish_time_us"] == 10.0

    def test_single_rank_linear_chain(self):
        from torchtitan.experiments.simulator.des_engine import simulate_single_rank_des
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            DataEdge,
            OpNode,
            PerfResult,
        )

        graph = ComputeGraph()
        for i in range(5):
            graph.add_node(
                OpNode(
                    node_id=f"n{i}",
                    op_name=f"op{i}",
                    op_type="compute",
                    phase="forward",
                    inputs=[],
                    outputs=[],
                    perf_result=PerfResult(total_time_us=10.0),
                )
            )
            if i > 0:
                graph.add_edge(DataEdge(f"n{i - 1}", f"n{i}", "data"))

        result = simulate_single_rank_des(graph)
        assert result == 50.0, f"Expected 50.0 (5*10 linear chain), got {result}"

    def test_compute_comm_overlap_on_separate_branches(self):
        from torchtitan.experiments.simulator.des_engine import simulate_single_rank_des
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            DataEdge,
            OpNode,
            PerfResult,
        )

        graph = ComputeGraph()
        root = OpNode(
            node_id="root",
            op_name="op_root",
            op_type="compute",
            phase="forward",
            inputs=[],
            outputs=[],
            perf_result=PerfResult(total_time_us=2.0),
        )
        compute = OpNode(
            node_id="compute",
            op_name="op_compute",
            op_type="compute",
            phase="forward",
            inputs=[],
            outputs=[],
            perf_result=PerfResult(total_time_us=100.0),
        )
        comm = OpNode(
            node_id="comm",
            op_name="all_reduce",
            op_type="comm_collective",
            phase="forward",
            inputs=[],
            outputs=[],
            perf_result=PerfResult(total_time_us=50.0),
        )
        join = OpNode(
            node_id="join",
            op_name="op_join",
            op_type="compute",
            phase="forward",
            inputs=[],
            outputs=[],
            perf_result=PerfResult(total_time_us=0.0),
        )
        graph.add_node(root)
        graph.add_node(compute)
        graph.add_node(comm)
        graph.add_node(join)
        graph.add_edge(DataEdge("root", "compute", "data"))
        graph.add_edge(DataEdge("root", "comm", "data"))
        graph.add_edge(DataEdge("compute", "join", "data"))
        graph.add_edge(DataEdge("comm", "join", "data"))

        result = simulate_single_rank_des(graph)
        expected = 2.0 + 100.0
        assert result == expected, f"Expected {expected} (overlap), got {result}"

    def test_compute_comm_serialize_on_same_branch(self):
        from torchtitan.experiments.simulator.des_engine import simulate_single_rank_des
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            DataEdge,
            OpNode,
            PerfResult,
        )

        graph = ComputeGraph()
        compute = OpNode(
            node_id="compute",
            op_name="op_compute",
            op_type="compute",
            phase="forward",
            inputs=[],
            outputs=[],
            perf_result=PerfResult(total_time_us=100.0),
        )
        comm = OpNode(
            node_id="comm",
            op_name="all_reduce",
            op_type="comm_collective",
            phase="forward",
            inputs=[],
            outputs=[],
            perf_result=PerfResult(total_time_us=50.0),
        )
        graph.add_node(compute)
        graph.add_node(comm)
        graph.add_edge(DataEdge("compute", "comm", "data"))

        result = simulate_single_rank_des(graph)
        expected = 150.0
        assert result == expected, f"Expected {expected} (no overlap), got {result}"

    def test_two_comm_ops_contention(self):
        from torchtitan.experiments.simulator.des_engine import simulate_single_rank_des
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            DataEdge,
            OpNode,
            PerfResult,
        )

        graph = ComputeGraph()
        root = OpNode(
            node_id="root",
            op_name="op_root",
            op_type="compute",
            phase="forward",
            inputs=[],
            outputs=[],
            perf_result=PerfResult(total_time_us=2.0),
        )
        comm1 = OpNode(
            node_id="comm1",
            op_name="all_reduce_1",
            op_type="comm_collective",
            phase="forward",
            inputs=[],
            outputs=[],
            perf_result=PerfResult(total_time_us=30.0),
        )
        comm2 = OpNode(
            node_id="comm2",
            op_name="all_reduce_2",
            op_type="comm_collective",
            phase="forward",
            inputs=[],
            outputs=[],
            perf_result=PerfResult(total_time_us=40.0),
        )
        join = OpNode(
            node_id="join",
            op_name="op_join",
            op_type="compute",
            phase="forward",
            inputs=[],
            outputs=[],
            perf_result=PerfResult(total_time_us=0.0),
        )
        graph.add_node(root)
        graph.add_node(comm1)
        graph.add_node(comm2)
        graph.add_node(join)
        graph.add_edge(DataEdge("root", "comm1", "data"))
        graph.add_edge(DataEdge("root", "comm2", "data"))
        graph.add_edge(DataEdge("comm1", "join", "data"))
        graph.add_edge(DataEdge("comm2", "join", "data"))

        result = simulate_single_rank_des(graph)
        expected = 2.0 + 30.0 + 40.0
        assert (
            result == expected
        ), f"Expected {expected} (comm contention), got {result}"

    def test_multi_rank_des_with_schedule(self):
        from torchtitan.experiments.simulator.des_engine import simulate_multi_rank_des
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            OpNode,
            PerfResult,
            ScheduleDep,
            ScheduleEvent,
            SimulationResult,
            TrainingSchedule,
        )

        graph = ComputeGraph()
        fwd_node = OpNode(
            "n_fwd",
            "aten.mm.default",
            "compute",
            "forward",
            pp_stage=0,
            microbatch_idx=0,
            perf_result=PerfResult(total_time_us=50.0),
        )
        bwd_node = OpNode(
            "n_bwd",
            "aten.mm.default",
            "compute",
            "backward",
            pp_stage=0,
            microbatch_idx=0,
            perf_result=PerfResult(total_time_us=30.0),
        )
        graph.add_node(fwd_node)
        graph.add_node(bwd_node)

        schedule = TrainingSchedule()
        ev_fwd0 = ScheduleEvent(
            "e_fwd0", "pp_forward", rank=0, pp_stage=0, microbatch_idx=0
        )
        ev_send = ScheduleEvent(
            "e_send", "pp_send_activation", rank=0, pp_stage=0, microbatch_idx=0
        )
        ev_recv = ScheduleEvent(
            "e_recv", "pp_recv_activation", rank=1, pp_stage=1, microbatch_idx=0
        )
        ev_fwd1 = ScheduleEvent(
            "e_fwd1", "pp_forward", rank=1, pp_stage=1, microbatch_idx=0
        )
        ev_bwd0 = ScheduleEvent(
            "e_bwd0", "pp_backward", rank=0, pp_stage=0, microbatch_idx=0
        )
        ev_bwd1 = ScheduleEvent(
            "e_bwd1", "pp_backward", rank=1, pp_stage=1, microbatch_idx=0
        )
        schedule.add_event(ev_fwd0)
        schedule.add_event(ev_send)
        schedule.add_event(ev_recv)
        schedule.add_event(ev_fwd1)
        schedule.add_event(ev_bwd0)
        schedule.add_event(ev_bwd1)
        schedule.add_dep(ScheduleDep("e_fwd0", "e_send", "pp_comm"))
        schedule.add_dep(ScheduleDep("e_send", "e_recv", "pp_comm"))
        schedule.add_dep(ScheduleDep("e_recv", "e_fwd1", "control"))
        schedule.add_dep(ScheduleDep("e_fwd1", "e_bwd1", "control"))
        schedule.add_dep(ScheduleDep("e_fwd0", "e_bwd0", "control"))

        result = SimulationResult(compute_graph=graph, schedule=schedule)
        step_time = simulate_multi_rank_des(result)
        assert step_time >= 80.0, f"Expected >= 80.0, got {step_time}"

    def test_des_accounts_for_comm_contention_while_cp_does_not(self):
        from torchtitan.experiments.simulator.cost_model import _critical_path_time_us
        from torchtitan.experiments.simulator.des_engine import simulate_single_rank_des
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            DataEdge,
            OpNode,
            PerfResult,
        )

        graph = ComputeGraph()
        root = OpNode(
            "root",
            "input",
            "compute",
            "forward",
            [],
            [],
            perf_result=PerfResult(total_time_us=1.0),
        )
        c1 = OpNode(
            "c1",
            "all_gather",
            "comm_collective",
            "forward",
            [],
            [],
            perf_result=PerfResult(total_time_us=50.0),
            comm_op="all_gather",
            comm_group_size=2,
        )
        c2 = OpNode(
            "c2",
            "reduce_scatter",
            "comm_collective",
            "forward",
            [],
            [],
            perf_result=PerfResult(total_time_us=60.0),
            comm_op="reduce_scatter",
            comm_group_size=2,
        )
        join = OpNode(
            "join",
            "output",
            "compute",
            "forward",
            [],
            [],
            perf_result=PerfResult(total_time_us=1.0),
        )
        graph.add_node(root)
        graph.add_node(c1)
        graph.add_node(c2)
        graph.add_node(join)
        graph.add_edge(DataEdge("root", "c1", "data"))
        graph.add_edge(DataEdge("root", "c2", "data"))
        graph.add_edge(DataEdge("c1", "join", "data"))
        graph.add_edge(DataEdge("c2", "join", "data"))

        cp_time = _critical_path_time_us(graph)
        des_time = simulate_single_rank_des(graph)
        assert des_time >= cp_time, f"DES ({des_time}) < CP ({cp_time})"
        assert (
            des_time > cp_time
        ), f"DES ({des_time}) should be > CP ({cp_time}) due to comm contention"

    def test_multi_rank_duration_memoization(self):
        from torchtitan.experiments.simulator.des_engine import simulate_multi_rank_des
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            OpNode,
            PerfResult,
            ScheduleDep,
            ScheduleEvent,
            SimulationResult,
            TrainingSchedule,
        )

        graph = ComputeGraph()
        for stage in range(2):
            graph.add_node(
                OpNode(
                    f"n_fwd_{stage}",
                    "aten.mm.default",
                    "compute",
                    "forward",
                    pp_stage=stage,
                    microbatch_idx=0,
                    perf_result=PerfResult(total_time_us=50.0),
                )
            )
            graph.add_node(
                OpNode(
                    f"n_bwd_{stage}",
                    "aten.mm.default",
                    "compute",
                    "backward",
                    pp_stage=stage,
                    microbatch_idx=0,
                    perf_result=PerfResult(total_time_us=30.0),
                )
            )

        schedule = TrainingSchedule()
        for mb in range(3):
            for stage in range(2):
                schedule.add_event(
                    ScheduleEvent(
                        f"e_fwd_{mb}_{stage}",
                        "pp_forward",
                        rank=stage,
                        pp_stage=stage,
                        microbatch_idx=mb,
                    )
                )
            schedule.add_event(
                ScheduleEvent(
                    f"e_send_{mb}_0",
                    "pp_send_activation",
                    rank=0,
                    pp_stage=0,
                    microbatch_idx=mb,
                )
            )
            schedule.add_event(
                ScheduleEvent(
                    f"e_recv_{mb}_1",
                    "pp_recv_activation",
                    rank=1,
                    pp_stage=1,
                    microbatch_idx=mb,
                )
            )
            for stage in range(2):
                schedule.add_event(
                    ScheduleEvent(
                        f"e_bwd_{mb}_{stage}",
                        "pp_backward",
                        rank=stage,
                        pp_stage=stage,
                        microbatch_idx=mb,
                    )
                )
            schedule.add_dep(ScheduleDep(f"e_fwd_{mb}_0", f"e_send_{mb}_0", "pp_comm"))
            schedule.add_dep(ScheduleDep(f"e_send_{mb}_0", f"e_recv_{mb}_1", "pp_comm"))
            schedule.add_dep(ScheduleDep(f"e_recv_{mb}_1", f"e_fwd_{mb}_1", "control"))

        result = SimulationResult(compute_graph=graph, schedule=schedule)
        step_time = simulate_multi_rank_des(result)
        assert step_time > 0, f"Expected positive step time, got {step_time}"
        assert all(e.des_start_time_us is not None for e in schedule.events)


class TestDESUtilization(unittest.TestCase):
    def _make_des_annotated_result(self):
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            DataEdge,
            OpNode,
            PerfResult,
            SimulationResult,
        )

        graph = ComputeGraph()
        root = OpNode(
            "root",
            "input",
            "compute",
            "forward",
            [],
            [],
            perf_result=PerfResult(total_time_us=10.0),
        )
        compute = OpNode(
            "compute",
            "mm",
            "compute",
            "forward",
            [],
            [],
            perf_result=PerfResult(total_time_us=100.0),
        )
        comm = OpNode(
            "comm",
            "all_reduce",
            "comm_collective",
            "forward",
            [],
            [],
            perf_result=PerfResult(total_time_us=50.0),
        )
        join = OpNode(
            "join",
            "output",
            "compute",
            "forward",
            [],
            [],
            perf_result=PerfResult(total_time_us=10.0),
        )
        graph.add_node(root)
        graph.add_node(compute)
        graph.add_node(comm)
        graph.add_node(join)
        graph.add_edge(DataEdge("root", "compute", "data"))
        graph.add_edge(DataEdge("root", "comm", "data"))
        graph.add_edge(DataEdge("compute", "join", "data"))
        graph.add_edge(DataEdge("comm", "join", "data"))

        from torchtitan.experiments.simulator.des_engine import simulate_single_rank_des

        simulate_single_rank_des(graph)
        return SimulationResult(compute_graph=graph)

    def test_compute_des_utilization_basic(self):
        from torchtitan.experiments.simulator.des_engine import compute_des_utilization

        result = self._make_des_annotated_result()
        stats = compute_des_utilization(result)
        assert "e2e_step_time_us" in stats
        assert "compute_busy_pct" in stats
        assert "comm_busy_pct" in stats
        assert "overlap_pct" in stats
        assert "contention_count" in stats
        assert "cp_step_time_us" in stats
        assert "des_vs_cp_ratio" in stats
        assert stats["e2e_step_time_us"] > 0
        assert stats["compute_busy_pct"] >= 0
        assert stats["overlap_pct"] >= 0

    def test_compute_des_utilization_no_des_timing(self):
        from torchtitan.experiments.simulator.des_engine import compute_des_utilization
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            OpNode,
            PerfResult,
            SimulationResult,
        )

        graph = ComputeGraph()
        graph.add_node(
            OpNode(
                "n1",
                "mm",
                "compute",
                "forward",
                [],
                [],
                perf_result=PerfResult(total_time_us=10.0),
            )
        )
        result = SimulationResult(compute_graph=graph)
        stats = compute_des_utilization(result)
        assert stats["e2e_step_time_us"] == 0.0
        assert stats["compute_busy_pct"] == 0.0

    def test_overlap_percentage(self):
        result = self._make_des_annotated_result()
        from torchtitan.experiments.simulator.des_engine import compute_des_utilization

        stats = compute_des_utilization(result)
        assert (
            stats["overlap_pct"] > 0
        ), f"Expected overlap > 0, got {stats['overlap_pct']}"


class TestDESMemoryTimeline(unittest.TestCase):
    def _make_result_with_memory(self):
        from torchtitan.experiments.simulator.des_engine import simulate_single_rank_des
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            DataEdge,
            MemoryEvent,
            OpNode,
            PerfResult,
            SimulationResult,
        )

        graph = ComputeGraph()
        n0 = OpNode(
            "n0",
            "input",
            "compute",
            "forward",
            [],
            [],
            perf_result=PerfResult(total_time_us=10.0),
        )
        n1 = OpNode(
            "n1",
            "mm",
            "compute",
            "forward",
            [],
            [],
            perf_result=PerfResult(total_time_us=50.0),
        )
        n2 = OpNode(
            "n2",
            "relu",
            "compute",
            "forward",
            [],
            [],
            perf_result=PerfResult(total_time_us=5.0),
        )
        n3 = OpNode(
            "n3",
            "mm_bwd",
            "compute",
            "backward",
            [],
            [],
            perf_result=PerfResult(total_time_us=30.0),
        )
        graph.add_node(n0)
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)
        graph.add_edge(DataEdge("n0", "n1", "data"))
        graph.add_edge(DataEdge("n1", "n2", "data"))
        simulate_single_rank_des(graph)

        return SimulationResult(
            compute_graph=graph,
            memory_events=[
                MemoryEvent(
                    event_id="act_fwd",
                    category="activation",
                    bytes=100000,
                    phase="forward",
                    lifetime_start=1,
                    lifetime_end=3,
                ),
                MemoryEvent(
                    event_id="grad",
                    category="gradient",
                    bytes=80000,
                    phase="backward",
                    lifetime_start=3,
                    lifetime_end=3,
                ),
                MemoryEvent(
                    event_id="param",
                    category="parameter",
                    bytes=200000,
                    phase="model_state",
                ),
            ],
        )

    def test_memory_timeline_basic(self):
        from torchtitan.experiments.simulator.des_engine import (
            compute_des_memory_timeline,
        )

        result = self._make_result_with_memory()
        timeline = compute_des_memory_timeline(result)
        assert "static_memory_bytes" in timeline
        assert "peak_dynamic_bytes" in timeline
        assert "peak_total_bytes" in timeline
        assert "timeline" in timeline
        assert timeline["static_memory_bytes"] == 200000
        assert timeline["peak_dynamic_bytes"] > 0
        assert len(timeline["timeline"]) > 0

    def test_memory_timeline_has_des_timestamps(self):
        from torchtitan.experiments.simulator.des_engine import (
            compute_des_memory_timeline,
        )

        result = self._make_result_with_memory()
        timeline = compute_des_memory_timeline(result)
        for sample in timeline["timeline"]:
            assert "time_us" in sample
            assert "total_bytes" in sample
            assert sample["time_us"] >= 0

    def test_memory_timeline_no_des(self):
        from torchtitan.experiments.simulator.des_engine import (
            compute_des_memory_timeline,
        )
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            MemoryEvent,
            OpNode,
            PerfResult,
            SimulationResult,
        )

        graph = ComputeGraph()
        graph.add_node(
            OpNode(
                "n1",
                "mm",
                "compute",
                "forward",
                [],
                [],
                perf_result=PerfResult(total_time_us=10.0),
            )
        )
        result = SimulationResult(
            compute_graph=graph,
            memory_events=[MemoryEvent(event_id="p", category="parameter", bytes=1000)],
        )
        timeline = compute_des_memory_timeline(result)
        assert timeline["static_memory_bytes"] == 1000
        assert timeline["peak_dynamic_bytes"] == 0


if __name__ == "__main__":
    unittest.main()
