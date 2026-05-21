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

        tm = TensorMeta(shape=(2, 16), dtype="torch.float32", device="cpu", requires_grad=True)
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
        from torchtitan.experiments.simulator.nodes import ComputeGraph, DataEdge, OpNode

        g = ComputeGraph()
        n1 = OpNode("n1", "aten.mm.default", "compute", "forward", [], [])
        n2 = OpNode("n2", "aten.relu.default", "compute", "forward", [], [])
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(DataEdge("n1", "n2", "data"))
        assert len(g.nodes) == 2
        assert len(g.edges) == 1

    def test_to_dict_serializable(self):
        from torchtitan.experiments.simulator.nodes import ComputeGraph, DataEdge, OpNode

        g = ComputeGraph(metadata={"rank": 0})
        g.add_node(OpNode("n1", "aten.mm.default", "compute", "forward", [], []))
        d = g.to_dict()
        # Must be JSON-serializable
        json.dumps(d)


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
# Dispatch interceptor tests
# ===========================================================================

class TestOpCaptureMode(unittest.TestCase):
    def test_captures_matmul(self):
        from torchtitan.experiments.simulator.dispatch_interceptor import (
            OpRecorder,
            capture_ops,
        )

        recorder = OpRecorder()
        a = torch.randn(4, 8)
        b = torch.randn(8, 4)
        with capture_ops(recorder, phase="forward"):
            torch.mm(a, b)

        assert len(recorder.nodes) > 0
        op_names = [n.op_name for n in recorder.nodes]
        assert any("mm" in name.lower() for name in op_names), f"ops: {op_names}"

    def test_phase_labelling(self):
        from torchtitan.experiments.simulator.dispatch_interceptor import (
            OpRecorder,
            capture_ops,
        )

        recorder = OpRecorder()
        x = torch.randn(3, 3)
        with capture_ops(recorder, phase="backward"):
            _ = x @ x

        phases = {n.phase for n in recorder.nodes}
        assert "backward" in phases

    def test_categorizes_relu(self):
        from torchtitan.experiments.simulator.dispatch_interceptor import (
            OpRecorder,
            capture_ops,
        )

        recorder = OpRecorder()
        x = torch.randn(4)
        with capture_ops(recorder, phase="forward"):
            torch.relu(x)

        # relu is a compute op
        types = {n.op_type for n in recorder.nodes}
        assert "compute" in types

    def test_tensor_meta_captured(self):
        from torchtitan.experiments.simulator.dispatch_interceptor import (
            OpRecorder,
            capture_ops,
        )

        recorder = OpRecorder()
        x = torch.randn(3, 5)
        with capture_ops(recorder, phase="forward"):
            torch.sigmoid(x)

        # At least one output shape should be (3, 5)
        shapes = [tuple(m.shape) for n in recorder.nodes for m in n.outputs]
        assert (3, 5) in shapes, f"shapes={shapes}"

    def test_runtime_data_edges_from_tensor_producers(self):
        from torchtitan.experiments.simulator.dispatch_interceptor import (
            OpRecorder,
            capture_ops,
        )

        recorder = OpRecorder()
        x = torch.randn(2, 2)
        with capture_ops(recorder, phase="forward"):
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
            CommRecorder,
            capture_comms,
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
            CommRecorder,
            capture_comms,
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
            CommRecorder,
            capture_comms,
        )
        from torchtitan.experiments.simulator.dispatch_interceptor import (
            OpRecorder,
            capture_ops,
        )

        comm = CommRecorder(rank=0)
        ops = OpRecorder()
        x = torch.ones(4)
        with capture_ops(ops, phase="forward"):
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

        all_shapes = [
            tuple(m.shape)
            for n in graph.nodes.values()
            for m in n.outputs
        ]
        assert len(all_shapes) > 0, "No output shapes captured"

    def test_graph_to_compute_graph(self):
        """Test fx_graph_to_compute_graph directly."""
        import torch.fx as fx

        from torchtitan.experiments.simulator.fx_capture import fx_graph_to_compute_graph

        model = nn.Linear(4, 4)
        gm = fx.symbolic_trace(model)
        graph = fx_graph_to_compute_graph(gm, phase="forward")
        # symbolic_trace does include call_function nodes for mm/add
        assert isinstance(graph.nodes, dict)


# ===========================================================================
# Graph assembler tests
# ===========================================================================

class TestGraphAssembler(unittest.TestCase):
    def _make_nodes(self, n: int):
        from torchtitan.experiments.simulator.nodes import OpNode

        return [
            OpNode(
                node_id=f"n{i}",
                op_name=f"aten.op_{i}",
                op_type="compute",
                phase="forward",
                inputs=[],
                outputs=[],
            )
            for i in range(n)
        ]

    def test_from_runtime_sequential_edges(self):
        from torchtitan.experiments.simulator.graph_assembler import GraphAssembler

        nodes = self._make_nodes(4)
        graph = GraphAssembler.from_runtime(nodes)
        assert len(graph.nodes) == 4
        # 3 sequential edges
        assert len(graph.edges) == 3

    def test_merge_comm_events(self):
        from torchtitan.experiments.simulator.graph_assembler import GraphAssembler
        from torchtitan.experiments.simulator.nodes import ComputeGraph

        graph = ComputeGraph()
        comm_events = [
            {"op": "all_reduce", "op_type": "comm_collective", "phase": "forward"},
            {"op": "reduce_scatter_tensor", "op_type": "comm_collective", "phase": "backward"},
        ]
        GraphAssembler.merge_comm_events(graph, comm_events)
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
        n2 = OpNode("n2", "aten.all_reduce.default", "comm_collective", "forward", [], [])
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
                )
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
            assert "schedule swimlanes" in content
            assert "canvas" in content
            assert "drawSchedule" in content
            assert "drawDag" in content
            assert "Zoom in" in content
            assert "rank-tabs" in content
            assert "Global rank" in content
            assert "PP rank" in content
            assert "Estimated live memory peak" in content
            assert "Memory estimate summary" in content


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
# PP schedule extractor tests (mock schedule)
# ===========================================================================

class TestPPScheduleExtractor(unittest.TestCase):
    def _make_mock_schedule(self, n_stages: int = 2, n_microbatches: int = 4):
        """Create a minimal mock PP schedule."""

        class MockAction:
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
        from torchtitan.experiments.simulator.pp_schedule_extractor import PPScheduleExtractor

        schedule_obj = self._make_mock_schedule(n_stages=2, n_microbatches=2)
        extractor = PPScheduleExtractor(schedule=schedule_obj, pp_rank=-1)
        sched = extractor.extract()

        assert len(sched.events) > 0
        event_types = {e.event_type for e in sched.events}
        # Should have forward and backward events
        assert len(event_types) > 0

    def test_schedule_has_deps(self):
        from torchtitan.experiments.simulator.pp_schedule_extractor import PPScheduleExtractor

        schedule_obj = self._make_mock_schedule(n_stages=2, n_microbatches=2)
        extractor = PPScheduleExtractor(schedule=schedule_obj, pp_rank=-1)
        sched = extractor.extract()

        assert len(sched.deps) > 0


# ===========================================================================
# Simulator integration test (CPU, no PP)
# ===========================================================================

class TestSimulatorIntegration(unittest.TestCase):
    def test_simulate_fx_small_model(self):
        from torchtitan.experiments.simulator import Simulator

        sim = Simulator(rank=0, verbose=False)
        model = _small_linear()
        result = sim.simulate_fx(model, _example_inputs())

        assert len(result.compute_graph.nodes) > 0

    def test_simulate_runtime_small_model(self):
        from torchtitan.experiments.simulator import Simulator

        # Ensure dist is initialized for CommRecorder
        import torch.distributed as dist

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
        from torchtitan.experiments.simulator import Simulator

        import torch.distributed as dist

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


if __name__ == "__main__":
    unittest.main()
