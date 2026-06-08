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
            capture_ops,
            OpRecorder,
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
            capture_ops,
            OpRecorder,
        )

        recorder = OpRecorder()
        x = torch.randn(3, 3)
        with capture_ops(recorder, phase="backward"):
            _ = x @ x

        phases = {n.phase for n in recorder.nodes}
        assert "backward" in phases

    def test_categorizes_relu(self):
        from torchtitan.experiments.simulator.dispatch_interceptor import (
            capture_ops,
            OpRecorder,
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
            capture_ops,
            OpRecorder,
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
            capture_ops,
            OpRecorder,
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
        from torchtitan.experiments.simulator.dispatch_interceptor import (
            capture_ops,
            OpRecorder,
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
            {
                "op": "reduce_scatter_tensor",
                "op_type": "comm_collective",
                "phase": "backward",
            },
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
        from torchtitan.experiments.simulator.trainer_runner import (
            _inject_synthetic_comm_events,
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

        _inject_synthetic_comm_events(result, MockTrainer(), sim_opts)

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
        from torchtitan.experiments.simulator.pp_schedule_extractor import (
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
        from torchtitan.experiments.simulator.pp_schedule_extractor import (
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

        step_time = predict_multi_rank_step_time_us(result)
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
        from torchtitan.experiments.simulator.trainer_runner import _infer_num_layers

        class MockConfig:
            n_layers = 8

        class MockModel(nn.Module):
            config = MockConfig()

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

        assert _infer_num_layers([MockModel()]) == 8

    def test_from_layers_attr(self):
        from torchtitan.experiments.simulator.trainer_runner import _infer_num_layers

        class MockModel(nn.Module):
            layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])

        assert _infer_num_layers([MockModel()]) == 4

    def test_fallback_prefix_count(self):
        from torchtitan.experiments.simulator.trainer_runner import _infer_num_layers

        model = nn.Linear(16, 4)
        result = _infer_num_layers([model])
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


if __name__ == "__main__":
    unittest.main()
