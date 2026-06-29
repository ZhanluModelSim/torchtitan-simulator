# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the layered IR projection (L0-L3).

Run with::

    pytest torchtitan/experiments/simulator/tests/test_ir.py -v
"""

from __future__ import annotations

import unittest


def _mixed_phase_graph():
    """Build a small captured ComputeGraph spanning fwd/bwd/optimizer.

    Topology (data edges):
        f1 -> f2 -> b1 -> b2 -> o1
    with f* in forward, b* in backward, o1 in optimizer.
    """
    from torchtitan.experiments.simulator.nodes import (
        ComputeGraph,
        DataEdge,
        OpNode,
        PerfResult,
        TensorMeta,
    )

    g = ComputeGraph(metadata={"rank": 0})
    g.add_node(
        OpNode(
            "f1",
            "aten.mm.default",
            "compute",
            "forward",
            inputs=[TensorMeta((4, 8), "torch.float32", "cpu", requires_grad=True)],
            outputs=[TensorMeta((4, 8), "torch.float32", "cpu")],
            perf_result=PerfResult(total_time_us=10.0, flops=128, bytes_written=128),
        )
    )
    g.add_node(
        OpNode(
            "f2",
            "aten.relu.default",
            "compute",
            "forward",
            outputs=[TensorMeta((4, 8), "torch.float32", "cpu")],
            perf_result=PerfResult(total_time_us=5.0, flops=32, bytes_written=128),
        )
    )
    g.add_node(
        OpNode(
            "b1",
            "aten.mm.default",
            "compute",
            "backward",
            outputs=[TensorMeta((4, 8), "torch.float32", "cpu")],
            perf_result=PerfResult(total_time_us=20.0, flops=256, bytes_written=128),
        )
    )
    g.add_node(
        OpNode(
            "b2",
            "all_reduce",
            "comm_collective",
            "backward",
            comm_op="all_reduce",
            comm_group_size=8,
            outputs=[TensorMeta((4, 8), "torch.float32", "cpu")],
            perf_result=PerfResult(
                comm_time_us=7.0, total_time_us=7.0, bytes_written=128
            ),
        )
    )
    g.add_node(
        OpNode(
            "o1",
            "adam_step",
            "compute",
            "optimizer",
            outputs=[TensorMeta((4, 8), "torch.float32", "cpu")],
            perf_result=PerfResult(total_time_us=3.0, flops=64),
        )
    )
    g.add_edge(DataEdge("f1", "f2", "data"))
    g.add_edge(DataEdge("f2", "b1", "data"))
    g.add_edge(DataEdge("b1", "b2", "data"))
    g.add_edge(DataEdge("b2", "o1", "data"))
    return g


# ===========================================================================
# L0 projection
# ===========================================================================


class TestSpecOpNodeProjection(unittest.TestCase):
    def test_projects_predecessors_and_successors(self):
        from torchtitan.experiments.simulator.ir.op_node import project_op_nodes

        g = _mixed_phase_graph()
        spec_nodes = project_op_nodes(g)

        self.assertEqual(spec_nodes["f2"].predecessors, ["f1"])
        self.assertEqual(spec_nodes["f2"].successors, ["b1"])
        self.assertEqual(spec_nodes["f1"].predecessors, [])

    def test_projects_cost_fields(self):
        from torchtitan.experiments.simulator.ir.op_node import project_op_nodes

        g = _mixed_phase_graph()
        spec_nodes = project_op_nodes(g)

        self.assertEqual(spec_nodes["f1"].flops, 128)
        # comm node should carry comm_bytes derived from tensor volume
        self.assertGreater(spec_nodes["b2"].comm_bytes, 0)
        self.assertEqual(spec_nodes["b2"].op_type, "all_reduce")


# ===========================================================================
# L1 StepGraph
# ===========================================================================


class TestStepBuilder(unittest.TestCase):
    def test_splits_into_three_step_graphs(self):
        from torchtitan.experiments.simulator.ir.step_graph import StepBuilder

        g = _mixed_phase_graph()
        steps = StepBuilder.from_compute_graph(g)

        self.assertEqual(
            {s.step_type for s in steps.values()},
            {"forward", "backward", "optimizer"},
        )

    def test_entry_and_exit_nodes(self):
        from torchtitan.experiments.simulator.ir.step_graph import StepBuilder

        g = _mixed_phase_graph()
        steps = StepBuilder.from_compute_graph(g)
        fwd = next(s for s in steps.values() if s.step_type == "forward")

        # within forward partition: f1 is entry, f2 is exit
        self.assertIn("f1", fwd.entry_nodes)
        self.assertIn("f2", fwd.exit_nodes)
        self.assertNotIn("f1", fwd.exit_nodes)

    def test_totals_aggregated(self):
        from torchtitan.experiments.simulator.ir.step_graph import StepBuilder

        g = _mixed_phase_graph()
        steps = StepBuilder.from_compute_graph(g)
        fwd = next(s for s in steps.values() if s.step_type == "forward")

        # f1 flops=128, f2 flops=32
        self.assertEqual(fwd.total_flops, 160)
        self.assertTrue(fwd.is_acyclic)

    def test_backward_comm_volume(self):
        from torchtitan.experiments.simulator.ir.step_graph import StepBuilder

        g = _mixed_phase_graph()
        steps = StepBuilder.from_compute_graph(g)
        bwd = next(s for s in steps.values() if s.step_type == "backward")

        self.assertGreater(bwd.comm_volume, 0)

    def test_cycle_detected(self):
        from torchtitan.experiments.simulator.ir.step_graph import StepBuilder
        from torchtitan.experiments.simulator.nodes import (
            ComputeGraph,
            DataEdge,
            OpNode,
        )

        g = ComputeGraph()
        g.add_node(OpNode("a", "op", "compute", "forward"))
        g.add_node(OpNode("b", "op", "compute", "forward"))
        g.add_edge(DataEdge("a", "b", "data"))
        g.add_edge(DataEdge("b", "a", "data"))
        steps = StepBuilder.from_compute_graph(g)
        fwd = next(s for s in steps.values() if s.step_type == "forward")
        self.assertFalse(fwd.is_acyclic)


# ===========================================================================
# L2 ScheduleGraph
# ===========================================================================


class _Parallelism:
    def __init__(self, **kw):
        self.pipeline_parallel_degree = kw.get("pp", 1)
        self.tensor_parallel_degree = kw.get("tp", 1)
        self.expert_parallel_degree = kw.get("ep", 1)
        self.context_parallel_degree = kw.get("cp", 1)
        self.data_parallel_shard_degree = kw.get("dp_shard", 1)
        self.data_parallel_replicate_degree = kw.get("dp_repl", 1)
        self.pipeline_parallel_schedule = kw.get("schedule", "1F1B")


class TestScheduleBuilder(unittest.TestCase):
    def _templates(self):
        from torchtitan.experiments.simulator.ir.step_graph import StepBuilder

        return StepBuilder.from_compute_graph(_mixed_phase_graph())

    def test_degrees_propagated(self):
        from torchtitan.experiments.simulator.ir.schedule_graph import ScheduleBuilder

        sg = ScheduleBuilder.from_capture(
            self._templates(),
            None,
            _Parallelism(pp=4, tp=8, ep=2, cp=2, dp_shard=3),
            gradient_accumulation=2,
        )
        self.assertEqual(sg.pp_degree, 4)
        self.assertEqual(sg.tp_degree, 8)
        self.assertEqual(sg.ep_degree, 2)
        self.assertEqual(sg.cp_degree, 2)
        self.assertEqual(sg.dp_degree, 3)
        self.assertEqual(sg.pipeline_schedule, "1F1B")

    def test_fallback_instances_without_pp_schedule(self):
        from torchtitan.experiments.simulator.ir.schedule_graph import ScheduleBuilder

        sg = ScheduleBuilder.from_capture(
            self._templates(), None, _Parallelism(), gradient_accumulation=3
        )
        self.assertEqual(sg.num_micro_batches, 3)
        # 3 fwd + 3 bwd + 1 opt
        self.assertEqual(len(sg.instances), 7)
        # at least one activations data pass
        self.assertTrue(
            any(
                slot.name == "activations" for dp in sg.data_passes for slot in dp.slots
            )
        )

    def test_instances_from_captured_pp_events(self):
        from torchtitan.experiments.simulator.ir.schedule_graph import ScheduleBuilder
        from torchtitan.experiments.simulator.nodes import (
            ScheduleEvent,
            TrainingSchedule,
        )

        sched = TrainingSchedule()
        # 2 stages x 2 microbatches forward
        for stage in (0, 1):
            for mb in (0, 1):
                sched.add_event(
                    ScheduleEvent(
                        event_id=f"f_{stage}_{mb}",
                        event_type="fwd",
                        rank=stage,
                        pp_stage=stage,
                        microbatch_idx=mb,
                    )
                )
        sg = ScheduleBuilder.from_capture(
            self._templates(),
            sched,
            _Parallelism(pp=2),
            gradient_accumulation=2,
        )
        self.assertEqual(sg.num_micro_batches, 2)
        fwd_instances = [i for i in sg.instances if i.step_type == "forward"]
        self.assertEqual(len(fwd_instances), 4)
        # PP forward pass stage0->stage1 should exist for each microbatch
        pp_passes = [
            dp for dp in sg.data_passes if dp.comm_primitive == "p2p_send_recv"
        ]
        self.assertGreaterEqual(len(pp_passes), 2)


# ===========================================================================
# L3 WorkloadGraph + orchestrator
# ===========================================================================


class _Training:
    steps = 5
    warmup_steps = 1
    seq_len = 128
    local_batch_size = 2
    gradient_accumulation_steps = 3


class _Config:
    def __init__(self):
        self.parallelism = _Parallelism(pp=2, tp=2)
        self.training = _Training()


class TestWorkloadBuilder(unittest.TestCase):
    def test_build_workload_graph_from_result(self):
        from torchtitan.experiments.simulator.ir import build_workload_graph
        from torchtitan.experiments.simulator.nodes import SimulationResult

        result = SimulationResult(compute_graph=_mixed_phase_graph())
        wg = build_workload_graph(result, _Config())

        self.assertEqual(wg.workload_type, "train")
        self.assertEqual(wg.num_iterations, 5)
        self.assertEqual(wg.warmup_iterations, 1)
        self.assertIn("forward", wg.step_templates)
        self.assertIn("backward", wg.step_templates)
        self.assertIn("optimizer", wg.step_templates)
        # dataloader DataFlow present
        self.assertTrue(any(d.source == "dataloader" for d in wg.data_inputs))
        # cross-iteration parameter pass present
        self.assertTrue(
            any(
                slot.name == "parameters"
                for dp in wg.cross_iter_passes
                for slot in dp.slots
            )
        )

    def test_gradient_accumulation_from_captured_metadata(self):
        from torchtitan.experiments.simulator.ir import build_workload_graph
        from torchtitan.experiments.simulator.nodes import SimulationResult

        result = SimulationResult(
            compute_graph=_mixed_phase_graph(),
            metadata={"gradient_accumulation_steps": 16},
        )
        wg = build_workload_graph(result, _Config())
        # captured Trainer value (16) must win over config.training (3)
        self.assertEqual(wg.iteration.schedule.gradient_accumulation, 16)
        self.assertEqual(wg.iteration.schedule.num_micro_batches, 16)

    def test_workload_graph_serializable(self):
        import json

        from torchtitan.experiments.simulator.ir import build_workload_graph
        from torchtitan.experiments.simulator.nodes import SimulationResult

        result = SimulationResult(compute_graph=_mixed_phase_graph())
        wg = build_workload_graph(result, _Config())
        json.dumps(wg.to_dict())


if __name__ == "__main__":
    unittest.main()
