# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.models.deepseek_v4 import model_registry

from ..synthetic_dataloader import SyntheticTokenDataLoader
from ..trainer import SimulationConfig, SimulationTrainer


def deepseek_v4_sim_smoketest() -> SimulationTrainer.Config:
    """
    DeepSeek V4 smoketest simulation config.

    Uses the minimal 2-layer smoketest model with PP=2, TP=2, DP=2 (8 ranks).
    """
    return SimulationTrainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("smoketest"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        training=TrainingConfig(local_batch_size=4, seq_len=128, steps=1),
        dataloader=SyntheticTokenDataLoader.Config(vocab_size=129280, seed=42),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(
            pipeline_parallel_degree=2,
            pipeline_parallel_schedule="Interleaved1F1B",
            pipeline_parallel_microbatch_size=8,
            tensor_parallel_degree=2,
            data_parallel_shard_degree=2,
            data_parallel_replicate_degree=1,
        ),
        checkpoint=CheckpointManager.Config(enable=False),
        simulation=SimulationConfig(
            output_dir="./simulator_output",
            output_formats=["json", "dot", "chrome_trace", "html", "text", "csv"],
            semantic_schedule=True,
            cost_model=True,
            comm_backend="gloo",
        ),
    )


def deepseek_v4_pro_sim_smoketest() -> SimulationTrainer.Config:
    """
    DeepSeek V4 Pro 61-layer simulation config with full parallelism.

    Topology: pp=8, tp=8, dp_shard=-1(auto), dp_replicate=1, ep=192
    Global batch = 384, seq_len = 4096
    """
    return SimulationTrainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("v4_pro_debug_61_layers"),
        optimizer=OptimizersContainer.Config(lr=1e-5),
        training=TrainingConfig(
            global_batch_size=384,
            local_batch_size=1,
            seq_len=4096,
            steps=1,
            max_norm=1.0,
        ),
        dataloader=SyntheticTokenDataLoader.Config(vocab_size=129280, seed=42),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(
            pipeline_parallel_degree=8,
            pipeline_parallel_schedule="Interleaved1F1B",
            pipeline_parallel_microbatch_size=8,
            tensor_parallel_degree=8,
            data_parallel_shard_degree=-1,
            data_parallel_replicate_degree=1,
            expert_parallel_degree=192,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(enable=False),
        simulation=SimulationConfig(
            output_dir="./simulator_output",
            output_formats=["json", "dot", "chrome_trace", "html", "text", "csv"],
            semantic_schedule=True,
            cost_model=True,
            comm_backend="gloo",
        ),
    )
