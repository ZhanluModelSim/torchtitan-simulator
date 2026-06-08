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
            output_formats=["json", "dot", "chrome_trace", "html", "text"],
            semantic_schedule=True,
            cost_model=True,
            comm_backend="gloo",
        ),
    )
