# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ActivationCheckpointConfig, ParallelismConfig, TrainingConfig
from torchtitan.models.llama3.config_registry import model_registry

from ..synthetic_dataloader import SyntheticTokenDataLoader
from ..trainer import SimulationConfig, SimulationTrainer


def llama3_sim_debugmodel() -> SimulationTrainer.Config:
    """
    CPU-friendly simulation config side-loaded via:
      --module simulator.llama3 --config llama3_sim_debugmodel
    """
    return SimulationTrainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debugmodel"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=64,
            steps=1,
        ),
        dataloader=SyntheticTokenDataLoader.Config(
            vocab_size=2048,
            seed=42,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
        ),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=1000,
            last_save_model_only=True,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        simulation=SimulationConfig(
            output_dir="./simulator_output",
            output_formats=["json", "dot", "chrome_trace", "html", "text"],
            capture_joint_fx=False,
        ),
    )


def llama3_sim_1024gpu() -> SimulationTrainer.Config:
    """
    1024-GPU simulation config with PP / TP / DP / FSDP2 semantics.

    Topology: pp=16, tp=8, dp_shard=4, dp_replicate=2 → 16×8×4×2 = 1024

    The semantic Interleaved1F1B schedule mirrors a real 1024-GPU run:
      16 pipeline ranks × 2 virtual stages = 32 pipeline stages,
      8 tensor-parallel ranks per TP group,
      4 FSDP shard ranks × 2 HSDP replicas per DP group,
      8 microbatches for PP interleaving.

    Usage::

      PYTHON=~/.local/bin/python3.11 \\
        NGPU=1 MODULE=simulator.llama3 CONFIG=llama3_sim_1024gpu \\
        COMM_MODE=fake_backend ./run_train.sh
    """
    return SimulationTrainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debugmodel"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=64,
            steps=1,
        ),
        dataloader=SyntheticTokenDataLoader.Config(
            vocab_size=2048,
            seed=42,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(
            pipeline_parallel_degree=16,
            pipeline_parallel_schedule="Interleaved1F1B",
            tensor_parallel_degree=8,
            data_parallel_shard_degree=4,
            data_parallel_replicate_degree=2,
        ),
        checkpoint=CheckpointManager.Config(enable=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        simulation=SimulationConfig(
            output_dir="./simulator_output",
            output_formats=["json", "dot", "chrome_trace", "html", "text"],
            capture_joint_fx=False,
            semantic_schedule=True,
        ),
    )
