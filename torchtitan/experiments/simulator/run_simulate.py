#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

def main() -> None:
    from torchtitan.config import ConfigManager
    config_manager = ConfigManager()
    config = config_manager.parse_args()

    from torchtitan.experiments.simulator.meta_env import patch_device_type_to_meta
    from torchtitan.experiments.simulator.cpu_env import init_cpu_distributed, destroy_cpu_distributed, patch_device_type_to_cpu

    # Determine device patch based on comm_backend
    comm_backend = getattr(config.simulation, "comm_backend", "") or ""
    actual_comm_mode = getattr(config.comm, "mode", "") or ""
    if actual_comm_mode == "fake_backend":
        comm_backend = ""

    if comm_backend == "gloo":
        patch_device_type_to_cpu()
    else:
        patch_device_type_to_meta()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    print(f"[run_simulate] rank={rank}/{world_size}, mode={config.simulation.mode}")

    if comm_backend == "gloo":
        init_cpu_distributed(rank, world_size)

    try:
        trainer = config.build()
        
        mode = config.simulation.mode
        if mode == "runtime" or mode == "all":
            # This triggers run_trainer_simulation which we already updated
            trainer.train()
        elif mode == "schedule":
            from torchtitan.experiments.simulator.simulator import Simulator
            sim = Simulator(rank=rank, world_size=world_size, verbose=(rank == 0))
            if hasattr(trainer, "_pp_schedule") and trainer._pp_schedule is not None:
                result = sim.simulate_pp_schedule(trainer._pp_schedule)
                from torchtitan.experiments.simulator.trainer_runner import _export_result
                _export_result(result, config.simulation.output_dir, config.simulation.output_formats)
            else:
                print("No PP schedule found; nothing to export.")
    finally:
        if comm_backend == "gloo":
            destroy_cpu_distributed()

if __name__ == "__main__":
    main()
