# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CPU environment setup utilities for the TorchTitan simulator.

Provides context managers that configure a pure-CPU, single-process (or
multi-process with gloo) environment so that the rest of the simulator can
run without any GPU hardware.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


@contextmanager
def cpu_only_env() -> Generator[None, None, None]:
    """
    Context manager that hides all GPUs and forces PyTorch to use CPU.

    Sets ``CUDA_VISIBLE_DEVICES=""`` so that ``torch.cuda.is_available()``
    returns False and ``_get_available_device_type()`` falls back to ``cpu``.
    Also sets ``PYTORCH_ENABLE_MPS_FALLBACK=1`` to avoid MPS on macOS.

    Must be entered *before* importing torchtitan.tools.utils (which caches
    ``device_type`` at import time).  In practice, call this at the very top
    of ``run_simulate.py`` before any torchtitan imports.
    """
    saved = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def init_cpu_distributed(
    rank: int = 0,
    world_size: int = 1,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
) -> None:
    """
    Initialize ``torch.distributed`` with the ``gloo`` backend on CPU.

    Sets the standard ``MASTER_ADDR``, ``MASTER_PORT``, ``RANK``,
    ``LOCAL_RANK``, and ``WORLD_SIZE`` environment variables so that
    TorchTitan's ``init_distributed`` helper works without modification.

    Args:
        rank: Global rank of this process (0 for single-process simulation).
        world_size: Total number of simulated ranks.
        master_addr: Rendezvous address.
        master_port: Rendezvous port.
    """
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
        )


def destroy_cpu_distributed() -> None:
    """Tear down a previously initialised process group."""
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


@contextmanager
def cpu_distributed_context(
    rank: int = 0,
    world_size: int = 1,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
) -> Generator[None, None, None]:
    """Context manager that initialises and then tears down a gloo process group."""
    init_cpu_distributed(rank, world_size, master_addr, master_port)
    try:
        yield
    finally:
        destroy_cpu_distributed()


_PATCHED_MODULES: dict[str, tuple[str, ...]] = {
    "torchtitan.components.metrics": ("device_module",),
    "torchtitan.distributed.parallel_dims": ("device_type",),
    "torchtitan.distributed.utils": ("device_module", "device_type"),
}


def make_device_module(
    device_count: int = 1,
    device_name: str = "CPU_Simulator",
    total_memory: int = 1,
) -> Any:
    """Build a namespace that quacks like torch.cuda on a simulated device."""
    import types

    return types.SimpleNamespace(
        set_device=lambda device: None,
        current_device=lambda: 0,
        device_count=lambda: device_count,
        device_capability=lambda device=None: (0, 0),
        get_device_name=lambda device=None: device_name,
        get_device_properties=lambda device=None: types.SimpleNamespace(
            name=device_name, total_memory=total_memory
        ),
        get_arch_list=lambda: [],
        synchronize=lambda: None,
        memory_allocated=lambda device=None: 0,
        max_memory_allocated=lambda device=None: 0,
        memory_reserved=lambda device=None: 0,
        max_memory_reserved=lambda device=None: 0,
        reset_peak_memory_stats=lambda device=None: None,
        memory_stats=lambda device=None: {},
        empty_cache=lambda: None,
    )


def _patch_downstream_modules(device_type: str, device_module: Any) -> None:
    """Rebind device_module/device_type in downstream torchtitan modules."""
    for mod_name, attrs in _PATCHED_MODULES.items():
        try:
            mod = __import__(mod_name, fromlist=list(attrs))
        except ImportError:
            continue
        for attr in attrs:
            if hasattr(mod, attr):
                if attr == "device_module":
                    setattr(mod, attr, device_module)
                else:
                    setattr(mod, attr, device_type)


def _patch_torch_cuda(device_module: Any) -> None:
    """Replace key torch.cuda entrypoints with stubs from device_module."""
    import torch
    import torch.cuda

    torch.cuda.is_available = lambda: False
    torch.cuda._lazy_init = lambda: None
    torch.cuda.current_device = device_module.current_device
    torch.cuda.device_count = device_module.device_count
    torch.cuda.get_device_name = device_module.get_device_name
    torch.cuda.get_device_properties = device_module.get_device_properties
    torch.cuda.synchronize = device_module.synchronize
    torch.cuda.memory_allocated = device_module.memory_allocated
    torch.cuda.max_memory_allocated = device_module.max_memory_allocated
    torch.cuda.memory_reserved = device_module.memory_reserved
    torch.cuda.max_memory_reserved = device_module.max_memory_reserved
    torch.cuda.reset_peak_memory_stats = device_module.reset_peak_memory_stats
    torch.cuda.memory_stats = device_module.memory_stats
    torch.cuda.empty_cache = device_module.empty_cache
    if not hasattr(torch.cuda, "set_device"):
        torch.cuda.set_device = device_module.set_device
    if not hasattr(torch.cuda, "get_arch_list"):
        torch.cuda.get_arch_list = device_module.get_arch_list
    if not hasattr(torch.cuda, "device_capability"):
        torch.cuda.device_capability = device_module.device_capability


def _make_cpu_device_module():
    """Lazily build a namespace that quacks like torch.cuda / torch.npu on CPU."""
    return make_device_module(
        device_count=1, device_name="CPU_Simulator", total_memory=1
    )


def patch_device_type_to_cpu() -> None:
    """
    Monkey-patch ``torchtitan.tools.utils.device_type`` and
    ``torchtitan.tools.utils.device_module`` to ``cpu``.

    Call this *after* importing torchtitan.tools.utils but *before* any
    TorchTitan component that reads those module-level variables.

    Also patches downstream modules that have already imported
    ``device_module`` at module scope (e.g. ``torchtitan.components.metrics``)
    so that their local references stay in sync.
    """
    cpu_mod = _make_cpu_device_module()

    try:
        import torchtitan.tools.utils as tt_utils

        tt_utils.device_type = "cpu"
        tt_utils.device_module = cpu_mod
    except ImportError:
        pass

    _patch_downstream_modules("cpu", cpu_mod)
    _patch_torch_cuda(cpu_mod)
