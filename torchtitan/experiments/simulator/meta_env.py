# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Meta device environment setup utilities for the TorchTitan simulator.

Provides device patching that redirects model construction and tensor
operations to ``torch.device("meta")`` so that no real memory is
allocated for parameters or activations.  This enables simulating
arbitrarily large models (e.g. Llama 3 70B) on a CPU-only host with
minimal RAM.

Meta patching is used for ``fake_backend`` mode (no real distributed
communication).  For ``gloo`` backend mode, the CPU patching in
``cpu_env.py`` is still required because real tensors must be exchanged
between processes.

Usage::

    patch_device_type_to_meta()
    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)
    # model parameters are now shape-only (0 bytes memory)
"""

from __future__ import annotations

import types


def _make_meta_device_module():
    """Build a namespace that quacks like ``torch.cuda`` but reports
    zero real devices (meta has no hardware backend)."""
    return types.SimpleNamespace(
        set_device=lambda device: None,
        current_device=lambda: 0,
        device_count=lambda: 0,
        device_capability=lambda device=None: (0, 0),
        get_device_name=lambda device=None: "Meta_Simulator",
        get_device_properties=lambda device=None: types.SimpleNamespace(
            name="Meta_Simulator", total_memory=0
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


def patch_device_type_to_meta() -> None:
    """Monkey-patch ``torchtitan.tools.utils.device_type`` and
    ``torchtitan.tools.utils.device_module`` to ``\"meta\"``.

    Also patches downstream modules that have already imported
    ``device_module`` / ``device_type`` at module scope.

    Additionally patches ``torch.cuda`` entrypoints with meta stubs
    (same approach as ``cpu_env._patch_torch_cuda_for_cpu``, but
    reporting 0 devices since meta tensors have no hardware backend).

    This is a **global, irreversible monkey-patch** — call it once at
    startup before any TorchTitan component reads device settings.
    """
    meta_mod = _make_meta_device_module()

    try:
        import torchtitan.tools.utils as tt_utils

        tt_utils.device_type = "meta"
        tt_utils.device_module = meta_mod
    except ImportError:
        pass

    _PATCHED_MODULES = {
        "torchtitan.components.metrics": ("device_module",),
        "torchtitan.distributed.parallel_dims": ("device_type",),
        "torchtitan.distributed.utils": ("device_module", "device_type"),
    }
    for mod_name, attrs in _PATCHED_MODULES.items():
        try:
            mod = __import__(mod_name, fromlist=list(attrs))
        except ImportError:
            continue
        for attr in attrs:
            if hasattr(mod, attr):
                if attr == "device_module":
                    setattr(mod, attr, meta_mod)
                else:
                    setattr(mod, attr, "meta")

    import torch
    import torch.cuda

    torch.cuda.is_available = lambda: False
    torch.cuda._lazy_init = lambda: None
    torch.cuda.current_device = meta_mod.current_device
    torch.cuda.device_count = meta_mod.device_count
    torch.cuda.get_device_name = meta_mod.get_device_name
    torch.cuda.get_device_properties = meta_mod.get_device_properties
    torch.cuda.synchronize = meta_mod.synchronize
    torch.cuda.memory_allocated = meta_mod.memory_allocated
    torch.cuda.max_memory_allocated = meta_mod.max_memory_allocated
    torch.cuda.memory_reserved = meta_mod.memory_reserved
    torch.cuda.max_memory_reserved = meta_mod.max_memory_reserved
    torch.cuda.reset_peak_memory_stats = meta_mod.reset_peak_memory_stats
    torch.cuda.memory_stats = meta_mod.memory_stats
    torch.cuda.empty_cache = meta_mod.empty_cache
    if not hasattr(torch.cuda, "set_device"):
        torch.cuda.set_device = meta_mod.set_device
    if not hasattr(torch.cuda, "get_arch_list"):
        torch.cuda.get_arch_list = meta_mod.get_arch_list
    if not hasattr(torch.cuda, "device_capability"):
        torch.cuda.device_capability = meta_mod.device_capability
