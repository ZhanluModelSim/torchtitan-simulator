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
    class FakeStream:
        def __init__(self, *args, **kwargs): pass
        def wait_stream(self, *args, **kwargs): pass
        def wait_event(self, *args, **kwargs): pass
        def record_event(self, *args, **kwargs): pass
        def query(self): return True
        def synchronize(self): pass
        def __enter__(self): pass
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def __eq__(self, other): return True
        def __hash__(self): return 0

    class FakeEvent:
        def __init__(self, *args, **kwargs): pass
        def record(self, *args, **kwargs): pass
        def wait(self, *args, **kwargs): pass
        def query(self): return True
        def elapsed_time(self, *args, **kwargs): return 0.0
        def synchronize(self): pass

    return types.SimpleNamespace(
        Stream=FakeStream,
        Event=FakeEvent,
        current_stream=lambda: FakeStream(),
        set_device=lambda device: None,
        is_initialized=lambda: True,
        current_device=lambda: 0,
        device_count=lambda: 0,
        device_capability=lambda device=None: (0, 0),
        get_device_name=lambda device=None: "Meta_Simulator",
        get_device_properties=lambda device=None: types.SimpleNamespace(
            name="Meta_Simulator", total_memory=80 * 1024**3
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

    # Patch FSDP2 so it accepts meta device meshes
    try:
        import torch.distributed.fsdp._fully_shard._fsdp_init as fsdp_init
        orig_get_device_from_mesh = fsdp_init._get_device_from_mesh
        def _get_device_from_mesh_meta(mesh):
            if mesh.device_type == "meta":
                return torch.device("meta")
            return orig_get_device_from_mesh(mesh)
        fsdp_init._get_device_from_mesh = _get_device_from_mesh_meta
        
        # Patch fully_shard directly since it imports it
        import torch.distributed.fsdp._fully_shard._fully_shard as fully_shard_mod
        fully_shard_mod._get_device_from_mesh = _get_device_from_mesh_meta
    except ImportError:
        pass

    # Patch buffer init for meta simulation
    try:
        from torchtitan.models.common.decoder import Decoder
        orig_init_self_buffers = Decoder._init_self_buffers
        def _init_self_buffers_meta(self, *, buffer_device=None):
            if buffer_device is not None and buffer_device.type == "meta":
                buffer_device = None
            return orig_init_self_buffers(self, buffer_device=buffer_device)
        Decoder._init_self_buffers = _init_self_buffers_meta
    except ImportError:
        pass

    # Patch torch.distributed.device_mesh._get_device_handle
    try:
        import torch.distributed.device_mesh as device_mesh_mod
        orig_get_device_handle = device_mesh_mod._get_device_handle
        def _get_device_handle_meta(device_type):
            if device_type == "meta":
                return meta_mod
            return orig_get_device_handle(device_type)
        device_mesh_mod._get_device_handle = _get_device_handle_meta
        
        # Patch where it's imported
        import torch.distributed.fsdp._fully_shard._fsdp_param_group as fsdp_param_group
        fsdp_param_group._get_device_handle = _get_device_handle_meta
        
        import torch.distributed.fsdp._fully_shard._fsdp_init as fsdp_init_mod
        fsdp_init_mod._get_device_handle = _get_device_handle_meta
        
        import torch.distributed.fsdp._fully_shard._fsdp_state as fsdp_state_mod
        fsdp_state_mod._get_device_handle = _get_device_handle_meta
        import torch.distributed.fsdp._fully_shard._fsdp_collectives as fsdp_collectives_mod
        fsdp_collectives_mod._get_device_handle = _get_device_handle_meta
        
        # Patch FSDP parameter validation to allow meta parameters
        import torch.distributed.fsdp._fully_shard._fsdp_param_group as param_group_mod
        param_group_mod.FSDPParamGroup._validate_no_meta_params = lambda self: None
    except ImportError:
        pass

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
