# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Meta device patches for FSDP2 simulation.

Applies 7 patches to enable FSDP2/DTensor/MoE to run on meta device and
naturally emit communication operators:

1. Skip _validate_no_meta_params -- FSDP2 normally rejects meta parameters
2. Patch FakeTensor._find_common_device -- Allow mixed meta/cpu tensors
3. Patch wrap_meta_outputs -- Convert CPU tensors to meta for FSDP buffers
4. Patch foreach_reduce -- Coerce mixed-dtype gradients to uniform dtype
5. Patch _unimplemented_deepcopy -- Allow FSDP module deepcopy (PP splitting)
6. Patch nn.Module.to_empty -- No-op on meta (preserve FSDP2 sharding state)
7. Patch repeat_interleave -- Return placeholder for dynamic output shapes

Usage:
    from torchtitan.experiments.simulator.meta_device_patches import (
        apply_meta_device_patches,
        restore_meta_device_patches,
    )

    # Apply patches before unified_trace
    apply_meta_device_patches()

    # ... run unified_trace with FSDP2 on meta device ...

    # Restore patches after unified_trace
    restore_meta_device_patches()
"""

from __future__ import annotations

import copy

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.fsdp._fully_shard import _fsdp_collectives, _fsdp_param_group
from torch.distributed.fsdp._fully_shard import _fully_shard as _fsdp_fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup


# Store original functions for restoration
_ORIGINAL_FUNCTIONS = {
    "validate_no_meta_params": None,
    "find_common_device": None,
    "wrap_meta_outputs": None,
    "foreach_reduce": None,
    "unimplemented_deepcopy": None,
    "to_empty": None,
    "repeat_interleave": None,
}

# FSDP ops that can have mixed meta/cpu tensors
_FSDP_MIXED_OPS = set()


def _patched_find_common_device(func, flat_args):
    """Allow FSDP ops with mixed meta/cpu tensors by preferring meta device."""
    # Check if this is an FSDP op with mixed devices
    if func in _FSDP_MIXED_OPS:
        has_meta = any(
            isinstance(a, torch.Tensor) and a.device.type == "meta"
            for a in flat_args
        )
        has_cpu = any(
            isinstance(a, torch.Tensor) and a.device.type == "cpu"
            for a in flat_args
        )
        # Prefer meta device for FSDP ops
        if has_meta and has_cpu:
            return torch.device("meta"), False
    
    # Fall back to original logic
    return _ORIGINAL_FUNCTIONS["find_common_device"](func, flat_args)


def _patched_wrap_meta_outputs(self, r, func, flat_args, device):
    """Convert CPU tensors to meta tensors for FSDP internal buffers."""
    import torch.utils._pytree as pytree
    
    def wrap(e):
        if not isinstance(e, torch.Tensor):
            return e
        # If already a FakeTensor, return it
        if self.is_our_fake(e):
            return e
        # Convert CPU tensor to meta
        if e.device.type == "cpu":
            meta_t = torch.empty(e.shape, dtype=e.dtype, device="meta")
            return self.from_tensor(meta_t, static_shapes=True)
        # Convert other tensors normally
        return self.from_tensor(e, static_shapes=True)
    
    return pytree.tree_map(wrap, r)


def _make_foreach_reduce_dtype_coercer(original_foreach_reduce):
    """Wrap FSDP2 ``foreach_reduce`` to coerce ``unsharded_grads`` to one dtype.

    On meta device under FakeTensorMode, mixed-precision training can produce
    gradients with more than one dtype (e.g. ``{bfloat16, float32}``). FSDP2's
    ``foreach_reduce`` asserts a single gradient dtype before reduce-scatter.
    For simulation we only need the reduce-scatter *communication operator* to
    be emitted, so we coerce every gradient to the first gradient's dtype --
    the values are fake and the exact dtype is irrelevant for trace capture.
    """

    def _patched_foreach_reduce(*args, **kwargs):
        # unsharded_grads is the 2nd positional argument
        if len(args) > 1 and isinstance(args[1], list):
            grads = args[1]
            dtypes = {
                g.dtype for g in grads if isinstance(g, torch.Tensor)
            }
            if len(dtypes) > 1:
                target_dtype = grads[0].dtype
                coerced = [
                    g.to(target_dtype) if g.dtype != target_dtype else g
                    for g in grads
                ]
                args = (*args[:1], coerced, *args[2:])
        return original_foreach_reduce(*args, **kwargs)

    return _patched_foreach_reduce


def _fsdp_meta_deepcopy(self, memo=None):
    """Permit FSDP module deepcopy on meta device.

    FSDP blocks deepcopy by default (``_unimplemented_deepcopy`` raises).
    On meta device the tensors are fake (0 bytes), so a structural deepcopy is
    safe and is needed for PP module splitting -- ``_split_module`` deep-copies
    the whole FSDP-wrapped model to produce per-stage parts.

    We bypass three obstacles:
    1. The ``__deepcopy__`` block -- temporarily remove it from the
       dynamically-created wrapper class and fall through to the standard
       ``nn.Module`` deepcopy (which deep-copies ``__dict__``).
    2. ``FSDPModule.__new__`` calling ``__init__`` during reconstruction --
       wrap the deepcopy in ``disable_fsdp_module_new_init()`` so the model's
       ``__init__(config, ...)`` is not re-invoked with no arguments.
    3. C++ ``ProcessGroup`` objects in FSDP comm state cannot be pickled --
       pre-scan the object tree and add them to the deepcopy ``memo`` as
       identity copies (safe on meta/fake device: they hold no real state).
    """
    if memo is None:
        memo = {}
    _collect_identity_copy_objects(self, memo)
    cls = self.__class__
    saved = cls.__dict__.get("__deepcopy__")
    if saved is not None:
        try:
            delattr(cls, "__deepcopy__")
        except (AttributeError, TypeError):
            saved = None
    try:
        with _fsdp_fully_shard.disable_fsdp_module_new_init():
            return copy.deepcopy(self, memo)
    finally:
        if saved is not None:
            try:
                type.__setattr__(cls, "__deepcopy__", saved)
            except (AttributeError, TypeError):
                pass


def _collect_identity_copy_objects(obj, memo, seen=None):
    """Pre-populate ``memo`` with identity copies for uncopyable C++ objects.

    Walks ``obj``'s attribute tree and registers every ``ProcessGroup``
    instance (and other known-unpicklable C++ types) as an identity copy in
    ``memo`` so ``copy.deepcopy`` skips them instead of raising.
    """
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen or oid in memo:
        return
    seen.add(oid)
    # Identity-copy uncopyable / stateful C++ objects and DeviceMesh.
    # DeviceMesh must be shared (not copied) so that DTensor placements on
    # deepcopied model parts reference the *same* mesh object as the FSDP2
    # re-application — otherwise DTensor ops reject "different meshes".
    if isinstance(obj, (ProcessGroup, DeviceMesh)):
        memo[oid] = obj
        return
    # Recurse into __dict__
    obj_dict = getattr(obj, "__dict__", None)
    if obj_dict is not None:
        for v in obj_dict.values():
            if isinstance(v, (ProcessGroup,)):
                memo.setdefault(id(v), v)
            else:
                _collect_identity_copy_objects(v, memo, seen)
    # Recurse into common containers
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_identity_copy_objects(v, memo, seen)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_identity_copy_objects(v, memo, seen)


def apply_meta_device_patches() -> None:
    """
    Apply 7 patches to enable FSDP2/DTensor/MoE on meta device.

    Patches:
    1. FSDPParamGroup._validate_no_meta_params -- Skip meta parameter validation
    2. FakeTensor._find_common_device -- Allow mixed meta/cpu for FSDP ops
    3. FakeTensorMode.wrap_meta_outputs_with_default_device_logic -- Convert CPU to meta
    4. _fsdp_collectives.foreach_reduce -- Coerce mixed-dtype gradients to uniform dtype
    5. _fully_shard._unimplemented_deepcopy -- Allow FSDP module deepcopy for PP splitting
    6. nn.Module.to_empty -- No-op on meta (preserve FSDP2 sharding state)
    7. fake_impls repeat_interleave -- Return placeholder for dynamic output shapes

    Must be called before running unified_trace with FSDP2 on meta device.
    Call restore_meta_device_patches() after unified_trace completes.
    """
    # Store originals for restoration
    _ORIGINAL_FUNCTIONS["validate_no_meta_params"] = FSDPParamGroup._validate_no_meta_params
    _ORIGINAL_FUNCTIONS["find_common_device"] = FakeTensor._find_common_device
    _ORIGINAL_FUNCTIONS["wrap_meta_outputs"] = (
        FakeTensorMode.wrap_meta_outputs_with_default_device_logic
    )
    _ORIGINAL_FUNCTIONS["foreach_reduce"] = _fsdp_collectives.foreach_reduce
    _ORIGINAL_FUNCTIONS["unimplemented_deepcopy"] = _fsdp_fully_shard._unimplemented_deepcopy
    _ORIGINAL_FUNCTIONS["to_empty"] = torch.nn.Module.to_empty
    from torch._subclasses import fake_impls as _fake_impls
    _ri_key = torch.ops.aten.repeat_interleave.Tensor
    _ORIGINAL_FUNCTIONS["repeat_interleave"] = _fake_impls.op_implementations_dict.get(_ri_key)
    
    # Build list of FSDP ops that can have mixed devices
    _FSDP_MIXED_OPS.clear()
    for op_name in ("all_gather_copy_in", "reduce_scatter_copy_out"):
        op = getattr(torch.ops.fsdp, op_name, None)
        if op is not None:
            _FSDP_MIXED_OPS.add(op.default)
    for op_name in ("_allgather_base_", "_reduce_scatter_base_"):
        op = getattr(torch.ops.c10d, op_name, None)
        if op is not None:
            _FSDP_MIXED_OPS.add(op.default)
    
    # Patch 1: Skip meta parameter validation
    FSDPParamGroup._validate_no_meta_params = lambda self: None
    
    # Patch 2: Allow mixed meta/cpu for FSDP ops
    FakeTensor._find_common_device = staticmethod(_patched_find_common_device)
    
    # Patch 3: Convert CPU tensors to meta
    FakeTensorMode.wrap_meta_outputs_with_default_device_logic = _patched_wrap_meta_outputs

    # Patch 4: Coerce mixed-dtype gradients for FSDP2 reduce-scatter on meta.
    # Patch both the defining module and the importing module (which holds its
    # own local reference from ``from ._fsdp_collectives import foreach_reduce``).
    coerced = _make_foreach_reduce_dtype_coercer(_fsdp_collectives.foreach_reduce)
    _fsdp_collectives.foreach_reduce = coerced
    _fsdp_param_group.foreach_reduce = coerced

    # Patch 5: Allow FSDP module deepcopy on meta device for PP splitting.
    # _unimplemented_deepcopy is looked up by name in _fully_shard globals when
    # fully_shard() creates the wrapper class, so patching the module attribute
    # before the parallelize call is sufficient. The patched __deepcopy__ also
    # pre-populates the memo with identity copies for uncopyable C++ ProcessGroup
    # objects found in FSDP comm state.
    _fsdp_fully_shard._unimplemented_deepcopy = _fsdp_meta_deepcopy

    # Patch 6: Make nn.Module.to_empty a no-op when target device is meta.
    # to_empty creates fresh tensors, discarding FSDP2's sharding registration.
    # On meta device the tensors are already 0-byte, so to_empty is redundant --
    # making it a no-op preserves FSDP2's DTensor placements so unshard/reshard
    # emit their all_gather/reduce_scatter communication operators during trace.
    _nn_module = torch.nn.Module

    def _to_empty_meta_noop(self, *, device=None, **kw):
        if str(device) == "meta":
            return self
        return _ORIGINAL_FUNCTIONS["to_empty"](self, device=device, **kw)

    torch.nn.Module.to_empty = _to_empty_meta_noop

    # Patch 7: Override FakeTensorMode's repeat_interleave fake impl.
    # The real impl raises DynamicOutputShapeException because the output
    # size depends on sum(repeats) — a runtime value.  On meta the exact
    # size is irrelevant; return a placeholder so downstream ops and FSDP2
    # comm operators are still emitted.
    _ri_key = torch.ops.aten.repeat_interleave.Tensor
    _orig_ri = _fake_impls.op_implementations_dict.get(_ri_key)

    def _patched_repeat_interleave(fake_mode, func, *args, **kwargs):
        try:
            return _orig_ri(fake_mode, func, *args, **kwargs)
        except torch._subclasses.fake_tensor.DynamicOutputShapeException:
            # Return a placeholder with shape [1, ...] — size 1 on the
            # repeat dim is broadcastable to any downstream shape.
            tensors = [a for a in args if isinstance(a, torch.Tensor)]
            if tensors:
                t = tensors[0]
                shape = list(t.shape)
                if shape:
                    shape[0] = 1
                return t.new_empty(tuple(shape))
            raise

    _fake_impls.op_implementations_dict[_ri_key] = _patched_repeat_interleave


def restore_meta_device_patches() -> None:
    """
    Restore original PyTorch functions.
    
    Must be called after unified_trace completes to avoid affecting
    other PyTorch operations.
    """
    if _ORIGINAL_FUNCTIONS["validate_no_meta_params"] is not None:
        FSDPParamGroup._validate_no_meta_params = _ORIGINAL_FUNCTIONS[
            "validate_no_meta_params"
        ]
    
    if _ORIGINAL_FUNCTIONS["find_common_device"] is not None:
        FakeTensor._find_common_device = _ORIGINAL_FUNCTIONS["find_common_device"]
    
    if _ORIGINAL_FUNCTIONS["wrap_meta_outputs"] is not None:
        FakeTensorMode.wrap_meta_outputs_with_default_device_logic = (
            _ORIGINAL_FUNCTIONS["wrap_meta_outputs"]
        )

    if _ORIGINAL_FUNCTIONS["foreach_reduce"] is not None:
        _fsdp_collectives.foreach_reduce = _ORIGINAL_FUNCTIONS["foreach_reduce"]
        _fsdp_param_group.foreach_reduce = _ORIGINAL_FUNCTIONS["foreach_reduce"]

    if _ORIGINAL_FUNCTIONS["unimplemented_deepcopy"] is not None:
        _fsdp_fully_shard._unimplemented_deepcopy = _ORIGINAL_FUNCTIONS[
            "unimplemented_deepcopy"
        ]

    if _ORIGINAL_FUNCTIONS["to_empty"] is not None:
        torch.nn.Module.to_empty = _ORIGINAL_FUNCTIONS["to_empty"]

    if _ORIGINAL_FUNCTIONS["repeat_interleave"] is not None:
        from torch._subclasses import fake_impls as _fake_impls
        _fake_impls.op_implementations_dict[
            torch.ops.aten.repeat_interleave.Tensor
        ] = _ORIGINAL_FUNCTIONS["repeat_interleave"]
    
    # Clear FSDP ops list
    _FSDP_MIXED_OPS.clear()
