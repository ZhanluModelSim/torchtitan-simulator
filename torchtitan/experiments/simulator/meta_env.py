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

from .cpu_env import _patch_downstream_modules, _patch_torch_cuda, make_device_module


def _make_meta_device_module():
    return make_device_module(
        device_count=0, device_name="Meta_Simulator", total_memory=80 * 1024**3
    )


def patch_device_type_to_meta() -> None:
    """Monkey-patch ``torchtitan.tools.utils.device_type`` and
    ``torchtitan.tools.utils.device_module`` to ``"meta"``.

    Also patches downstream modules that have already imported
    ``device_module`` / ``device_type`` at module scope.

    Additionally patches ``torch.cuda`` entrypoints with meta stubs
    (same approach as ``cpu_env._patch_torch_cuda``, but
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

    _patch_downstream_modules("meta", meta_mod)
    _patch_torch_cuda(meta_mod)
