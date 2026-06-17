# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

_RECORDER_STACK: list[Any] = []


def get_current_recorder() -> Any | None:
    return _RECORDER_STACK[-1] if _RECORDER_STACK else None


def push_recorder(recorder: Any) -> None:
    _RECORDER_STACK.append(recorder)


def pop_recorder() -> Any | None:
    return _RECORDER_STACK.pop() if _RECORDER_STACK else None
