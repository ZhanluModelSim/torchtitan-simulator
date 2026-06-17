# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from typing import Any

from .chrome_trace import export_chrome_trace
from .dot_export import export_dot
from .html_export import export_html
from .json_export import export_json
from .text_summary import export_text_summary


def export_result(
    result: Any,
    output_dir: str,
    output_formats: list[str],
    log_fn: Any | None = None,
    print_summary: bool = False,
) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    if "json" in output_formats:
        p = os.path.join(output_dir, "simulation_result.json")
        export_json(result, p)
        if log_fn:
            log_fn(f"JSON → {p}")
    if "dot" in output_formats:
        p = os.path.join(output_dir, "compute_graph.dot")
        export_dot(result.compute_graph, p)
        if log_fn:
            log_fn(f"DOT  → {p}")
    if "chrome_trace" in output_formats:
        p = os.path.join(output_dir, "trace.json")
        export_chrome_trace(result, p)
        if log_fn:
            log_fn(f"Chrome trace → {p}")
    if "html" in output_formats:
        p = os.path.join(output_dir, "trace.html")
        export_html(result, p)
        if log_fn:
            log_fn(f"HTML trace → {p}")
    if "text" in output_formats:
        summary = export_text_summary(result)
        p = os.path.join(output_dir, "summary.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(summary)
        if log_fn:
            log_fn(f"Text summary → {p}")
        if print_summary:
            print(summary)
