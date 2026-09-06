#!/usr/bin/env python3
"""Render vLLM V1 scheduler events as a compact dependency-free SVG."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from llm_serving_lab.tracing import TraceRecorder


def render(trace_path: Path, output_path: Path) -> None:
    events = [
        event
        for event in TraceRecorder.read(trace_path)
        if event.kind == "vllm_scheduler_step"
    ]
    request_ids = sorted(
        {
            str(request_id)
            for event in events
            for request_id in event.fields.get("scheduled_tokens", {})
        }
    )
    rows = {request_id: index for index, request_id in enumerate(request_ids)}
    max_step = max((event.step for event in events), default=0)
    cell_width, row_height, left, top = 70, 44, 145, 58
    width = left + (max_step + 1) * cell_width + 30
    height = top + max(1, len(rows)) * row_height + 55
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        "<style>text{font-family:ui-monospace,monospace;font-size:12px}.title{font-family:system-ui,sans-serif;font-size:18px;font-weight:700}</style>",
        '<text class="title" x="20" y="28">Measured vLLM scheduler trace</text>',
    ]
    for step in range(max_step + 1):
        x = left + step * cell_width
        parts.append(f'<text x="{x + 10}" y="50">step {step}</text>')
        parts.append(
            f'<line x1="{x}" y1="{top}" x2="{x}" y2="{height - 30}" stroke="#e5e7eb"/>'
        )
    for request_id, row in rows.items():
        y = top + row * row_height
        label = html.escape(request_id[:16] + ("…" if len(request_id) > 16 else ""))
        parts.append(f'<text x="20" y="{y + 28}">{label}</text>')
        for event in events:
            count = event.fields.get("scheduled_tokens", {}).get(request_id)
            if count is None:
                continue
            phase = "prefill" if int(count) > 1 else "decode"
            color = "#4C78A8" if phase == "prefill" else "#F58518"
            x = left + event.step * cell_width + 5
            parts.append(
                f'<rect x="{x}" y="{y + 7}" width="{cell_width - 10}" '
                f'height="28" rx="5" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{x + 8}" y="{y + 26}" fill="white">'
                f"{phase[0].upper()}:{count}</text>"
            )
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.trace, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
