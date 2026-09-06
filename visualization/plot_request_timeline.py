#!/usr/bin/env python3
import argparse
import html
from collections import defaultdict
from pathlib import Path

from llm_serving_lab.tracing import TraceRecorder

COLORS = {"prefill": "#4C78A8", "decode": "#F58518"}


def render_timeline(trace_path: Path, output_path: Path) -> None:
    events = [
        event
        for event in TraceRecorder.read(trace_path)
        if event.kind == "request_scheduled" and event.request_id is not None
    ]
    request_ids = sorted({event.request_id for event in events if event.request_id})
    row_by_request = {request_id: index for index, request_id in enumerate(request_ids)}
    max_step = max((event.step for event in events), default=0)
    cell_width = 64
    row_height = 42
    left = 100
    top = 55
    width = left + (max_step + 1) * cell_width + 30
    height = top + max(1, len(request_ids)) * row_height + 65
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>text{font-family:ui-monospace,monospace;font-size:12px}.title{font-family:system-ui,sans-serif;font-size:18px;font-weight:700}</style>",
        '<text class="title" x="20" y="28">Request scheduling timeline</text>',
    ]
    for step in range(max_step + 1):
        x = left + step * cell_width
        parts.append(f'<text x="{x + 8}" y="48">step {step}</text>')
        parts.append(
            f'<line x1="{x}" y1="{top}" x2="{x}" y2="{height - 45}" stroke="#e5e7eb"/>'
        )
    grouped: dict[tuple[str, int], list] = defaultdict(list)
    for event in events:
        grouped[(event.request_id or "", event.step)].append(event)
    for request_id, row in row_by_request.items():
        y = top + row * row_height
        parts.append(f'<text x="20" y="{y + 25}">{html.escape(request_id)}</text>')
        parts.append(
            f'<line x1="{left}" y1="{y + row_height}" '
            f'x2="{width - 20}" y2="{y + row_height}" '
            'stroke="#f0f0f0"/>'
        )
        for step in range(max_step + 1):
            items = grouped.get((request_id, step), [])
            if not items:
                continue
            event = items[0]
            phase = str(event.fields.get("phase", "unknown"))
            count = event.fields.get("num_scheduled_tokens", "?")
            color = COLORS.get(phase, "#9CA3AF")
            x = left + step * cell_width + 4
            parts.append(
                f'<rect x="{x}" y="{y + 6}" width="{cell_width - 8}" '
                f'height="28" rx="5" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{x + 8}" y="{y + 25}" fill="white">'
                f"{phase[0].upper()}:{count}</text>"
            )
    legend_y = height - 20
    parts.extend(
        [
            (
                f'<rect x="{left}" y="{legend_y - 12}" width="14" '
                f'height="14" fill="{COLORS["prefill"]}"/>'
            ),
            f'<text x="{left + 20}" y="{legend_y}">prefill</text>',
            (
                f'<rect x="{left + 100}" y="{legend_y - 12}" width="14" '
                f'height="14" fill="{COLORS["decode"]}"/>'
            ),
            f'<text x="{left + 120}" y="{legend_y}">decode</text>',
        ]
    )
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render_timeline(args.trace, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
