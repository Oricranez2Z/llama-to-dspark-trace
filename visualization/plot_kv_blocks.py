#!/usr/bin/env python3
import argparse
import html
from pathlib import Path

from llm_serving_lab.tracing import TraceRecorder

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]


def render_blocks(trace_path: Path, output_path: Path) -> None:
    events = [
        event
        for event in TraceRecorder.read(trace_path)
        if event.kind == "request_scheduled" and event.request_id is not None
    ]
    request_ids = sorted({event.request_id for event in events if event.request_id})
    color_by_request = {
        request_id: PALETTE[index % len(PALETTE)]
        for index, request_id in enumerate(request_ids)
    }
    max_step = max((event.step for event in events), default=0)
    snapshots: dict[int, dict[int, str]] = {}
    current: dict[int, str] = {}
    finished_by_step: dict[int, list] = {}
    for event in TraceRecorder.read(trace_path):
        if event.kind == "request_finished":
            finished_by_step.setdefault(event.step, []).append(event)
    for step in range(max_step + 1):
        for event in events:
            if event.step != step:
                continue
            for block_id in event.fields.get("block_ids", []):
                current[int(block_id)] = event.request_id or ""
        for finished in finished_by_step.get(step, []):
            for block_id in finished.fields.get("released_block_ids", []):
                current.pop(int(block_id), None)
        snapshots[step] = current.copy()

    all_blocks = sorted(
        {block for snapshot in snapshots.values() for block in snapshot}
    )
    width = max(720, 130 + max(1, len(all_blocks)) * 36)
    height = 90 + (max_step + 1) * 42 + 50
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:ui-monospace,monospace;font-size:11px}.title{font-family:system-ui,sans-serif;font-size:18px;font-weight:700}</style>",
        '<text class="title" x="20" y="28">Physical KV block ownership</text>',
    ]
    for column, block_id in enumerate(all_blocks):
        x = 110 + column * 36
        parts.append(f'<text x="{x + 7}" y="57">{block_id}</text>')
    for step, snapshot in snapshots.items():
        y = 68 + step * 42
        parts.append(f'<text x="20" y="{y + 23}">step {step}</text>')
        for column, block_id in enumerate(all_blocks):
            x = 110 + column * 36
            owner = snapshot.get(block_id)
            color = "#F3F4F6" if owner is None else color_by_request[owner]
            parts.append(
                f'<rect x="{x}" y="{y}" width="30" height="30" '
                f'rx="4" fill="{color}" stroke="#d1d5db"/>'
            )
            if owner:
                parts.append(
                    f'<text x="{x + 9}" y="{y + 20}" fill="white">'
                    f"{html.escape(owner[:2])}</text>"
                )
    legend_x = 110
    legend_y = height - 22
    for index, request_id in enumerate(request_ids):
        x = legend_x + index * 95
        parts.append(
            f'<rect x="{x}" y="{legend_y - 12}" width="14" '
            f'height="14" fill="{color_by_request[request_id]}"/>'
        )
        parts.append(
            f'<text x="{x + 20}" y="{legend_y}">{html.escape(request_id)}</text>'
        )
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render_blocks(args.trace, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
