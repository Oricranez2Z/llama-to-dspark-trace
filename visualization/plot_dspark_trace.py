#!/usr/bin/env python3
"""Render accepted and rejected draft tokens from a DeepSpec JSONL trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(trace_path: Path, output_path: Path) -> None:
    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    bar_width, gap, left, top = 54, 18, 70, 58
    plot_height = 180
    width = left + len(rows) * (bar_width + gap) + 40
    height = top + plot_height + 65
    max_tokens = max((int(row["effective_proposal_length"]) for row in rows), default=1)
    scale = plot_height / max_tokens
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        "<style>text{font-family:ui-monospace,monospace;font-size:12px}.title{font-family:system-ui,sans-serif;font-size:18px;font-weight:700}</style>",
        '<text class="title" x="20" y="28">Measured DSpark verification rounds</text>',
    ]
    baseline = top + plot_height
    parts.append(
        f'<line x1="{left - 8}" y1="{baseline}" x2="{width - 20}" '
        f'y2="{baseline}" stroke="#64748b"/>'
    )
    for index, row in enumerate(rows):
        accepted = int(row["accepted_draft_tokens"])
        proposed = int(row["effective_proposal_length"])
        rejected = proposed - accepted
        x = left + index * (bar_width + gap)
        accepted_height = accepted * scale
        rejected_height = rejected * scale
        parts.append(
            f'<rect x="{x}" y="{baseline - accepted_height}" '
            f'width="{bar_width}" height="{accepted_height}" fill="#54A24B"/>'
        )
        parts.append(
            f'<rect x="{x}" y="{baseline - accepted_height - rejected_height}" '
            f'width="{bar_width}" height="{rejected_height}" fill="#E45756"/>'
        )
        parts.append(f'<text x="{x + 15}" y="{baseline + 20}">r{index}</text>')
        parts.append(
            f'<text x="{x + 13}" y="{baseline - accepted_height + 16}" '
            f'fill="white">{accepted}/{proposed}</text>'
        )
    parts.extend(
        [
            f'<rect x="{left}" y="{height - 22}" width="12" '
            'height="12" fill="#54A24B"/>',
            f'<text x="{left + 18}" y="{height - 12}">accepted</text>',
            f'<rect x="{left + 110}" y="{height - 22}" width="12" '
            'height="12" fill="#E45756"/>',
            f'<text x="{left + 128}" y="{height - 12}">rejected</text>',
            "</svg>",
        ]
    )
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
