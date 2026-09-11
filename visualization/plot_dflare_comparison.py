#!/usr/bin/env python3
"""Render a dependency-free SVG comparison of measured DFlare methods."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

COLORS = {"ar": "#64748b", "dflash": "#f59e0b", "dflare": "#2563eb"}


def render(source: Path, destination: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = data["methods"]
    maximum = max(row["speedup_vs_ar"] for row in rows)
    width, height = 760, 120 + 86 * len(rows)
    chart_left, chart_width = 180, 500
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="40" y="42" font-family="sans-serif" font-size="24" '
        'font-weight="700" fill="#0f172a">RTX 8000 FP16/SDPA smoke</text>',
        '<text x="40" y="68" font-family="sans-serif" font-size="14" '
        'fill="#475569">Relative speed from four prompts; not a serving '
        "benchmark</text>",
    ]
    for index, row in enumerate(rows):
        y = 105 + index * 86
        method = str(row["method"])
        speedup = float(row["speedup_vs_ar"])
        bar_width = chart_width * speedup / maximum
        label = html.escape(method.upper())
        color = COLORS.get(method, "#7c3aed")
        pieces.extend(
            [
                f'<text x="40" y="{y + 25}" font-family="sans-serif" '
                f'font-size="17" font-weight="700" fill="#0f172a">'
                f"{label}</text>",
                f'<rect x="{chart_left}" y="{y}" width="{bar_width:.1f}" '
                f'height="34" rx="6" fill="{color}"/>',
                f'<text x="{chart_left + bar_width + 10:.1f}" y="{y + 23}" '
                f'font-family="monospace" font-size="15" fill="#0f172a">'
                f"{speedup:.2f}x</text>",
                f'<text x="{chart_left}" y="{y + 56}" '
                f'font-family="sans-serif" font-size="13" fill="#475569">'
                "mean committed/round "
                f"{row['mean_committed_tokens_per_round']:.2f}</text>",
            ]
        )
    pieces.append("</svg>")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    render(args.source, args.destination)


if __name__ == "__main__":
    main()
