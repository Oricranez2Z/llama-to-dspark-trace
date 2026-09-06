#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def render_acceptance(summary_path: Path, output_path: Path) -> None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload["results"]
    width = 820
    row_height = 38
    height = 85 + len(rows) * row_height
    max_value = max((row["mean_accepted_length"] for row in rows), default=1) or 1
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:ui-monospace,monospace;font-size:12px}.title{font-family:system-ui,sans-serif;font-size:18px;font-weight:700}</style>",
        '<text class="title" x="20" y="28">Mean accepted draft length</text>',
    ]
    for index, row in enumerate(rows):
        y = 55 + index * row_height
        label = row["method"]
        if row["method"] == "fixed":
            label += f" k={row['draft_length']}"
        else:
            label += f" c={row['concurrency']}"
        value = float(row["mean_accepted_length"])
        bar_width = 500 * value / max_value
        color = "#4C78A8" if row["method"] == "fixed" else "#54A24B"
        parts.append(f'<text x="20" y="{y + 20}">{label}</text>')
        parts.append(
            f'<rect x="170" y="{y}" width="{bar_width:.1f}" '
            f'height="25" rx="4" fill="{color}"/>'
        )
        parts.append(f'<text x="{180 + bar_width:.1f}" y="{y + 18}">{value:.2f}</text>')
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render_acceptance(args.summary, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
