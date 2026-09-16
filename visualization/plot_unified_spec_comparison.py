#!/usr/bin/env python3
"""Render the unified speculative benchmark as a dependency-free SVG."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

COLORS = {
    "ar": "#64748b",
    "eagle3": "#8b5cf6",
    "dflash": "#f59e0b",
    "dflare": "#2563eb",
    "dspark": "#10b981",
}


def render(source: Path, destination: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = data["methods"]
    maximum = max(float(row["speedup_vs_ar"]) for row in rows)
    width, height = 900, 150 + 94 * len(rows)
    chart_left, chart_width = 190, 570
    gpu = html.escape(data["environment"]["gpu"])
    title = html.escape(data["target"]["label"])
    validity = data.get("measurement_validity", {})
    provisional = validity.get("status") == "provisional_shared_gpu"
    status_text = (
        "PROVISIONAL: another process occupied the GPU during measurement"
        if provisional
        else "VALID: no pre-method GPU contention detected"
    )
    status_color = "#b45309" if provisional else "#047857"
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="40" y="42" font-family="sans-serif" font-size="24" '
        f'font-weight="700" fill="#0f172a">{title} unified vLLM benchmark</text>',
        f'<text x="40" y="70" font-family="sans-serif" font-size="14" '
        f'fill="#475569">{gpu}; median of controlled offline batches</text>',
        '<text x="40" y="96" font-family="sans-serif" font-size="13" '
        'fill="#64748b">Bar: speedup vs AR · detail: output tok/s and '
        "committed tokens/step</text>",
        f'<text x="40" y="118" font-family="sans-serif" font-size="13" '
        f'font-weight="700" fill="{status_color}">{status_text}</text>',
    ]
    for index, row in enumerate(rows):
        y = 140 + index * 94
        method = str(row["method"])
        speedup = float(row["speedup_vs_ar"])
        bar_width = chart_width * speedup / maximum
        color = COLORS.get(method, "#334155")
        semantic_exact = row.get("exact_match_ar_until_stop", row["exact_match_ar"])
        strict_exact = row.get("exact_match_ar_full_length", row["exact_match_ar"])
        correctness = "lossless-to-EOS" if semantic_exact else "MISMATCH"
        if semantic_exact and not strict_exact:
            correctness += "; post-EOS differs"
        pieces.extend(
            [
                f'<text x="40" y="{y + 25}" font-family="sans-serif" '
                f'font-size="17" font-weight="700" fill="#0f172a">'
                f"{html.escape(method.upper())}</text>",
                f'<rect x="{chart_left}" y="{y}" width="{bar_width:.1f}" '
                f'height="34" rx="6" fill="{color}"/>',
                f'<text x="{chart_left + bar_width + 10:.1f}" y="{y + 23}" '
                f'font-family="monospace" font-size="15" fill="#0f172a">'
                f"{speedup:.2f}x</text>",
                f'<text x="{chart_left}" y="{y + 57}" '
                f'font-family="sans-serif" font-size="13" fill="#475569">'
                f"{row['median_output_tokens_per_second']:.2f} tok/s · "
                f"{row['mean_committed_tokens_per_decode_step']:.2f} committed/step "
                f"· K={row['num_speculative_tokens']} · {correctness}</text>",
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
