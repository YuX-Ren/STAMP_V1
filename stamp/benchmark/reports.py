from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def save_experiment_markdown_tables(
    representation_stats_json: str | Path,
    denovo_metrics_jsons: Sequence[str | Path],
    output_md: str | Path,
) -> None:
    rep = json.loads(Path(representation_stats_json).read_text(encoding="utf-8"))

    lines = [
        "# Experiment Tables",
        "",
        "## Table 1: Representation statistics",
        "",
        "| Repr | Avg char | Avg tok | P95 tok | Max tok | Vocab | Decode valid | Special ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rep.get("results", []):
        if row.get("error"):
            lines.append(f"| {row.get('repr', '')} | - | - | - | - | - | - | - |")
            continue
        lines.append(
            "| {repr} | {avg_char:.2f} | {avg_tok:.2f} | {p95:.2f} | {mx} | {vocab} | {dec:.4f} | {spec:.4f} |".format(
                repr=row.get("repr", ""),
                avg_char=float(row.get("avg_char_length", 0.0)),
                avg_tok=float(row.get("avg_token_length", 0.0)),
                p95=float(row.get("p95_token_length", 0.0)),
                mx=int(row.get("max_token_length", 0)),
                vocab=int(row.get("vocab_size", 0)),
                dec=float(row.get("decode_valid_rate", 0.0)),
                spec=float(row.get("special_token_ratio", 0.0)),
            )
        )

    lines.extend([
        "",
        "## Table 2: De novo generation metrics",
        "",
        "| Repr | Validity | Uniqueness | Novelty | Internal diversity | Quality |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])

    for path in denovo_metrics_jsons:
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        lines.append(
            "| {repr} | {val:.4f} | {uniq:.4f} | {nov:.4f} | {div:.4f} | {quality:.4f} |".format(
                repr=m.get("representation", ""),
                val=float(m.get("validity", 0.0)),
                uniq=float(m.get("uniqueness", 0.0)),
                nov=float(m.get("novelty", 0.0)),
                div=float(m.get("internal_diversity", 0.0)),
                quality=float(m.get("quality", 0.0)),
            )
        )

    out = Path(output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
