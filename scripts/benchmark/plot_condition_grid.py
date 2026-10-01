#!/usr/bin/env python3
"""Plot QED/SA condition-grid metrics from eval_condition_grid.py output."""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _plot_metric(ax, rows: list[dict[str, str]], qed_bins: list[str], sa_bins: list[str], metric: str, title: str):
    import numpy as np

    grid = np.full((len(qed_bins), len(sa_bins)), np.nan, dtype=float)
    qpos = {name: i for i, name in enumerate(qed_bins)}
    spos = {name: i for i, name in enumerate(sa_bins)}
    for row in rows:
        qbin = str(row["target_qed_bin"]).upper()
        sbin = str(row["target_sa_bin"]).upper()
        if qbin in qpos and sbin in spos:
            grid[qpos[qbin], spos[sbin]] = _to_float(row[metric])

    image = ax.imshow(grid, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(sa_bins)), sa_bins)
    ax.set_yticks(range(len(qed_bins)), qed_bins)
    ax.set_xlabel("SA condition")
    ax.set_ylabel("QED condition")
    for i in range(len(qed_bins)):
        for j in range(len(sa_bins)):
            value = grid[i, j]
            label = "nan" if not math.isfinite(float(value)) else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", color="white", fontsize=8)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="condition_grid_summary.tsv")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/stamp_mpl_config")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_rows(Path(args.summary))
    qed_bins = ["LOW", "MID", "HIGH", "VHIGH"]
    sa_bins = ["LOW", "MID", "HIGH"]
    metrics = [
        ("mean_qed", "Mean QED"),
        ("mean_sa", "Mean SA"),
        ("quality", "Quality (%)"),
        ("target_qed_hit", "QED Target Hit (%)"),
        ("target_sa_hit", "SA Target Hit (%)"),
        ("uniqueness", "Uniqueness (%)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        image = _plot_metric(ax, rows, qed_bins, sa_bins, metric, title)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle("STAMP QED/SA Condition Grid", fontsize=14)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    fig.savefig(prefix.with_suffix(".pdf"))
    print(prefix.with_suffix(".png"))
    print(prefix.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
