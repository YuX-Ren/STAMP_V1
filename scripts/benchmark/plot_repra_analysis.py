#!/usr/bin/env python3
"""Plot aggregate RePRA distance-analysis figures from repra_full.json."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",   # editable text in SVG
    "pdf.fonttype": 42,       # editable TrueType in PDF
    "ps.fonttype": 42,
})


def _save_fig(fig, out_path: Path) -> None:
    """Save raster PNG + editable vector PDF/SVG (publication export contract)."""
    base = Path(out_path).with_suffix("")
    fig.savefig(base.with_suffix(".png"), dpi=220)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")


def load_rows(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[dict] = []
    for task, reps in data.items():
        if not isinstance(reps, dict):
            continue
        for rep, stats in reps.items():
            if not isinstance(stats, dict) or "C_AC" not in stats:
                continue
            total_pairs = max(float(stats.get("total_pairs", 0)), 1.0)
            c_ac = float(stats.get("C_AC", math.nan))
            c_sh = float(stats.get("C_SH", math.nan))
            rows.append(
                {
                    "task": task,
                    "rep": rep,
                    "C_AC": c_ac,
                    "C_SH": c_sh,
                    "total_pairs": total_pairs,
                    "AC_rate": c_ac / total_pairs,
                    "SH_rate": c_sh / total_pairs,
                    "noise_rate": (c_ac + c_sh) / total_pairs,
                    "delta_AC": float(stats.get("delta_AC", math.nan)),
                    "delta_SH": float(stats.get("delta_SH", math.nan)),
                    "eps_AC": float(stats.get("eps_AC", math.nan)),
                    "eps_SH": float(stats.get("eps_SH", math.nan)),
                    "e": float(stats.get("e", math.nan)),
                    "dbar": float(stats.get("dbar", math.nan)),
                    "sAD": float(stats.get("sAD", math.nan)),
                    "sIR": float(stats.get("sIR", math.nan)),
                    "n_valid": int(stats.get("n_valid", 0)),
                    "n_label_dims": int(stats.get("n_label_dims", 0)),
                }
            )
    return pd.DataFrame(rows)


def _ordered(values: Iterable[str], preferred: List[str]) -> List[str]:
    values_set = set(values)
    out = [v for v in preferred if v in values_set]
    out.extend(sorted(values_set - set(out)))
    return out


def _rep_colors(reps: List[str]) -> List[str]:
    palette = {
        "MorganFP": "#6b7280",
        "stamp_mlm": "#2f6f73",
        "stamp_mtr": "#c46a2b",
        "chemberta": "#8c4a8f",
    }
    fallback = ["#4f5f9f", "#b84a62", "#7a9b3f", "#6c4a8f"]
    colors = []
    for i, rep in enumerate(reps):
        colors.append(palette.get(rep, fallback[i % len(fallback)]))
    return colors


def plot_grouped_bar(
    df: pd.DataFrame,
    *,
    value: str,
    ylabel: str,
    out_path: Path,
    title: str,
    baseline_y: float | None = None,
    reps: List[str],
    tasks: List[str],
) -> None:
    pivot = df.pivot_table(index="task", columns="rep", values=value, aggfunc="mean").reindex(tasks)
    reps = [r for r in reps if r in pivot.columns]
    x = np.arange(len(tasks), dtype=float)
    width = min(0.78 / max(len(reps), 1), 0.28)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.62 * len(tasks)), 4.8))
    colors = _rep_colors(reps)
    for i, (rep, color) in enumerate(zip(reps, colors)):
        offset = (i - (len(reps) - 1) / 2.0) * width
        ax.bar(x + offset, pivot[rep].to_numpy(dtype=float), width=width, label=rep, color=color)
    if baseline_y is not None:
        ax.axhline(baseline_y, color="#111827", linestyle="--", linewidth=1.1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncols=min(len(reps), 3))
    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_ac_sh_rates(df: pd.DataFrame, out_path: Path, reps: List[str], tasks: List[str]) -> None:
    rows = df[df["rep"].isin(reps)].copy()
    fig, axes = plt.subplots(len(reps), 1, figsize=(max(8.0, 0.62 * len(tasks)), 3.0 * len(reps)), sharex=True)
    if len(reps) == 1:
        axes = [axes]
    x = np.arange(len(tasks), dtype=float)
    for ax, rep in zip(axes, reps):
        sub = rows[rows["rep"] == rep].set_index("task").reindex(tasks)
        ac = sub["AC_rate"].to_numpy(dtype=float)
        sh = sub["SH_rate"].to_numpy(dtype=float)
        ax.bar(x, ac, label="AC rate", color="#b84a62")
        ax.bar(x, sh, bottom=ac, label="SH rate", color="#4f5f9f")
        ax.set_ylabel(rep)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, ncols=2, loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(tasks, rotation=35, ha="right")
    fig.suptitle("RePRA noise-pair rates: activity cliffs + scaffold hops", y=0.995)
    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_delta_scatter(df: pd.DataFrame, out_path: Path, reps: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    colors = _rep_colors(reps)
    for rep, color in zip(reps, colors):
        sub = df[df["rep"] == rep]
        ax.scatter(
            sub["delta_AC"],
            sub["delta_SH"],
            s=58,
            alpha=0.82,
            label=rep,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
        for _, row in sub.iterrows():
            ax.text(row["delta_AC"] + 0.006, row["delta_SH"] + 0.006, str(row["task"]), fontsize=7, alpha=0.75)
    lim_hi = float(np.nanmax([df["delta_AC"].max(), df["delta_SH"].max(), 1.0]))
    ax.plot([0, lim_hi], [0, lim_hi], color="#111827", linestyle="--", alpha=0.45, linewidth=1.0)
    ax.set_xlabel("delta_AC: mean representation distance among similar-label pairs")
    ax.set_ylabel("delta_SH: mean representation distance among different-label pairs")
    ax.set_title("RePRA distance-threshold separation")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repra-json", default="/scratch/yuxuan.ren/data/pubchem10m/repra_v4_full/repra_full.json")
    ap.add_argument("--output-dir", default="results/repra_analysis_figures")
    ap.add_argument("--reps", nargs="+", default=["MorganFP", "stamp_mlm", "stamp_mtr", "chemberta"])
    ap.add_argument("--tasks", nargs="*", default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(Path(args.repra_json))
    if df.empty:
        raise SystemExit(f"No RePRA rows found in {args.repra_json}")
    reps = _ordered(df["rep"].unique(), args.reps)
    if args.tasks:
        tasks = [t for t in args.tasks if t in set(df["task"])]
    else:
        preferred_tasks = ["bace", "bbbp", "clintox", "sider", "tox21", "esol", "freesolv", "lipophilicity"]
        tasks = _ordered(df["task"].unique(), preferred_tasks)
    df = df[df["task"].isin(tasks) & df["rep"].isin(reps)].copy()
    df.to_csv(out_dir / "repra_metrics_table.csv", index=False)

    plot_grouped_bar(
        df,
        value="sIR",
        ylabel="sIR (lower is better; MorganFP baseline = 2)",
        out_path=out_dir / "repra_sir_bar.png",
        title="RePRA instability ratio",
        baseline_y=2.0,
        reps=reps,
        tasks=tasks,
    )
    plot_grouped_bar(
        df,
        value="sAD",
        ylabel="sAD",
        out_path=out_dir / "repra_sad_bar.png",
        title="RePRA structural-property abnormality degree",
        reps=reps,
        tasks=tasks,
    )
    plot_ac_sh_rates(df, out_dir / "repra_ac_sh_noise_rates.png", reps=reps, tasks=tasks)
    plot_delta_scatter(df, out_dir / "repra_delta_ac_delta_sh.png", reps=reps)
    print(f"saved figures and table to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
