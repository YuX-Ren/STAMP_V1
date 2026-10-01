#!/usr/bin/env python3
"""Recompute MMP-aware edit-locality metrics from MoleculeACE pair CSV.

This intentionally avoids full-sequence normalized edit distance. The main
quantities are raw token edit distance and token edit distance per MMP changed
heavy-atom count, derived from the MMP variable fragments.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(x, y, nan_policy="omit").correlation)
    except Exception:
        xr = pd.Series(x).rank(method="average").to_numpy()
        yr = pd.Series(y).rank(method="average").to_numpy()
        if len(xr) < 2 or np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
            return math.nan
        return float(np.corrcoef(xr, yr)[0, 1])


def _display_rep_name(rep: str) -> str:
    return {
        "stamp": "STAMP",
        "safe": "SAFE",
        "smiles_bpe": "SMILES-BPE",
    }.get(rep, rep)


def _rep_colors(reps: Sequence[str]) -> List[str]:
    palette = {
        "stamp": "#2f6f73",
        "safe": "#c46a2b",
        "smiles_bpe": "#4f5f9f",
    }
    fallback = ["#b84a62", "#7a9b3f", "#6c4a8f"]
    out: List[str] = []
    for i, rep in enumerate(reps):
        out.append(palette.get(rep, fallback[i % len(fallback)]))
    return out


def infer_representations(columns: Iterable[str]) -> List[str]:
    reps: List[str] = []
    for col in columns:
        if col.endswith("_edit_distance") and not col.endswith("_norm_edit_distance"):
            reps.append(col[: -len("_edit_distance")])
    preferred = ["stamp", "safe", "smiles_bpe"]
    return [r for r in preferred if r in reps] + sorted(set(reps) - set(preferred))


def load_metrics(input_csv: Path, reps: Sequence[str]) -> pd.DataFrame:
    base_cols = [
        "task",
        "ecfp4_tanimoto_distance",
        "variable_a_heavy_atoms",
        "variable_b_heavy_atoms",
        "context_heavy_atoms",
        "n_mmp_contexts",
    ]
    edit_cols = [f"{rep}_edit_distance" for rep in reps]
    len_cols = [c for rep in reps for c in (f"{rep}_len_a", f"{rep}_len_b") if c]
    usecols = [c for c in base_cols + edit_cols + len_cols if c]
    df = pd.read_csv(input_csv, usecols=lambda c: c in set(usecols))

    df["variable_heavy_atoms_sum"] = (
        df["variable_a_heavy_atoms"].fillna(0).astype(float)
        + df["variable_b_heavy_atoms"].fillna(0).astype(float)
    ).clip(lower=1.0)
    df["variable_heavy_atoms_max"] = np.maximum(
        df["variable_a_heavy_atoms"].fillna(0).astype(float),
        df["variable_b_heavy_atoms"].fillna(0).astype(float),
    ).clip(lower=1.0)
    df["variable_heavy_atoms_delta_abs"] = np.abs(
        df["variable_a_heavy_atoms"].fillna(0).astype(float)
        - df["variable_b_heavy_atoms"].fillna(0).astype(float)
    )
    df["low_chem_distance"] = df["ecfp4_tanimoto_distance"] <= 0.3

    long_parts: List[pd.DataFrame] = []
    for rep in reps:
        edit_col = f"{rep}_edit_distance"
        if edit_col not in df:
            continue
        part = df[
            [
                "task",
                "ecfp4_tanimoto_distance",
                "variable_heavy_atoms_sum",
                "variable_heavy_atoms_max",
                "variable_heavy_atoms_delta_abs",
                "context_heavy_atoms",
                "n_mmp_contexts",
                "low_chem_distance",
                edit_col,
            ]
        ].copy()
        part["rep"] = rep
        part["edit_distance"] = part[edit_col].astype(float)
        part.drop(columns=[edit_col], inplace=True)
        part["edit_per_variable_atom_sum"] = part["edit_distance"] / part["variable_heavy_atoms_sum"]
        part["edit_per_variable_atom_max"] = part["edit_distance"] / part["variable_heavy_atoms_max"]
        long_parts.append(part)
    return pd.concat(long_parts, ignore_index=True)


def summarize(long_df: pd.DataFrame, reps: Sequence[str]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "benchmark_type": "mmp_aware_token_edit_locality",
        "scope": (
            "Tokenizer-level edit locality. Primary metrics are raw token edit "
            "distance and token edit distance per MMP variable heavy atom. "
            "Full-sequence normalized edit distance is intentionally not used."
        ),
        "n_pairs": int(len(long_df) / max(len(reps), 1)),
        "n_tasks": int(long_df["task"].nunique()),
        "low_chem_threshold": 0.3,
        "representations": {},
    }
    for rep in reps:
        sub = long_df[long_df["rep"] == rep].copy()
        valid = np.isfinite(sub["edit_distance"].to_numpy(dtype=float))
        sub = sub[valid]
        low = sub[sub["low_chem_distance"]]
        rep_stats = {
            "valid_pairs": int(len(sub)),
            "median_raw_edit": float(sub["edit_distance"].median()),
            "mean_raw_edit": float(sub["edit_distance"].mean()),
            "median_edit_per_variable_atom_sum": float(sub["edit_per_variable_atom_sum"].median()),
            "mean_edit_per_variable_atom_sum": float(sub["edit_per_variable_atom_sum"].mean()),
            "median_edit_per_variable_atom_max": float(sub["edit_per_variable_atom_max"].median()),
            "mean_edit_per_variable_atom_max": float(sub["edit_per_variable_atom_max"].mean()),
            "spearman_chem_distance_vs_raw_edit": _spearman(
                sub["ecfp4_tanimoto_distance"].to_numpy(dtype=float),
                sub["edit_distance"].to_numpy(dtype=float),
            ),
            "spearman_variable_sum_vs_raw_edit": _spearman(
                sub["variable_heavy_atoms_sum"].to_numpy(dtype=float),
                sub["edit_distance"].to_numpy(dtype=float),
            ),
            "low_chem_pairs": int(len(low)),
        }
        if len(low):
            rep_stats.update(
                {
                    "low_chem_median_raw_edit": float(low["edit_distance"].median()),
                    "low_chem_mean_raw_edit": float(low["edit_distance"].mean()),
                    "low_chem_median_edit_per_variable_atom_sum": float(
                        low["edit_per_variable_atom_sum"].median()
                    ),
                    "low_chem_mean_edit_per_variable_atom_sum": float(
                        low["edit_per_variable_atom_sum"].mean()
                    ),
                    "low_chem_median_edit_per_variable_atom_max": float(
                        low["edit_per_variable_atom_max"].median()
                    ),
                }
            )
        summary["representations"][rep] = rep_stats
    return summary


def per_task_table(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for (task, rep), sub in long_df.groupby(["task", "rep"], sort=True):
        low = sub[sub["low_chem_distance"]]
        rows.append(
            {
                "task": task,
                "rep": rep,
                "n_pairs": int(len(sub)),
                "median_raw_edit": float(sub["edit_distance"].median()),
                "median_edit_per_variable_atom_sum": float(sub["edit_per_variable_atom_sum"].median()),
                "median_edit_per_variable_atom_max": float(sub["edit_per_variable_atom_max"].median()),
                "low_chem_n": int(len(low)),
                "low_chem_median_raw_edit": float(low["edit_distance"].median()) if len(low) else math.nan,
                "low_chem_median_edit_per_variable_atom_sum": (
                    float(low["edit_per_variable_atom_sum"].median()) if len(low) else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_win_rates(input_csv: Path, reps: Sequence[str]) -> pd.DataFrame:
    cols = ["variable_a_heavy_atoms", "variable_b_heavy_atoms"] + [f"{r}_edit_distance" for r in reps]
    df = pd.read_csv(input_csv, usecols=lambda c: c in set(cols))
    denom_sum = (
        df["variable_a_heavy_atoms"].fillna(0).astype(float)
        + df["variable_b_heavy_atoms"].fillna(0).astype(float)
    ).clip(lower=1.0)
    rows: List[dict] = []
    for a, b in combinations(reps, 2):
        av = df[f"{a}_edit_distance"].astype(float)
        bv = df[f"{b}_edit_distance"].astype(float)
        valid = np.isfinite(av) & np.isfinite(bv)
        for metric_name, left, right in [
            ("raw_edit", av[valid], bv[valid]),
            ("edit_per_variable_atom_sum", av[valid] / denom_sum[valid], bv[valid] / denom_sum[valid]),
        ]:
            n = int(len(left))
            rows.append(
                {
                    "metric": metric_name,
                    "left_rep": a,
                    "right_rep": b,
                    "n_pairs": n,
                    "left_lower_rate": float((left < right).sum() / max(n, 1)),
                    "tie_rate": float((left == right).sum() / max(n, 1)),
                    "right_lower_rate": float((left > right).sum() / max(n, 1)),
                    "median_left_minus_right": float((left - right).median()),
                }
            )
    return pd.DataFrame(rows)


def plot_metric_bars(long_df: pd.DataFrame, reps: Sequence[str], out_dir: Path) -> None:
    metrics = [
        ("edit_distance", "median raw token edit", "mmpaware_median_raw_edit.png"),
        (
            "edit_per_variable_atom_sum",
            "median token edit per changed heavy atom (sum)",
            "mmpaware_median_edit_per_variable_atom_sum.png",
        ),
        (
            "edit_per_variable_atom_max",
            "median token edit per changed heavy atom (max)",
            "mmpaware_median_edit_per_variable_atom_max.png",
        ),
    ]
    colors = _rep_colors(reps)
    for metric, ylabel, filename in metrics:
        all_vals = [long_df[long_df["rep"] == rep][metric].median() for rep in reps]
        low_vals = [
            long_df[(long_df["rep"] == rep) & (long_df["low_chem_distance"])][metric].median()
            for rep in reps
        ]
        x = np.arange(len(reps), dtype=float)
        width = 0.36
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.bar(x - width / 2, all_vals, width=width, label="all MMPs", color=colors, alpha=0.82)
        ax.bar(x + width / 2, low_vals, width=width, label="ECFP4 distance <= 0.3", color=colors, alpha=0.42)
        ax.set_xticks(x)
        ax.set_xticklabels([_display_rep_name(r) for r in reps])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=220)
        plt.close(fig)


def plot_by_bins(long_df: pd.DataFrame, reps: Sequence[str], out_dir: Path) -> None:
    chem_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    chem_labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    size_bins = [0, 4, 8, 12, 20, np.inf]
    size_labels = ["1-4", "5-8", "9-12", "13-20", ">20"]

    for metric, ylabel, filename, bin_col, bins, labels in [
        (
            "edit_distance",
            "raw token edit",
            "mmpaware_raw_edit_by_chem_bin.png",
            "ecfp4_tanimoto_distance",
            chem_bins,
            chem_labels,
        ),
        (
            "edit_per_variable_atom_sum",
            "token edit per changed heavy atom (sum)",
            "mmpaware_edit_per_atom_by_variable_size.png",
            "variable_heavy_atoms_sum",
            size_bins,
            size_labels,
        ),
    ]:
        fig, axes = plt.subplots(1, len(reps), figsize=(5.0 * len(reps), 4.2), squeeze=False)
        for ax, rep in zip(axes[0], reps):
            sub = long_df[long_df["rep"] == rep]
            values = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (sub[bin_col] > lo) & (sub[bin_col] <= hi)
                values.append(sub.loc[mask, metric].dropna().to_numpy(dtype=float))
            ax.boxplot(values, tick_labels=labels, showfliers=False)
            ax.set_title(_display_rep_name(rep))
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.22)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=220)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--reps", nargs="*", default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(input_csv, nrows=0)
    reps = args.reps or infer_representations(header.columns)
    long_df = load_metrics(input_csv, reps)
    long_df.to_csv(out_dir / "mmpaware_long_metrics.csv", index=False)

    summary = summarize(long_df, reps)
    (out_dir / "mmpaware_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    per_task_table(long_df).to_csv(out_dir / "mmpaware_per_task_metrics.csv", index=False)
    paired_win_rates(input_csv, reps).to_csv(out_dir / "mmpaware_paired_win_rates.csv", index=False)

    plot_metric_bars(long_df, reps, out_dir)
    plot_by_bins(long_df, reps, out_dir)
    print(f"saved MMP-aware locality outputs to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
