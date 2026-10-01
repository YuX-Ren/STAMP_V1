#!/usr/bin/env python
"""Create focused edit-locality tables and figures from mechanism outputs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stamp_mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATASETS = {
    "MoleculeACE cliffs": "edit_mechanism",
    "ChEMBL MMP": "edit_mechanism_chembl_mmp",
    "FC-STAMP generated": "edit_mechanism_fcstamp_generation",
}

REPRESENTATION_ORDER = [
    "smiles",
    "selfies",
    "safe",
    "fragsmiles",
    "group-selfies",
    "stamp",
]

REPRESENTATION_LABELS = {
    "smiles": "SMILES",
    "selfies": "SELFIES",
    "safe": "SAFE",
    "fragsmiles": "fragSMILES",
    "group-selfies": "Group-SELFIES",
    "stamp": "STAMP",
}

PALETTE = {
    "SMILES": "#6F6F6F",
    "SELFIES": "#4A8C8A",
    "SAFE": "#3C73B9",
    "fragSMILES": "#55934D",
    "Group-SELFIES": "#8D6CAB",
    "STAMP": "#B64032",
}

PATCH_KIND_ORDER = [
    "motif_patch",
    "joint_patch",
    "motif_joint_patch",
    "branch_topology_patch",
    "subtree_patch",
    "no_change",
    "unknown",
]

PATCH_KIND_LABELS = {
    "motif_patch": "Motif",
    "joint_patch": "Joint",
    "motif_joint_patch": "Motif+joint",
    "branch_topology_patch": "Branch",
    "subtree_patch": "Subtree",
    "no_change": "No change",
    "unknown": "Unknown",
}

PATCH_KIND_PALETTE = {
    "motif_patch": "#B64032",
    "joint_patch": "#3C73B9",
    "motif_joint_patch": "#8D6CAB",
    "branch_topology_patch": "#D9A35F",
    "subtree_patch": "#55934D",
    "no_change": "#B7B7B7",
    "unknown": "#6F6F6F",
}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for suffix in (".png", ".pdf"):
        fig.savefig(out_dir / f"{name}{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_cross_tables(root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for dataset, subdir in DATASETS.items():
        path = root / subdir / "cross_language_edit_locality_pairs.csv"
        df = pd.read_csv(path)
        df["dataset"] = dataset
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["representation"] = pd.Categorical(
        out["representation"],
        categories=REPRESENTATION_ORDER,
        ordered=True,
    )
    out["representation_label"] = out["representation"].map(REPRESENTATION_LABELS)
    out["dataset"] = pd.Categorical(out["dataset"], categories=list(DATASETS), ordered=True)
    out["subtree_contiguous_edit"] = (
        out["subtree_contiguous_edit"].astype("boolean").fillna(False).astype(bool)
    )
    out["one_local_span"] = out["one_local_span"].astype("boolean").fillna(False).astype(bool)
    return out


def _load_decomposition_tables(root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for dataset, subdir in DATASETS.items():
        path = root / subdir / "stamp_edit_decomposition_pairs.csv"
        df = pd.read_csv(path)
        df["dataset"] = dataset
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["dataset"] = pd.Categorical(out["dataset"], categories=list(DATASETS), ordered=True)
    for col in (
        "stamp_one_local_span",
        "stamp_subtree_contiguous_edit",
        "stamp_executable_as_one_typed_patch",
        "stamp_transferable_to_compatible_scaffold",
    ):
        if col in out:
            out[col] = out[col].astype("boolean").fillna(False).astype(bool)
    return out


def summarize_cross(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["dataset", "representation", "representation_label"], observed=True)
    rows = []
    for keys, group in grouped:
        dataset, representation, label = keys
        rows.append(
            {
                "dataset": dataset,
                "representation": representation,
                "representation_label": label,
                "n_pairs": group["pair_id"].nunique(),
                "mean_norm_edit": group["normalized_token_edit_distance"].mean(),
                "median_norm_edit": group["normalized_token_edit_distance"].median(),
                "mean_edit_script_length": group["edit_script_length"].mean(),
                "median_edit_script_length": group["edit_script_length"].median(),
                "mean_changed_token_fraction": group["changed_token_fraction"].mean(),
                "median_changed_token_fraction": group["changed_token_fraction"].median(),
                "mean_changed_spans": group["changed_span_count"].mean(),
                "median_changed_spans": group["changed_span_count"].median(),
                "mean_disconnected_edit_regions": group["disconnected_edit_regions"].mean(),
                "median_disconnected_edit_regions": group["disconnected_edit_regions"].median(),
                "mean_largest_span": group["largest_changed_span"].mean(),
                "median_largest_span": group["largest_changed_span"].median(),
                "p90_largest_span": group["largest_changed_span"].quantile(0.90),
                "mean_minimal_contiguous_edit_span": group["minimal_contiguous_edit_span"].mean(),
                "median_minimal_contiguous_edit_span": group["minimal_contiguous_edit_span"].median(),
                "one_span_rate": group["one_local_span"].mean(),
                "subtree_contiguous_rate": group["subtree_contiguous_edit"].mean(),
            }
        )
    return pd.DataFrame(rows)


def compute_stamp_advantage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, dataset_df in df.groupby("dataset", observed=True):
        for metric in ("largest_changed_span", "changed_span_count"):
            pivot = dataset_df.pivot_table(
                index="pair_id",
                columns="representation",
                values=metric,
                aggfunc="first",
                observed=True,
            )
            if "stamp" not in pivot:
                continue
            stamp = pivot["stamp"]
            for rep in REPRESENTATION_ORDER:
                if rep == "stamp" or rep not in pivot:
                    continue
                comp = pivot[rep]
                matched = pd.concat([stamp, comp], axis=1, keys=["stamp", "comp"]).dropna()
                if matched.empty:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "comparison": f"STAMP vs {REPRESENTATION_LABELS[rep]}",
                        "representation": rep,
                        "n_pairs": int(len(matched)),
                        "stamp_mean": matched["stamp"].mean(),
                        "comparison_mean": matched["comp"].mean(),
                        "compression_ratio": matched["comp"].mean()
                        / max(1e-9, matched["stamp"].mean()),
                        "mean_percent_reduction": (
                            matched["comp"].mean() - matched["stamp"].mean()
                        )
                        / max(1e-9, matched["comp"].mean()),
                        "stamp_win_rate": (matched["stamp"] < matched["comp"]).mean(),
                        "tie_rate": (matched["stamp"] == matched["comp"]).mean(),
                        "stamp_loss_rate": (matched["stamp"] > matched["comp"]).mean(),
                    }
                )
    return pd.DataFrame(rows)


def compute_stamp_vs_fragsmiles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pivot = df[df["representation"].isin(["stamp", "fragsmiles"])].pivot_table(
        index=["dataset", "edit_class", "pair_id"],
        columns="representation",
        values=["largest_changed_span", "changed_span_count", "normalized_token_edit_distance"],
        aggfunc="first",
        observed=True,
    )
    pivot.columns = [f"{metric}_{representation}" for metric, representation in pivot.columns]
    pivot = pivot.reset_index().dropna()
    pivot["delta_largest_span_frag_minus_stamp"] = (
        pivot["largest_changed_span_fragsmiles"] - pivot["largest_changed_span_stamp"]
    )
    pivot["delta_changed_spans_frag_minus_stamp"] = (
        pivot["changed_span_count_fragsmiles"] - pivot["changed_span_count_stamp"]
    )
    pivot["delta_norm_edit_frag_minus_stamp"] = (
        pivot["normalized_token_edit_distance_fragsmiles"]
        - pivot["normalized_token_edit_distance_stamp"]
    )
    pivot["largest_span_outcome"] = pivot["delta_largest_span_frag_minus_stamp"].map(
        lambda value: "STAMP shorter" if value > 0 else ("Tie" if value == 0 else "fragSMILES shorter")
    )
    pivot["changed_span_outcome"] = pivot["delta_changed_spans_frag_minus_stamp"].map(
        lambda value: "STAMP fewer" if value > 0 else ("Tie" if value == 0 else "fragSMILES fewer")
    )

    rows = []
    for dataset, group in pivot.groupby("dataset", observed=True):
        rows.append(_summarize_stamp_frag_group(dataset, "all", group))
    by_dataset = pd.DataFrame(rows)

    rows = []
    for (dataset, edit_class), group in pivot.groupby(["dataset", "edit_class"], observed=True):
        rows.append(_summarize_stamp_frag_group(dataset, edit_class, group))
    by_edit_class = pd.DataFrame(rows)
    return pivot, by_dataset, by_edit_class


def _summarize_stamp_frag_group(dataset: str, edit_class: str, group: pd.DataFrame) -> Dict[str, object]:
    largest_delta = group["delta_largest_span_frag_minus_stamp"]
    span_delta = group["delta_changed_spans_frag_minus_stamp"]
    return {
        "dataset": dataset,
        "edit_class": edit_class,
        "n_pairs": int(len(group)),
        "stamp_mean_largest_span": group["largest_changed_span_stamp"].mean(),
        "fragsmiles_mean_largest_span": group["largest_changed_span_fragsmiles"].mean(),
        "largest_span_ratio_frag_over_stamp": group["largest_changed_span_fragsmiles"].mean()
        / max(1e-9, group["largest_changed_span_stamp"].mean()),
        "mean_largest_span_delta_frag_minus_stamp": largest_delta.mean(),
        "median_largest_span_delta_frag_minus_stamp": largest_delta.median(),
        "stamp_shorter_largest_rate": (largest_delta > 0).mean(),
        "tie_largest_rate": (largest_delta == 0).mean(),
        "fragsmiles_shorter_largest_rate": (largest_delta < 0).mean(),
        "stamp_mean_changed_spans": group["changed_span_count_stamp"].mean(),
        "fragsmiles_mean_changed_spans": group["changed_span_count_fragsmiles"].mean(),
        "changed_span_ratio_frag_over_stamp": group["changed_span_count_fragsmiles"].mean()
        / max(1e-9, group["changed_span_count_stamp"].mean()),
        "mean_changed_span_delta_frag_minus_stamp": span_delta.mean(),
        "stamp_fewer_spans_rate": (span_delta > 0).mean(),
        "tie_span_count_rate": (span_delta == 0).mean(),
        "fragsmiles_fewer_spans_rate": (span_delta < 0).mean(),
    }


def summarize_axes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, edit_class), group in df.groupby(["dataset", "edit_class"], observed=True):
        motif = group["delta_motif"].mean()
        joint = group["delta_joint"].mean()
        branch = group["delta_branch"].mean()
        total = max(1e-9, motif + joint + branch)
        rows.append(
            {
                "dataset": dataset,
                "edit_class": edit_class,
                "n_pairs": int(len(group)),
                "mean_delta_motif": motif,
                "mean_delta_joint": joint,
                "mean_delta_branch": branch,
                "motif_fraction": motif / total,
                "joint_fraction": joint / total,
                "branch_fraction": branch / total,
                "dominant_axis": max(
                    {"motif": motif, "joint": joint, "branch": branch},
                    key={"motif": motif, "joint": joint, "branch": branch}.get,
                ),
                "mean_changed_spans": group["stamp_changed_span_count"].mean(),
                "mean_largest_span": group["stamp_largest_changed_span"].mean(),
                "subtree_contiguous_rate": group["stamp_subtree_contiguous_edit"].mean(),
            }
        )
    return pd.DataFrame(rows)


def summarize_stamp_patches(df: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_keys, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_keys, keys))
        row.update(
            {
                "n_pairs": int(len(group)),
                "mean_edit_script_length": group["stamp_edit_script_length"].mean(),
                "median_edit_script_length": group["stamp_edit_script_length"].median(),
                "mean_changed_token_fraction": group["stamp_changed_token_fraction"].mean(),
                "median_changed_token_fraction": group["stamp_changed_token_fraction"].median(),
                "mean_disconnected_edit_regions": group["stamp_disconnected_edit_regions"].mean(),
                "median_disconnected_edit_regions": group["stamp_disconnected_edit_regions"].median(),
                "mean_minimal_contiguous_edit_span": group[
                    "stamp_minimal_contiguous_edit_span"
                ].mean(),
                "median_minimal_contiguous_edit_span": group[
                    "stamp_minimal_contiguous_edit_span"
                ].median(),
                "one_span_rate": group["stamp_one_local_span"].mean(),
                "subtree_contiguous_rate": group["stamp_subtree_contiguous_edit"].mean(),
                "executable_as_one_typed_patch_rate": group[
                    "stamp_executable_as_one_typed_patch"
                ].mean(),
                "transferable_to_compatible_scaffold_rate": group[
                    "stamp_transferable_to_compatible_scaffold"
                ].mean(),
                "mean_source_patch_span": group["stamp_source_patch_span"].mean(),
                "mean_target_patch_span": group["stamp_target_patch_span"].mean(),
                "mean_executable_patch_span": group["stamp_executable_patch_span"].mean(),
                "median_executable_patch_span": group["stamp_executable_patch_span"].median(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_typed_patch_kinds(df: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_keys, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        counts = (
            group["stamp_typed_patch_kind"]
            .fillna("unknown")
            .replace("", "unknown")
            .value_counts()
        )
        total = max(1, int(counts.sum()))
        for kind in PATCH_KIND_ORDER:
            count = int(counts.get(kind, 0))
            if count == 0 and kind == "unknown":
                continue
            row = dict(zip(group_keys, keys))
            row.update(
                {
                    "patch_kind": kind,
                    "patch_kind_label": PATCH_KIND_LABELS.get(kind, kind),
                    "n_pairs": int(total),
                    "count": count,
                    "rate": count / total,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_largest_span(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.barplot(
        data=summary,
        x="dataset",
        y="mean_largest_span",
        hue="representation_label",
        hue_order=[REPRESENTATION_LABELS[r] for r in REPRESENTATION_ORDER],
        palette=PALETTE,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean largest changed span (tokens)")
    ax.set_title("Largest Changed Span", fontsize=12.5, pad=12)
    ax.legend(title="", ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20), fontsize=8.5)
    sns.despine(fig=fig, ax=ax)
    _save(fig, out_dir, "fig1_largest_changed_span")


def plot_changed_spans(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.barplot(
        data=summary,
        x="dataset",
        y="mean_changed_spans",
        hue="representation_label",
        hue_order=[REPRESENTATION_LABELS[r] for r in REPRESENTATION_ORDER],
        palette=PALETTE,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean number of changed spans")
    ax.set_title("Changed Span Count", fontsize=12.5, pad=12)
    ax.legend(title="", ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20), fontsize=8.5)
    sns.despine(fig=fig, ax=ax)
    _save(fig, out_dir, "fig2_changed_span_count")


def plot_pareto(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    for dataset, group in summary.groupby("dataset", observed=True):
        for _, row in group.iterrows():
            label = str(row["representation_label"])
            ax.scatter(
                row["mean_norm_edit"],
                row["mean_largest_span"],
                s=90 if label == "STAMP" else 52,
                marker="*" if label == "STAMP" else "o",
                color=PALETTE[label],
                edgecolor="black" if label == "STAMP" else "white",
                linewidth=0.9,
                alpha=0.92,
            )
            if label == "STAMP":
                ax.annotate(
                    str(dataset).replace(" ", "\n"),
                    (row["mean_norm_edit"], row["mean_largest_span"]),
                    xytext=(6, 5),
                    textcoords="offset points",
                    fontsize=8.5,
                )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="*" if label == "STAMP" else "o",
            color="w",
            label=label,
            markerfacecolor=PALETTE[label],
            markeredgecolor="black" if label == "STAMP" else "white",
            markersize=11 if label == "STAMP" else 8,
        )
        for label in [REPRESENTATION_LABELS[r] for r in REPRESENTATION_ORDER]
    ]
    ax.legend(handles=handles, title="", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_xlabel("Mean normalized token edit distance")
    ax.set_ylabel("Mean largest changed span (tokens)")
    ax.set_title("Edit Locality Tradeoff", fontsize=12.5, pad=10)
    sns.despine(fig=fig, ax=ax)
    _save(fig, out_dir, "fig3_locality_pareto")


def plot_subtree_rate(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    sns.barplot(
        data=summary,
        x="dataset",
        y="subtree_contiguous_rate",
        hue="representation_label",
        hue_order=[REPRESENTATION_LABELS[r] for r in REPRESENTATION_ORDER],
        palette=PALETTE,
        ax=ax,
    )
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("")
    ax.set_ylabel("Subtree-contiguous edit rate")
    ax.set_title("Subtree-Contiguous Edit Rate", fontsize=12.5, pad=12)
    ax.legend(title="", ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=8.5)
    sns.despine(fig=fig, ax=ax)
    _save(fig, out_dir, "fig4_subtree_contiguity")


def plot_axes_heatmap(axes_summary: pd.DataFrame, out_dir: Path) -> None:
    plot_df = axes_summary.copy()
    plot_df["row_label"] = plot_df["dataset"].astype(str) + " | " + plot_df["edit_class"].astype(str)
    heat = plot_df.set_index("row_label")[
        ["mean_delta_motif", "mean_delta_joint", "mean_delta_branch"]
    ]
    heat.columns = ["Motif", "Joint", "Branch"]
    fig, ax = plt.subplots(figsize=(7.6, max(4.6, 0.36 * len(heat))))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Mean STAMP edit contribution"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("STAMP Edit Axes", fontsize=12.5, pad=10)
    _save(fig, out_dir, "fig5_stamp_edit_axis_heatmap")


def plot_compression_heatmap(advantage: pd.DataFrame, out_dir: Path) -> None:
    largest = advantage[advantage["metric"] == "largest_changed_span"].copy()
    largest["comparison_label"] = largest["representation"].map(REPRESENTATION_LABELS)
    heat = largest.pivot(index="comparison_label", columns="dataset", values="compression_ratio")
    heat = heat.loc[[REPRESENTATION_LABELS[r] for r in REPRESENTATION_ORDER if r != "stamp"]]
    heat = heat[list(DATASETS)]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        vmin=1.0,
        linewidths=0.5,
        cbar_kws={"label": "Comparator largest span / STAMP largest span"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Comparator")
    ax.set_title("Largest-Span Compression vs STAMP", fontsize=12.5, pad=10)
    _save(fig, out_dir, "fig6_largest_span_compression")


def plot_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    keep = df[df["representation"].isin(["smiles", "safe", "fragsmiles", "stamp"])].copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    sns.boxplot(
        data=keep,
        x="dataset",
        y="largest_changed_span",
        hue="representation_label",
        hue_order=["SMILES", "SAFE", "fragSMILES", "STAMP"],
        palette=PALETTE,
        showfliers=False,
        width=0.72,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Largest changed span (tokens, outliers hidden)")
    ax.set_title("Pair-Level Largest Span Distribution", fontsize=12.5, pad=12)
    ax.legend(title="", ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=8.5)
    sns.despine(fig=fig, ax=ax)
    _save(fig, out_dir, "fig7_largest_span_distribution")


def plot_stamp_fragsmiles_outcomes(stamp_frag_summary: pd.DataFrame, out_dir: Path) -> None:
    plot_rows = []
    for _, row in stamp_frag_summary.iterrows():
        plot_rows.extend(
            [
                {
                    "dataset": row["dataset"],
                    "outcome": "STAMP shorter",
                    "rate": row["stamp_shorter_largest_rate"],
                },
                {"dataset": row["dataset"], "outcome": "Tie", "rate": row["tie_largest_rate"]},
                {
                    "dataset": row["dataset"],
                    "outcome": "fragSMILES shorter",
                    "rate": row["fragsmiles_shorter_largest_rate"],
                },
            ]
        )
    plot_df = pd.DataFrame(plot_rows)
    plot_df["dataset"] = pd.Categorical(plot_df["dataset"], categories=list(DATASETS), ordered=True)
    plot_df["outcome"] = pd.Categorical(
        plot_df["outcome"],
        categories=["STAMP shorter", "Tie", "fragSMILES shorter"],
        ordered=True,
    )
    colors = {
        "STAMP shorter": "#B64032",
        "Tie": "#B7B7B7",
        "fragSMILES shorter": "#55934D",
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bottoms = {dataset: 0.0 for dataset in DATASETS}
    x_values = list(DATASETS)
    for outcome in ["STAMP shorter", "Tie", "fragSMILES shorter"]:
        values = []
        for dataset in x_values:
            value = float(
                plot_df[
                    (plot_df["dataset"].astype(str) == dataset)
                    & (plot_df["outcome"].astype(str) == outcome)
                ]["rate"].iloc[0]
            )
            values.append(value)
        ax.bar(
            x_values,
            values,
            bottom=[bottoms[dataset] for dataset in x_values],
            color=colors[outcome],
            edgecolor="white",
            linewidth=0.8,
            label=outcome,
        )
        for dataset, value in zip(x_values, values):
            bottoms[dataset] += value
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Pairwise outcome rate")
    ax.set_xlabel("")
    ax.set_title("STAMP vs fragSMILES: Largest-Span Outcome", fontsize=12.5, pad=10)
    ax.legend(title="", ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.17), fontsize=8.5)
    sns.despine(fig=fig, ax=ax)
    _save(fig, out_dir, "fig8_stamp_vs_fragsmiles_outcomes")


def plot_stamp_fragsmiles_delta(stamp_frag_pairs: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    sns.boxplot(
        data=stamp_frag_pairs,
        x="dataset",
        y="delta_largest_span_frag_minus_stamp",
        color="#D9A35F",
        showfliers=False,
        width=0.55,
        ax=ax,
    )
    ax.axhline(0, color="#333333", linewidth=1.1, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel("fragSMILES largest span - STAMP largest span")
    ax.set_title("Positive Values Favor STAMP", fontsize=12.5, pad=10)
    sns.despine(fig=fig, ax=ax)
    _save(fig, out_dir, "fig9_stamp_vs_fragsmiles_delta")


def plot_requested_language_metrics(summary: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("mean_edit_script_length", "Edit script length"),
        ("mean_changed_token_fraction", "Changed-token fraction"),
        ("mean_disconnected_edit_regions", "Disconnected edit regions"),
        ("mean_minimal_contiguous_edit_span", "Minimal contiguous edit span"),
    ]
    plot_df = summary.melt(
        id_vars=["dataset", "representation_label"],
        value_vars=[metric for metric, _label in metrics],
        var_name="metric",
        value_name="value",
    )
    labels = {metric: label for metric, label in metrics}
    plot_df["metric_label"] = plot_df["metric"].map(labels)
    plot_df["metric_label"] = pd.Categorical(
        plot_df["metric_label"],
        categories=[label for _metric, label in metrics],
        ordered=True,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), sharex=False)
    axes_flat = axes.flatten()
    for ax, (_metric, label) in zip(axes_flat, metrics):
        subset = plot_df[plot_df["metric_label"].astype(str) == label]
        sns.barplot(
            data=subset,
            x="dataset",
            y="value",
            hue="representation_label",
            hue_order=[REPRESENTATION_LABELS[r] for r in REPRESENTATION_ORDER],
            palette=PALETTE,
            ax=ax,
        )
        ax.set_title(label, fontsize=11.5, pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("Mean")
        ax.tick_params(axis="x", rotation=0)
        if ax.legend_ is not None:
            ax.legend_.remove()
        sns.despine(fig=fig, ax=ax)
    handles, labels_out = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_out,
        title="",
        ncol=6,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        fontsize=8.5,
    )
    fig.suptitle("Edit Locality Metrics Beyond Sequence Length", fontsize=13.2, fontweight="bold", y=1.07)
    fig.tight_layout()
    _save(fig, out_dir, "fig10_requested_edit_locality_metrics")


def plot_patch_executability(patch_summary: pd.DataFrame, out_dir: Path) -> None:
    plot_df = patch_summary.melt(
        id_vars=["dataset"],
        value_vars=[
            "executable_as_one_typed_patch_rate",
            "transferable_to_compatible_scaffold_rate",
        ],
        var_name="metric",
        value_name="rate",
    )
    plot_df["metric_label"] = plot_df["metric"].map(
        {
            "executable_as_one_typed_patch_rate": "Executable typed patch",
            "transferable_to_compatible_scaffold_rate": "Transferable patch",
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    sns.barplot(
        data=plot_df,
        x="dataset",
        y="rate",
        hue="metric_label",
        palette={
            "Executable typed patch": "#B64032",
            "Transferable patch": "#3C73B9",
        },
        ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Rate")
    ax.set_title("STAMP Patch Executability", fontsize=12.5, pad=10)
    ax.legend(title="", frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    sns.despine(fig=fig, ax=ax)
    _save(fig, out_dir, "fig11_stamp_patch_executability")


def plot_patch_size(patch_summary: pd.DataFrame, out_dir: Path) -> None:
    plot_df = patch_summary.melt(
        id_vars=["dataset"],
        value_vars=[
            "mean_edit_script_length",
            "mean_minimal_contiguous_edit_span",
            "mean_executable_patch_span",
        ],
        var_name="metric",
        value_name="tokens",
    )
    plot_df["metric_label"] = plot_df["metric"].map(
        {
            "mean_edit_script_length": "Edit script",
            "mean_minimal_contiguous_edit_span": "Minimal span",
            "mean_executable_patch_span": "Executable subtree",
        }
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    sns.barplot(
        data=plot_df,
        x="dataset",
        y="tokens",
        hue="metric_label",
        palette={
            "Edit script": "#6F6F6F",
            "Minimal span": "#D9A35F",
            "Executable subtree": "#B64032",
        },
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean tokens")
    ax.set_title("STAMP Patch Size", fontsize=12.5, pad=10)
    ax.legend(title="", frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    sns.despine(fig=fig, ax=ax)
    _save(fig, out_dir, "fig12_stamp_patch_size")


def plot_typed_patch_kinds(typed_summary: pd.DataFrame, out_dir: Path) -> None:
    plot_df = typed_summary.copy()
    plot_df = plot_df[plot_df["patch_kind"].isin(PATCH_KIND_ORDER)]
    plot_df["dataset"] = pd.Categorical(plot_df["dataset"], categories=list(DATASETS), ordered=True)
    x_values = list(DATASETS)
    bottoms = {dataset: 0.0 for dataset in x_values}

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for kind in PATCH_KIND_ORDER:
        subset = plot_df[plot_df["patch_kind"] == kind]
        if subset.empty:
            continue
        values = []
        for dataset in x_values:
            match = subset[subset["dataset"].astype(str) == dataset]
            values.append(float(match["rate"].iloc[0]) if not match.empty else 0.0)
        ax.bar(
            x_values,
            values,
            bottom=[bottoms[dataset] for dataset in x_values],
            color=PATCH_KIND_PALETTE.get(kind, "#6F6F6F"),
            edgecolor="white",
            linewidth=0.8,
            label=PATCH_KIND_LABELS.get(kind, kind),
        )
        for dataset, value in zip(x_values, values):
            bottoms[dataset] += value
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Rate")
    ax.set_title("STAMP Typed Patch Composition", fontsize=12.5, pad=10)
    ax.legend(title="", ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=8.5)
    sns.despine(fig=fig, ax=ax)
    _save(fig, out_dir, "fig13_stamp_typed_patch_kinds")


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    advantage: pd.DataFrame,
    axes_summary: pd.DataFrame,
    stamp_frag_summary: pd.DataFrame,
    stamp_frag_by_edit: pd.DataFrame,
    patch_summary: pd.DataFrame,
    patch_by_edit: pd.DataFrame,
    typed_by_dataset: pd.DataFrame,
) -> None:
    largest_adv = advantage[advantage["metric"] == "largest_changed_span"].copy()
    changed_adv = advantage[advantage["metric"] == "changed_span_count"].copy()
    stamp_rows = summary[summary["representation"].astype(str) == "stamp"].copy()
    dataset_order = {name: i for i, name in enumerate(DATASETS)}

    lines: List[str] = []
    lines.append("# Edit Locality Focused Report")
    lines.append("")
    lines.append("Scope: MoleculeACE strict cliffs, sampled ChEMBL MMP pairs, and FC-STAMP source/generated pairs.")
    lines.append("")
    lines.append("## Main Claim")
    lines.append("")
    lines.append(
        "Real medicinal-chemistry transformations in MMP and activity-cliff pairs often collapse into short executable "
        "STAMP patches. The evidence is not that STAMP is merely shorter as a sequence; the key measurements are edit "
        "script length, changed-token fraction, disconnected edit regions, minimal contiguous edit span, typed-patch "
        "executability, and transferability to compatible scaffold contexts. FC-STAMP source/generated pairs are included "
        "as a harder generation stress test, where many outputs require multi-site construction rather than one typed patch."
    )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for name, caption in [
        ("fig1_largest_changed_span.png", "Mean largest changed span across complete languages."),
        ("fig2_changed_span_count.png", "Mean number of changed spans."),
        ("fig3_locality_pareto.png", "Normalized edit distance versus largest changed span."),
        ("fig4_subtree_contiguity.png", "Subtree-contiguous edit rate."),
        ("fig5_stamp_edit_axis_heatmap.png", "Motif/joint/branch edit contribution heatmap."),
        ("fig6_largest_span_compression.png", "Comparator largest-span compression relative to STAMP."),
        ("fig7_largest_span_distribution.png", "Pair-level largest-span distribution."),
        ("fig8_stamp_vs_fragsmiles_outcomes.png", "Pairwise STAMP-vs-fragSMILES largest-span outcomes."),
        ("fig9_stamp_vs_fragsmiles_delta.png", "Pairwise STAMP-vs-fragSMILES largest-span deltas."),
        ("fig10_requested_edit_locality_metrics.png", "Requested edit-locality metrics across complete languages."),
        ("fig11_stamp_patch_executability.png", "STAMP typed-patch executability and transferability."),
        ("fig12_stamp_patch_size.png", "STAMP edit script, minimal span, and executable subtree size."),
        ("fig13_stamp_typed_patch_kinds.png", "STAMP typed patch composition."),
    ]:
        lines.append(f"![{caption}]({name})")
        lines.append("")
    lines.append("## STAMP Locality Summary")
    lines.append("")
    lines.append(
        "| dataset | n pairs | edit script | changed-token frac | disconnected regions | minimal span | executable typed patch | transferable patch |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in patch_summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(int(row["n_pairs"])),
                    _format_float(row["mean_edit_script_length"]),
                    _format_float(row["mean_changed_token_fraction"]),
                    _format_float(row["mean_disconnected_edit_regions"]),
                    _format_float(row["mean_minimal_contiguous_edit_span"]),
                    _format_float(row["executable_as_one_typed_patch_rate"]),
                    _format_float(row["transferable_to_compatible_scaffold_rate"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Patch metrics by edit class:")
    lines.append("")
    lines.append(
        "| dataset | edit class | n | edit script | minimal span | executable subtree | executable | transferable |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in patch_by_edit.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["edit_class"]),
                    str(int(row["n_pairs"])),
                    _format_float(row["mean_edit_script_length"]),
                    _format_float(row["mean_minimal_contiguous_edit_span"]),
                    _format_float(row["mean_executable_patch_span"]),
                    _format_float(row["executable_as_one_typed_patch_rate"]),
                    _format_float(row["transferable_to_compatible_scaffold_rate"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Typed patch composition:")
    lines.append("")
    lines.append("| dataset | patch type | rate | count |")
    lines.append("| --- | --- | ---: | ---: |")
    typed_dataset_order = {name: i for i, name in enumerate(DATASETS)}
    patch_kind_order = {kind: i for i, kind in enumerate(PATCH_KIND_ORDER)}
    typed_sorted = typed_by_dataset.copy()
    typed_sorted["_dataset_order"] = typed_sorted["dataset"].map(typed_dataset_order)
    typed_sorted["_patch_order"] = typed_sorted["patch_kind"].map(patch_kind_order)
    typed_sorted = typed_sorted.sort_values(["_dataset_order", "_patch_order"])
    for _, row in typed_sorted.iterrows():
        if int(row["count"]) == 0:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["patch_kind_label"]),
                    _format_float(row["rate"]),
                    str(int(row["count"])),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Largest-Span Compression")
    lines.append("")
    lines.append("| dataset | comparator | compression ratio | STAMP win rate |")
    lines.append("| --- | --- | ---: | ---: |")
    largest_adv["_dataset_order"] = largest_adv["dataset"].map(dataset_order)
    largest_adv = largest_adv.sort_values(
        ["_dataset_order", "compression_ratio"],
        ascending=[True, False],
    )
    for _, row in largest_adv.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["comparison"]),
                    _format_float(row["compression_ratio"]),
                    _format_float(row["stamp_win_rate"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## fragSMILES Is The Nearest Baseline")
    lines.append("")
    lines.append(
        "fragSMILES is consistently the closest competitor because it already uses fragment-scale tokens. "
        "This is useful evidence: locality improves first from atom strings to fragment strings, then STAMP adds topology."
    )
    lines.append("")
    lines.append("| dataset | largest-span ratio frag/STAMP | STAMP shorter | tie | fragSMILES shorter | changed-span ratio frag/STAMP |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in stamp_frag_summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    _format_float(row["largest_span_ratio_frag_over_stamp"]),
                    _format_float(row["stamp_shorter_largest_rate"]),
                    _format_float(row["tie_largest_rate"]),
                    _format_float(row["fragsmiles_shorter_largest_rate"]),
                    _format_float(row["changed_span_ratio_frag_over_stamp"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "Takeaway: do not frame fragSMILES as a weak baseline. It is the strongest locality baseline. "
        "The STAMP advantage over fragSMILES is modest on MoleculeACE cliffs and ChEMBL MMP, but larger on FC generated pairs; "
        "the qualitative distinction is that fragSMILES remains flat, while STAMP provides typed motif/joint/branch patches "
        "with subtree spans and compatible-scaffold port signatures."
    )
    lines.append("")
    lines.append("By edit class:")
    lines.append("")
    lines.append("| dataset | edit class | n | ratio frag/STAMP | STAMP shorter | fragSMILES shorter |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for _, row in stamp_frag_by_edit.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["edit_class"]),
                    str(int(row["n_pairs"])),
                    _format_float(row["largest_span_ratio_frag_over_stamp"]),
                    _format_float(row["stamp_shorter_largest_rate"]),
                    _format_float(row["fragsmiles_shorter_largest_rate"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Edit-Axis Interpretation")
    lines.append("")
    lines.append("| dataset | edit class | n | motif | joint | branch | dominant |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for _, row in axes_summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["edit_class"]),
                    str(int(row["n_pairs"])),
                    _format_float(row["mean_delta_motif"]),
                    _format_float(row["mean_delta_joint"]),
                    _format_float(row["mean_delta_branch"]),
                    str(row["dominant_axis"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Nuanced Takeaways")
    lines.append("")
    lines.append("- STAMP's normalized edit distance can be larger because motif/joint/branch tokens are semantically coarser than atom characters.")
    lines.append("- The decisive evidence is patch executability: many edits can be represented as one typed STAMP patch, often with a transferable subtree context.")
    lines.append("- fragSMILES is close to STAMP on token-span metrics; use it as the main flat-fragment comparator, not as a strawman.")
    lines.append("- The flat-fragment limitation is not span length alone. fragSMILES does not expose a parent-child subtree or a port signature for compatible-scaffold transfer.")
    lines.append("- SAFE is high-level and valid, but flattening loses parent-child topology; its changed spans are longer, especially on ChEMBL MMP and FC generated pairs.")
    lines.append("- The tree advantage is strongest for fragment-constrained generation, where linker and superstructure edits require branch/subtree edits rather than isolated token substitutions.")
    lines.append("")

    (out_dir / "edit_locality_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot edit-locality mechanism evidence.")
    parser.add_argument(
        "--input-root",
        default="results/mechanistic_full_20260701",
        help="Directory containing edit_mechanism* result subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/mechanistic_full_20260701/edit_locality_figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.titlesize": 12.5,
            "font.family": "DejaVu Sans",
        }
    )

    cross = _load_cross_tables(root)
    decomposition = _load_decomposition_tables(root)
    summary = summarize_cross(cross)
    advantage = compute_stamp_advantage(cross)
    stamp_frag_pairs, stamp_frag_summary, stamp_frag_by_edit = compute_stamp_vs_fragsmiles(cross)
    axes_summary = summarize_axes(decomposition)
    patch_summary = summarize_stamp_patches(decomposition, ["dataset"])
    patch_by_edit = summarize_stamp_patches(decomposition, ["dataset", "edit_class"])
    typed_by_dataset = summarize_typed_patch_kinds(decomposition, ["dataset"])
    typed_by_edit = summarize_typed_patch_kinds(decomposition, ["dataset", "edit_class"])

    summary.to_csv(out_dir / "edit_locality_summary_by_representation.csv", index=False)
    advantage.to_csv(out_dir / "stamp_pairwise_advantage.csv", index=False)
    stamp_frag_pairs.to_csv(out_dir / "stamp_vs_fragsmiles_pairs.csv", index=False)
    stamp_frag_summary.to_csv(out_dir / "stamp_vs_fragsmiles_by_dataset.csv", index=False)
    stamp_frag_by_edit.to_csv(out_dir / "stamp_vs_fragsmiles_by_edit_class.csv", index=False)
    axes_summary.to_csv(out_dir / "stamp_edit_axis_summary.csv", index=False)
    patch_summary.to_csv(out_dir / "patch_executability_by_dataset.csv", index=False)
    patch_by_edit.to_csv(out_dir / "patch_executability_by_edit_class.csv", index=False)
    typed_by_dataset.to_csv(out_dir / "typed_patch_kind_distribution_by_dataset.csv", index=False)
    typed_by_edit.to_csv(out_dir / "typed_patch_kind_distribution_by_edit_class.csv", index=False)

    plot_largest_span(summary, out_dir)
    plot_changed_spans(summary, out_dir)
    plot_pareto(summary, out_dir)
    plot_subtree_rate(summary, out_dir)
    plot_axes_heatmap(axes_summary, out_dir)
    plot_compression_heatmap(advantage, out_dir)
    plot_distribution(cross, out_dir)
    plot_stamp_fragsmiles_outcomes(stamp_frag_summary, out_dir)
    plot_stamp_fragsmiles_delta(stamp_frag_pairs, out_dir)
    plot_requested_language_metrics(summary, out_dir)
    plot_patch_executability(patch_summary, out_dir)
    plot_patch_size(patch_summary, out_dir)
    plot_typed_patch_kinds(typed_by_dataset, out_dir)
    write_report(
        out_dir,
        summary,
        advantage,
        axes_summary,
        stamp_frag_summary,
        stamp_frag_by_edit,
        patch_summary,
        patch_by_edit,
        typed_by_dataset,
    )

    print(f"[edit-locality] wrote figures and tables to {out_dir}")
    print(f"[edit-locality] wrote {out_dir / 'edit_locality_report.md'}")


if __name__ == "__main__":
    main()
