#!/usr/bin/env python3
"""Combine denovo parameter sweep CSVs and plot a joint Quality/Diversity Pareto front."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


GENMOL_PAPER_PARETO: tuple[dict[str, Any], ...] = (
    # GenMol arXiv:2501.06158 Table 1 / Figure 4, N=1.
    {"config": "GenMol_tau0.5_r0.5", "temperature": 0.5, "randomness": 0.5, "validity_mean": 100.0, "validity_std": 0.0, "uniqueness_mean": 99.7, "uniqueness_std": 0.1, "quality_mean": 84.6, "quality_std": 0.8, "diversity_mean": 0.818, "diversity_std": 0.001, "sampling_time_mean": 21.1, "sampling_time_std": 0.4},
    {"config": "GenMol_tau0.5_r1.0", "temperature": 0.5, "randomness": 1.0, "validity_mean": 99.7, "validity_std": 0.1, "uniqueness_mean": 100.0, "uniqueness_std": 0.0, "quality_mean": 83.8, "quality_std": 0.5, "diversity_mean": 0.832, "diversity_std": 0.001, "sampling_time_mean": 20.5, "sampling_time_std": 0.6},
    # Present in the arXiv TeX source as the Figure 4 trade-off point.
    {"config": "GenMol_tau0.5_r10.0", "temperature": 0.5, "randomness": 10.0, "validity_mean": 99.6, "validity_std": 0.1, "uniqueness_mean": 99.9, "uniqueness_std": 0.0, "quality_mean": 75.0, "quality_std": 0.6, "diversity_mean": 0.858, "diversity_std": 0.002, "sampling_time_mean": 21.6, "sampling_time_std": 0.4},
    {"config": "GenMol_tau1.0_r10.0", "temperature": 1.0, "randomness": 10.0, "validity_mean": 99.8, "validity_std": 0.1, "uniqueness_mean": 99.6, "uniqueness_std": 0.1, "quality_mean": 63.0, "quality_std": 0.4, "diversity_mean": 0.882, "diversity_std": 0.003, "sampling_time_mean": 21.5, "sampling_time_std": 0.5},
    {"config": "GenMol_tau1.5_r10.0", "temperature": 1.5, "randomness": 10.0, "validity_mean": 95.6, "validity_std": 0.3, "uniqueness_mean": 98.3, "uniqueness_std": 0.2, "quality_mean": 39.7, "quality_std": 0.5, "diversity_mean": 0.911, "diversity_std": 0.004, "sampling_time_mean": 20.9, "sampling_time_std": 0.5},
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _as_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _as_int(row: dict[str, Any], key: str) -> int:
    return int(float(row[key]))


def _pareto_front(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        q = _as_float(row, "quality_mean")
        d = _as_float(row, "diversity_mean")
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            oq = _as_float(other, "quality_mean")
            od = _as_float(other, "diversity_mean")
            if oq >= q and od >= d and (oq > q or od > d):
                dominated = True
                break
        if not dominated:
            front.append(dict(row))
    return sorted(front, key=lambda r: (_as_float(r, "diversity_mean"), _as_float(r, "quality_mean")))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _genmol_paper_front() -> list[dict[str, Any]]:
    return sorted((dict(row) for row in GENMOL_PAPER_PARETO), key=lambda r: float(r["diversity_mean"]))


def _plot(
    rows: Sequence[dict[str, Any]],
    front: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    output_prefix: str,
    front_only: bool,
    genmol_front: Sequence[dict[str, Any]],
    annotate_points: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from stamp.benchmark.plot_style import (
        COL_DOUBLE,
        NATURE,
        apply_nature_style,
        despine,
        save_publication,
        soft_grid,
        style_colorbar,
        style_legend,
    )

    apply_nature_style()
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 4.6))
    shown_rows = list(front if front_only else rows)
    xs = [_as_float(r, "diversity_mean") for r in shown_rows]
    ys = [_as_float(r, "quality_mean") for r in shown_rows]
    temps = [_as_float(r, "temperature") for r in shown_rows]
    topks = [_as_int(r, "top_k") for r in shown_rows]
    sizes = [34 if k == 0 else 50 if k <= 10 else 66 for k in topks]

    stamp_color = NATURE["ours"]
    genmol_color = NATURE["baseline"]

    # GenMol reference front first, so it sits visually behind the STAMP front.
    if genmol_front:
        gx = [_as_float(r, "diversity_mean") for r in genmol_front]
        gy = [_as_float(r, "quality_mean") for r in genmol_front]
        ax.plot(
            gx,
            gy,
            color=genmol_color,
            linewidth=1.4,
            linestyle=(0, (5, 2)),
            marker="s",
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor=genmol_color,
            markeredgewidth=1.2,
            label="GenMol (paper)",
            zorder=2,
        )
        if annotate_points:
            for r in genmol_front:
                label = f"τ={_as_float(r, 'temperature'):g}, r={_as_float(r, 'randomness'):g}"
                ax.annotate(
                    label,
                    (_as_float(r, "diversity_mean"), _as_float(r, "quality_mean")),
                    xytext=(5, -10),
                    textcoords="offset points",
                    fontsize=6.5,
                    color=genmol_color,
                )

    if front:
        px = [_as_float(r, "diversity_mean") for r in front]
        py = [_as_float(r, "quality_mean") for r in front]
        ax.plot(
            px,
            py,
            color=stamp_color,
            linewidth=1.6,
            marker="None",
            label="STAMP (ours)",
            zorder=3,
        )

    sc = ax.scatter(
        xs,
        ys,
        c=temps,
        s=sizes,
        cmap="viridis",
        alpha=0.95,
        edgecolors="white",
        linewidths=0.6,
        zorder=4,
        label="_nolegend_" if front_only else "STAMP sweep",
    )
    if front and annotate_points:
        for r in front:
            label = f"T={_as_float(r, 'temperature'):g}, p={_as_float(r, 'top_p'):g}, k={_as_int(r, 'top_k')}"
            ax.annotate(
                label,
                (_as_float(r, "diversity_mean"), _as_float(r, "quality_mean")),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=6.5,
            )

    ax.set_xlabel("Internal diversity", labelpad=6)
    ax.set_ylabel("Quality (%)", labelpad=6)
    ax.set_title("Unconditional generation: quality–diversity trade-off", pad=10)
    ax.margins(x=0.06, y=0.12)

    despine(ax)
    soft_grid(ax, axis="y", alpha=0.06)

    # Direct labels in place of a legend — the two fronts are spatially separated,
    # so labelling them in situ avoids the eye-travel a corner legend would force.
    if front:
        ax.text(
            px[len(px) // 2], max(py) + 1.6, "STAMP (ours)",
            color=stamp_color, fontsize=9, fontweight="bold", ha="center", va="bottom",
        )
    if genmol_front:
        gmid = len(gx) // 2
        ax.text(
            gx[gmid] + 0.004, gy[gmid] - 1.0, "GenMol (paper)",
            color=genmol_color, fontsize=9, fontweight="bold", ha="left", va="top",
        )

    # Matched-diversity advantage: compare the STAMP and GenMol points sitting at
    # (nearly) the same internal diversity to quantify the quality gain.
    if front and genmol_front:
        s_pt = min(front, key=lambda r: _as_float(r, "diversity_mean"))
        sx, sy = _as_float(s_pt, "diversity_mean"), _as_float(s_pt, "quality_mean")
        g_pt = min(genmol_front, key=lambda r: abs(_as_float(r, "diversity_mean") - sx))
        gx2, gy2 = _as_float(g_pt, "diversity_mean"), _as_float(g_pt, "quality_mean")
        xa = (sx + gx2) / 2.0
        ax.annotate(
            "", xy=(xa, sy), xytext=(xa, gy2),
            arrowprops=dict(arrowstyle="<->", color=NATURE["neutral"], lw=1.0),
        )
        ax.text(
            xa - 0.0015, (sy + gy2) / 2.0,
            f"+{sy - gy2:.0f} pp quality\nat matched diversity",
            fontsize=7.5, va="center", ha="right", color=NATURE["neutral"],
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    style_colorbar(cbar, "STAMP sampling temperature")

    # Inset: zoom the tightly-clustered STAMP front so the temperature trajectory
    # (compressed to a few pixels in the main axes) becomes legible.
    if front:
        axins = ax.inset_axes([0.07, 0.12, 0.33, 0.40])
        axins.plot(px, py, color=stamp_color, linewidth=1.4, zorder=3)
        axins.scatter(
            xs, ys, c=temps, s=[sz * 0.8 for sz in sizes], cmap="viridis",
            alpha=0.95, edgecolors="white", linewidths=0.5, zorder=4,
        )
        pad_x = (max(xs) - min(xs)) * 0.18 or 0.002
        pad_y = (max(ys) - min(ys)) * 0.12 or 1.0
        axins.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        axins.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
        axins.tick_params(labelsize=6, length=2, width=0.6, colors=NATURE["neutral"])
        axins.set_title("STAMP front (zoom)", fontsize=7, color=NATURE["neutral"], pad=3)
        despine(axins)
        _, connectors = ax.indicate_inset_zoom(
            axins, edgecolor=NATURE["muted"], alpha=0.7, linewidth=0.8,
        )
        for c in connectors:  # mark the zoomed region but drop the crossing connector lines
            c.set_visible(False)

    fig.tight_layout()
    save_publication(fig, out_dir / output_prefix, formats=("svg", "pdf", "png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="sweep_summary.csv files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="quality_diversity_pareto_combined")
    parser.add_argument("--front-only", action="store_true", help="Plot only non-dominated STAMP points.")
    parser.add_argument("--include-genmol-paper", action="store_true", help="Overlay the GenMol paper Figure 4 Pareto front.")
    parser.add_argument("--annotate-points", action="store_true", help="Label each plotted Pareto point with its sampling parameters.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[float, float, int]] = set()
    for item in args.inputs:
        for row in _read_csv(Path(item)):
            key = (_as_float(row, "temperature"), _as_float(row, "top_p"), _as_int(row, "top_k"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    rows = sorted(rows, key=lambda r: (_as_float(r, "temperature"), _as_float(r, "top_p"), _as_int(r, "top_k")))
    front = _pareto_front(rows)
    genmol_front = _genmol_paper_front() if args.include_genmol_paper else []
    _write_csv(out_dir / "combined_sweep_summary.csv", rows)
    _write_csv(out_dir / "combined_pareto_front.csv", front)
    (out_dir / "combined_pareto_front.json").write_text(json.dumps(front, indent=2), encoding="utf-8")
    if genmol_front:
        _write_csv(out_dir / "genmol_paper_pareto_front.csv", genmol_front)
        (out_dir / "genmol_paper_pareto_front.json").write_text(json.dumps(genmol_front, indent=2), encoding="utf-8")
    _plot(
        rows,
        front,
        out_dir,
        output_prefix=str(args.output_prefix),
        front_only=bool(args.front_only),
        genmol_front=genmol_front,
        annotate_points=bool(args.annotate_points),
    )
    for row in front:
        print(
            f"{row['config']}: Q={_as_float(row, 'quality_mean'):.2f} "
            f"D={_as_float(row, 'diversity_mean'):.3f} "
            f"V={_as_float(row, 'validity_mean'):.2f} U={_as_float(row, 'uniqueness_mean'):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
