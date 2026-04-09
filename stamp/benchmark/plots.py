from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


def _clean_number(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    if math.isnan(x) or math.isinf(x):
        return float("nan")
    return x


def _nice_top(vmax: float) -> float:
    if vmax <= 0 or math.isnan(vmax):
        return 1.0
    p = 10 ** math.floor(math.log10(vmax))
    for m in (1.0, 2.0, 5.0, 10.0):
        t = m * p
        if t >= vmax:
            return t
    return vmax


def _default_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        return ImageFont.load_default()


def _draw_bar_chart(
    categories: Sequence[str],
    values: Sequence[float],
    title: str,
    y_label: str,
    out_path: str | Path,
    y_max: float | None = None,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    w, h = 1400, 900
    left, right, top, bottom = 100, 40, 100, 220
    plot_w = w - left - right
    plot_h = h - top - bottom

    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    font = _default_font()

    draw.text((left, 40), title, fill="black", font=font)
    draw.text((20, top - 10), y_label, fill="black", font=font)

    clean_vals = [_clean_number(v) for v in values]
    vmax = max([v for v in clean_vals if not math.isnan(v)] or [1.0])
    y_top = _nice_top(y_max if y_max is not None else vmax * 1.1)

    draw.line((left, top, left, top + plot_h), fill="black", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=2)

    n_ticks = 5
    for i in range(n_ticks + 1):
        t = i / n_ticks
        y = top + plot_h - int(t * plot_h)
        val = t * y_top
        draw.line((left - 6, y, left, y), fill="black", width=1)
        draw.text((10, y - 8), f"{val:.2g}", fill="black", font=font)
        if i > 0:
            draw.line((left, y, left + plot_w, y), fill=(235, 235, 235), width=1)

    n = max(1, len(categories))
    slot = plot_w / n
    bar_w = max(8, int(slot * 0.6))
    for i, (cat, v) in enumerate(zip(categories, clean_vals)):
        cx = left + int((i + 0.5) * slot)
        x0 = cx - bar_w // 2
        x1 = cx + bar_w // 2
        if math.isnan(v):
            y1 = top + plot_h
            y0 = y1 - 4
            color = (180, 180, 180)
        else:
            y1 = top + plot_h
            y0 = top + plot_h - int((v / y_top) * plot_h) if y_top > 0 else y1
            color = (52, 120, 246)
        draw.rectangle((x0, y0, x1, y1), fill=color, outline=(40, 40, 40))
        if not math.isnan(v):
            draw.text((x0, max(top + 2, y0 - 20)), f"{v:.3g}", fill="black", font=font)
        draw.text((x0 - 10, top + plot_h + 16), cat, fill="black", font=font)

    img.save(out)


def _draw_grouped_bar_chart(
    categories: Sequence[str],
    series: Dict[str, Sequence[float]],
    title: str,
    y_label: str,
    out_path: str | Path,
    y_max: float | None = None,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    names = list(series.keys())
    if not names:
        return

    w, h = 1600, 950
    left, right, top, bottom = 100, 40, 100, 250
    plot_w = w - left - right
    plot_h = h - top - bottom

    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    font = _default_font()

    draw.text((left, 40), title, fill="black", font=font)
    draw.text((20, top - 10), y_label, fill="black", font=font)

    vals_all = []
    clean_series: Dict[str, list[float]] = {}
    for k, arr in series.items():
        clean = [_clean_number(v) for v in arr]
        clean_series[k] = clean
        vals_all.extend([v for v in clean if not math.isnan(v)])

    vmax = max(vals_all or [1.0])
    y_top = _nice_top(y_max if y_max is not None else vmax * 1.1)

    draw.line((left, top, left, top + plot_h), fill="black", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=2)

    n_ticks = 5
    for i in range(n_ticks + 1):
        t = i / n_ticks
        y = top + plot_h - int(t * plot_h)
        val = t * y_top
        draw.line((left - 6, y, left, y), fill="black", width=1)
        draw.text((10, y - 8), f"{val:.2g}", fill="black", font=font)
        if i > 0:
            draw.line((left, y, left + plot_w, y), fill=(235, 235, 235), width=1)

    palette = [
        (52, 120, 246),
        (255, 140, 0),
        (34, 166, 89),
        (220, 70, 70),
        (130, 90, 220),
        (0, 160, 160),
    ]

    n_cat = max(1, len(categories))
    n_ser = max(1, len(names))
    slot = plot_w / n_cat
    group_w = slot * 0.8
    bar_w = max(4, int(group_w / n_ser))
    for i, cat in enumerate(categories):
        gx0 = left + int(i * slot + (slot - group_w) / 2)
        for j, name in enumerate(names):
            arr = clean_series[name]
            v = arr[i] if i < len(arr) else float("nan")
            x0 = gx0 + j * bar_w
            x1 = x0 + max(2, bar_w - 2)
            y1 = top + plot_h
            if math.isnan(v):
                y0 = y1 - 4
                color = (180, 180, 180)
            else:
                y0 = top + plot_h - int((v / y_top) * plot_h) if y_top > 0 else y1
                color = palette[j % len(palette)]
            draw.rectangle((x0, y0, x1, y1), fill=color, outline=(40, 40, 40))
        draw.text((gx0, top + plot_h + 18), cat, fill="black", font=font)

    lx = left
    ly = h - 120
    for i, name in enumerate(names):
        c = palette[i % len(palette)]
        x0 = lx + i * 220
        draw.rectangle((x0, ly, x0 + 24, ly + 16), fill=c, outline=(30, 30, 30))
        draw.text((x0 + 30, ly - 2), name, fill="black", font=font)

    img.save(out)


def _draw_line_chart(
    xs: Sequence[float],
    series: Dict[str, Sequence[float]],
    title: str,
    y_label: str,
    out_path: str | Path,
    y_max: float | None = None,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    w, h = 1400, 900
    left, right, top, bottom = 100, 40, 100, 140
    plot_w = w - left - right
    plot_h = h - top - bottom

    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    font = _default_font()

    draw.text((left, 40), title, fill="black", font=font)
    draw.text((20, top - 10), y_label, fill="black", font=font)

    xvals = [_clean_number(x) for x in xs]
    x_clean = [x for x in xvals if not math.isnan(x)]
    if not x_clean:
        return
    xmin, xmax = min(x_clean), max(x_clean)
    if xmax <= xmin:
        xmax = xmin + 1.0

    vals_all = []
    clean_series: Dict[str, list[float]] = {}
    for k, arr in series.items():
        clean = [_clean_number(v) for v in arr]
        clean_series[k] = clean
        vals_all.extend([v for v in clean if not math.isnan(v)])
    vmax = max(vals_all or [1.0])
    y_top = _nice_top(y_max if y_max is not None else vmax * 1.1)

    draw.line((left, top, left, top + plot_h), fill="black", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=2)

    n_ticks = 5
    for i in range(n_ticks + 1):
        t = i / n_ticks
        y = top + plot_h - int(t * plot_h)
        val = t * y_top
        draw.line((left - 6, y, left, y), fill="black", width=1)
        draw.text((10, y - 8), f"{val:.2g}", fill="black", font=font)
        if i > 0:
            draw.line((left, y, left + plot_w, y), fill=(235, 235, 235), width=1)

    unique_x = sorted(set(x_clean))
    for x in unique_x:
        tx = left + int(((x - xmin) / (xmax - xmin)) * plot_w)
        draw.line((tx, top + plot_h, tx, top + plot_h + 6), fill="black", width=1)
        draw.text((tx - 16, top + plot_h + 12), str(int(x)), fill="black", font=font)

    palette = [
        (52, 120, 246),
        (255, 140, 0),
        (34, 166, 89),
        (220, 70, 70),
        (130, 90, 220),
        (0, 160, 160),
    ]
    for i, (name, arr) in enumerate(clean_series.items()):
        color = palette[i % len(palette)]
        points = []
        for x, y in zip(xvals, arr):
            if math.isnan(x) or math.isnan(y):
                continue
            px = left + int(((x - xmin) / (xmax - xmin)) * plot_w)
            py = top + plot_h - int((y / y_top) * plot_h) if y_top > 0 else top + plot_h
            points.append((px, py))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for px, py in points:
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color, outline=(30, 30, 30))

    lx = left
    ly = h - 70
    for i, name in enumerate(clean_series.keys()):
        c = palette[i % len(palette)]
        x0 = lx + i * 220
        draw.rectangle((x0, ly, x0 + 24, ly + 16), fill=c, outline=(30, 30, 30))
        draw.text((x0 + 30, ly - 2), name, fill="black", font=font)

    img.save(out)


def _plot_representation_stats(rep_json: str | Path, out_dir: str | Path) -> list[str]:
    payload = json.loads(Path(rep_json).read_text(encoding="utf-8"))
    rows = [r for r in payload.get("results", []) if not r.get("error")]
    if not rows:
        return []

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reps = [str(r.get("repr", "")) for r in rows]

    files = []
    p = out / "expA_lengths.png"
    _draw_grouped_bar_chart(
        categories=reps,
        series={
            "avg_char_length": [float(r.get("avg_char_length", 0.0)) for r in rows],
            "avg_token_length": [float(r.get("avg_token_length", 0.0)) for r in rows],
            "p95_token_length": [float(r.get("p95_token_length", 0.0)) for r in rows],
        },
        title="Experiment A: Sequence Length Statistics",
        y_label="Length",
        out_path=p,
    )
    files.append(str(p))

    p = out / "expA_validity.png"
    _draw_grouped_bar_chart(
        categories=reps,
        series={
            "encode_valid_rate": [float(r.get("encode_valid_rate", 0.0)) for r in rows],
            "decode_valid_rate": [float(r.get("decode_valid_rate", 0.0)) for r in rows],
            "decode_exact_rate": [float(r.get("decode_exact_rate", 0.0)) for r in rows],
        },
        title="Experiment A: Encode/Decode Validity",
        y_label="Rate",
        out_path=p,
        y_max=1.0,
    )
    files.append(str(p))

    p = out / "expA_vocab.png"
    _draw_bar_chart(
        categories=reps,
        values=[float(r.get("vocab_size", 0.0)) for r in rows],
        title="Experiment A: Vocabulary Size",
        y_label="Vocab Size",
        out_path=p,
    )
    files.append(str(p))

    p = out / "expA_ratio_vs_smiles.png"
    _draw_grouped_bar_chart(
        categories=reps,
        series={
            "char_len_ratio_vs_smiles": [float(r.get("char_len_ratio_vs_smiles", 0.0)) for r in rows],
            "token_len_ratio_vs_smiles": [float(r.get("token_len_ratio_vs_smiles", 0.0)) for r in rows],
        },
        title="Experiment A: Length Ratios vs SMILES",
        y_label="Ratio",
        out_path=p,
    )
    files.append(str(p))

    return files


def _plot_denovo_metrics(denovo_jsons: Sequence[str | Path], out_dir: str | Path) -> list[str]:
    rows = []
    for p in denovo_jsons:
        try:
            rows.append(json.loads(Path(p).read_text(encoding="utf-8")))
        except Exception:
            continue
    if not rows:
        return []

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reps = [str(r.get("representation", Path(str(p)).parent.name)) for r, p in zip(rows, denovo_jsons)]

    p = out / "expB_denovo_core_metrics.png"
    _draw_grouped_bar_chart(
        categories=reps,
        series={
            "validity": [float(r.get("validity", 0.0)) for r in rows],
            "uniqueness": [float(r.get("uniqueness", 0.0)) for r in rows],
            "novelty": [float(r.get("novelty", 0.0)) for r in rows],
            "internal_diversity": [float(r.get("internal_diversity", 0.0)) for r in rows],
            "quality": [float(r.get("quality", 0.0)) for r in rows],
        },
        title="Experiment B: De Novo Core Metrics",
        y_label="Rate",
        out_path=p,
        y_max=1.0,
    )
    return [str(p)]


def _plot_task_metric_card(metric_json: str | Path, out_path: str | Path, title: str) -> str:
    m = json.loads(Path(metric_json).read_text(encoding="utf-8"))
    cols = [k for k, v in m.items() if isinstance(v, (int, float)) and not str(k).startswith("n_")]
    cols = [c for c in cols if _clean_number(m.get(c)) == _clean_number(m.get(c))]
    if not cols:
        cols = [k for k in m.keys() if isinstance(m.get(k), (int, float))]

    _draw_bar_chart(
        categories=cols,
        values=[float(m.get(c, 0.0)) for c in cols],
        title=title,
        y_label="Metric",
        out_path=out_path,
        y_max=1.0,
    )
    return str(out_path)


def _plot_low_resource(csv_path: str | Path, out_dir: str | Path) -> list[str]:
    rows = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return []

    ok_rows = [r for r in rows if str(r.get("status", "")).startswith("ok")]
    if not ok_rows:
        ok_rows = rows

    reps = sorted(set(str(r.get("representation", "")) for r in ok_rows))
    sizes = sorted(set(int(float(r.get("train_size", 0))) for r in ok_rows))

    by_rep: Dict[str, Dict[int, Dict[str, float]]] = {rep: {} for rep in reps}
    for r in ok_rows:
        rep = str(r.get("representation", ""))
        size = int(float(r.get("train_size", 0)))
        by_rep.setdefault(rep, {})[size] = {
            "validity": _clean_number(r.get("validity")),
            "novelty": _clean_number(r.get("novelty")),
            "internal_diversity": _clean_number(r.get("internal_diversity")),
            "quality": _clean_number(r.get("quality")),
        }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for metric in ("validity", "novelty", "internal_diversity", "quality"):
        series: Dict[str, list[float]] = {}
        for rep in reps:
            vals = []
            rep_map = by_rep.get(rep, {})
            for s in sizes:
                vals.append(rep_map.get(s, {}).get(metric, float("nan")))
            series[rep] = vals
        p = out / f"expE_low_resource_{metric}.png"
        _draw_line_chart(
            xs=sizes,
            series=series,
            title=f"Experiment E: Low-Resource Scaling ({metric})",
            y_label=metric,
            out_path=p,
            y_max=1.0 if metric != "internal_diversity" else None,
        )
        files.append(str(p))
    return files


def _plot_property_prediction(prop_jsons: Sequence[str | Path], out_dir: str | Path) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    for p in prop_jsons:
        payload = json.loads(Path(p).read_text(encoding="utf-8"))
        rows = payload.get("results", [])
        rows = [r for r in rows if not r.get("error")]
        if not rows:
            continue

        reps = [str(r.get("repr", "")) for r in rows]
        stem = Path(p).parent.name

        p_err = out / f"{stem}_prop_pred_error.png"
        _draw_grouped_bar_chart(
            categories=reps,
            series={
                "rmse": [float(r.get("rmse", 0.0)) for r in rows],
                "mae": [float(r.get("mae", 0.0)) for r in rows],
            },
            title=f"Experiment F: Property Prediction Error ({stem})",
            y_label="Error",
            out_path=p_err,
        )
        files.append(str(p_err))

        p_corr = out / f"{stem}_prop_pred_correlation.png"
        r2_vals = []
        pearson_vals = []
        for r in rows:
            r2 = _clean_number(r.get("r2"))
            pr = _clean_number(r.get("pearson_r"))
            r2_vals.append(r2 if (not math.isnan(r2) and r2 >= 0.0) else float("nan"))
            pearson_vals.append(pr if (not math.isnan(pr) and pr >= 0.0) else float("nan"))
        _draw_grouped_bar_chart(
            categories=reps,
            series={
                "r2 (>=0)": r2_vals,
                "pearson_r (>=0)": pearson_vals,
            },
            title=f"Experiment F: Property Prediction Correlation ({stem})",
            y_label="Score",
            out_path=p_corr,
            y_max=1.0,
        )
        files.append(str(p_corr))

    return files


def save_benchmark_plots(
    run_root: str | Path,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    root = Path(run_root)
    out = Path(output_dir) if output_dir is not None else root / "plots"
    out.mkdir(parents=True, exist_ok=True)

    generated: Dict[str, list[str]] = {}

    rep_json = root / "expA_repr_stats" / "representation_stats.json"
    if rep_json.exists():
        generated["expA"] = _plot_representation_stats(rep_json, out / "expA")

    denovo_jsons = sorted(root.glob("expB_*_denovo/denovo_metrics.json"))
    if denovo_jsons:
        generated["expB"] = _plot_denovo_metrics(denovo_jsons, out / "expB")

    scaffold_jsons = sorted(root.glob("expC_scaffold/**/scaffold_completion_metrics.json"))
    if scaffold_jsons:
        generated["expC"] = [
            _plot_task_metric_card(
                metric_json=p,
                out_path=out / "expC" / f"{p.parent.name}_scaffold_metrics.png",
                title=f"Experiment C: Scaffold Completion ({p.parent.name})",
            )
            for p in scaffold_jsons
        ]

    linker_jsons = sorted(root.glob("expD_linker/**/linker_design_metrics.json"))
    if linker_jsons:
        generated["expD"] = [
            _plot_task_metric_card(
                metric_json=p,
                out_path=out / "expD" / f"{p.parent.name}_linker_metrics.png",
                title=f"Experiment D: Linker Design ({p.parent.name})",
            )
            for p in linker_jsons
        ]

    low_csv = root / "expE_low_resource" / "low_resource_scaling.csv"
    if low_csv.exists():
        generated["expE"] = _plot_low_resource(low_csv, out / "expE")

    prop_jsons = sorted(root.glob("expF*/property_prediction_metrics.json"))
    if prop_jsons:
        generated["expF"] = _plot_property_prediction(prop_jsons, out / "expF")

    summary = {
        "run_root": str(root),
        "output_dir": str(out),
        "generated": generated,
    }
    (out / "plots_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
