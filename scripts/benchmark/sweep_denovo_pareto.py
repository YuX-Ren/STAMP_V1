#!/usr/bin/env python3
"""Sweep unconditional generation parameters and plot Quality/Diversity Pareto front.

This uses direct sampling, not dedup-fill/oversampling, so the Quality and
Diversity tradeoff reflects the model/sampler rather than post-processing.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCH = ROOT / "scripts" / "benchmark"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from eval_denovo_taskaware import (  # noqa: E402
    _canonical,
    _hybrid_tokens_to_stamp_tokens,
    _load_hybrid_vocab,
    _parse_condition_tokens,
    _quality_count,
    _strip_condition_tokens,
)
from stamp.benchmark.lm import load_trained_model, sample_model_sequences  # noqa: E402
from stamp.benchmark.metrics import _internal_diversity, _load_sa_scorer  # noqa: E402

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class SamplerConfig:
    temperature: float
    top_p: float
    top_k: int

    @property
    def name(self) -> str:
        return f"T{self.temperature:g}_P{self.top_p:g}_K{self.top_k}"


def _parse_float_list(text: str) -> list[float]:
    return [float(x) for x in str(text).replace(",", " ").split() if x]


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in str(text).replace(",", " ").split() if x]


def _parse_grid(args: argparse.Namespace) -> list[SamplerConfig]:
    configs: list[SamplerConfig] = []
    seen: set[tuple[float, float, int]] = set()
    for temp in _parse_float_list(args.temperatures):
        for top_p in _parse_float_list(args.top_ps):
            for top_k in _parse_int_list(args.top_ks):
                key = (float(temp), float(top_p), int(top_k))
                if key in seen:
                    continue
                seen.add(key)
                configs.append(SamplerConfig(temperature=float(temp), top_p=float(top_p), top_k=int(top_k)))
    if not configs:
        raise ValueError("Empty parameter grid.")
    return configs


def _decode_batch(
    seqs: Sequence[Sequence[str]],
    *,
    rep: Any,
    hybrid_vocab: Any,
) -> list[str]:
    out: list[str] = []
    for toks in seqs:
        try:
            if hybrid_vocab is not None:
                stamp_toks = _hybrid_tokens_to_stamp_tokens(toks, hybrid_vocab)
            else:
                stamp_toks = _strip_condition_tokens(toks)
            smi = rep.decode(rep.detokenize(stamp_toks))
            can = _canonical(smi)
        except Exception:
            can = None
        if can:
            out.append(can)
    return out


def _evaluate_one(
    *,
    model: Any,
    vocab: Any,
    rep: Any,
    dev: torch.device,
    hybrid_vocab: Any,
    sa: Any,
    config: SamplerConfig,
    seed: int,
    num_samples: int,
    batch_size: int,
    max_new: int,
    condition_tokens: Sequence[str],
    ao_decode_mode: str,
    ao_canvas_placeholder: str,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    valid: list[str] = []
    remaining = int(num_samples)
    batch_idx = 0
    while remaining > 0:
        cur_n = min(int(batch_size), remaining)
        batch_idx += 1
        seqs = sample_model_sequences(
            model,
            vocab,
            n_samples=cur_n,
            max_new_tokens=int(max_new),
            temperature=float(config.temperature),
            top_k=int(config.top_k),
            top_p=float(config.top_p),
            prompt_tokens=condition_tokens,
            device=dev,
            ao_decode_mode=str(ao_decode_mode),
            ao_canvas_placeholder=str(ao_canvas_placeholder),
        )
        valid.extend(_decode_batch(seqs, rep=rep, hybrid_vocab=hybrid_vocab))
        remaining -= cur_n
        print(
            f"[sweep] {config.name} seed={seed} batch={batch_idx} "
            f"valid={len(valid)}/{num_samples}",
            flush=True,
        )

    unique_valid = sorted(set(valid))
    n_valid = len(valid)
    n_unique = len(unique_valid)
    n_quality = _quality_count(unique_valid, sa)
    diversity = _internal_diversity(unique_valid) if n_unique >= 2 else float("nan")
    return {
        "config": config.name,
        "seed": int(seed),
        "temperature": float(config.temperature),
        "top_p": float(config.top_p),
        "top_k": int(config.top_k),
        "num_samples": int(num_samples),
        "n_valid": int(n_valid),
        "n_unique": int(n_unique),
        "n_quality": int(n_quality),
        "validity": 100.0 * n_valid / max(1, int(num_samples)),
        "uniqueness": 100.0 * n_unique / max(1, int(num_samples)),
        "quality": 100.0 * n_quality / max(1, int(num_samples)),
        "diversity": float(diversity),
    }


def _mean(vals: Sequence[float]) -> float:
    vals = [float(x) for x in vals]
    return sum(vals) / len(vals) if vals else float("nan")


def _std(vals: Sequence[float]) -> float:
    vals = [float(x) for x in vals]
    if len(vals) < 2:
        return float("nan")
    m = _mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def _summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["temperature"]), float(row["top_p"]), int(row["top_k"]))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (temp, top_p, top_k), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "config": f"T{temp:g}_P{top_p:g}_K{top_k}",
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "n_seeds": len(group),
            "num_samples_per_seed": int(group[0]["num_samples"]),
        }
        for key in ("validity", "uniqueness", "quality", "diversity"):
            vals = [float(r[key]) for r in group]
            item[f"{key}_mean"] = _mean(vals)
            item[f"{key}_std"] = _std(vals)
        out.append(item)
    return out


def _pareto_front(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        q = float(row["quality_mean"])
        d = float(row["diversity_mean"])
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            oq = float(other["quality_mean"])
            od = float(other["diversity_mean"])
            if oq >= q and od >= d and (oq > q or od > d):
                dominated = True
                break
        if not dominated:
            front.append(dict(row))
    return sorted(front, key=lambda r: (float(r["diversity_mean"]), float(r["quality_mean"])))


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


def _plot(summary: Sequence[dict[str, Any]], pareto: Sequence[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.8), dpi=160)
    xs = [float(r["diversity_mean"]) for r in summary]
    ys = [float(r["quality_mean"]) for r in summary]
    cs = [float(r["temperature"]) for r in summary]
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=58, alpha=0.82, edgecolors="black", linewidths=0.35)

    if pareto:
        px = [float(r["diversity_mean"]) for r in pareto]
        py = [float(r["quality_mean"]) for r in pareto]
        ax.plot(px, py, color="#d62728", linewidth=2.0, marker="o", markersize=4.5, label="Pareto front")
        for r in pareto:
            label = f"T={float(r['temperature']):g},p={float(r['top_p']):g},k={int(r['top_k'])}"
            ax.annotate(
                label,
                (float(r["diversity_mean"]), float(r["quality_mean"])),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
            )

    ax.set_xlabel("Internal diversity")
    ax.set_ylabel("Quality (%)")
    ax.set_title("Unconditional STAMP Generation: Quality/Diversity Pareto Sweep")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Temperature")
    fig.tight_layout()
    fig.savefig(out_dir / "quality_diversity_pareto.png")
    fig.savefig(out_dir / "quality_diversity_pareto.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-new", type=int, default=255)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", default="20260617")
    p.add_argument("--temperatures", default="0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2")
    p.add_argument("--top-ps", default="0.90 0.95 0.98 1.0")
    p.add_argument("--top-ks", default="0")
    p.add_argument("--condition-mode", default="none", choices=["none", "qed", "qed_sa"])
    p.add_argument("--condition-tokens", default="")
    p.add_argument("--ao-decode-mode", default="ar", choices=["ar", "ao_canvas_confidence"])
    p.add_argument("--ao-canvas-placeholder", default="unk", choices=["unk", "pad"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = _parse_grid(args)
    seeds = _parse_int_list(args.seeds)
    print(f"[sweep] configs={len(configs)} seeds={seeds}", flush=True)
    print(f"[sweep] model={args.model_dir}", flush=True)

    model, vocab, rep, dev = load_trained_model(args.model_dir, device=args.device)
    hybrid_vocab = _load_hybrid_vocab(args.model_dir)
    if hybrid_vocab is not None:
        print(f"[sweep] hybrid vocab enabled size={len(hybrid_vocab.itos)}", flush=True)
    sa = _load_sa_scorer()
    condition_tokens = _parse_condition_tokens(args.condition_tokens, args.condition_mode)

    rows: list[dict[str, Any]] = []
    for config in configs:
        for seed in seeds:
            row = _evaluate_one(
                model=model,
                vocab=vocab,
                rep=rep,
                dev=dev,
                hybrid_vocab=hybrid_vocab,
                sa=sa,
                config=config,
                seed=seed,
                num_samples=int(args.num_samples),
                batch_size=int(args.batch_size),
                max_new=int(args.max_new),
                condition_tokens=condition_tokens,
                ao_decode_mode=str(args.ao_decode_mode),
                ao_canvas_placeholder=str(args.ao_canvas_placeholder),
            )
            rows.append(row)
            print(
                f"[sweep] done {config.name} seed={seed} "
                f"V={row['validity']:.2f} U={row['uniqueness']:.2f} "
                f"Q={row['quality']:.2f} D={row['diversity']:.3f}",
                flush=True,
            )
            (out_dir / "raw_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            _write_csv(out_dir / "raw_results.csv", rows)

    summary = _summarize(rows)
    pareto = _pareto_front(summary)
    payload = {
        "model_dir": str(args.model_dir),
        "num_samples": int(args.num_samples),
        "seeds": seeds,
        "grid": {
            "temperatures": _parse_float_list(args.temperatures),
            "top_ps": _parse_float_list(args.top_ps),
            "top_ks": _parse_int_list(args.top_ks),
        },
        "summary": summary,
        "pareto_front": pareto,
    }
    (out_dir / "sweep_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(out_dir / "sweep_summary.csv", summary)
    _write_csv(out_dir / "pareto_front.csv", pareto)
    _plot(summary, pareto, out_dir)

    print("\n=== Pareto front ===", flush=True)
    for row in pareto:
        print(
            f"{row['config']}: Q={float(row['quality_mean']):.2f} "
            f"D={float(row['diversity_mean']):.3f} "
            f"V={float(row['validity_mean']):.2f} U={float(row['uniqueness_mean']):.2f}",
            flush=True,
        )
    print(f"[sweep] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
