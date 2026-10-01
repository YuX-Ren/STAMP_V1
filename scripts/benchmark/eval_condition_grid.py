#!/usr/bin/env python3
"""Evaluate discrete condition-token controllability across a QED/SA grid."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import QED

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.lm import load_trained_model, sample_model_sequences  # noqa: E402
from stamp.benchmark.metrics import _internal_diversity, _load_sa_scorer  # noqa: E402

RDLogger.DisableLog("rdApp.*")


QED_TOKENS = {
    "LOW": "[C_QED_LOW]",
    "MID": "[C_QED_MID]",
    "HIGH": "[C_QED_HIGH]",
    "VHIGH": "[C_QED_VHIGH]",
}
SA_TOKENS = {
    "LOW": "[C_SA_LOW]",
    "MID": "[C_SA_MID]",
    "HIGH": "[C_SA_HIGH]",
    "UNK": "[C_SA_UNK]",
}


def _canonical(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _load_hybrid_vocab(model_dir: str | Path):
    vocab_path = Path(model_dir) / "vocab.json"
    if not vocab_path.exists():
        return None
    try:
        payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if "atomic_motifs" not in payload or "motif_expansion" not in payload:
        return None
    from stamp.benchmark.hybrid_vocab import HybridVocab

    return HybridVocab.from_dict(payload)


def _hybrid_tokens_to_stamp_tokens(tokens: Sequence[str]) -> list[str]:
    from stamp.benchmark.hybrid_vocab import BOS, EOS, MASK, PAD, UNK, is_condition_token, is_stamp_structural

    special = {PAD, BOS, EOS, MASK, UNK}
    out: list[str] = []
    buf: list[str] = []
    for tok in tokens:
        if tok in special or is_condition_token(tok):
            continue
        if is_stamp_structural(tok):
            if buf:
                out.append("".join(buf))
                buf = []
            out.append(tok)
        else:
            buf.append(tok)
    if buf:
        out.append("".join(buf))
    return out


def _strip_condition_tokens(tokens: Sequence[str]) -> list[str]:
    return [tok for tok in tokens if not (tok.startswith("[C_") and tok.endswith("]"))]


def _qed_bin(value: float) -> str:
    if value < 0.4:
        return "LOW"
    if value < 0.6:
        return "MID"
    if value < 0.8:
        return "HIGH"
    return "VHIGH"


def _sa_bin(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "UNK"
    if float(value) <= 3.0:
        return "LOW"
    if float(value) <= 4.0:
        return "MID"
    return "HIGH"


def _parse_bins(raw: str, allowed: dict[str, str]) -> list[str]:
    out: list[str] = []
    for item in str(raw or "").replace(",", " ").split():
        key = item.strip().upper()
        if not key:
            continue
        if key not in allowed:
            raise ValueError(f"Unsupported bin {item!r}; allowed={sorted(allowed)}")
        out.append(key)
    return out


def _condition_specs(condition_mode: str, qed_bins: Sequence[str], sa_bins: Sequence[str]) -> list[dict[str, Any]]:
    mode = str(condition_mode or "qed_sa").strip().lower()
    specs: list[dict[str, Any]] = []
    if mode == "qed":
        for qbin in qed_bins:
            specs.append(
                {
                    "name": f"qed_{qbin.lower()}",
                    "tokens": [QED_TOKENS[qbin]],
                    "target_qed_bin": qbin,
                    "target_sa_bin": "",
                }
            )
        return specs
    if mode == "qed_sa":
        for qbin in qed_bins:
            for sbin in sa_bins:
                specs.append(
                    {
                        "name": f"qed_{qbin.lower()}__sa_{sbin.lower()}",
                        "tokens": [QED_TOKENS[qbin], SA_TOKENS[sbin]],
                        "target_qed_bin": qbin,
                        "target_sa_bin": sbin,
                    }
                )
        return specs
    raise ValueError(f"Unsupported condition_mode: {condition_mode!r}")


def _decode_tokens(tokens: Sequence[str], *, hybrid_vocab: Any, rep: Any) -> str | None:
    try:
        if hybrid_vocab is not None:
            stamp_tokens = _hybrid_tokens_to_stamp_tokens(tokens)
        else:
            stamp_tokens = _strip_condition_tokens(tokens)
        text = rep.detokenize(stamp_tokens)
        return _canonical(rep.decode(text))
    except Exception:
        return None


def _sample_one_condition(
    *,
    model: Any,
    vocab: Any,
    rep: Any,
    hybrid_vocab: Any,
    sa_scorer: Any,
    device: torch.device,
    prompt_tokens: Sequence[str],
    num_samples: int,
    batch_size: int,
    max_new: int,
    temperature: float,
    top_k: int,
    top_p: float,
    dedup_fill: bool,
    oversample_factor: float,
) -> tuple[dict[str, Any], list[str]]:
    target = int(num_samples)
    max_attempts = (
        int(math.ceil(target * max(1.0, float(oversample_factor))))
        if bool(dedup_fill)
        else target
    )
    valid_smiles: list[str] = []
    unique_seen: dict[str, None] = {}
    attempts = 0

    while attempts < max_attempts:
        if dedup_fill and len(unique_seen) >= target:
            break
        cur_n = min(max(1, int(batch_size)), max_attempts - attempts)
        seqs = sample_model_sequences(
            model,
            vocab,
            n_samples=cur_n,
            max_new_tokens=int(max_new),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            prompt_tokens=prompt_tokens,
            device=device,
            ao_decode_mode="ar",
        )
        for toks in seqs:
            smi = _decode_tokens(toks, hybrid_vocab=hybrid_vocab, rep=rep)
            if smi:
                valid_smiles.append(smi)
                unique_seen.setdefault(smi, None)
        attempts += cur_n

    unique_smiles = list(unique_seen.keys())
    eval_smiles = unique_smiles[:target] if dedup_fill else unique_smiles
    qed_values: list[float] = []
    sa_values: list[float] = []
    qed_bins: Counter[str] = Counter()
    sa_bins: Counter[str] = Counter()
    n_high_quality = 0
    for smi in eval_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            qed_value = float(QED.qed(mol))
        except Exception:
            continue
        sa_value: float | None = None
        if sa_scorer is not None:
            try:
                sa_value = float(sa_scorer.calculateScore(mol))
            except Exception:
                sa_value = None
        qed_values.append(qed_value)
        if sa_value is not None:
            sa_values.append(sa_value)
        qed_bins[_qed_bin(qed_value)] += 1
        sa_bins[_sa_bin(sa_value)] += 1
        if qed_value >= 0.6 and sa_value is not None and sa_value <= 4.0:
            n_high_quality += 1

    summary = {
        "num_samples": target,
        "num_attempts": attempts,
        "n_valid": len(valid_smiles) if not dedup_fill else len(eval_smiles),
        "n_unique": len(eval_smiles),
        "n_high_quality_unique": n_high_quality,
        "validity": 100.0 * (len(valid_smiles) if not dedup_fill else len(eval_smiles)) / max(1, target),
        "uniqueness": 100.0 * len(eval_smiles) / max(1, target),
        "quality": 100.0 * n_high_quality / max(1, target),
        "raw_validity": 100.0 * len(valid_smiles) / max(1, attempts),
        "raw_uniqueness": 100.0 * len(unique_smiles) / max(1, attempts),
        "mean_qed": sum(qed_values) / len(qed_values) if qed_values else float("nan"),
        "mean_sa": sum(sa_values) / len(sa_values) if sa_values else float("nan"),
        "qed_bin_counts": dict(sorted(qed_bins.items())),
        "sa_bin_counts": dict(sorted(sa_bins.items())),
        "diversity": _internal_diversity(eval_smiles) if len(eval_smiles) >= 2 else float("nan"),
    }
    return summary, eval_smiles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition-mode", default="qed_sa", choices=["qed", "qed_sa"])
    parser.add_argument("--qed-bins", default="LOW,MID,HIGH,VHIGH")
    parser.add_argument("--sa-bins", default="LOW,MID,HIGH")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dedup-fill", action="store_true")
    parser.add_argument("--oversample-factor", type=float, default=3.0)
    parser.add_argument("--save-smiles", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, vocab, rep, dev = load_trained_model(args.model_dir, device=args.device)
    hybrid_vocab = _load_hybrid_vocab(args.model_dir)
    sa_scorer = _load_sa_scorer()

    qed_bins = _parse_bins(args.qed_bins, QED_TOKENS)
    sa_bins = _parse_bins(args.sa_bins, SA_TOKENS)
    specs = _condition_specs(args.condition_mode, qed_bins, sa_bins)

    results: list[dict[str, Any]] = []
    for i, spec in enumerate(specs):
        torch.manual_seed(int(args.seed) + i)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(int(args.seed) + i)
        print(f"[condition-grid] {spec['name']} prompt={' '.join(spec['tokens'])}", flush=True)
        metrics, smiles = _sample_one_condition(
            model=model,
            vocab=vocab,
            rep=rep,
            hybrid_vocab=hybrid_vocab,
            sa_scorer=sa_scorer,
            device=dev,
            prompt_tokens=spec["tokens"],
            num_samples=int(args.num_samples),
            batch_size=int(args.batch_size),
            max_new=int(args.max_new),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            dedup_fill=bool(args.dedup_fill),
            oversample_factor=float(args.oversample_factor),
        )
        target_qed = str(spec["target_qed_bin"])
        target_sa = str(spec["target_sa_bin"])
        metrics["target_qed_hit"] = 100.0 * int(metrics["qed_bin_counts"].get(target_qed, 0)) / max(1, metrics["n_unique"])
        metrics["target_sa_hit"] = (
            100.0 * int(metrics["sa_bin_counts"].get(target_sa, 0)) / max(1, metrics["n_unique"])
            if target_sa
            else float("nan")
        )
        row = {**spec, **metrics}
        results.append(row)
        print(
            f"[condition-grid] {spec['name']} V={row['validity']:.2f} "
            f"U={row['uniqueness']:.2f} Q={row['quality']:.2f} "
            f"mean_qed={row['mean_qed']:.3f} mean_sa={row['mean_sa']:.3f} "
            f"qed_hit={row['target_qed_hit']:.2f} sa_hit={row['target_sa_hit']:.2f}",
            flush=True,
        )
        if args.save_smiles:
            (out_dir / f"{spec['name']}.smi").write_text("\n".join(smiles) + "\n", encoding="utf-8")

    payload = {
        "model_dir": str(args.model_dir),
        "condition_mode": str(args.condition_mode),
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "max_new": int(args.max_new),
        "dedup_fill": bool(args.dedup_fill),
        "oversample_factor": float(args.oversample_factor),
        "hybrid_vocab": bool(hybrid_vocab is not None),
        "results": results,
    }
    (out_dir / "condition_grid_eval.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    tsv_path = out_dir / "condition_grid_summary.tsv"
    fieldnames = [
        "name",
        "tokens",
        "target_qed_bin",
        "target_sa_bin",
        "validity",
        "uniqueness",
        "quality",
        "raw_validity",
        "raw_uniqueness",
        "mean_qed",
        "mean_sa",
        "target_qed_hit",
        "target_sa_hit",
        "n_unique",
        "n_valid",
        "num_attempts",
    ]
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in results:
            flat = dict(row)
            flat["tokens"] = " ".join(row["tokens"])
            writer.writerow(flat)
    print(f"[condition-grid] wrote {out_dir / 'condition_grid_eval.json'}")
    print(f"[condition-grid] wrote {tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
