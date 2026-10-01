#!/usr/bin/env python3
"""Evaluate AO-GPT fixed-length random-order AR sampling on hybrid STAMP vocab."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import QED

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.hybrid_vocab import (
    BOS,
    EOS,
    MASK,
    PAD,
    UNK,
    HybridVocab,
    is_condition_token,
    is_stamp_structural,
)
from stamp.benchmark.lm import (
    _sample_next_tokens,
    _trim_special_tail,
    load_trained_model,
    sample_model_sequences,
)
from stamp.benchmark.metrics import _internal_diversity, _load_sa_scorer

RDLogger.DisableLog("rdApp.*")


def _canonical(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _load_hybrid_vocab(model_dir: str | Path) -> HybridVocab | None:
    vocab_path = Path(model_dir) / "vocab.json"
    if not vocab_path.exists():
        return None
    try:
        payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if "atomic_motifs" not in payload or "motif_expansion" not in payload:
        return None
    return HybridVocab.from_dict(payload)


def _hybrid_tokens_to_stamp_tokens(tokens: Sequence[str]) -> list[str]:
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


def _read_train_ids_length_distribution(model_dir: str | Path, max_len: int) -> np.ndarray:
    meta_path = Path(model_dir) / "train_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing train_meta.json in {model_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    hist = meta.get("train_token_length_histogram") or {}
    lengths: list[int] = []
    weights: list[int] = []
    for key, value in hist.items():
        try:
            tok_len = int(key)
            count = int(value)
        except Exception:
            continue
        if count <= 0:
            continue
        ids_len = min(max_len, max(4, tok_len + 2))
        lengths.append(ids_len)
        weights.append(count)
    if not lengths:
        stats = meta.get("train_token_length_stats") or {}
        mean_len = int(round(float(stats.get("mean", 32)))) + 2
        lengths = [min(max_len, max(4, mean_len))]
        weights = [1]
    arr = np.asarray(lengths, dtype=np.int64)
    prob = np.asarray(weights, dtype=np.float64)
    prob = prob / prob.sum()
    return np.column_stack([arr, prob])


def _sample_lengths(length_dist: np.ndarray, n: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    lengths = length_dist[:, 0].astype(np.int64)
    probs = length_dist[:, 1].astype(np.float64)
    sampled = rng.choice(lengths, size=n, replace=True, p=probs)
    return torch.tensor(sampled, dtype=torch.long)


def random_order_ar(
    model,
    vocab,
    n_samples: int,
    lengths: torch.Tensor,
    *,
    max_len: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    device: torch.device,
) -> list[list[str]]:
    """Fixed-length random-order AR aligned to the AO-GPT next-slot objective."""
    if int(lengths.max().item()) > max_len:
        raise ValueError("sampled lengths exceed max_len")

    generator = random.Random(seed)
    rows = torch.arange(n_samples, device=device)
    lengths = lengths.to(device=device, dtype=torch.long)
    eos_pos = lengths - 1
    last_valid_y = eos_pos - 1
    n_valid = last_valid_y + 1
    y_len = max_len - 1

    ids = torch.full((n_samples, max_len), vocab.pad_id, dtype=torch.long, device=device)
    ids[:, 0] = vocab.bos_id
    ids[rows, eos_pos] = vocab.eos_id

    orders = torch.zeros(n_samples, y_len, dtype=torch.long, device=device)
    for b in range(n_samples):
        lv = int(last_valid_y[b].item())
        mid = list(range(1, lv))
        generator.shuffle(mid)
        row = [0] + mid + [lv] + list(range(lv + 1, y_len))
        orders[b] = torch.tensor(row, dtype=torch.long, device=device)

    max_steps = int((n_valid - 1).max().item())
    t0 = time.time()
    for k in range(max_steps):
        active = torch.full((n_samples,), k, dtype=torch.long, device=device) < (n_valid - 1)
        if not bool(active.any().item()):
            break
        target_y = orders[rows, torch.full((n_samples,), k, device=device, dtype=torch.long)]
        x_in = ids[:, :-1].contiguous()
        with torch.no_grad():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(x_in, orders=orders).float()
        logits_k = logits[:, k, :].clone()
        for suppressed_id in (vocab.pad_id, vocab.bos_id, vocab.eos_id, vocab.mask_id, vocab.unk_id):
            if 0 <= int(suppressed_id) < logits_k.size(-1):
                logits_k[:, int(suppressed_id)] = float("-inf")
        sampled = _sample_next_tokens(
            logits_k,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        target_ids_pos = target_y + 1
        ids[rows, target_ids_pos] = torch.where(active, sampled, ids[rows, target_ids_pos])

    print(f"[INFO] random-order AR decode finished in {time.time() - t0:.1f}s steps={max_steps}", flush=True)
    out: list[list[str]] = []
    for row in ids[:, 1:].detach().cpu().tolist():
        row = _trim_special_tail(row, vocab)
        out.append(vocab.decode_ids(row))
    return out


def _decode_canonical_smiles(tokens_list, rep, hybrid_vocab) -> tuple[list[str | None], list[str | None]]:
    raw_smiles: list[str | None] = []
    canon: list[str | None] = []
    for toks in tokens_list:
        try:
            stamp_toks = _hybrid_tokens_to_stamp_tokens(toks) if hybrid_vocab is not None else list(toks)
            smi = rep.decode(rep.detokenize(stamp_toks))
        except Exception:
            smi = None
        raw_smiles.append(smi)
        if smi:
            try:
                canon.append(_canonical(smi))
            except Exception:
                canon.append(None)
        else:
            canon.append(None)
    return raw_smiles, canon


def _quality_count(smiles: Sequence[str], sa_scorer) -> int:
    n_pass = 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            if QED.qed(mol) >= 0.6 and sa_scorer.calculateScore(mol) <= 4.0:
                n_pass += 1
        except Exception:
            continue
    return n_pass


def score_sequences(name: str, seqs, rep, hybrid_vocab, sa_scorer) -> dict:
    raw_smiles, canon = _decode_canonical_smiles(seqs, rep, hybrid_vocab)
    valid = [s for s in canon if s]
    unique = list(dict.fromkeys(valid))
    n_total = len(canon)
    n_valid = len(valid)
    n_unique = len(unique)
    n_hq_unique = _quality_count(unique, sa_scorer)
    n_hq_valid = _quality_count(valid, sa_scorer)
    diversity = _internal_diversity(unique) if n_unique >= 2 else 0.0
    legacy_genmol = (
        (n_valid / max(1, n_total))
        * (n_unique / max(1, n_valid))
        * (n_hq_valid / max(1, n_valid))
        * 100.0
    )
    return {
        "name": name,
        "num_samples": n_total,
        "n_valid": n_valid,
        "n_unique": n_unique,
        "n_high_quality_unique": n_hq_unique,
        "validity": 100.0 * n_valid / max(1, n_total),
        "uniqueness": 100.0 * n_unique / max(1, n_total),
        "quality": 100.0 * n_hq_unique / max(1, n_total),
        "unique_per_valid": 100.0 * n_unique / max(1, n_valid),
        "quality_per_valid": 100.0 * n_hq_valid / max(1, n_valid),
        "legacy_genmol": legacy_genmol,
        "diversity": diversity,
        "examples": [s for s in raw_smiles if s][:10],
    }


def _parse_configs(text: str) -> list[tuple[str, float, int, float]]:
    out: list[tuple[str, float, int, float]] = []
    for raw in str(text).split(";"):
        item = raw.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(",")]
        if len(parts) not in {3, 4}:
            raise ValueError(f"Bad config {item!r}; expected mode,temp,top_k[,top_p]")
        mode = parts[0].lower()
        if mode not in {"ar", "rand"}:
            raise ValueError(f"Bad config mode {mode!r}; expected ar|rand")
        temp = float(parts[1])
        top_k = int(parts[2])
        top_p = float(parts[3]) if len(parts) == 4 else 0.0
        out.append((mode, temp, top_k, top_p))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=96, help="Total ids length including BOS/EOS.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--configs",
        default="ar,0.6,10,0;rand,0.6,10,0;rand,0.7,10,0",
        help="Semicolon-separated mode,temp,top_k,top_p entries. mode is ar or rand.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    model, vocab, rep, device = load_trained_model(args.model_dir, device=args.device)
    hybrid_vocab = _load_hybrid_vocab(args.model_dir)
    print(
        f"[INFO] model_dir={args.model_dir} vocab={len(vocab.itos)} "
        f"hybrid_decode={hybrid_vocab is not None} max_len={args.max_len}",
        flush=True,
    )

    length_dist = _read_train_ids_length_distribution(args.model_dir, int(args.max_len))
    sampled_lengths = _sample_lengths(length_dist, int(args.num_samples), int(args.seed))
    print(
        f"[INFO] sampled ids lengths mean={sampled_lengths.float().mean().item():.1f} "
        f"min={int(sampled_lengths.min().item())} max={int(sampled_lengths.max().item())}",
        flush=True,
    )

    sa_scorer = _load_sa_scorer()
    results: list[dict] = []
    for mode, temp, top_k, top_p in _parse_configs(args.configs):
        name = f"{mode} T={temp:g} K={top_k} P={top_p:g}"
        print(f"[INFO] sampling {name}", flush=True)
        t0 = time.time()
        if mode == "ar":
            seqs = sample_model_sequences(
                model,
                vocab,
                n_samples=int(args.num_samples),
                max_new_tokens=max(1, int(args.max_len) - 1),
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                device=device,
                ao_decode_mode="ar",
            )
        else:
            seqs = random_order_ar(
                model,
                vocab,
                int(args.num_samples),
                sampled_lengths,
                max_len=int(args.max_len),
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                seed=int(args.seed),
                device=device,
            )
        result = score_sequences(name, seqs, rep, hybrid_vocab, sa_scorer)
        result.update(
            {
                "mode": mode,
                "temperature": temp,
                "top_k": top_k,
                "top_p": top_p,
                "elapsed_sec": time.time() - t0,
            }
        )
        results.append(result)
        print(
            f"[RESULT] {name} V={result['validity']:.2f} U={result['uniqueness']:.2f} "
            f"Q={result['quality']:.2f} U|V={result['unique_per_valid']:.2f} "
            f"Q|V={result['quality_per_valid']:.2f} legacy={result['legacy_genmol']:.2f} "
            f"D={result['diversity']:.3f}",
            flush=True,
        )

    summary = {
        "model_dir": str(args.model_dir),
        "num_samples": int(args.num_samples),
        "max_len": int(args.max_len),
        "seed": int(args.seed),
        "configs": str(args.configs),
        "hybrid_decode": hybrid_vocab is not None,
        "results": results,
    }
    out_json = out_dir / "random_order_ar_hybrid_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
