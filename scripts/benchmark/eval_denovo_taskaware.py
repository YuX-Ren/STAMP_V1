#!/usr/bin/env python3
"""De novo generation eval: sample N mols from BOS with the trained AO-GPT model.

Metrics match GenMol convention:
  V = n_valid / num_samples
  U = n_unique_valid / num_samples
  Q = n_unique_high_quality (QED>=0.6, SA<=4) / num_samples
  Diversity = avg pairwise Tanimoto distance over unique valid mols.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Sequence

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import QED

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.lm import load_trained_model, sample_model_sequences
from stamp.benchmark.metrics import _internal_diversity, _load_sa_scorer

RDLogger.DisableLog("rdApp.*")


def _canonical(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) if m else None


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


def _hybrid_tokens_to_stamp_tokens(tokens: Sequence[str], hybrid_vocab) -> list[str]:
    from stamp.benchmark.hybrid_vocab import MASK, PAD, BOS, EOS, UNK, is_condition_token, is_stamp_structural

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
            # Atomic motif tokens and fallback SMILES regex tokens are both
            # SMILES text. Structural tokens are the only motif boundaries.
            buf.append(tok)
    if buf:
        out.append("".join(buf))
    return out


def _is_condition_token(tok: str) -> bool:
    return tok.startswith("[C_") and tok.endswith("]")


def _strip_condition_tokens(tokens: Sequence[str]) -> list[str]:
    return [tok for tok in tokens if not _is_condition_token(tok)]


def _default_condition_tokens(condition_mode: str) -> list[str]:
    mode = str(condition_mode or "none").strip().lower()
    if mode == "qed":
        return ["[C_QED_HIGH]"]
    if mode == "qed_sa":
        return ["[C_QED_HIGH]", "[C_SA_LOW]"]
    return []


def _parse_condition_tokens(text: str, condition_mode: str) -> list[str]:
    raw = str(text or "").strip()
    if raw:
        return [tok for tok in raw.replace(",", " ").split() if tok]
    return _default_condition_tokens(condition_mode)


def _quality_count(smiles: Sequence[str], sa) -> int:
    n_pass = 0
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            q = QED.qed(m)
            sa_val = sa.calculateScore(m)
        except Exception:
            continue
        if q >= 0.6 and sa_val <= 4.0:
            n_pass += 1
    return n_pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=200, help="Generation batch size")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ao-decode-mode",
        default="ar",
        choices=["ar", "ao_canvas_confidence"],
        help="For ao_gpt checkpoints: identity-order AR or any-order canvas confidence filling.",
    )
    p.add_argument(
        "--ao-canvas-placeholder",
        default="unk",
        choices=["unk", "pad"],
        help="Placeholder token for unfilled positions in ao_canvas_confidence mode.",
    )
    p.add_argument("--condition-mode", default="none", choices=["none", "qed", "qed_sa"])
    p.add_argument(
        "--condition-tokens",
        default="",
        help="Prompt condition tokens. Empty uses a high-quality default for --condition-mode.",
    )
    p.add_argument(
        "--dedup-fill",
        action="store_true",
        help="Keep sampling until num-samples unique valid canonical SMILES are "
             "collected, or until max attempts is reached. Raw direct-sampling "
             "metrics are still reported as raw_*.",
    )
    p.add_argument(
        "--oversample-factor",
        type=float,
        default=3.0,
        help="Maximum attempts for --dedup-fill as num-samples * factor.",
    )
    p.add_argument(
        "--no-motif-tokenizer",
        action="store_true",
        help="Force codec to atomic-level (no motif tokenizer). Required for "
             "old AO-GPT checkpoints whose decode path was 'stamp w/o vocab'.",
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, vocab, rep, dev = load_trained_model(args.model_dir, device=args.device)
    hybrid_vocab = _load_hybrid_vocab(args.model_dir)
    if hybrid_vocab is not None:
        print(f"[INFO] Hybrid vocab decode enabled; vocab_size={len(hybrid_vocab.itos)}", flush=True)
    if args.no_motif_tokenizer:
        # Override codec to atomic-level (matches the prior 79% Q grid eval).
        from stamp.benchmark.representations import build_representation
        rep = build_representation("stamp")
        # The model device may have shifted; re-resolve.
        print("[INFO] Forced atomic-level codec (no motif tokenizer)", flush=True)
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    condition_tokens = _parse_condition_tokens(args.condition_tokens, args.condition_mode)
    if condition_tokens:
        print(f"[INFO] condition prompt: {' '.join(condition_tokens)}", flush=True)

    model_max_seq = None
    for sub in (model, getattr(model, "module", None)):
        if sub is None:
            continue
        cfg = getattr(sub, "cfg", None)
        if cfg is not None and hasattr(cfg, "max_seq_len"):
            model_max_seq = int(cfg.max_seq_len)
            break
    if model_max_seq is None:
        model_max_seq = int(args.max_new) + 4

    sa = _load_sa_scorer()
    num_total = int(args.num_samples)
    bs = max(1, int(args.batch_size))
    if bool(args.dedup_fill):
        max_attempts = int(math.ceil(num_total * max(1.0, float(args.oversample_factor))))
    else:
        max_attempts = num_total
    n_batches = math.ceil(max_attempts / bs)
    mode = "dedup-fill" if bool(args.dedup_fill) else "direct"
    ao_decode_mode = str(args.ao_decode_mode).strip().lower()
    print(
        f"[INFO] target={num_total} mode={mode} max_attempts={max_attempts} "
        f"in up to {n_batches} batches of {bs}",
        flush=True,
    )
    print(
        f"[INFO] ao_decode_mode={ao_decode_mode} "
        f"ao_canvas_placeholder={args.ao_canvas_placeholder}",
        flush=True,
    )

    all_smiles_valid: List[str] = []
    unique_seen: dict[str, None] = {}
    attempts = 0
    batch_idx = 0
    while attempts < max_attempts:
        if bool(args.dedup_fill) and len(unique_seen) >= num_total:
            break
        cur_n = min(bs, max_attempts - attempts)
        batch_idx += 1
        # For AO-GPT, `ar` uses identity target order while
        # `ao_canvas_confidence` exercises true any-order canvas filling.
        seqs = sample_model_sequences(
            model, vocab, n_samples=cur_n, max_new_tokens=int(args.max_new),
            temperature=float(args.temperature),
            top_k=int(args.top_k), top_p=float(args.top_p),
            prompt_tokens=condition_tokens,
            device=dev,
            ao_decode_mode=ao_decode_mode,
            ao_canvas_placeholder=str(args.ao_canvas_placeholder),
        )
        for toks in seqs:
            try:
                if hybrid_vocab is not None:
                    toks = _hybrid_tokens_to_stamp_tokens(toks, hybrid_vocab)
                else:
                    toks = _strip_condition_tokens(toks)
                text = rep.detokenize(toks)
                smi = rep.decode(text)
            except Exception:
                continue
            c = _canonical(smi)
            if c:
                all_smiles_valid.append(c)
                unique_seen.setdefault(c, None)
        attempts += cur_n
        print(
            f"[INFO] batch {batch_idx}/{n_batches}  attempts={attempts:,} "
            f"cum_valid={len(all_smiles_valid):,} cum_unique={len(unique_seen):,}",
            flush=True,
        )

    # Metrics (GenMol-style)
    raw_unique = list(unique_seen.keys())
    raw_n_valid = len(all_smiles_valid)
    raw_n_unique = len(raw_unique)
    raw_n_pass = _quality_count(raw_unique, sa)
    raw_validity = 100.0 * raw_n_valid / max(1, attempts)
    raw_uniqueness = 100.0 * raw_n_unique / max(1, attempts)
    raw_quality = 100.0 * raw_n_pass / max(1, attempts)

    if bool(args.dedup_fill):
        eval_unique = raw_unique[:num_total]
        n_valid = len(eval_unique)
        n_unique = len(eval_unique)
    else:
        eval_unique = raw_unique
        n_valid = raw_n_valid
        n_unique = raw_n_unique

    n_pass = _quality_count(eval_unique, sa)
    div = _internal_diversity(eval_unique) if n_unique >= 2 else float("nan")

    v_rate = 100.0 * n_valid / num_total
    u_rate = 100.0 * n_unique / num_total
    q_rate = 100.0 * n_pass / num_total

    summary = {
        "num_samples": num_total,
        "num_attempts": attempts,
        "dedup_fill": bool(args.dedup_fill),
        "oversample_factor": float(args.oversample_factor),
        "n_valid": n_valid,
        "n_unique": n_unique,
        "n_high_quality_unique": n_pass,
        "validity": v_rate,
        "uniqueness": u_rate,
        "quality": q_rate,
        "diversity": div,
        "raw_n_valid": raw_n_valid,
        "raw_n_unique": raw_n_unique,
        "raw_n_high_quality_unique": raw_n_pass,
        "raw_validity": raw_validity,
        "raw_uniqueness": raw_uniqueness,
        "raw_quality": raw_quality,
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "ao_decode_mode": ao_decode_mode,
        "ao_canvas_placeholder": str(args.ao_canvas_placeholder),
        "condition_mode": str(args.condition_mode),
        "condition_tokens": list(condition_tokens),
    }
    out_json = out_dir / "denovo_eval.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}\n")

    print(f"=== DE-NOVO SUMMARY ===")
    print(f"  num_samples:           {num_total}")
    print(f"  V (n_valid/N):         {v_rate:.2f}%   ({n_valid}/{num_total})")
    print(f"  U (n_unique/N):        {u_rate:.2f}%   ({n_unique}/{num_total})")
    print(f"  Q (n_unique_HQ/N):     {q_rate:.2f}%   ({n_pass}/{num_total})")
    print(f"  Diversity:             {div:.3f}")
    if bool(args.dedup_fill):
        print(f"  Attempts used:          {attempts}/{max_attempts}")
        print(f"  Raw U before dedup:     {raw_uniqueness:.2f}%   ({raw_n_unique}/{attempts})")
        print(f"  Raw Q before dedup:     {raw_quality:.2f}%   ({raw_n_pass}/{attempts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
