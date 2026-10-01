#!/usr/bin/env python3
"""Quantify generated-token -> canonical-SMILES many-to-one collapse.

The script samples from a checkpoint and decomposes molecule-level duplicates
into:
  1) exact raw-token sequence repeats,
  2) hybrid vocab aliases that normalize to the same STAMP token stream, and
  3) distinct normalized STAMP streams that decode to the same canonical SMILES.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.hybrid_vocab import (  # noqa: E402
    BOS,
    EOS,
    MASK,
    PAD,
    UNK,
    HybridVocab,
    is_stamp_structural,
)
from stamp.benchmark.lm import (  # noqa: E402
    LMConfig,
    LlamaDecoderLM,
    TinyDecoderLM,
    TokenVocab,
    _normalize_model_type,
    _resolve_torch_device,
    sample_model_sequences,
)
from stamp.core.stamp_codec import STAMPEncoder  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def _unwrap_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    if any(k.startswith("_orig_mod.") for k in state_dict):
        return {
            (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
            for k, v in state_dict.items()
        }
    return state_dict


def _load_vocab(model_dir: Path) -> tuple[Any, TokenVocab, bool]:
    payload = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    is_hybrid = "atomic_motifs" in payload and "motif_expansion" in payload
    if is_hybrid:
        vocab = HybridVocab.from_dict(payload)
        sample_vocab = TokenVocab(stoi=dict(vocab.stoi), itos=list(vocab.itos))
    else:
        vocab = TokenVocab.from_dict(payload)
        sample_vocab = vocab
    return vocab, sample_vocab, is_hybrid


def _load_model(checkpoint: Path, model_dir: Path, device: str | None):
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = LMConfig(**ckpt["cfg"])
    model_type = _normalize_model_type(str(ckpt.get("model_type", cfg.model_type)))
    cfg.model_type = model_type
    vocab, sample_vocab, is_hybrid = _load_vocab(model_dir)
    if model_type == "llama":
        model = LlamaDecoderLM(vocab_size=len(sample_vocab.itos), cfg=cfg)
    else:
        model = TinyDecoderLM(vocab_size=len(sample_vocab.itos), cfg=cfg)
    model.load_state_dict(_unwrap_state_dict(ckpt["model_state"]))
    dev = _resolve_torch_device(device)
    model.to(dev)
    model.eval()
    return model, vocab, sample_vocab, is_hybrid, dev


def _hybrid_tokens_to_stamp_tokens(tokens: Iterable[str]) -> tuple[str, ...]:
    special = {PAD, BOS, EOS, MASK, UNK}
    out: list[str] = []
    buf: list[str] = []
    for tok in tokens:
        if tok in special:
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
    return tuple(out)


def _canonical_from_stamp(stamp_tokens: Sequence[str], decoder: STAMPEncoder) -> str | None:
    try:
        mol = decoder.decode_stamp_to_mol(" ".join(stamp_tokens))
    except Exception:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _example_tokens(seq: Sequence[str], limit: int = 80) -> str:
    text = " ".join(seq)
    return text if len(text) <= limit else text[:limit] + " ..."


def _summarize(
    rows: list[dict[str, Any]],
    *,
    num_samples: int,
    checkpoint: Path,
    temperature: float,
    top_k: int,
    top_p: float,
    max_new: int,
    is_hybrid: bool,
    max_examples: int,
) -> dict[str, Any]:
    valid_rows = [r for r in rows if r["canonical"]]
    valid = len(valid_rows)
    raw_tokens = [tuple(r["raw_tokens"]) for r in valid_rows]
    stamp_tokens = [tuple(r["stamp_tokens"]) for r in valid_rows]
    canonicals = [str(r["canonical"]) for r in valid_rows]

    unique_raw = len(set(raw_tokens))
    unique_stamp = len(set(stamp_tokens))
    unique_canonical = len(set(canonicals))

    raw_by_canonical: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    stamp_by_canonical: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    samples_by_canonical: dict[str, int] = defaultdict(int)
    raw_by_stamp: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    samples_by_stamp: dict[tuple[str, ...], int] = defaultdict(int)
    for r in valid_rows:
        raw = tuple(r["raw_tokens"])
        stamp = tuple(r["stamp_tokens"])
        can = str(r["canonical"])
        raw_by_canonical[can].add(raw)
        stamp_by_canonical[can].add(stamp)
        samples_by_canonical[can] += 1
        raw_by_stamp[stamp].add(raw)
        samples_by_stamp[stamp] += 1

    hybrid_alias_stamps = {
        stamp: raws for stamp, raws in raw_by_stamp.items() if len(raws) > 1
    }
    canonical_alias_groups = {
        can: raws for can, raws in raw_by_canonical.items() if len(raws) > 1
    }
    stamp_alias_groups = {
        can: stamps for can, stamps in stamp_by_canonical.items() if len(stamps) > 1
    }

    top_canonical = sorted(
        canonical_alias_groups,
        key=lambda can: (
            len(raw_by_canonical[can]),
            len(stamp_by_canonical[can]),
            samples_by_canonical[can],
        ),
        reverse=True,
    )[:max_examples]
    examples = []
    for can in top_canonical:
        raw_examples = list(raw_by_canonical[can])[:3]
        stamp_examples = list(stamp_by_canonical[can])[:3]
        examples.append(
            {
                "canonical": can,
                "n_samples": int(samples_by_canonical[can]),
                "n_distinct_raw_token_sequences": int(len(raw_by_canonical[can])),
                "n_distinct_normalized_stamp_sequences": int(len(stamp_by_canonical[can])),
                "raw_token_examples": [_example_tokens(x) for x in raw_examples],
                "stamp_token_examples": [_example_tokens(x) for x in stamp_examples],
            }
        )

    return {
        "checkpoint": str(checkpoint),
        "num_samples": int(num_samples),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "max_new": int(max_new),
        "is_hybrid_vocab": bool(is_hybrid),
        "n_valid": int(valid),
        "n_invalid": int(num_samples - valid),
        "n_unique_raw_token_sequences": int(unique_raw),
        "n_unique_normalized_stamp_sequences": int(unique_stamp),
        "n_unique_canonical_smiles": int(unique_canonical),
        "raw_uniqueness_pct": 100.0 * unique_canonical / max(1, num_samples),
        "valid_unique_over_valid_pct": 100.0 * unique_canonical / max(1, valid),
        "exact_raw_token_repeat_samples": int(valid - unique_raw),
        "hybrid_alias_unique_excess": int(unique_raw - unique_stamp),
        "stamp_to_canonical_alias_unique_excess": int(unique_stamp - unique_canonical),
        "total_unique_sequence_alias_excess": int(unique_raw - unique_canonical),
        "canonical_groups_with_multiple_raw_sequences": int(len(canonical_alias_groups)),
        "canonical_groups_with_multiple_stamp_sequences": int(len(stamp_alias_groups)),
        "stamp_groups_with_multiple_raw_sequences": int(len(hybrid_alias_stamps)),
        "samples_in_multi_raw_canonical_groups": int(
            sum(samples_by_canonical[can] for can in canonical_alias_groups)
        ),
        "samples_in_multi_stamp_canonical_groups": int(
            sum(samples_by_canonical[can] for can in stamp_alias_groups)
        ),
        "samples_in_hybrid_alias_stamp_groups": int(
            sum(samples_by_stamp[stamp] for stamp in hybrid_alias_stamps)
        ),
        "examples": examples,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-dir", default="", help="Defaults to checkpoint parent/..")
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-samples", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=127)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-examples", type=int, default=20)
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    model_dir = Path(args.model_dir) if args.model_dir else checkpoint.parent.parent
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, vocab, sample_vocab, is_hybrid, dev = _load_model(checkpoint, model_dir, args.device)
    decoder = STAMPEncoder(motif_tokenizer=None)
    torch.manual_seed(int(args.seed))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    rows: list[dict[str, Any]] = []
    remaining = int(args.num_samples)
    while remaining > 0:
        cur_n = min(int(args.batch_size), remaining)
        seqs = sample_model_sequences(
            model,
            sample_vocab,
            n_samples=cur_n,
            max_new_tokens=int(args.max_new),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            device=dev,
            ao_decode_mode="ar",
        )
        for raw in seqs:
            stamp = _hybrid_tokens_to_stamp_tokens(raw) if is_hybrid else tuple(raw)
            canonical = _canonical_from_stamp(stamp, decoder)
            rows.append(
                {
                    "raw_tokens": list(raw),
                    "stamp_tokens": list(stamp),
                    "canonical": canonical,
                }
            )
        remaining -= cur_n

    summary = _summarize(
        rows,
        num_samples=int(args.num_samples),
        checkpoint=checkpoint,
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        max_new=int(args.max_new),
        is_hybrid=bool(is_hybrid),
        max_examples=int(args.max_examples),
    )
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
