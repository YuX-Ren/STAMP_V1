#!/usr/bin/env python3
"""Evaluate a STAMP-MolT5 checkpoint with generative span infilling."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark.train_stamp_molt5_parquet import (  # noqa: E402
    StampT5ParquetDataset,
    StampT5Vocab,
    _autocast,
    _collate,
    _make_model,
    _resolve_paths,
    _span_corrupt,
)
from stamp.benchmark.hybrid_vocab import is_condition_token, is_fcstamp_control_token, is_stamp_structural  # noqa: E402
from stamp.core.stamp_codec import STAMPEncoder  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def _trim(ids: Sequence[int], *, eos_id: int, pad_id: int = -1, ignore_id: int = -100) -> list[int]:
    out: list[int] = []
    for raw in ids:
        idx = int(raw)
        if idx == ignore_id or idx == pad_id:
            continue
        out.append(idx)
        if idx == eos_id:
            break
    return out


def _is_sentinel(idx: int, vocab: StampT5Vocab) -> bool:
    return vocab.sentinel_start_id <= int(idx) < vocab.sentinel_start_id + int(vocab.num_extra_ids)


def _parse_fills(target_ids: Sequence[int], vocab: StampT5Vocab) -> dict[int, list[int]]:
    fills: dict[int, list[int]] = {}
    cur: int | None = None
    for idx in target_ids:
        idx = int(idx)
        if idx == vocab.eos_id:
            break
        if _is_sentinel(idx, vocab):
            cur = idx
            fills.setdefault(cur, [])
            continue
        if cur is not None and idx != vocab.pad_id:
            fills[cur].append(idx)
    return fills


def _splice(source_ids: Sequence[int], target_ids: Sequence[int], vocab: StampT5Vocab) -> list[int] | None:
    fills = _parse_fills(target_ids, vocab)
    out: list[int] = []
    used = 0
    for idx in source_ids:
        idx = int(idx)
        if idx == vocab.eos_id:
            break
        if _is_sentinel(idx, vocab):
            fill = fills.get(idx)
            if fill is None:
                return None
            out.extend(fill)
            used += 1
        elif idx != vocab.pad_id:
            out.append(idx)
    if used == 0:
        return None
    return out


def _ids_to_stamp_tokens(ids: Sequence[int], vocab: StampT5Vocab) -> list[str]:
    special = {"<pad>", "<eos>", "<unk>", "<mask>", "<bos>"}
    out: list[str] = []
    buf: list[str] = []
    for idx in ids:
        idx = int(idx)
        if idx < 0 or idx >= len(vocab.itos):
            continue
        tok = vocab.itos[idx]
        if tok in special or tok.startswith("<extra_id_") or is_condition_token(tok) or is_fcstamp_control_token(tok):
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


def _decode_ids(ids: Sequence[int] | None, vocab: StampT5Vocab, codec: STAMPEncoder) -> str | None:
    if ids is None:
        return None
    toks = _ids_to_stamp_tokens(ids, vocab)
    if not toks:
        return None
    try:
        smi = codec.decode_stamp(" ".join(toks), canonical=True)
    except Exception:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _token_accuracy(pred: Sequence[int], gold: Sequence[int], eos_id: int) -> tuple[int, int]:
    pred_t = _trim(pred, eos_id=eos_id)
    gold_t = _trim(gold, eos_id=eos_id)
    denom = max(len(pred_t), len(gold_t))
    if denom == 0:
        return 0, 0
    num = sum(1 for a, b in zip(pred_t, gold_t) if int(a) == int(b))
    return int(num), int(denom)


def _load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = StampT5Vocab.from_dict(ckpt["vocab"])
    ns = argparse.Namespace(**ckpt["args"])
    model = _make_model(ns, vocab)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, vocab, ns, int(ckpt.get("step", -1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid-parquet", default="/scratch/yuxuan.ren/data/zinc_l3_full_condition/val_*_stamp.parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-column", default="stamp")
    parser.add_argument("--num-examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-source-len", type=int, default=256)
    parser.add_argument("--max-target-len", type=int, default=128)
    parser.add_argument("--noise-density", type=float, default=0.15)
    parser.add_argument("--mean-noise-span-length", type=float, default=3.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--save-examples", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, vocab, train_args, step = _load_model(ckpt_path, device)
    valid_paths = _resolve_paths(args.valid_parquet)
    if not valid_paths:
        raise FileNotFoundError(args.valid_parquet)
    dataset = StampT5ParquetDataset(
        valid_paths,
        vocab=vocab,
        text_column=args.text_column,
        max_rows=int(args.num_examples),
        seed=int(args.seed),
        max_source_len=int(args.max_source_len),
        max_target_len=int(args.max_target_len),
        noise_density=float(args.noise_density),
        mean_span_length=float(args.mean_noise_span_length),
        fallback_training=False,
        fallback_p_low=float(getattr(train_args, "fallback_p_low", 0.02)),
        fallback_p_high=float(getattr(train_args, "fallback_p_high", 0.15)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        num_workers=0,
        collate_fn=lambda batch: _collate(batch, vocab.pad_id),
        pin_memory=device.type == "cuda",
    )
    codec = STAMPEncoder()
    rng = random.Random(args.seed)

    n = 0
    loss_sum = 0.0
    exact = 0
    tok_num = 0
    tok_den = 0
    gold_valid = 0
    pred_valid = 0
    smiles_match = 0
    examples: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with _autocast(device, getattr(train_args, "amp", "bf16")):
                out = model(**batch)
            loss_sum += float(out.loss.detach().item()) * int(batch["input_ids"].shape[0])
            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=int(args.max_target_len),
                num_beams=int(args.num_beams),
                do_sample=False,
                eos_token_id=vocab.eos_id,
                pad_token_id=vocab.pad_id,
                decoder_start_token_id=vocab.pad_id,
            )
            for i in range(int(batch["input_ids"].shape[0])):
                src = _trim(batch["input_ids"][i].detach().cpu().tolist(), eos_id=vocab.eos_id, pad_id=vocab.pad_id)
                gold = _trim(batch["labels"][i].detach().cpu().tolist(), eos_id=vocab.eos_id)
                pred = _trim(generated[i].detach().cpu().tolist(), eos_id=vocab.eos_id, pad_id=vocab.pad_id)
                if pred and pred[0] == vocab.pad_id:
                    pred = pred[1:]
                exact += int(pred == gold)
                a, b = _token_accuracy(pred, gold, vocab.eos_id)
                tok_num += a
                tok_den += b
                gold_body = _splice(src, gold, vocab)
                pred_body = _splice(src, pred, vocab)
                gold_smi = _decode_ids(gold_body, vocab, codec)
                pred_smi = _decode_ids(pred_body, vocab, codec)
                gold_valid += int(gold_smi is not None)
                pred_valid += int(pred_smi is not None)
                smiles_match += int(gold_smi is not None and pred_smi is not None and gold_smi == pred_smi)
                if len(examples) < int(args.save_examples):
                    examples.append(
                        {
                            "source": " ".join(vocab.itos[x] for x in src if 0 <= x < len(vocab.itos)),
                            "gold_target": " ".join(vocab.itos[x] for x in gold if 0 <= x < len(vocab.itos)),
                            "pred_target": " ".join(vocab.itos[x] for x in pred if 0 <= x < len(vocab.itos)),
                            "gold_smiles": gold_smi,
                            "pred_smiles": pred_smi,
                            "exact_target": bool(pred == gold),
                        }
                    )
                n += 1
                if n >= int(args.num_examples):
                    break
            if n >= int(args.num_examples):
                break

    metrics = {
        "checkpoint": str(ckpt_path),
        "checkpoint_step": int(step),
        "num_examples": int(n),
        "loss": float(loss_sum / max(1, n)),
        "ppl": float(math.exp(min(20.0, loss_sum / max(1, n)))),
        "target_exact_match": 100.0 * exact / max(1, n),
        "target_token_accuracy": 100.0 * tok_num / max(1, tok_den),
        "gold_decode_validity": 100.0 * gold_valid / max(1, n),
        "pred_decode_validity": 100.0 * pred_valid / max(1, n),
        "canonical_smiles_match": 100.0 * smiles_match / max(1, n),
        "num_beams": int(args.num_beams),
        "seed": int(args.seed),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "examples.jsonl").open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "examples.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "gold_target", "pred_target", "gold_smiles", "pred_smiles", "exact_target"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(examples)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
