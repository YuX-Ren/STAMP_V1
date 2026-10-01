#!/usr/bin/env python3
"""Single-motif slot infill diagnostic for AO-GPT STAMP checkpoints."""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import pyarrow.parquet as pq
import torch
from rdkit import Chem, RDLogger

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
    is_fcstamp_control_token,
    is_stamp_structural,
)
from stamp.benchmark.lm import load_trained_model

RDLogger.DisableLog("rdApp.*")


def _load_hybrid_vocab(model_dir: str | Path) -> HybridVocab | None:
    vocab_path = Path(model_dir) / "vocab.json"
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    if "atomic_motifs" not in payload or "motif_expansion" not in payload:
        return None
    return HybridVocab.from_dict(payload)


def _iter_stamp_rows(paths_glob: str, *, stamp_column: str, smiles_column: str, max_scan_rows: int):
    paths = sorted(glob.glob(paths_glob))
    if not paths:
        raise FileNotFoundError(f"No parquet files matched: {paths_glob}")
    scanned = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        columns = [stamp_column]
        if smiles_column:
            columns.append(smiles_column)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=columns)
            data = table.to_pydict()
            stamps = data.get(stamp_column) or []
            smiles = data.get(smiles_column) if smiles_column else None
            for i, stamp in enumerate(stamps):
                scanned += 1
                yield {
                    "path": path,
                    "row_in_scan": scanned,
                    "stamp": stamp,
                    "smiles": (smiles[i] if smiles is not None else None),
                }
                if max_scan_rows > 0 and scanned >= max_scan_rows:
                    return


def _encode_with_token_spans(tokens: Sequence[str], vocab, hybrid_vocab: HybridVocab | None) -> tuple[list[int], list[tuple[int, int]]]:
    ids = [vocab.bos_id]
    spans: list[tuple[int, int]] = []
    if hybrid_vocab is not None:
        for tok in tokens:
            start = len(ids)
            ids.extend(hybrid_vocab.encode_stream([tok], training=False))
            spans.append((start, len(ids)))
    else:
        for tok in tokens:
            start = len(ids)
            ids.extend(vocab.encode_tokens([tok]))
            spans.append((start, len(ids)))
    ids.append(vocab.eos_id)
    return ids, spans


def _is_maskable_motif(tok: str, start: int, end: int, vocab, hybrid_vocab: HybridVocab | None) -> bool:
    if is_stamp_structural(tok) or is_condition_token(tok) or is_fcstamp_control_token(tok):
        return False
    if (end - start) != 1:
        return False
    if hybrid_vocab is not None:
        return tok in hybrid_vocab.atomic_motifs
    return tok in vocab.stoi and int(vocab.stoi[tok]) != int(vocab.unk_id)


def _choose_maskable_token(
    tokens: Sequence[str],
    spans: Sequence[tuple[int, int]],
    vocab,
    hybrid_vocab: HybridVocab | None,
    rng: random.Random,
) -> int | None:
    candidates = [
        i
        for i, tok in enumerate(tokens)
        if _is_maskable_motif(tok, spans[i][0], spans[i][1], vocab, hybrid_vocab)
    ]
    # Avoid first-token fills when possible; y[0] is a special first-valid case.
    non_first = [i for i in candidates if spans[i][0] > 1]
    if non_first:
        candidates = non_first
    if not candidates:
        return None
    return rng.choice(candidates)


def _build_no_leak_order(seqlen: int, target_ids_pos: int, *, mode: str) -> tuple[torch.Tensor, int]:
    """Return AO order and output index for predicting ids[target_ids_pos].

    Target ids position p is label slot y[p-1]. The leaking input slot is x[p],
    which contains the target token itself. We place x[p] after the target slot.
    """
    target_y = target_ids_pos - 1
    leak_x = target_ids_pos
    if target_y < 0 or target_y >= seqlen:
        raise ValueError(f"Invalid target slot p={target_ids_pos} for seqlen={seqlen}")
    if leak_x >= seqlen:
        leak_x = -1

    held_after: list[int] = []
    if leak_x >= 0 and leak_x != target_y:
        held_after.append(leak_x)

    if mode == "trainlike":
        # Match _sample_ao_orders as much as possible: first valid first, EOS
        # target slot last. The target itself is inserted before leak_x/EOS.
        eos_y = seqlen - 1
        if eos_y != target_y and eos_y not in held_after:
            held_after.append(eos_y)
        prefix = [0] if target_y != 0 and 0 not in held_after else []
        prefix.extend(
            j for j in range(seqlen)
            if j not in set(prefix) and j != target_y and j not in set(held_after)
        )
    elif mode == "fullctx":
        # True in-place infill: expose every clean x slot except x[p] before
        # predicting y[p-1]. This is less training-like but closer to masking.
        prefix = [j for j in range(seqlen) if j != target_y and j not in set(held_after)]
    else:
        raise ValueError(f"Unsupported order mode: {mode}")

    order = prefix + [target_y] + [j for j in held_after if j != target_y]
    used = set(order)
    order.extend(j for j in range(seqlen) if j not in used)
    if len(order) != seqlen or len(set(order)) != seqlen:
        raise AssertionError("AO order is not a permutation")
    return torch.tensor(order, dtype=torch.long), order.index(target_y)


def _valid_fill_token_mask(vocab) -> torch.Tensor:
    bad = {PAD, BOS, EOS, MASK, UNK}
    allowed = []
    for tok in vocab.itos:
        ok = (
            tok not in bad
            and not is_stamp_structural(tok)
            and not is_condition_token(tok)
            and not is_fcstamp_control_token(tok)
        )
        allowed.append(ok)
    return torch.tensor(allowed, dtype=torch.bool)


def _decode_spliced(tokens: Sequence[str], token_index: int, fill_tok: str, rep) -> tuple[str | None, str | None]:
    if (
        fill_tok in {PAD, BOS, EOS, MASK, UNK}
        or is_stamp_structural(fill_tok)
        or is_condition_token(fill_tok)
        or is_fcstamp_control_token(fill_tok)
    ):
        return None, None
    filled = list(tokens)
    filled[token_index] = fill_tok
    try:
        smiles = rep.decode(rep.detokenize(filled))
        mol = Chem.MolFromSmiles(smiles)
        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else None
        return smiles, canon
    except Exception:
        return None, None


def _run_one(model, vocab, rep, hybrid_vocab, row, token_index: int, *, order_mode: str, top_k: int, device):
    tokens = str(row["stamp"]).split()
    ids, spans = _encode_with_token_spans(tokens, vocab, hybrid_vocab)
    start, end = spans[token_index]
    if end - start != 1:
        raise ValueError("selected motif is not single-token encoded")
    target_id = ids[start]
    clean_ids = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    order, target_order_idx = _build_no_leak_order(clean_ids.size(1), start, mode=order_mode)
    order = order.to(device).unsqueeze(0)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(clean_ids, orders=order)[0, target_order_idx, :].float()

    probs = torch.softmax(logits, dim=-1)
    target_prob = float(probs[target_id].item())
    target_rank = int((probs > probs[target_id]).sum().item()) + 1

    constrained = logits.clone()
    allowed = _valid_fill_token_mask(vocab).to(device)
    constrained[~allowed] = float("-inf")
    cprobs = torch.softmax(constrained, dim=-1)
    vals, idxs = torch.topk(cprobs, k=min(top_k, cprobs.numel()))
    top = []
    for prob, idx in zip(vals.detach().cpu().tolist(), idxs.detach().cpu().tolist()):
        fill_tok = vocab.itos[int(idx)]
        smiles, canon = _decode_spliced(tokens, token_index, fill_tok, rep)
        top.append(
            {
                "token": fill_tok,
                "prob": float(prob),
                "match": int(idx) == int(target_id),
                "decode_valid": canon is not None,
                "smiles": smiles,
                "canonical_smiles": canon,
            }
        )

    top_ids = [int(x) for x in idxs.detach().cpu().tolist()]
    original_smiles, original_canon = _decode_spliced(tokens, token_index, tokens[token_index], rep)
    return {
        "source_smiles": row.get("smiles"),
        "source_stamp": row["stamp"],
        "original_decode_smiles": original_smiles,
        "original_decode_canonical": original_canon,
        "masked_token_index": token_index,
        "masked_ids_pos": start,
        "target_token": tokens[token_index],
        "target_id": int(target_id),
        "target_prob_raw": target_prob,
        "target_rank_raw": target_rank,
        "target_in_constrained_topk": int(target_id) in top_ids,
        "top1_match": bool(top and top[0]["match"]),
        "top5_match": int(target_id) in top_ids[:5],
        "top1_decode_valid": bool(top and top[0]["decode_valid"]),
        "topk": top,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--stamp-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-examples", type=int, default=32)
    parser.add_argument("--max-scan-rows", type=int, default=2000)
    parser.add_argument("--stamp-column", default="stamp")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument(
        "--cases-json",
        default="",
        help="Optional fixed cases JSON. Expected either a list or {'cases': list}; "
             "each case needs stamp and masked_token_index.",
    )
    parser.add_argument("--order-mode", choices=["trainlike", "fullctx"], default="trainlike")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    model, vocab, rep, device = load_trained_model(args.model_dir, device=args.device)
    hybrid_vocab = _load_hybrid_vocab(args.model_dir)
    print(
        f"[INFO] model={args.model_dir} vocab={len(vocab.itos)} "
        f"vocab_type={'hybrid' if hybrid_vocab is not None else 'direct'} "
        f"order_mode={args.order_mode}",
        flush=True,
    )

    results = []
    scanned = 0
    if str(args.cases_json).strip():
        payload = json.loads(Path(args.cases_json).read_text(encoding="utf-8"))
        raw_cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
        row_iter = (
            {
                "stamp": c["stamp"],
                "smiles": c.get("smiles"),
                "row_in_scan": c.get("row_in_scan"),
                "masked_token_index": int(c["masked_token_index"]),
            }
            for c in raw_cases
        )
    else:
        row_iter = _iter_stamp_rows(
            args.stamp_parquet,
            stamp_column=args.stamp_column,
            smiles_column=args.smiles_column,
            max_scan_rows=int(args.max_scan_rows),
        )

    for row in row_iter:
        scanned += 1
        stamp = row.get("stamp")
        if not stamp:
            continue
        tokens = str(stamp).split()
        ids, spans = _encode_with_token_spans(tokens, vocab, hybrid_vocab)
        if len(ids) - 1 > model.cfg.max_seq_len:
            continue
        if "masked_token_index" in row:
            token_index = int(row["masked_token_index"])
            if token_index < 0 or token_index >= len(tokens):
                continue
            if not _is_maskable_motif(tokens[token_index], spans[token_index][0], spans[token_index][1], vocab, hybrid_vocab):
                continue
        else:
            token_index = _choose_maskable_token(tokens, spans, vocab, hybrid_vocab, rng)
        if token_index is None:
            continue
        try:
            results.append(
                _run_one(
                    model,
                    vocab,
                    rep,
                    hybrid_vocab,
                    row,
                    token_index,
                    order_mode=str(args.order_mode),
                    top_k=int(args.top_k),
                    device=device,
                )
            )
        except Exception as exc:
            print(f"[WARN] skip row={row.get('row_in_scan')} error={exc}", flush=True)
            continue
        if len(results) >= int(args.num_examples):
            break

    if not results:
        raise RuntimeError("No maskable examples were evaluated.")

    n = len(results)
    summary = {
        "model_dir": str(args.model_dir),
        "stamp_parquet": str(args.stamp_parquet),
        "num_examples": n,
        "scanned_rows": scanned,
        "order_mode": str(args.order_mode),
        "top_k": int(args.top_k),
        "cases_json": str(args.cases_json),
        "top1_exact": 100.0 * sum(r["top1_match"] for r in results) / n,
        "top5_exact": 100.0 * sum(r["top5_match"] for r in results) / n,
        "topk_contains_target": 100.0 * sum(r["target_in_constrained_topk"] for r in results) / n,
        "top1_decode_valid": 100.0 * sum(r["top1_decode_valid"] for r in results) / n,
        "mean_target_prob_raw": sum(float(r["target_prob_raw"]) for r in results) / n,
        "median_target_rank_raw": sorted(int(r["target_rank_raw"]) for r in results)[n // 2],
        "examples": results,
    }
    out_json = out_dir / f"motif_slot_infill_{args.order_mode}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[SUMMARY]", flush=True)
    print(
        f"n={n} top1={summary['top1_exact']:.2f}% top5={summary['top5_exact']:.2f}% "
        f"top{args.top_k}={summary['topk_contains_target']:.2f}% "
        f"top1_decode_valid={summary['top1_decode_valid']:.2f}% "
        f"mean_target_prob_raw={summary['mean_target_prob_raw']:.4f} "
        f"median_target_rank_raw={summary['median_target_rank_raw']}",
        flush=True,
    )
    print("[EXAMPLES]", flush=True)
    for i, r in enumerate(results[: min(8, len(results))], start=1):
        top_str = ", ".join(f"{x['token']}:{x['prob']:.3f}" for x in r["topk"][:5])
        print(
            f"{i}. target={r['target_token']} rank={r['target_rank_raw']} "
            f"top1={r['topk'][0]['token']} valid={r['top1_decode_valid']} top5=[{top_str}]",
            flush=True,
        )
        print(f"   stamp={r['source_stamp']}", flush=True)
        if r["topk"][0].get("smiles"):
            print(f"   top1_smiles={r['topk'][0]['smiles']}", flush=True)
    print(f"[INFO] wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
