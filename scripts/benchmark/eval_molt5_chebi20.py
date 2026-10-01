#!/usr/bin/env python3
"""Evaluate a finetuned MolT5 checkpoint on ChEBI-20 test.

Two directions selected by --direction:
  - mol2text: source=STAMP tokens, target=English description.
              Metrics: BLEU-2, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, METEOR.
  - text2mol: source=English description, target=STAMP tokens.
              Predictions decoded STAMP -> SMILES via STAMPRepresentation, then
              Metrics: validity, exact-match, BLEU-4, Levenshtein,
              MACCS/RDK/Morgan Tanimoto (over valid predictions).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_test(parquet_path: str, source_col: str, target_col: str, gold_smiles_col: str | None):
    tbl = pq.read_table(parquet_path)
    src = tbl.column(source_col).to_pylist()
    tgt = tbl.column(target_col).to_pylist()
    gold_smi = tbl.column(gold_smiles_col).to_pylist() if gold_smiles_col else None
    cid = tbl.column("cid").to_pylist() if "cid" in tbl.schema.names else [str(i) for i in range(len(src))]
    return cid, src, tgt, gold_smi


_STAMP_STRUCTURAL_EXACT = frozenset({
    "[Ring]", "[END]", "[BACK]", "[BOS]", "[EOS]", "[CONCAT]", ".",
    "[S_R]", "[S_S]", "[S_E]", "[S_Z]",
})


def _is_stamp_structural(tok: str) -> bool:
    return tok in _STAMP_STRUCTURAL_EXACT or tok.startswith(("[J_", "[B_", "[S_"))


def _decode_output(tokenizer, output_ids, *, stamp_mode: bool):
    """Decode T5 output ids.

    stamp_mode=True: for STAMP targets. The tokenizer holds two kinds of
    single-id entries: STAMP structural tokens ([J_*]/[B_*]/[S_*]/[Ring]/...)
    and motif SMILES tokens like 'CCO' or 'OC[C@@H]1C[C@H](O)CCO1' — plus a
    T5 SentencePiece base for OOV motifs which get emitted as subword pieces
    (e.g. 'O=[As]' -> ['▁O', '=', '[', 'As', '▁', ']']). The STAMP decoder
    expects each MOTIF to be one whitespace-separated token, so we merge
    every consecutive run of non-structural pieces via
    ``convert_tokens_to_string`` (handles ▁ properly) before space-joining
    with structural tokens.
    stamp_mode=False: plain batch_decode for English text output.
    """
    if not stamp_mode:
        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    outs = []
    special_ids = set(tokenizer.all_special_ids)
    for row in output_ids:
        parts: list[str] = []
        current: list[str] = []
        for tid in row.tolist():
            if tid in special_ids:
                continue
            tok = tokenizer.convert_ids_to_tokens(int(tid))
            if tok is None or tok == "":
                continue
            if _is_stamp_structural(tok):
                if current:
                    motif = tokenizer.convert_tokens_to_string(current).replace(" ", "").strip()
                    if motif:
                        parts.append(motif)
                    current = []
                parts.append(tok)
            else:
                current.append(tok)
        if current:
            motif = tokenizer.convert_tokens_to_string(current).replace(" ", "").strip()
            if motif:
                parts.append(motif)
        outs.append(" ".join(parts))
    return outs


def _generate_batch(model, tokenizer, srcs, *, max_source_len: int, max_new_tokens: int, num_beams: int, device, stamp_mode: bool):
    enc = tokenizer(list(srcs), max_length=max_source_len, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            do_sample=False,
        )
    return _decode_output(tokenizer, out, stamp_mode=stamp_mode)


def _mol2text_metrics(preds: list[str], refs: list[str]) -> dict:
    """Report both a whitespace-split and the MolT5-official SciBERT-tokenized
    versions of BLEU / METEOR. ROUGE uses raw strings on both.
    """
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import single_meteor_score, meteor_score
    from rouge_score import rouge_scorer
    from transformers import BertTokenizerFast

    smooth = SmoothingFunction().method1
    # --- whitespace-split BLEU / METEOR ---
    ref_tokens = [[r.split()] for r in refs]
    pred_tokens = [p.split() for p in preds]
    bleu2_ws = corpus_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu4_ws = corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    meteor_ws = [single_meteor_score(r.split(), p.split()) for p, r in zip(preds, refs)]

    # --- SciBERT wordpiece BLEU / METEOR (MolT5 official recipe) ---
    tk = BertTokenizerFast.from_pretrained("allenai/scibert_scivocab_uncased")
    trunc = 512
    scibert_refs, scibert_hyps = [], []
    for p, r in zip(preds, refs):
        gt_t = tk.tokenize(r, truncation=True, max_length=trunc, padding="max_length")
        gt_t = [t for t in gt_t if t not in ("[PAD]", "[CLS]", "[SEP]")]
        ot_t = tk.tokenize(p, truncation=True, max_length=trunc, padding="max_length")
        ot_t = [t for t in ot_t if t not in ("[PAD]", "[CLS]", "[SEP]")]
        scibert_refs.append([gt_t])
        scibert_hyps.append(ot_t)
    bleu2_sci = corpus_bleu(scibert_refs, scibert_hyps, weights=(0.5, 0.5, 0, 0))
    bleu4_sci = corpus_bleu(scibert_refs, scibert_hyps, weights=(0.25, 0.25, 0.25, 0.25))
    meteor_sci = [meteor_score(refs_list, hyp) for refs_list, hyp in zip(scibert_refs, scibert_hyps)]

    # --- ROUGE on raw strings ---
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)

    return {
        "n": len(preds),
        # MolT5 official (SciBERT wordpiece) — primary numbers for comparison
        "BLEU-2": float(bleu2_sci),
        "BLEU-4": float(bleu4_sci),
        "METEOR": float(sum(meteor_sci) / len(meteor_sci)),
        "ROUGE-1": float(sum(r1) / len(r1)),
        "ROUGE-2": float(sum(r2) / len(r2)),
        "ROUGE-L": float(sum(rL) / len(rL)),
        # Also report whitespace-split for reference
        "BLEU-2_whitespace": float(bleu2_ws),
        "BLEU-4_whitespace": float(bleu4_ws),
        "METEOR_whitespace": float(sum(meteor_ws) / len(meteor_ws)),
    }


def _canon(smi: str):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True), m


def _text2mol_metrics(pred_stamps: list[str], gold_smiles: list[str], *, vocab_path: str) -> dict:
    from stamp.benchmark.representations import STAMPRepresentation
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, MACCSkeys
    from rdkit.DataStructs import TanimotoSimilarity
    import Levenshtein
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    RDLogger.DisableLog("rdApp.*")
    rep = STAMPRepresentation(vocab_path=vocab_path)

    # Decode each predicted STAMP -> SMILES
    pred_smiles: list[str | None] = []
    n_decode_fail = 0
    for st in pred_stamps:
        try:
            pred_smiles.append(rep.decode(st))
        except Exception:
            pred_smiles.append(None)
            n_decode_fail += 1

    # Canonicalize gold
    gold_canon = []
    gold_mols = []
    for s in gold_smiles:
        c = _canon(s)
        if c is None:
            gold_canon.append(None); gold_mols.append(None)
        else:
            gold_canon.append(c[0]); gold_mols.append(c[1])

    # Validity: predicted SMILES parses to a valid mol
    n_valid = 0
    valid_pairs = []      # (pred_canon, pred_mol, gold_canon, gold_mol, orig_pred_smi)
    for ps, gc, gm in zip(pred_smiles, gold_canon, gold_mols):
        if ps is None:
            continue
        c = _canon(ps)
        if c is None:
            continue
        n_valid += 1
        valid_pairs.append((c[0], c[1], gc, gm, ps))

    n_total = len(pred_stamps)
    validity = n_valid / n_total

    # Exact match (over ALL rows: unparseable predictions count as no-match)
    n_exact = sum(1 for (pc, _, gc, _, _) in valid_pairs if gc is not None and pc == gc)
    exact = n_exact / n_total

    # BLEU-4 on character-level (as in MolT5 paper: SMILES character tokens)
    smooth = SmoothingFunction().method1
    pred_char_tokens = [list(ps if ps else "") for ps in pred_smiles]
    gold_char_tokens = [[list(gc)] if gc else [[""]] for gc in gold_canon]
    bleu4_smiles = corpus_bleu(gold_char_tokens, pred_char_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    # Levenshtein (over all rows, dist over char strings; use empty for None)
    lev = [Levenshtein.distance((ps or ""), (gc or "")) for ps, gc in zip(pred_smiles, gold_canon)]
    lev_mean = sum(lev) / len(lev)

    # Fingerprint similarities: only over valid_pairs where BOTH gold and pred parseable
    maccs, rdk, morgan = [], [], []
    for pc, pm, gc, gm, _ in valid_pairs:
        if gm is None:
            continue
        try:
            m_p, m_g = MACCSkeys.GenMACCSKeys(pm), MACCSkeys.GenMACCSKeys(gm)
            maccs.append(TanimotoSimilarity(m_p, m_g))
        except Exception:
            pass
        try:
            rdk_p, rdk_g = Chem.RDKFingerprint(pm), Chem.RDKFingerprint(gm)
            rdk.append(TanimotoSimilarity(rdk_p, rdk_g))
        except Exception:
            pass
        try:
            mo_p = AllChem.GetMorganFingerprintAsBitVect(pm, 2, nBits=2048)
            mo_g = AllChem.GetMorganFingerprintAsBitVect(gm, 2, nBits=2048)
            morgan.append(TanimotoSimilarity(mo_p, mo_g))
        except Exception:
            pass

    def _mean(xs): return float(sum(xs) / len(xs)) if xs else 0.0

    return {
        "n": n_total,
        "decode_fail": n_decode_fail,
        "validity": validity,
        "exact_match": exact,
        "BLEU-4": float(bleu4_smiles),
        "Levenshtein_mean": lev_mean,
        "MACCS_FTS": _mean(maccs),
        "RDK_FTS": _mean(rdk),
        "Morgan_FTS": _mean(morgan),
        "n_valid_pairs": len(valid_pairs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--direction", choices=["mol2text", "text2mol"], required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--stamp-vocab", default="/scratch/yuxuan.ren/data/ckpt_stamp/zinc_curated_vocab_30k_legacy/motif_vocab.txt")
    ap.add_argument("--max-source-len", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 = full test.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoTokenizer, T5ForConditionalGeneration

    print(f"[eval] loading model from {args.ckpt_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(args.ckpt_dir, local_files_only=True)
    device = torch.device(args.device)
    model.to(device).eval()

    src_col = "mol2text_source" if args.direction == "mol2text" else "text2mol_source"
    tgt_col = "mol2text_target" if args.direction == "mol2text" else "text2mol_target"
    gold_smi_col = "smiles" if args.direction == "text2mol" else None
    cid, srcs, tgts, gold_smi = _load_test(args.test_parquet, src_col, tgt_col, gold_smi_col)
    if args.limit > 0:
        cid = cid[:args.limit]; srcs = srcs[:args.limit]; tgts = tgts[:args.limit]
        gold_smi = gold_smi[:args.limit] if gold_smi else None
    print(f"[eval] test rows: {len(srcs)}  direction={args.direction}", flush=True)

    preds: list[str] = []
    t0 = time.time()
    for i in range(0, len(srcs), args.batch_size):
        batch = srcs[i:i + args.batch_size]
        out = _generate_batch(model, tokenizer, batch,
                              max_source_len=args.max_source_len,
                              max_new_tokens=args.max_new_tokens,
                              num_beams=args.num_beams,
                              device=device,
                              stamp_mode=(args.direction == "text2mol"))
        preds.extend(out)
        if (i // args.batch_size) % 10 == 0:
            dt = max(1e-6, time.time() - t0)
            print(f"[eval] generated {min(i + args.batch_size, len(srcs))}/{len(srcs)}  rate={(i + args.batch_size)/dt:.1f}/s", flush=True)
    dt = time.time() - t0
    print(f"[eval] generation wall: {dt:.1f}s  rate={len(srcs)/dt:.1f}/s", flush=True)

    if args.direction == "mol2text":
        metrics = _mol2text_metrics(preds, tgts)
    else:
        metrics = _text2mol_metrics(preds, gold_smi, vocab_path=args.stamp_vocab)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save side-by-side predictions
    csv_path = out_dir / f"predictions_{args.direction}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["cid", "source", "gold_target", "prediction"]
        if gold_smi is not None:
            header.insert(2, "gold_smiles")
        w.writerow(header)
        for i, (c, s, t, p) in enumerate(zip(cid, srcs, tgts, preds)):
            # Store FULL strings — earlier version truncated to 400 chars, which
            # broke STAMP grammar on ~8% of long molecules and made offline
            # metric recomputation from this CSV misleading.
            row = [c, s, t, p]
            if gold_smi is not None:
                row.insert(2, (gold_smi[i] or ""))
            w.writerow(row)

    metrics_path = out_dir / f"metrics_{args.direction}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print()
    print(f"=== {args.direction} metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20} {v:.4f}")
        else:
            print(f"  {k:<20} {v}")
    print(f"\npredictions -> {csv_path}")
    print(f"metrics     -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
