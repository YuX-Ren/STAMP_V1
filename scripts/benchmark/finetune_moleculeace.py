#!/usr/bin/env python3
"""MoleculeACE benchmark fine-tune for STAMP encoder.

Each MoleculeACE CSV (Tilborg et al., JCIM 2022) has columns:
    smiles, exp_mean [nM], y, cliff_mol, split, y [pEC50/pKi]

We use `y [pEC50/pKi]` as the regression target and report:
    test_RMSE_all     — RMSE on all test molecules
    test_RMSE_cliff   — RMSE on the cliff_mol==1 subset of test

Usage:
    python scripts/benchmark/finetune_moleculeace.py \
        --pretrained-dir /path/to/pretrain_stamp_top4666_mtr \
        --moleculeace-dir /tmp/MoleculeACE/MoleculeACE/Data/benchmark_data \
        --output-dir /scratch/.../moleculeace_stamp_mtr
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.roberta_property import (  # noqa: E402
    load_pretrained_roberta, RegressionHead, _resolve_amp_mode,
)
from stamp.benchmark.common import canonicalize_smiles  # noqa: E402


def encode_smiles_to_ids(rep, vocab, max_len: int, smiles: List[str]) -> Tuple[List[List[int]], List[int]]:
    ids_list, kept = [], []
    for i, smi in enumerate(smiles):
        try:
            text = rep.encode(smi)
            toks = rep.tokenize(text)
            if not toks:
                continue
            ids = [vocab.bos_id] + vocab.encode_tokens(toks) + [vocab.eos_id]
            ids = ids[:max_len]
            ids_list.append(ids)
            kept.append(i)
        except Exception:
            continue
    return ids_list, kept


class _CSVRegressionDataset(Dataset):
    def __init__(self, ids_list, labels, max_len, pad_id):
        self.samples = [(ids, float(y)) for ids, y in zip(ids_list, labels) if len(ids) >= 2]
        self.max_len = max_len
        self.pad_id = pad_id

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        ids, y = self.samples[i]
        return ids[: self.max_len], y


def collate_reg(pad_id):
    def f(batch):
        max_len = max(len(x) for x, _ in batch)
        bsz = len(batch)
        x_out = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        y_out = torch.zeros(bsz, dtype=torch.float)
        for i, (x, y) in enumerate(batch):
            x_out[i, : len(x)] = torch.tensor(x, dtype=torch.long)
            y_out[i] = y
        return x_out, y_out
    return f


def finetune_one(csv_path: Path, encoder, vocab, rep, cfg, dev,
                 epochs: int, lr: float, bs: int, dropout: float, amp: str, seed: int):
    df = pd.read_csv(csv_path)
    target_col = "y [pEC50/pKi]"
    if target_col not in df.columns:
        raise KeyError(f"{csv_path}: no '{target_col}' column")
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    # Canonicalize + drop invalid
    def _clean(d):
        smis, ys, cliff = [], [], []
        for _, row in d.iterrows():
            can = canonicalize_smiles(str(row["smiles"]), strip_salts=True)
            if can is None:
                continue
            y = float(row[target_col])
            if not math.isfinite(y):
                continue
            smis.append(can)
            ys.append(y)
            cliff.append(int(row["cliff_mol"]))
        return smis, np.asarray(ys), np.asarray(cliff)

    tr_smi, tr_y, _ = _clean(train_df)
    te_smi, te_y, te_cliff = _clean(test_df)

    # Encode
    max_len = cfg.max_seq_len
    tr_ids, tr_keep = encode_smiles_to_ids(rep, vocab, max_len, tr_smi)
    te_ids, te_keep = encode_smiles_to_ids(rep, vocab, max_len, te_smi)
    tr_y = tr_y[tr_keep]
    te_y = te_y[te_keep]
    te_cliff = te_cliff[te_keep]

    if len(tr_ids) < 10 or len(te_ids) < 5:
        return {"task": csv_path.stem, "error": f"insufficient data: train={len(tr_ids)} test={len(te_ids)}"}

    # z-norm using train stats
    mean = float(np.mean(tr_y))
    std = float(np.std(tr_y))
    if std < 1e-8: std = 1.0
    tr_y_norm = (tr_y - mean) / std
    te_y_norm = (te_y - mean) / std

    torch.manual_seed(seed)
    np.random.seed(seed)
    head = RegressionHead(cfg.d_model, dropout=dropout).to(dev)
    encoder.train()

    ds_tr = _CSVRegressionDataset(tr_ids, tr_y_norm, max_len, vocab.pad_id)
    ds_te = _CSVRegressionDataset(te_ids, te_y_norm, max_len, vocab.pad_id)
    cf = collate_reg(vocab.pad_id)
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=False, collate_fn=cf, num_workers=2, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=bs, shuffle=False, collate_fn=cf, num_workers=2, pin_memory=True)

    use_amp, amp_dtype, use_scaler, _ = _resolve_amp_mode(amp, dev)
    params = list(encoder.parameters()) + list(head.parameters())
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    loss_fn = nn.MSELoss()

    best_state = None
    best_valid = float("inf")
    for ep in range(1, epochs + 1):
        encoder.train(); head.train()
        for x, y in dl_tr:
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True) if use_amp else nullcontext()
            with ctx:
                cls_repr = encoder(x, return_cls=True)
                pred = head(cls_repr).squeeze(-1)
                loss = loss_fn(pred, y)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optim); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()
        # Eval
        encoder.eval(); head.eval()
        preds = []
        with torch.no_grad():
            for x, _ in dl_te:
                x = x.to(dev)
                ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True) if use_amp else nullcontext()
                with ctx:
                    cls_repr = encoder(x, return_cls=True)
                    p = head(cls_repr).squeeze(-1).float()
                preds.extend(p.cpu().tolist())
        preds = np.asarray(preds)
        # Un-normalize
        pred_raw = preds * std + mean
        true_raw = te_y[: len(pred_raw)]
        rmse_all = float(np.sqrt(np.mean((pred_raw - true_raw) ** 2)))
        cliff_idx = te_cliff[: len(pred_raw)] == 1
        if cliff_idx.sum() > 0:
            rmse_cliff = float(np.sqrt(np.mean((pred_raw[cliff_idx] - true_raw[cliff_idx]) ** 2)))
        else:
            rmse_cliff = float("nan")
        if rmse_all < best_valid:
            best_valid = rmse_all
            best_rmse_cliff = rmse_cliff
            best_epoch = ep

    return {
        "task": csv_path.stem,
        "n_train": len(tr_ids), "n_test": len(te_ids),
        "n_test_cliff": int(te_cliff[: len(pred_raw)].sum()),
        "test_rmse_all": best_valid,
        "test_rmse_cliff": best_rmse_cliff,
        "best_epoch": best_epoch,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained-dir", required=True)
    ap.add_argument("--moleculeace-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--amp", default="bf16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"Loading encoder from {args.pretrained_dir} ...", flush=True)
    encoder, vocab, rep, cfg, dev = load_pretrained_roberta(args.pretrained_dir, device=args.device)
    encoder.to(dev)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ace_dir = Path(args.moleculeace_dir)
    csvs = sorted(ace_dir.glob("CHEMBL*_*.csv"))
    print(f"Found {len(csvs)} MoleculeACE CSVs", flush=True)

    results = []
    for i, csv in enumerate(csvs, 1):
        t0 = time.time()
        # Re-load encoder fresh for each task (fine-tuning mutates weights)
        encoder, vocab, rep, cfg, dev = load_pretrained_roberta(args.pretrained_dir, device=args.device)
        encoder.to(dev)
        try:
            res = finetune_one(
                csv, encoder, vocab, rep, cfg, dev,
                epochs=args.epochs, lr=args.lr, bs=args.bs, dropout=args.dropout,
                amp=args.amp, seed=args.seed,
            )
        except Exception as e:
            res = {"task": csv.stem, "error": str(e)}
        res["elapsed"] = round(time.time() - t0, 1)
        print(f"[{i:2d}/{len(csvs)}] {res}", flush=True)
        results.append(res)
        # Persist incrementally
        (out_dir / "moleculeace_results.json").write_text(json.dumps(results, indent=2))

    # Summary
    valid = [r for r in results if "error" not in r]
    if valid:
        rmse_all = np.array([r["test_rmse_all"] for r in valid])
        rmse_clf = np.array([r["test_rmse_cliff"] for r in valid if not math.isnan(r["test_rmse_cliff"])])
        print(f"\n=== Summary over {len(valid)} tasks ===")
        print(f"  test_rmse_all  mean={rmse_all.mean():.3f}  std={rmse_all.std():.3f}")
        print(f"  test_rmse_cliff mean={rmse_clf.mean():.3f}  std={rmse_clf.std():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
