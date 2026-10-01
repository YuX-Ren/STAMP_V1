#!/usr/bin/env python3
"""ChemBERTa-2 (HuggingFace) MoleculeACE 30-task fine-tune.

Mirrors `finetune_moleculeace.py` but uses a HF RoBERTa encoder
(default: seyonec/PubChem10M_SMILES_BPE_60k) instead of our STAMP
encoder, so the comparison is apples-to-apples on the same fine-tune
protocol (epochs / LR / batch / dropout / AMP).
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Tuple

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

from stamp.benchmark.common import canonicalize_smiles  # noqa: E402


class _CSVRegressionDataset(Dataset):
    def __init__(self, ids_list, attn_list, labels):
        assert len(ids_list) == len(attn_list) == len(labels)
        self.samples = [
            (ids, attn, float(y))
            for ids, attn, y in zip(ids_list, attn_list, labels)
            if len(ids) >= 2
        ]

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate_reg(pad_id):
    def f(batch):
        max_len = max(len(x) for x, _, _ in batch)
        bsz = len(batch)
        x_out = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        a_out = torch.zeros((bsz, max_len), dtype=torch.long)
        y_out = torch.zeros(bsz, dtype=torch.float)
        for i, (x, a, y) in enumerate(batch):
            x_out[i, : len(x)] = torch.tensor(x, dtype=torch.long)
            a_out[i, : len(a)] = torch.tensor(a, dtype=torch.long)
            y_out[i] = y
        return x_out, a_out, y_out
    return f


class ChemBERTaRegressor(nn.Module):
    def __init__(self, base_model, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = base_model
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        x = self.dropout(cls)
        x = nn.functional.gelu(self.dense(x))
        x = self.dropout(x)
        return self.out(x).squeeze(-1)


def finetune_one(csv_path: Path, model_name: str, max_len: int,
                 epochs: int, lr: float, bs: int, dropout: float, amp: str, seed: int, dev: torch.device):
    from transformers import AutoTokenizer, AutoModel
    df = pd.read_csv(csv_path)
    target_col = "y [pEC50/pKi]"
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    def _clean(d):
        smis, ys, cliff = [], [], []
        for _, row in d.iterrows():
            can = canonicalize_smiles(str(row["smiles"]), strip_salts=True)
            if can is None: continue
            y = float(row[target_col])
            if not math.isfinite(y): continue
            smis.append(can); ys.append(y); cliff.append(int(row["cliff_mol"]))
        return smis, np.asarray(ys), np.asarray(cliff)

    tr_smi, tr_y, _ = _clean(train_df)
    te_smi, te_y, te_cliff = _clean(test_df)

    if len(tr_smi) < 10 or len(te_smi) < 5:
        return {"task": csv_path.stem, "error": f"insufficient data"}

    tok = AutoTokenizer.from_pretrained(model_name)
    base = AutoModel.from_pretrained(model_name)
    hidden = base.config.hidden_size
    pad_id = tok.pad_token_id

    def encode_batch(smi_list):
        enc = tok(smi_list, max_length=max_len, padding=False, truncation=True, return_attention_mask=True)
        return enc["input_ids"], enc["attention_mask"]

    tr_ids, tr_attn = encode_batch(tr_smi)
    te_ids, te_attn = encode_batch(te_smi)

    mean = float(np.mean(tr_y)); std = float(np.std(tr_y))
    if std < 1e-8: std = 1.0
    tr_y_norm = (tr_y - mean) / std
    te_y_norm = (te_y - mean) / std

    torch.manual_seed(seed); np.random.seed(seed)
    model = ChemBERTaRegressor(base, hidden, dropout=dropout).to(dev)

    cf = collate_reg(pad_id)
    ds_tr = _CSVRegressionDataset(tr_ids, tr_attn, tr_y_norm)
    ds_te = _CSVRegressionDataset(te_ids, te_attn, te_y_norm)
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=False, collate_fn=cf, num_workers=2, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=bs, shuffle=False, collate_fn=cf, num_workers=2, pin_memory=True)

    use_amp = amp != "none"
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16 if amp == "fp16" else torch.float32
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.MSELoss()

    best_valid = float("inf")
    best_rmse_cliff = float("nan")
    best_epoch = 0
    for ep in range(1, epochs + 1):
        model.train()
        for x, a, y in dl_tr:
            x = x.to(dev, non_blocking=True); a = a.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True) if use_amp else nullcontext()
            with ctx:
                pred = model(x, a)
                loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        # Eval
        model.eval()
        preds = []
        with torch.no_grad():
            for x, a, _ in dl_te:
                x = x.to(dev); a = a.to(dev)
                ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True) if use_amp else nullcontext()
                with ctx:
                    p = model(x, a).float()
                preds.extend(p.cpu().tolist())
        preds = np.asarray(preds)
        pred_raw = preds * std + mean
        true_raw = te_y[: len(pred_raw)]
        rmse_all = float(np.sqrt(np.mean((pred_raw - true_raw) ** 2)))
        cmask = te_cliff[: len(pred_raw)] == 1
        rmse_cliff = float(np.sqrt(np.mean((pred_raw[cmask] - true_raw[cmask]) ** 2))) if cmask.sum() else float("nan")
        if rmse_all < best_valid:
            best_valid = rmse_all
            best_rmse_cliff = rmse_cliff
            best_epoch = ep
    return {
        "task": csv_path.stem,
        "n_train": len(tr_smi), "n_test": len(te_smi),
        "n_test_cliff": int(te_cliff[: len(pred_raw)].sum()),
        "test_rmse_all": best_valid,
        "test_rmse_cliff": best_rmse_cliff,
        "best_epoch": best_epoch,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="seyonec/PubChem10M_SMILES_BPE_60k",
                    help="HuggingFace ChemBERTa model id (default: PubChem-10M MLM, comparable to our STAMP-MLM scale).")
    ap.add_argument("--moleculeace-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--amp", default="bf16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ace_dir = Path(args.moleculeace_dir)
    csvs = sorted(ace_dir.glob("CHEMBL*_*.csv"))
    print(f"Found {len(csvs)} CSVs. Model={args.model_name}", flush=True)

    results = []
    for i, csv in enumerate(csvs, 1):
        t0 = time.time()
        try:
            res = finetune_one(
                csv, args.model_name, args.max_len,
                epochs=args.epochs, lr=args.lr, bs=args.bs, dropout=args.dropout,
                amp=args.amp, seed=args.seed, dev=dev,
            )
        except Exception as e:
            res = {"task": csv.stem, "error": str(e)}
        res["elapsed"] = round(time.time() - t0, 1)
        print(f"[{i:2d}/{len(csvs)}] {res}", flush=True)
        results.append(res)
        (out_dir / "moleculeace_results.json").write_text(json.dumps(results, indent=2))

    valid = [r for r in results if "error" not in r]
    if valid:
        rmse_all = np.array([r["test_rmse_all"] for r in valid])
        rmse_clf = np.array([r["test_rmse_cliff"] for r in valid if not math.isnan(r["test_rmse_cliff"])])
        print(f"\n=== Summary over {len(valid)} tasks ===")
        print(f"  test_rmse_all   mean={rmse_all.mean():.3f}  std={rmse_all.std():.3f}")
        print(f"  test_rmse_cliff mean={rmse_clf.mean():.3f}  std={rmse_clf.std():.3f}")


if __name__ == "__main__":
    main()
